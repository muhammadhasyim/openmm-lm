#!/usr/bin/env python3
"""Demonstrate leapfrog VerletIntegrator injects energy when dt changes externally.

OpenMM's plain VerletIntegrator stores half-step velocities (leapfrog).  When
adaptive timestepping calls setStepSize() between steps, that state is silently
reinterpreted and spurious kinetic energy accumulates.  The velocity-Verlet
CustomIntegrator from cavitymd.adaptive is immune to this artifact.

Run on Reference platform for deterministic energy accounting.
"""

from __future__ import annotations

import sys

import numpy as np
import openmm
from openmm import unit

N_STEPS = 20_000
# Stiff harmonic bond (order of real bond stiffness)
K_BOND = 500_000.0
R0_NM = 0.1
MASS_AMU = 12.0
# Toggle schedule mimicking adaptive recalibration
TOGGLE_SMALL_PS = 0.0005
TOGGLE_LARGE_PS = 0.001
TOGGLE_PERIOD = 100
# Velocity-Verlet drift must stay within this factor of fixed-dt control
VV_DRIFT_FACTOR = 5.0
# Leapfrog must exceed this factor vs fixed-dt (proves the artifact exists)
LF_DRIFT_FACTOR = 3.0


def create_velocity_verlet_integrator(dt_ps: float) -> openmm.CustomIntegrator:
    """Must match openmm.cavitymd.adaptive.create_velocity_verlet_integrator."""
    integrator = openmm.CustomIntegrator(dt_ps * unit.picosecond)
    integrator.addPerDofVariable("x1", 0)
    integrator.addUpdateContextState()
    integrator.addComputePerDof("v", "v + 0.5*dt*f/m")
    integrator.addComputePerDof("x", "x + dt*v")
    integrator.addComputePerDof("x1", "x")
    integrator.addConstrainPositions()
    integrator.addComputePerDof("v", "v + 0.5*dt*f/m + (x-x1)/dt")
    integrator.addConstrainVelocities()
    return integrator


def _build_system() -> openmm.System:
    system = openmm.System()
    system.addParticle(MASS_AMU)
    system.addParticle(MASS_AMU)
    force = openmm.HarmonicBondForce()
    force.addBond(
        0,
        1,
        R0_NM * unit.nanometer,
        K_BOND * unit.kilojoule_per_mole / unit.nanometer**2,
    )
    system.addForce(force)
    return system


def _make_leapfrog_verlet(dt0_ps: float) -> openmm.VerletIntegrator:
    return openmm.VerletIntegrator(dt0_ps * unit.picosecond)


def _total_energy_kj_mol(context) -> float:
    state = context.getState(getEnergy=True)
    pe = state.getPotentialEnergy().value_in_unit(unit.kilojoule_per_mole)
    ke = state.getKineticEnergy().value_in_unit(unit.kilojoule_per_mole)
    return float(pe + ke)


def _run_schedule(
    integrator_factory,
    dt_schedule,
    label: str,
) -> tuple[float, float]:
    system = _build_system()
    integrator = integrator_factory(TOGGLE_SMALL_PS)
    platform = openmm.Platform.getPlatformByName("Reference")
    context = openmm.Context(system, integrator, platform)
    context.setPositions(
        [
            openmm.Vec3(0.0, 0.0, 0.0) * unit.nanometer,
            openmm.Vec3(0.13, 0.0, 0.0) * unit.nanometer,
        ]
    )
    context.setVelocities(
        [
            openmm.Vec3(0.0, 0.0, 0.0) * (unit.nanometer / unit.picosecond),
            openmm.Vec3(0.0, 0.0, 0.0) * (unit.nanometer / unit.picosecond),
        ]
    )

    e0 = _total_energy_kj_mol(context)
    for step in range(N_STEPS):
        dt_ps = dt_schedule(step)
        integrator.setStepSize(dt_ps * unit.picosecond)
        integrator.step(1)
    e_final = _total_energy_kj_mol(context)
    drift = e_final - e0
    print(f"{label}: E0={e0:.4f} Efinal={e_final:.4f} drift={drift:+.4f} kJ/mol")
    return e0, drift


def _toggle_schedule(step: int) -> float:
    if (step // TOGGLE_PERIOD) % 2 == 0:
        return TOGGLE_SMALL_PS
    return TOGGLE_LARGE_PS


def _fixed_schedule(_step: int) -> float:
    return TOGGLE_LARGE_PS


def main() -> int:
    print(f"=== dt-churn energy injection test ({N_STEPS} steps, Reference) ===\n")

    _, drift_fixed_vv = _run_schedule(
        create_velocity_verlet_integrator,
        _fixed_schedule,
        "Velocity-Verlet fixed dt=1.0fs",
    )
    _, drift_toggle_vv = _run_schedule(
        create_velocity_verlet_integrator,
        _toggle_schedule,
        "Velocity-Verlet toggling 0.5/1.0fs every 100 steps",
    )
    _, drift_fixed_lf = _run_schedule(
        _make_leapfrog_verlet,
        _fixed_schedule,
        "Leapfrog VerletIntegrator fixed dt=1.0fs",
    )
    _, drift_toggle_lf = _run_schedule(
        _make_leapfrog_verlet,
        _toggle_schedule,
        "Leapfrog VerletIntegrator toggling 0.5/1.0fs every 100 steps",
    )

    abs_fixed_vv = abs(drift_fixed_vv)
    abs_toggle_vv = abs(drift_toggle_vv)
    abs_fixed_lf = abs(drift_fixed_lf)
    abs_toggle_lf = abs(drift_toggle_lf)

    print()
    vv_ok = abs_toggle_vv <= max(abs_fixed_vv * VV_DRIFT_FACTOR, 5.0)
    lf_bad = abs_toggle_lf >= max(abs_fixed_lf * LF_DRIFT_FACTOR, 5.0)

    print(f"Velocity-Verlet toggle drift / fixed drift = {abs_toggle_vv / max(abs_fixed_vv, 1e-9):.2f}x")
    print(f"Leapfrog toggle drift / fixed drift       = {abs_toggle_lf / max(abs_fixed_lf, 1e-9):.2f}x")

    if vv_ok and lf_bad:
        print(
            "\nPASS: velocity-Verlet is stable under dt churn; "
            "leapfrog VerletIntegrator injects spurious energy."
        )
        return 0

    print("\nFAIL:")
    if not vv_ok:
        print(
            f"  velocity-Verlet toggle drift {abs_toggle_vv:.4f} exceeds "
            f"{VV_DRIFT_FACTOR}x fixed-dt control {abs_fixed_vv:.4f}"
        )
    if not lf_bad:
        print(
            f"  leapfrog toggle drift {abs_toggle_lf:.4f} did not exceed "
            f"{LF_DRIFT_FACTOR}x fixed-dt control {abs_fixed_lf:.4f} (artifact not reproduced)"
        )
    return 1


if __name__ == "__main__":
    sys.exit(main())
