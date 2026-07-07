#!/usr/bin/env python3
"""Regression: CavityForce coupling turn-on and CavityParticleDisplacer fire together.

Before the fix, coupling activated one step before displaceToEquilibrium, injecting a
large impulse on the ultralight photon and blowing up simulations at realistic lambda.
"""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pytest

DIMER_DIR = Path(__file__).resolve().parents[2] / "examples" / "cavity" / "dimer_system"
sys.path.insert(0, str(DIMER_DIR))

openmm = pytest.importorskip("openmm")
from openmm import unit  # noqa: E402

import run_simulation as rs  # noqa: E402


def _select_platform() -> openmm.Platform:
    for name in ("CUDA", "Reference"):
        try:
            platform = openmm.Platform.getPlatformByName(name)
            if name == "CUDA":
                platform.setPropertyDefaultValue("Precision", "mixed")
            return platform
        except Exception:
            continue
    return openmm.Platform.getPlatformByName("Reference")


def _run_switch_test(*, lambda_coupling: float, switch_step: int, n_steps: int, seed: int = 42):
    # Need enough dimers so constant-density box exceeds 2*rcut (~1.59 nm).
    num_molecules = 120
    box_size_nm = rs.box_size_nm_at_constant_density(num_molecules)
    omegac_au = 1560.0 / 219474.63
    photon_mass = 1.0 / 1822.888
    temperature_K = 100.0
    dt_ps = 0.001

    result = rs.create_diamer_system_from_forcefield(
        num_molecules=num_molecules,
        fraction_OO=0.8,
        box_size_nm=box_size_nm,
        seed=seed,
        include_cavity=True,
    )
    system, positions, _topology, cavity_index = result
    system.setParticleMass(cavity_index, photon_mass)

    cavity_force = openmm.CavityForce(cavity_index, omegac_au, 0.0, photon_mass)
    cavity_force.setCouplingOnStep(switch_step, lambda_coupling)
    system.addForce(cavity_force)

    displacer = openmm.CavityParticleDisplacer(cavity_index, omegac_au, photon_mass)
    displacer.setSwitchOnLambda(lambda_coupling)
    displacer.setSwitchOnStep(switch_step)
    system.addForce(displacer)

    bussi = openmm.BussiThermostat(temperature_K, 1.0)
    bussi.setApplyToAllParticles(False)
    for i in range(cavity_index):
        bussi.addParticle(i)
    system.addForce(bussi)

    integrator = openmm.VerletIntegrator(dt_ps * unit.picosecond)
    platform = _select_platform()
    context = openmm.Context(system, integrator, platform)
    context.setPositions(positions)
    context.setVelocitiesToTemperature(temperature_K * unit.kelvin, seed)

    openmm.LocalEnergyMinimizer.minimize(context, maxIterations=50)

    pe_before = None
    pe_at_switch = None
    pe_after = None
    coupling_after = None
    q_at_switch = None
    max_q_xy = 0.0

    for step in range(n_steps + 1):
        state = context.getState(getEnergy=True, getPositions=True)
        pe = state.getPotentialEnergy().value_in_unit(unit.kilojoule_per_mole)
        assert np.isfinite(pe), f"NaN/inf PE at step {step}"

        q_xy = state.getPositions(asNumpy=True)[cavity_index, :2].value_in_unit(
            unit.nanometer
        )
        max_q_xy = max(max_q_xy, float(np.linalg.norm(q_xy)))

        if step == switch_step - 1:
            pe_before = pe
        if step == switch_step:
            pe_at_switch = pe
            q_at_switch = float(np.linalg.norm(q_xy))
            coupling_after = cavity_force.getCouplingEnergy(context).value_in_unit(
                unit.kilojoule_per_mole
            )
        if step == switch_step + 5:
            pe_after = pe

        if step < n_steps:
            integrator.step(1)

    ke = context.getState(getEnergy=True).getKineticEnergy().value_in_unit(
        unit.kilojoule_per_mole
    )
    assert np.isfinite(ke), "Kinetic energy blew up after switch"

    return {
        "platform": platform.getName(),
        "pe_before": pe_before,
        "pe_at_switch": pe_at_switch,
        "pe_after": pe_after,
        "coupling_after": coupling_after,
        "q_at_switch": q_at_switch,
        "max_q_xy": max_q_xy,
        "pe_jump": (pe_at_switch - pe_before) if pe_before is not None and pe_at_switch is not None else None,
    }


@pytest.mark.timeout(180)
def test_instant_switch_energy_continuity_realistic_lambda():
    """Coupling and displacer fire together: finite PE, active coupling, equilibrium q at switch."""
    switch_step = 50
    summary = _run_switch_test(
        lambda_coupling=0.1,
        switch_step=switch_step,
        n_steps=switch_step + 20,
    )

    assert summary["coupling_after"] is not None
    assert abs(summary["coupling_after"]) > 1e-6, "Coupling energy should be active after switch"

    # Displacer should place the photon near equilibrium (|q| ~ O(0.1) nm, not O(10)).
    assert summary["q_at_switch"] is not None
    assert summary["q_at_switch"] < 1.0, (
        f"Photon |q|={summary['q_at_switch']:.3f} nm at switch; displacer did not fire in sync"
    )

    pe_jump = summary["pe_jump"]
    assert pe_jump is not None
    # Mis-synced switch injected multi-thousand kJ/mol runaway spikes and often NaN.
    assert abs(pe_jump) < 2500.0, (
        f"Switch PE jump {pe_jump:.1f} kJ/mol too large on {summary['platform']}"
    )

    assert summary["pe_after"] is not None
    assert np.isfinite(summary["pe_after"])


@pytest.mark.timeout(120)
def test_instant_switch_survives_many_steps_post_switch():
    """Simulation remains finite for hundreds of steps after a realistic switch."""
    switch_step = 30
    _run_switch_test(
        lambda_coupling=0.098,
        switch_step=switch_step,
        n_steps=switch_step + 200,
    )
