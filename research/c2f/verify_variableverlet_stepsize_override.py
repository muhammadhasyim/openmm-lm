#!/usr/bin/env python3
"""Demonstrate that VariableVerletIntegrator.step() ignores manual setStepSize()."""

from __future__ import annotations

import sys

import openmm
from openmm import unit


def _step_delta_ps(integrator, context) -> float:
    t0 = context.getState().getTime().value_in_unit(unit.picosecond)
    integrator.step(1)
    t1 = context.getState().getTime().value_in_unit(unit.picosecond)
    return t1 - t0


def main() -> int:
    system = openmm.System()
    system.addParticle(1.0)
    system.addParticle(1.0)
    force = openmm.HarmonicBondForce()
    force.addBond(0, 1, 0.1 * unit.nanometer, 1000.0 * unit.kilojoule_per_mole / unit.nanometer**2)
    system.addForce(force)

    requested_ps = 0.0005  # 0.5 fs
    positions = [
        openmm.Vec3(0.0, 0.0, 0.0) * unit.nanometer,
        openmm.Vec3(0.1, 0.0, 0.0) * unit.nanometer,
    ]
    velocities = [
        openmm.Vec3(0.1, 0.0, 0.0) * (unit.nanometer / unit.picosecond),
        openmm.Vec3(-0.1, 0.0, 0.0) * (unit.nanometer / unit.picosecond),
    ]

    print("=== Plain VerletIntegrator ===")
    verlet = openmm.VerletIntegrator(0.001 * unit.picosecond)
    ctx_v = openmm.Context(system, verlet)
    ctx_v.setPositions(positions)
    ctx_v.setVelocities(velocities)
    verlet.setStepSize(requested_ps * unit.picosecond)
    dt_v = _step_delta_ps(verlet, ctx_v)
    print(f"  requested dt = {requested_ps * 1000:.3f} fs")
    print(f"  actual   dt  = {dt_v * 1000:.3f} fs")
    verlet_respects = abs(dt_v - requested_ps) < 1e-12
    print(f"  respects setStepSize: {verlet_respects}")

    print("\n=== VariableVerletIntegrator ===")
    var = openmm.VariableVerletIntegrator(5.0)
    var.setMaximumStepSize(0.001 * unit.picosecond)
    ctx_var = openmm.Context(system, var)
    ctx_var.setPositions(positions)
    ctx_var.setVelocities(velocities)
    var.setStepSize(requested_ps * unit.picosecond)
    dt_var = _step_delta_ps(var, ctx_var)
    print(f"  requested dt = {requested_ps * 1000:.3f} fs")
    print(f"  actual   dt  = {dt_var * 1000:.3f} fs")
    var_ignores = abs(dt_var - requested_ps) > 1e-12
    print(f"  ignores setStepSize: {var_ignores}")

    if verlet_respects and var_ignores:
        print("\nPASS: VariableVerletIntegrator overrides manual setStepSize(); Verlet does not.")
        return 0
    print("\nFAIL: unexpected integrator behavior")
    return 1


if __name__ == "__main__":
    sys.exit(main())
