#!/usr/bin/env python3
"""Multiple time stepping (r-RESPA) with RPMDIntegrator."""

import math

import numpy as np
import pytest

from openmm import Context, HarmonicBondForce, NonbondedForce, Platform, RPMDIntegrator, System
from openmm import unit

from rpmd_test_utils import is_finite_energy, rpmd_total_energy_kjmol

if not hasattr(RPMDIntegrator, "setNumInnerSteps"):
    pytest.skip("Extended RPMD API (MTS) not available", allow_module_level=True)


def _two_particle_system():
    """Particle 0--1 bond (fast, group 0) + soft LJ-like interaction via Nonbonded (slow, group 1)."""
    system = System()
    for _ in range(2):
        system.addParticle(1.0)
    bond = HarmonicBondForce()
    bond.addBond(0, 1, 0.14, 200000.0)
    bond.setForceGroup(0)
    system.addForce(bond)
    nb = NonbondedForce()
    nb.addParticle(0.0, 0.34, 1.2)
    nb.addParticle(0.0, 0.34, 1.2)
    nb.setForceGroup(1)
    nb.setNonbondedMethod(NonbondedForce.NoCutoff)
    system.addForce(nb)
    return system


def test_mts_energy_drift_nve_bounded():
    """NVE + MTS: total RPMD energy stays finite; drift modest over short run."""
    system = _two_particle_system()
    num_beads = 8
    dt_ps = 0.0005
    integrator = RPMDIntegrator(
        num_beads,
        300.0 * unit.kelvin,
        1.0 / unit.picosecond,
        dt_ps * unit.picosecond,
    )
    integrator.setThermostatType(RPMDIntegrator.NoneThermo)
    integrator.setApplyThermostat(False)
    integrator.setNumInnerSteps(4)
    integrator.setInnerForceGroups(1 << 0)

    context = Context(system, integrator, Platform.getPlatformByName("Reference"))
    rng = np.random.RandomState(0)
    for bead in range(num_beads):
        dx = 0.01 * rng.randn(2, 3)
        pos = [
            unit.Quantity(tuple(0.1 + dx[0]), unit.nanometer),
            unit.Quantity(tuple(0.24 + dx[1]), unit.nanometer),
        ]
        integrator.setPositions(bead, pos)
        integrator.setVelocities(
            bead,
            [
                unit.Quantity(tuple(0.5 * rng.randn(3)), unit.nanometer / unit.picosecond),
                unit.Quantity(tuple(0.5 * rng.randn(3)), unit.nanometer / unit.picosecond),
            ],
        )

    e0 = rpmd_total_energy_kjmol(integrator)
    integrator.step(200)
    e1 = rpmd_total_energy_kjmol(integrator)
    assert math.isfinite(e0) and math.isfinite(e1)
    # NVE + MTS (r-RESPA): modest drift over 200 steps on Reference
    assert abs(e1 - e0) < 0.02 * abs(e0) + 50.0


def test_mts_contraction_slow_group():
    """MTS with contraction on slow group: runs without error."""
    system = _two_particle_system()
    num_beads = 8
    contractions = {1: 4}
    integrator = RPMDIntegrator(
        num_beads,
        200.0 * unit.kelvin,
        10.0 / unit.picosecond,
        0.0005 * unit.picosecond,
        contractions,
    )
    integrator.setThermostatType(RPMDIntegrator.Pile)
    integrator.setNumInnerSteps(2)
    integrator.setInnerForceGroups(1 << 0)

    context = Context(system, integrator, Platform.getPlatformByName("Reference"))
    rng = np.random.RandomState(3)
    for bead in range(num_beads):
        dx = 0.008 * rng.randn(2, 3)
        pos = [
            unit.Quantity(tuple(0.1 + dx[0]), unit.nanometer),
            unit.Quantity(tuple(0.24 + dx[1]), unit.nanometer),
        ]
        integrator.setPositions(bead, pos)
    integrator.step(30)
    assert is_finite_energy(integrator)
