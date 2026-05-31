#!/usr/bin/env python3
"""RPMDIntegrator + CavityForce / MultiModeCavityForce integration (cavity-md)."""

import math

import numpy as np
import pytest

try:
    from openmm import CavityForce, Context, MultiModeCavityForce, NonbondedForce, Platform, RPMDIntegrator, System, Vec3
    from openmm import unit
except ImportError:
    pytest.skip("CavityForce / MultiModeCavityForce not in this OpenMM build", allow_module_level=True)

if not hasattr(RPMDIntegrator, "setParticleType"):
    pytest.skip("Extended RPMD API (hybrid particle types) not available", allow_module_level=True)

from rpmd_test_utils import is_finite_energy, rpmd_total_energy_kjmol


def _pick_platform(prefer_cuda: bool = False):
    if prefer_cuda:
        try:
            return Platform.getPlatformByName("CUDA")
        except Exception:
            pass
    return Platform.getPlatformByName("Reference")


def _init_rpmd_beads(context, integrator, system, cavity_indices, temperature_K, rng):
    """Set bead positions/velocities; cavity sites identical on every bead."""
    cavity_set = set(cavity_indices)
    num_beads = integrator.getNumCopies()
    num_particles = system.getNumParticles()

    for bead in range(num_beads):
        pos = []
        for i in range(num_particles):
            if i in cavity_set:
                pos.append((0.01, 0.0, 0.0))
            else:
                pos.append(
                    (
                        0.15 + 0.01 * rng.randn(),
                        0.15 + 0.01 * rng.randn(),
                        0.15 + 0.01 * rng.randn(),
                    )
                )
        integrator.setPositions(
            bead,
            [unit.Quantity(p, unit.nanometer) for p in pos],
        )

    context.setPositions(integrator.getState(0, getPositions=True).getPositions())
    context.setVelocitiesToTemperature(temperature_K * unit.kelvin)
    base_vel = context.getState(getVelocities=True).getVelocities()
    for bead in range(num_beads):
        bead_vel = []
        for i, v in enumerate(base_vel):
            vx = v[0].value_in_unit(unit.nanometers / unit.picoseconds)
            vy = v[1].value_in_unit(unit.nanometers / unit.picoseconds)
            vz = v[2].value_in_unit(unit.nanometers / unit.picoseconds)
            if i not in cavity_set:
                vx += 0.005 * rng.randn()
                vy += 0.005 * rng.randn()
                vz += 0.005 * rng.randn()
            bead_vel.append(Vec3(vx, vy, vz) * unit.nanometers / unit.picoseconds)
        integrator.setVelocities(bead, bead_vel)


def test_cavity_rpmd_single_mode_classical_cavity():
    """CavityForce + RPMD: cavity particle classical; molecules quantum."""
    num_molecular = 4
    photon_mass = 1.0
    omegac = 0.01
    lambda_c = 0.05
    num_beads = 4
    temperature_K = 100.0
    dt_ps = 0.001

    system = System()
    nb = NonbondedForce()
    for i in range(num_molecular):
        system.addParticle(12.0)
        # Neutral molecules: per-bead dipole fluctuations with charged RPMD beads
        # can destabilize the classical photon when lambda is large.
        nb.addParticle(0.0, 0.3, 1.0)
    cavity_index = system.addParticle(photon_mass)
    nb.addParticle(0.0, 0.1, 0.0)
    system.addForce(nb)

    cavity = CavityForce(cavity_index, omegac, lambda_c, photon_mass)
    system.addForce(cavity)

    integrator = RPMDIntegrator(
        num_beads,
        temperature_K * unit.kelvin,
        1.0 / unit.picosecond,
        dt_ps * unit.picosecond,
    )
    integrator.setThermostatType(RPMDIntegrator.PileG)
    integrator.setCentroidFriction(0.5 / unit.picosecond)
    integrator.setParticleType(cavity_index, 1)
    integrator.setQuantumParticleTypes({0})
    integrator.setClassicalThermostat(RPMDIntegrator.LangevinClassical)

    context = Context(system, integrator, _pick_platform(False))
    _init_rpmd_beads(context, integrator, system, [cavity_index], temperature_K, np.random.RandomState(1))
    context.setPeriodicBoxVectors(
        unit.Quantity([2.0, 0, 0], unit.nanometer),
        unit.Quantity([0, 2.0, 0], unit.nanometer),
        unit.Quantity([0, 0, 2.0], unit.nanometer),
    )

    integrator.step(50)
    spread = []
    for bead in range(num_beads):
        st = integrator.getState(bead, getPositions=True)
        p = st.getPositions(asNumpy=True).value_in_unit(unit.nanometer)
        spread.append(p[cavity_index])
    spread = np.array(spread)
    assert np.max(np.std(spread, axis=0)) < 0.01

    assert is_finite_energy(integrator)
    del context, integrator


def test_cavity_rpmd_reference_cuda_agreement():
    """Same initial beads on Reference vs CUDA: energies match within tolerance (if CUDA exists)."""
    try:
        cuda = Platform.getPlatformByName("CUDA")
    except Exception:
        pytest.skip("CUDA platform not available")

    num_molecular = 3
    photon_mass = 1.0
    omegac = 0.01
    lambda_c = 0.05
    num_beads = 4
    temperature_K = 100.0
    dt_ps = 0.001

    def build():
        system = System()
        nb = NonbondedForce()
        for i in range(num_molecular):
            system.addParticle(12.0)
            nb.addParticle(0.3, 0.3, 0.5)
        cavity_index = system.addParticle(photon_mass)
        nb.addParticle(0.0, 0.1, 0.0)
        system.addForce(nb)
        cavity = CavityForce(cavity_index, omegac, lambda_c, photon_mass)
        system.addForce(cavity)
        integrator = RPMDIntegrator(
            num_beads,
            temperature_K * unit.kelvin,
            1.0 / unit.picosecond,
            dt_ps * unit.picosecond,
        )
        integrator.setThermostatType(RPMDIntegrator.Pile)
        integrator.setCentroidFriction(0.5 / unit.picosecond)
        integrator.setParticleType(cavity_index, 1)
        integrator.setQuantumParticleTypes({0})
        integrator.setClassicalThermostat(RPMDIntegrator.LangevinClassical)
        return system, integrator, cavity_index

    system, int_ref, cavity_index = build()
    ctx_ref = Context(system, int_ref, Platform.getPlatformByName("Reference"))
    system2, int_cuda, _ = build()
    try:
        ctx_cuda = Context(system2, int_cuda, cuda)
    except Exception as exc:
        pytest.skip(f"CUDA platform lacks CavityForce kernels: {exc}")

    rng = np.random.RandomState(42)
    for ctx, integrator in ((ctx_ref, int_ref), (ctx_cuda, int_cuda)):
        _init_rpmd_beads(ctx, integrator, system, [cavity_index], temperature_K, rng)
        ctx.setPeriodicBoxVectors(
            unit.Quantity([2.0, 0, 0], unit.nanometer),
            unit.Quantity([0, 2.0, 0], unit.nanometer),
            unit.Quantity([0, 0, 2.0], unit.nanometer),
        )

    int_ref.step(10)
    int_cuda.step(10)
    e_ref = rpmd_total_energy_kjmol(int_ref)
    e_cuda = rpmd_total_energy_kjmol(int_cuda)
    assert abs(e_ref - e_cuda) < 5.0


def test_multimode_cavity_rpmd():
    """MultiModeCavityForce with classical cavity particles: identical beads per cavity site."""
    num_beads = 4
    temperature_K = 80.0
    dt_ps = 0.001
    cavity_indices = [4, 5, 6]

    system = System()
    nb = NonbondedForce()
    for i in range(4):
        system.addParticle(10.0)
        nb.addParticle(0.0, 0.3, 0.1)
    for _ in range(3):
        system.addParticle(1.0)
        nb.addParticle(0.0, 0.1, 0.0)
    system.addForce(nb)

    mm = MultiModeCavityForce(3, 0.02, 0.3, 2.0, 1.0)
    for k in cavity_indices:
        mm.addCavityParticle(k)
    system.addForce(mm)

    integrator = RPMDIntegrator(
        num_beads,
        temperature_K * unit.kelvin,
        1.0 / unit.picosecond,
        dt_ps * unit.picosecond,
    )
    integrator.setThermostatType(RPMDIntegrator.PileG)
    integrator.setCentroidFriction(0.5 / unit.picosecond)
    for k in cavity_indices:
        integrator.setParticleType(k, 1)
    integrator.setQuantumParticleTypes({0})
    integrator.setClassicalThermostat(RPMDIntegrator.LangevinClassical)

    context = Context(system, integrator, _pick_platform(False))
    rng = np.random.RandomState(7)
    for bead in range(num_beads):
        pos = []
        for i in range(system.getNumParticles()):
            if i in cavity_indices:
                pos.append((0.05 * (i - 4), 0.0, 0.0))
            else:
                pos.append(tuple(0.2 + 0.02 * rng.randn(3)))
        integrator.setPositions(
            bead,
            [unit.Quantity(p, unit.nanometer) for p in pos],
        )
    context.setPositions(integrator.getState(0, getPositions=True).getPositions())
    context.setVelocitiesToTemperature(temperature_K * unit.kelvin)
    context.setPeriodicBoxVectors(
        unit.Quantity([2.5, 0, 0], unit.nanometer),
        unit.Quantity([0, 2.5, 0], unit.nanometer),
        unit.Quantity([0, 0, 2.5], unit.nanometer),
    )

    integrator.step(80)
    for idx in cavity_indices:
        bead_pos = []
        for bead in range(num_beads):
            st = integrator.getState(bead, getPositions=True)
            p = st.getPositions(asNumpy=True).value_in_unit(unit.nanometer)
            bead_pos.append(p[idx])
        bead_pos = np.array(bead_pos)
        assert np.max(np.std(bead_pos, axis=0)) < 0.05
    assert is_finite_energy(integrator)
