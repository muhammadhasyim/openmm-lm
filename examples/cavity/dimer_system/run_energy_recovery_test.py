#!/usr/bin/env python3
"""Instant coupling switch energy-recovery test for mKA bulk glass.

Protocol (nonthermal-aging style):
  - NVT at T=100 K with lambda=0 until t_switch
  - Instant lambda jump at t_switch (displaceToEquilibrium via CavityParticleDisplacer)
  - VariableVerletIntegrator with ramped error tolerance around the switch
  - Track harmonic bond and nonbonded energies vs pre-switch baseline
"""

from __future__ import annotations

import argparse
import sys
import time
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
if str(HERE) not in sys.path:
    sys.path.insert(0, str(HERE))

from openmm import openmm, unit

import run_simulation as rs

GROUP_BONDS = 1
GROUP_NONBONDED = 2


def assign_energy_groups(system: openmm.System) -> None:
    """Put bond and nonbonded forces in separate energy groups."""
    for force in system.getForces():
        if isinstance(force, openmm.HarmonicBondForce):
            force.setForceGroup(GROUP_BONDS)
        elif isinstance(
            force, (openmm.NonbondedForce, openmm.CustomNonbondedForce)
        ):
            force.setForceGroup(GROUP_NONBONDED)


def group_energy(context: openmm.Context, group: int) -> float:
    mask = 1 << group
    state = context.getState(getEnergy=True, groups=mask)
    return state.getPotentialEnergy().value_in_unit(unit.kilojoule_per_mole)


def adaptive_epsilon(sim_time_ps: float, switch_time_ps: float) -> float:
    eps_min, eps_max, tau, pre_ramp = 1e-5, 1.0, 50.0, 1.0
    if sim_time_ps < switch_time_ps - pre_ramp:
        return eps_max
    t_since = max(0.0, sim_time_ps - switch_time_ps)
    return eps_min + (eps_max - eps_min) * (1.0 - np.exp(-t_since / tau))


def run_energy_recovery(
    *,
    num_molecules: int = 250,
    lambda_coupling: float = 0.005,
    switch_time_ps: float = 200.0,
    total_time_ps: float = 2500.0,
    temperature_K: float = 100.0,
    dt_ps: float = 0.001,
    seed: int = 42,
    sample_interval_ps: float = 10.0,
    output_npz: str | None = None,
) -> dict:
    box_size_nm = rs.box_size_nm_at_constant_density(num_molecules)
    omegac_au = 1560.0 / 219474.63
    photon_mass = 1.0 / 1822.888

    print("=" * 70)
    print("mKA energy recovery test (instant coupling switch)")
    print("=" * 70)
    print(f"  N dimers     : {num_molecules}")
    print(f"  Box          : {box_size_nm:.4f} nm (constant density)")
    print(f"  T            : {temperature_K} K")
    print(f"  lambda       : {lambda_coupling} (g = {lambda_coupling * np.sqrt(num_molecules):.4f})")
    print(f"  Switch time  : {switch_time_ps} ps")
    print(f"  Total time   : {total_time_ps} ps")
    print(f"  dt cap       : {dt_ps} ps (adaptive VariableVerlet)")

    result = rs.create_diamer_system_from_forcefield(
        num_molecules=num_molecules,
        fraction_OO=0.8,
        box_size_nm=box_size_nm,
        seed=seed,
        include_cavity=True,
    )
    system, positions, topology, cavity_index = result
    num_molecular = cavity_index
    system.setParticleMass(cavity_index, photon_mass)

    assign_energy_groups(system)

    switch_step = int(round(switch_time_ps / dt_ps))
    cavity_force = openmm.CavityForce(cavity_index, omegac_au, 0.0, photon_mass)
    cavity_force.setCouplingOnStep(switch_step, lambda_coupling)
    system.addForce(cavity_force)

    displacer = openmm.CavityParticleDisplacer(cavity_index, omegac_au, photon_mass)
    displacer.setSwitchOnLambda(lambda_coupling)
    displacer.setSwitchOnStep(switch_step)
    system.addForce(displacer)

    bussi = openmm.BussiThermostat(temperature_K, 1.0)
    bussi.setApplyToAllParticles(False)
    for i in range(num_molecular):
        bussi.addParticle(i)
    system.addForce(bussi)

    integrator = openmm.VariableVerletIntegrator(1.0)
    integrator.setMaximumStepSize(dt_ps * unit.picosecond)

    try:
        platform = openmm.Platform.getPlatformByName("CUDA")
        platform.setPropertyDefaultValue("Precision", "mixed")
        platform_name = "CUDA"
    except Exception:
        platform = openmm.Platform.getPlatformByName("Reference")
        platform_name = "Reference"

    context = openmm.Context(system, integrator, platform)
    context.setPositions(positions)
    context.setVelocitiesToTemperature(temperature_K * unit.kelvin, seed)

    print(f"  Platform     : {platform_name}")
    print("\n--- Energy minimization ---")
    openmm.LocalEnergyMinimizer.minimize(context, maxIterations=100)
    state = context.getState(getEnergy=True)
    print(f"  PE after min : {state.getPotentialEnergy()}")

    times: list[float] = []
    bond_es: list[float] = []
    nb_es: list[float] = []
    total_pes: list[float] = []
    dts: list[float] = []

    next_sample_ps = 0.0
    t0 = time.time()

    while True:
        sim_time_ps = context.getState().getTime().value_in_unit(unit.picosecond)
        if sim_time_ps >= total_time_ps - 1e-12:
            break

        integrator.setErrorTolerance(adaptive_epsilon(sim_time_ps, switch_time_ps))
        integrator.step(1)
        dt_step = integrator.getStepSize().value_in_unit(unit.picosecond)
        sim_time_ps = context.getState().getTime().value_in_unit(unit.picosecond)

        if sim_time_ps + 1e-12 >= next_sample_ps:
            bond_e = group_energy(context, GROUP_BONDS)
            nb_e = group_energy(context, GROUP_NONBONDED)
            pe = context.getState(getEnergy=True).getPotentialEnergy().value_in_unit(
                unit.kilojoule_per_mole
            )
            times.append(sim_time_ps)
            bond_es.append(bond_e)
            nb_es.append(nb_e)
            total_pes.append(pe)
            dts.append(dt_step)
            next_sample_ps += sample_interval_ps

            if len(times) % 50 == 1 or abs(sim_time_ps - switch_time_ps) < sample_interval_ps:
                print(
                    f"  t={sim_time_ps:7.1f} ps  bond={bond_e:10.2f}  "
                    f"nb={nb_e:10.2f}  dt={dt_step*1000:.3f} fs"
                )

    elapsed = time.time() - t0
    times_arr = np.array(times)
    bond_arr = np.array(bond_es)
    nb_arr = np.array(nb_es)

    pre = (times_arr >= switch_time_ps - 50) & (times_arr < switch_time_ps)
    post = times_arr >= total_time_ps - 50
    switch_win = (times_arr >= switch_time_ps) & (times_arr <= switch_time_ps + 20)

    def stats(mask: np.ndarray, arr: np.ndarray) -> tuple[float, float]:
        if not np.any(mask):
            return float("nan"), float("nan")
        sel = arr[mask]
        return float(np.mean(sel)), float(np.std(sel))

    pre_bond_mean, pre_bond_std = stats(pre, bond_arr)
    pre_nb_mean, pre_nb_std = stats(pre, nb_arr)
    post_bond_mean, post_bond_std = stats(post, bond_arr)
    post_nb_mean, post_nb_std = stats(post, nb_arr)
    sw_bond_mean, _ = stats(switch_win, bond_arr)
    sw_nb_mean, _ = stats(switch_win, nb_arr)

    bond_delta = post_bond_mean - pre_bond_mean
    nb_delta = post_nb_mean - pre_nb_mean
    bond_z = bond_delta / pre_bond_std if pre_bond_std > 0 else float("nan")
    nb_z = nb_delta / pre_nb_std if pre_nb_std > 0 else float("nan")

    print("\n" + "=" * 70)
    print("Energy recovery analysis")
    print("=" * 70)
    print(f"  Pre-switch baseline  (t in [{switch_time_ps-50}, {switch_time_ps}) ps):")
    print(f"    Harmonic bonds : {pre_bond_mean:10.2f} +/- {pre_bond_std:.2f} kJ/mol")
    print(f"    Nonbonded      : {pre_nb_mean:10.2f} +/- {pre_nb_std:.2f} kJ/mol")
    print(f"  At switch spike    (t in [{switch_time_ps}, {switch_time_ps+20}] ps):")
    print(f"    Harmonic bonds : {sw_bond_mean:10.2f} kJ/mol")
    print(f"    Nonbonded      : {sw_nb_mean:10.2f} kJ/mol")
    print(f"  At {total_time_ps:.0f} ps window (last 50 ps):")
    print(f"    Harmonic bonds : {post_bond_mean:10.2f} +/- {post_bond_std:.2f} kJ/mol")
    print(f"    Nonbonded      : {post_nb_mean:10.2f} +/- {post_nb_std:.2f} kJ/mol")
    print(f"  Delta (post - pre):")
    print(f"    Harmonic bonds : {bond_delta:+10.2f} kJ/mol  (z = {bond_z:+.2f})")
    print(f"    Nonbonded      : {nb_delta:+10.2f} kJ/mol  (z = {nb_z:+.2f})")

    recovered_bond = abs(bond_z) < 2.0 if np.isfinite(bond_z) else False
    recovered_nb = abs(nb_z) < 2.0 if np.isfinite(nb_z) else False
    print(f"\n  Recovered to pre-switch equilibrium (|z| < 2)?")
    print(f"    Harmonic bonds : {'YES' if recovered_bond else 'NO'}")
    print(f"    Nonbonded      : {'YES' if recovered_nb else 'NO'}")
    print(f"  Wall time: {elapsed/60:.1f} min")

    summary = {
        "pre_bond_mean": pre_bond_mean,
        "pre_bond_std": pre_bond_std,
        "pre_nb_mean": pre_nb_mean,
        "pre_nb_std": pre_nb_std,
        "post_bond_mean": post_bond_mean,
        "post_bond_std": post_bond_std,
        "post_nb_mean": post_nb_mean,
        "post_nb_std": post_nb_std,
        "bond_delta": bond_delta,
        "nb_delta": nb_delta,
        "bond_z": bond_z,
        "nb_z": nb_z,
        "recovered_bond": recovered_bond,
        "recovered_nb": recovered_nb,
        "platform": platform_name,
        "elapsed_s": elapsed,
    }

    out = output_npz or str(
        HERE / f"energy_recovery_lambda{lambda_coupling}_t{int(switch_time_ps)}.npz"
    )
    np.savez(
        out,
        time_ps=times_arr,
        bond_energy_kj_mol=bond_arr,
        nonbonded_energy_kj_mol=nb_arr,
        total_pe_kj_mol=np.array(total_pes),
        dt_ps=np.array(dts),
        metadata=summary,
    )
    print(f"\n  Trajectory saved: {out}")
    return summary


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dimers", type=int, default=250)
    parser.add_argument("--lambda", type=float, default=0.005, dest="lambda_coupling")
    parser.add_argument("--switch-time", type=float, default=200.0)
    parser.add_argument("--total-time", type=float, default=2500.0)
    parser.add_argument("--temp", type=float, default=100.0)
    parser.add_argument("--dt", type=float, default=0.001)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--sample-interval", type=float, default=10.0)
    parser.add_argument("--output", type=str, default=None)
    args = parser.parse_args()

    run_energy_recovery(
        num_molecules=args.dimers,
        lambda_coupling=args.lambda_coupling,
        switch_time_ps=args.switch_time,
        total_time_ps=args.total_time,
        temperature_K=args.temp,
        dt_ps=args.dt,
        seed=args.seed,
        sample_interval_ps=args.sample_interval,
        output_npz=args.output,
    )


if __name__ == "__main__":
    main()
