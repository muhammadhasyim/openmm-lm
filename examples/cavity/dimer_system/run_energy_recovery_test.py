#!/usr/bin/env python3
"""Instant coupling switch energy-recovery test for mKA bulk glass.

Reproduces the non-thermal-aging protocol of Hasyim, Damiani & Hoffmann,
"Non-Thermal Aging of Supercooled Liquids in Optical Cavities" (arXiv:2603.15693),
as implemented in cavHOOMD-blue:

  - NVT at T=100 K, molecules on a Bussi-Parrinello bath (tau_b = 1 ps)
  - Cavity photon on a Langevin bath (tau_c = 1 ps) so the pumped vibrational
    energy has a dissipation channel and reaches a bounded steady state instead
    of drifting without bound (bath temperature held fixed to prevent heating).
  - Optional NVT equilibration (default 4 ns) with lambda = 0 so stiff bond modes
    and the glass structure reach thermal equilibrium before the aging experiment.
  - lambda = 0 until t_switch, then an instant jump (displaceToEquilibrium via
    CavityParticleDisplacer). t_switch = equil_time + switch_delay (default 4 ns
    + 200 ps).
  - VariableVerletIntegrator with the paper's Eq. 3.16 adaptive error tolerance
    (eps* = 5.0, f = 1e-3, tau* = 50 ps): relaxed before the switch, reset strict
    at the switch, then ramped back up.
  - Track harmonic bond and nonbonded energies vs the pre-switch baseline:
    harmonic (fast modes) is pumped to a bounded steady state; nonbonded
    (structural) recovers to its pre-switch baseline.

Example (recommended: equilibrated baseline, then aging):
  python run_energy_recovery_test.py --equil-time 4000 --switch-time 200 \\
      --post-switch-time 2500 --cavity-friction 1.0

Legacy (no equilibration, switch at 200 ps, end at 2500 ps):
  python run_energy_recovery_test.py --equil-time 0 --total-time 2500 --switch-time 200
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

from openmm.cavitymd.adaptive import (
    DT_MAX_PS,
    EPS_STAR_NM,
    epsilon_tolerance,
)
from openmm.cavitymd.thermostats import DualThermostat

import run_simulation as rs

GROUP_BONDS = 1
GROUP_NONBONDED = 2

# Paper Methods (arXiv:2603.15693): tau_b = tau_c = 1.0 ps. cavHOOMD uses
# gamma = 1/tau (velocity relaxation time), so tau_c = 1 ps -> gamma_c = 1.0 ps^-1.
DEFAULT_CAVITY_FRICTION_PS_INV = 1.0


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


def run_energy_recovery(
    *,
    num_molecules: int = 250,
    lambda_coupling: float = 0.005,
    equil_time_ps: float = 4000.0,
    switch_time_ps: float = 200.0,
    post_switch_time_ps: float = 2500.0,
    total_time_ps: float | None = None,
    temperature_K: float = 100.0,
    dt_ps: float = DT_MAX_PS,
    bussi_tau_ps: float = 1.0,
    cavity_friction_ps_inv: float = DEFAULT_CAVITY_FRICTION_PS_INV,
    seed: int = 42,
    sample_interval_ps: float = 10.0,
    output_npz: str | None = None,
) -> dict:
    # switch_time_ps = delay after equil ends (or absolute switch when equil=0).
    absolute_switch_ps = equil_time_ps + switch_time_ps
    if total_time_ps is not None:
        total_end_ps = total_time_ps
    else:
        total_end_ps = absolute_switch_ps + post_switch_time_ps
    box_size_nm = rs.box_size_nm_at_constant_density(num_molecules)
    omegac_au = 1560.0 / 219474.63
    photon_mass = 1.0 / 1822.888

    print("=" * 70)
    print("mKA energy recovery test (instant coupling switch)")
    print("=" * 70)
    print(f"  N dimers       : {num_molecules}")
    print(f"  Box            : {box_size_nm:.4f} nm (constant density)")
    print(f"  T              : {temperature_K} K")
    print(f"  lambda         : {lambda_coupling} (g = {lambda_coupling * np.sqrt(num_molecules):.4f})")
    print(f"  Equilibration  : {equil_time_ps} ps ({equil_time_ps/1000:.1f} ns)")
    print(f"  Switch delay   : {switch_time_ps} ps after equil (absolute t = {absolute_switch_ps} ps)")
    print(f"  Post-switch    : {total_end_ps - absolute_switch_ps:.0f} ps")
    print(f"  Total time     : {total_end_ps} ps ({total_end_ps/1000:.1f} ns)")
    print(f"  dt cap         : {dt_ps} ps (adaptive VariableVerlet, eps*={EPS_STAR_NM})")
    print(f"  Molecular bath : Bussi, tau_b = {bussi_tau_ps} ps")
    if cavity_friction_ps_inv > 0.0:
        print(f"  Cavity bath    : Langevin, gamma_c = {cavity_friction_ps_inv} ps^-1 "
              f"(tau_c = {1.0 / cavity_friction_ps_inv:.3g} ps)")
    else:
        print(f"  Cavity bath    : NONE (photon is NVE) -- expect runaway pumping")

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

    switch_step = int(round(absolute_switch_ps / dt_ps))
    cavity_force = openmm.CavityForce(cavity_index, omegac_au, 0.0, photon_mass)
    cavity_force.setCouplingOnStep(switch_step, lambda_coupling)
    system.addForce(cavity_force)

    displacer = openmm.CavityParticleDisplacer(cavity_index, omegac_au, photon_mass)
    displacer.setSwitchOnLambda(lambda_coupling)
    displacer.setSwitchOnStep(switch_step)
    system.addForce(displacer)

    bussi = openmm.BussiThermostat(temperature_K, bussi_tau_ps)
    bussi.setApplyToAllParticles(False)
    for i in range(num_molecular):
        bussi.addParticle(i)
    system.addForce(bussi)

    integrator = openmm.VariableVerletIntegrator(EPS_STAR_NM)
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

    # Langevin bath on the cavity photon (Caldeira-Leggett cavity loss). Applied
    # as a per-step Ornstein-Uhlenbeck velocity update, operator-split with the
    # Verlet propagation of the deterministic cavity force. This is the missing
    # dissipation channel: without it the resonant photon pumps the O-O stretch
    # without bound; with it the fast-mode energy saturates to a steady state.
    cavity_thermostat = DualThermostat(
        context,
        system,
        cavity_particle_index=cavity_index,
        cavity_friction_ps_inv=cavity_friction_ps_inv,
        cavity_temperature_K=temperature_K,
    )

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

    if equil_time_ps > 0:
        print(f"\n--- NVT equilibration ({equil_time_ps} ps, lambda=0) ---")
        next_sample_ps = 0.0
        next_equil_report = 500.0
        while True:
            sim_time_ps = context.getState().getTime().value_in_unit(unit.picosecond)
            if sim_time_ps >= equil_time_ps - 1e-12:
                break
            integrator.setErrorTolerance(epsilon_tolerance(sim_time_ps, absolute_switch_ps))
            integrator.step(1)
            dt_step = integrator.getStepSize().value_in_unit(unit.picosecond)
            cavity_thermostat.apply_cavity_thermostat_step(dt_step)
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

            if sim_time_ps + 1e-12 >= next_equil_report:
                bond_e = group_energy(context, GROUP_BONDS)
                nb_e = group_energy(context, GROUP_NONBONDED)
                print(
                    f"  equil t={sim_time_ps:7.1f} ps  bond={bond_e:10.2f}  "
                    f"nb={nb_e:10.2f}  dt={dt_step*1000:.3f} fs"
                )
                next_equil_report += 500.0

        bond_e = group_energy(context, GROUP_BONDS)
        nb_e = group_energy(context, GROUP_NONBONDED)
        t_equil = context.getState().getTime().value_in_unit(unit.picosecond)
        print(
            f"  Equilibration complete at t={t_equil:.1f} ps: "
            f"bond={bond_e:.2f} kJ/mol, nb={nb_e:.2f} kJ/mol"
        )
        next_sample_ps = t_equil + sample_interval_ps
    else:
        next_sample_ps = 0.0

    print(f"\n--- Non-thermal aging (switch at t={absolute_switch_ps} ps) ---")
    t0 = time.time()

    while True:
        sim_time_ps = context.getState().getTime().value_in_unit(unit.picosecond)
        if sim_time_ps >= total_end_ps - 1e-12:
            break

        integrator.setErrorTolerance(epsilon_tolerance(sim_time_ps, absolute_switch_ps))
        integrator.step(1)
        dt_step = integrator.getStepSize().value_in_unit(unit.picosecond)
        # Cavity Langevin bath (no-op when friction == 0).
        cavity_thermostat.apply_cavity_thermostat_step(dt_step)
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

            if (
                len(times) % 50 == 1
                or abs(sim_time_ps - absolute_switch_ps) < sample_interval_ps
            ):
                print(
                    f"  t={sim_time_ps:7.1f} ps  bond={bond_e:10.2f}  "
                    f"nb={nb_e:10.2f}  dt={dt_step*1000:.3f} fs"
                )

    elapsed = time.time() - t0
    times_arr = np.array(times)
    bond_arr = np.array(bond_es)
    nb_arr = np.array(nb_es)

    pre = (times_arr >= absolute_switch_ps - 50) & (times_arr < absolute_switch_ps)
    post = times_arr >= total_end_ps - 50
    switch_win = (times_arr >= absolute_switch_ps) & (
        times_arr <= absolute_switch_ps + 20
    )

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

    # Bounded-steady-state check: slope of the harmonic energy over the last
    # 500 ps. Non-thermal aging pumps fast modes to a *bounded* plateau, so a
    # converged run has a late-time slope consistent with zero. The pre-fix
    # (no cavity bath) run instead shows a clear positive slope (runaway).
    late = times_arr >= (total_end_ps - 500.0)
    if np.count_nonzero(late) >= 3:
        bond_slope = float(np.polyfit(times_arr[late], bond_arr[late], 1)[0])
    else:
        bond_slope = float("nan")
    # Treat |slope| < ~1 kJ/mol per 100 ps as effectively stationary.
    bounded_bond = abs(bond_slope) < 0.01 if np.isfinite(bond_slope) else False

    print("\n" + "=" * 70)
    print("Energy recovery analysis")
    print("=" * 70)
    print(f"  Pre-switch baseline  (t in [{absolute_switch_ps-50}, {absolute_switch_ps}) ps):")
    print(f"    Harmonic bonds : {pre_bond_mean:10.2f} +/- {pre_bond_std:.2f} kJ/mol")
    print(f"    Nonbonded      : {pre_nb_mean:10.2f} +/- {pre_nb_std:.2f} kJ/mol")
    print(f"  At switch spike    (t in [{absolute_switch_ps}, {absolute_switch_ps+20}] ps):")
    print(f"    Harmonic bonds : {sw_bond_mean:10.2f} kJ/mol")
    print(f"    Nonbonded      : {sw_nb_mean:10.2f} kJ/mol")
    print(f"  At {total_end_ps:.0f} ps window (last 50 ps):")
    print(f"    Harmonic bonds : {post_bond_mean:10.2f} +/- {post_bond_std:.2f} kJ/mol")
    print(f"    Nonbonded      : {post_nb_mean:10.2f} +/- {post_nb_std:.2f} kJ/mol")
    print(f"  Delta (post - pre):")
    print(f"    Harmonic bonds : {bond_delta:+10.2f} kJ/mol  (z = {bond_z:+.2f})")
    print(f"    Nonbonded      : {nb_delta:+10.2f} kJ/mol  (z = {nb_z:+.2f})")
    print(f"  Late-time harmonic slope (last 500 ps): {bond_slope*100:+.3f} kJ/mol per 100 ps")

    recovered_nb = abs(nb_z) < 2.0 if np.isfinite(nb_z) else False
    print(f"\n  Non-thermal-aging criteria:")
    print(f"    Harmonic (fast modes) BOUNDED steady state : {'YES' if bounded_bond else 'NO'}")
    print(f"    Nonbonded (structural) recovered to baseline: {'YES' if recovered_nb else 'NO'}")
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
        "bond_late_slope_kj_per_ps": bond_slope,
        "bounded_bond": bounded_bond,
        "recovered_nb": recovered_nb,
        "cavity_friction_ps_inv": cavity_friction_ps_inv,
        "bussi_tau_ps": bussi_tau_ps,
        "lambda_coupling": lambda_coupling,
        "equil_time_ps": equil_time_ps,
        "switch_delay_ps": switch_time_ps,
        "switch_time_ps": absolute_switch_ps,
        "absolute_switch_ps": absolute_switch_ps,
        "post_switch_time_ps": total_end_ps - absolute_switch_ps,
        "total_end_ps": total_end_ps,
        "platform": platform_name,
        "elapsed_s": elapsed,
    }

    out = output_npz or str(
        HERE
        / f"energy_recovery_eq{int(equil_time_ps)}_lambda{lambda_coupling}_sw{int(switch_time_ps)}.npz"
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
    parser = argparse.ArgumentParser(
        description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    parser.add_argument("--dimers", type=int, default=250)
    parser.add_argument("--lambda", type=float, default=0.005, dest="lambda_coupling")
    parser.add_argument(
        "--equil-time",
        type=float,
        default=4000.0,
        help="NVT equilibration before aging experiment in ps (default: 4000 = 4 ns). "
        "Set 0 to skip (legacy cold-start).",
    )
    parser.add_argument(
        "--switch-time",
        type=float,
        default=200.0,
        help="Coupling switch delay in ps after equil ends (default: 200). "
        "Absolute switch time = equil-time + switch-time.",
    )
    parser.add_argument(
        "--post-switch-time",
        type=float,
        default=2500.0,
        help="Aging observation window after the switch in ps (default: 2500). "
        "Ignored if --total-time is set.",
    )
    parser.add_argument(
        "--total-time",
        type=float,
        default=None,
        help="Absolute simulation end time in ps (overrides --post-switch-time). "
        "Use with --equil-time 0 for legacy runs ending at 2500 ps.",
    )
    parser.add_argument("--temp", type=float, default=100.0)
    parser.add_argument("--dt", type=float, default=DT_MAX_PS,
                        help=f"Adaptive dt cap in ps (default: {DT_MAX_PS} = paper ~1.5 fs)")
    parser.add_argument("--bussi-tau", type=float, default=1.0,
                        help="Molecular Bussi thermostat time constant in ps (default: 1.0, paper)")
    parser.add_argument("--cavity-friction", type=float,
                        default=DEFAULT_CAVITY_FRICTION_PS_INV,
                        help="Cavity-photon Langevin friction gamma_c in ps^-1 "
                             f"(default: {DEFAULT_CAVITY_FRICTION_PS_INV} = 1/tau_c with tau_c=1 ps, paper). "
                             "Set 0 to disable the cavity bath (reproduces the runaway bug).")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--sample-interval", type=float, default=10.0)
    parser.add_argument("--output", type=str, default=None)
    args = parser.parse_args()

    run_energy_recovery(
        num_molecules=args.dimers,
        lambda_coupling=args.lambda_coupling,
        equil_time_ps=args.equil_time,
        switch_time_ps=args.switch_time,
        post_switch_time_ps=args.post_switch_time,
        total_time_ps=args.total_time,
        temperature_K=args.temp,
        dt_ps=args.dt,
        bussi_tau_ps=args.bussi_tau,
        cavity_friction_ps_inv=args.cavity_friction,
        seed=args.seed,
        sample_interval_ps=args.sample_interval,
        output_npz=args.output,
    )


if __name__ == "__main__":
    main()
