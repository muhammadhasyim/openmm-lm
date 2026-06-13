#!/usr/bin/env python
"""NVT cavity equilibrium for the mKA system at fixed bath temperature."""

from __future__ import annotations

import argparse
import sys
import time as wall_time
from pathlib import Path

import numpy as np

try:
    import openmm
    from openmm import unit
except ImportError:
    sys.exit("OpenMM (cavity-md) required.")

from openmm.cavitymd import (
    DualThermostat,
    EnergyTracker,
    TemperatureTracker,
    EmpiricalTemperatureData,
    assign_force_groups,
    setup_gpu_step,
)

from run_c2f import (
    REFERENCE_CALIBRATION_FILE,
    OMEGA_C_CM1,
    PHOTON_MASS_AMU,
    NUM_MOL,
    HARTREE_TO_CM1,
    BUSSI_TAU_PS,
    BOHR_TO_NM,
    FKT_KMAG_AU,
    build_mka_system,
    add_cavity_particle,
    initialize_cavity_position,
    remove_molecular_com_velocity,
    _select_platform,
)

FKT_KMAG_NM_INV_DEFAULT = FKT_KMAG_AU / BOHR_TO_NM

_SCRIPT_DIR = Path(__file__).resolve().parent


def _load_initial_state(path: Path) -> tuple[np.ndarray, np.ndarray | None]:
    data = np.load(path)
    if "positions_nm" not in data:
        raise KeyError(f"{path} missing positions_nm")
    positions = np.asarray(data["positions_nm"], dtype=float)
    velocities = None
    if "velocities_nm_per_ps" in data:
        velocities = np.asarray(data["velocities_nm_per_ps"], dtype=float)
    return positions, velocities


def run_cavity_equilibrium(
    *,
    temperature_K: float,
    runtime_ps: float,
    lambda_coupling: float,
    include_dipole_self_energy: bool,
    output_prefix: str,
    seed: int = 42,
    dt_ps: float = 0.001,
    sample_interval_ps: float = 1.0,
    initial_state: Path | None = None,
    calibration_file: str | Path = REFERENCE_CALIBRATION_FILE,
    platform_name: str | None = None,
    finite_q: bool = True,
    omega_c_cm1: float = OMEGA_C_CM1,
    snapshot_interval_ps: float = 0.0,
    snapshots_out: Path | None = None,
    coupling_start_ps: float = 0.0,
    resample_velocities: bool = True,
    enable_fkt: bool = False,
    fkt_kmag_nm_inv: float | None = None,
    fkt_num_wavevectors: int = 50,
    fkt_ref_interval_ps: float = 200.0,
    fkt_output_period_ps: float = 1.0,
    fkt_max_refs: int = 13,
    fkt_start_ps: float | None = None,
    fkt_sites: str = "atomic",
) -> Path:
    """Run fixed-T NVT with cavity coupling; return final-state path.

    When *coupling_start_ps* > 0, lambda stays at 0 until that time then steps
    to *lambda_coupling* and remains on (single turn-on, no square wave).
    """
    if fkt_kmag_nm_inv is None:
        fkt_kmag_nm_inv = FKT_KMAG_NM_INV_DEFAULT

    print("\n=== Cavity NVT equilibrium ===")
    print(f"  T_bath           = {temperature_K} K")
    print(f"  runtime          = {runtime_ps} ps")
    print(f"  lambda           = {lambda_coupling}")
    print(f"  omega_c          = {omega_c_cm1} cm^-1")
    print(f"  dipole self-energy = {'ON' if include_dipole_self_energy else 'OFF'}")
    print(f"  finite-q shift   = {'ON' if finite_q else 'OFF'}")
    if coupling_start_ps > 0:
        print(f"  coupling on at   = {coupling_start_ps} ps (step, stays on)")
    print(f"  sample interval  = {sample_interval_ps} ps")
    if initial_state is not None:
        print(f"  resample vel     = {resample_velocities}")
    if snapshot_interval_ps > 0 and snapshots_out is not None:
        print(f"  snapshot interval= {snapshot_interval_ps} ps -> {snapshots_out}")
    if enable_fkt:
        fkt_t0 = fkt_start_ps if fkt_start_ps is not None else coupling_start_ps
        print(
            f"  F(k,t)         = ON  |k|={fkt_kmag_nm_inv} nm^-1, "
            f"sites={fkt_sites}, refs every {fkt_ref_interval_ps} ps from t={fkt_t0} ps"
        )
    print(f"  initial state    = {initial_state or '(fresh build, seed=' + str(seed) + ')'}")

    omegac_au = omega_c_cm1 / HARTREE_TO_CM1
    system, positions, n_atoms = build_mka_system(
        seed=seed,
        sample_bonds_at_T=temperature_K if initial_state is None else None,
    )
    cavity_index = add_cavity_particle(system, positions)
    initial_velocities = None
    if initial_state is not None:
        pos_nm, vel_nm = _load_initial_state(initial_state)
        if pos_nm.shape[0] != n_atoms + 1:
            raise ValueError(
                f"Expected {n_atoms + 1} particles in {initial_state}, got {pos_nm.shape[0]}"
            )
        positions = [
            openmm.Vec3(*pos_nm[i]) * unit.nanometer for i in range(pos_nm.shape[0])
        ]
        if vel_nm is not None:
            initial_velocities = vel_nm

    cavity_force = openmm.CavityForce(cavity_index, omegac_au, 0.0, PHOTON_MASS_AMU)
    cavity_force.setIncludeDipoleSelfEnergy(include_dipole_self_energy)
    setup_gpu_step(
        cavity_force, lambda_coupling, start_time_ps=coupling_start_ps
    )
    system.addForce(cavity_force)

    displacer = None
    if finite_q and lambda_coupling > 0.0:
        displacer = openmm.CavityParticleDisplacer(
            cavity_index, omegac_au, PHOTON_MASS_AMU
        )
        displacer.setSwitchOnLambda(lambda_coupling)
        displacer.setSwitchOnStep(2**31 - 1)
        system.addForce(displacer)

    DualThermostat.setup_bussi_for_system(
        system, list(range(n_atoms)), temperature_K, BUSSI_TAU_PS
    )
    group_map = assign_force_groups(
        system, include_dipole_self_energy=include_dipole_self_energy
    )

    integrator = openmm.VerletIntegrator(dt_ps * unit.picosecond)
    platform = _select_platform(platform_name)
    context = openmm.Context(system, integrator, platform)
    context.setPositions(positions)
    if initial_state is not None and not resample_velocities and initial_velocities is not None:
        context.setVelocities(
            [
                openmm.Vec3(*initial_velocities[i])
                * (unit.nanometer / unit.picosecond)
                for i in range(len(initial_velocities))
            ]
        )
    else:
        context.setVelocitiesToTemperature(temperature_K * unit.kelvin)
        if initial_state is None and not finite_q:
            initialize_cavity_position(
                context, cavity_index, temperature_K, omegac_au
            )

    remove_molecular_com_velocity(context, system, n_atoms)

    if displacer is not None and coupling_start_ps <= 0:
        displacer.displaceToEquilibrium(context, lambda_coupling)
        print(f"  Photon displaced to finite-q equilibrium (lambda={lambda_coupling})")

    energy_tracker = EnergyTracker(
        context, cavity_force, group_map, n_atoms, cavity_index
    )
    empirical_structural = EmpiricalTemperatureData(
        str(calibration_file), energy_component="lj_coulombic"
    )
    empirical_harmonic = EmpiricalTemperatureData(
        str(calibration_file), energy_component="harmonic"
    )
    temp_tracker = TemperatureTracker(
        energy_tracker,
        num_molecular_particles=n_atoms,
        num_molecules=NUM_MOL,
        empirical_structural=empirical_structural,
        empirical_harmonic=empirical_harmonic,
    )
    thermostat = DualThermostat(
        context,
        system,
        cavity_index,
        cavity_friction_ps_inv=0.5,
        cavity_temperature_K=temperature_K,
    )

    csv_path = f"{output_prefix}_energies.csv"
    n_samples = max(1, int(round(runtime_ps / sample_interval_ps)))
    sample_dt_ps = runtime_ps / n_samples
    print(f"  {n_samples} samples over {runtime_ps} ps")

    snapshot_times: list[float] = []
    snapshot_positions: list[np.ndarray] = []
    next_snapshot_ps = snapshot_interval_ps if snapshot_interval_ps > 0 else None

    fkt_tracker = None
    fkt_t0_ps = 0.0
    next_fkt_ps = 0.0
    if enable_fkt:
        from fkt_tracker import FKTTracker, fkt_positions_nm

        fkt_t0_ps = fkt_start_ps if fkt_start_ps is not None else coupling_start_ps
        fkt_tracker = FKTTracker(
            kmag_nm_inv=fkt_kmag_nm_inv,
            num_wavevectors=fkt_num_wavevectors,
            reference_interval_ps=fkt_ref_interval_ps,
            max_references=fkt_max_refs,
            output_period_ps=fkt_output_period_ps,
            output_prefix=output_prefix,
        )
        next_fkt_ps = fkt_t0_ps

    def _guard_molecular_com_velocity() -> None:
        remove_molecular_com_velocity(context, system, n_atoms)

    def _maybe_update_fkt(time_ps: float) -> None:
        nonlocal next_fkt_ps
        if fkt_tracker is None or time_ps + 1e-9 < fkt_t0_ps:
            return
        while time_ps + 1e-9 >= next_fkt_ps:
            _guard_molecular_com_velocity()
            state_fkt = context.getState(getPositions=True, enforcePeriodicBox=True)
            pos_all = state_fkt.getPositions(asNumpy=True).value_in_unit(unit.nanometer)
            pos_atomic = np.asarray(pos_all[:n_atoms], dtype=np.float64)
            pos_fkt = fkt_positions_nm(pos_atomic, NUM_MOL, site_mode=fkt_sites)
            fkt_tracker.update(time_ps - fkt_t0_ps, pos_fkt)
            next_fkt_ps += fkt_output_period_ps

    with open(csv_path, "w", encoding="utf-8") as csv_file:
        csv_file.write(
            "time_ps,T_bath_K,T_kinetic_K,T_v_fictive_K,T_s_fictive_K,"
            "E_kinetic_kjmol,E_potential_kjmol,E_mech_kjmol,"
            "E_bond_kjmol,E_nonbonded_kjmol,E_cav_harmonic_kjmol,"
            "E_cav_coupling_kjmol,E_cav_dse_kjmol\n"
        )
        t0 = wall_time.time()
        for sample_idx in range(1, n_samples + 1):
            target_time = sample_idx * sample_dt_ps
            steps = max(1, int(round((target_time - context.getState().getTime().value_in_unit(unit.picosecond)) / dt_ps)))
            for _ in range(steps):
                integrator.step(1)
                thermostat.apply_cavity_thermostat_step(dt_ps)
                step_time_ps = context.getState().getTime().value_in_unit(unit.picosecond)
                _maybe_update_fkt(step_time_ps)

            time_ps = context.getState().getTime().value_in_unit(unit.picosecond)
            energies = energy_tracker.get_energies()
            temps = temp_tracker.get_all()
            T_kin = temps.get("kinetic", float("nan"))
            T_v = temps.get("harmonic_equipartition", float("nan"))
            T_s = temps.get("structural_fictive")
            T_s_str = "" if T_s is None else f"{T_s:.4f}"
            e_kin = energies.get("total_kinetic", 0.0)
            e_pot = energies.get("total_potential", 0.0)
            csv_file.write(
                f"{time_ps:.6f},{temperature_K:.4f},{T_kin:.4f},{T_v:.4f},{T_s_str},"
                f"{e_kin:.6f},{e_pot:.6f},{e_kin + e_pot:.6f},"
                f"{energies.get('harmonic_bond', 0.0):.6f},"
                f"{energies.get('nonbonded', 0.0):.6f},"
                f"{energies.get('cavity_harmonic', 0.0):.6f},"
                f"{energies.get('cavity_coupling', 0.0):.6f},"
                f"{energies.get('cavity_dipole_self', 0.0):.6f}\n"
            )
            csv_file.flush()

            if next_snapshot_ps is not None and time_ps + 1e-9 >= next_snapshot_ps:
                _guard_molecular_com_velocity()
                pos_snap = context.getState(getPositions=True).getPositions(asNumpy=True)
                snapshot_times.append(time_ps)
                snapshot_positions.append(
                    np.asarray(pos_snap.value_in_unit(unit.nanometer), dtype=float)
                )
                while next_snapshot_ps is not None and time_ps + 1e-9 >= next_snapshot_ps:
                    next_snapshot_ps += snapshot_interval_ps

            if sample_idx % max(1, n_samples // 20) == 0 or sample_idx == n_samples:
                elapsed = wall_time.time() - t0
                rate = time_ps / elapsed if elapsed > 0 else 0.0
                print(
                    f"  t={time_ps:8.2f} ps  T_kin={T_kin:7.2f} K  "
                    f"T_v={T_v:7.2f} K  T_s={T_s if T_s is not None else 0:7.2f} K  "
                    f"[{rate:.1f} ps/s]"
                )

    elapsed = wall_time.time() - t0
    print(f"\nSimulation complete: {runtime_ps:.1f} ps in {elapsed:.1f} s")
    if fkt_tracker is not None:
        fkt_tracker.finalize()
        print(f"F(k,t) files written with prefix {output_prefix}_fkt_ref_*.txt")

    state = context.getState(getPositions=True, getVelocities=True)
    pos_final = state.getPositions(asNumpy=True).value_in_unit(unit.nanometer)
    vel_final = state.getVelocities(asNumpy=True).value_in_unit(
        unit.nanometer / unit.picosecond
    )
    final_path = Path(f"{output_prefix}_final_state.npz")
    np.savez(
        final_path,
        positions_nm=pos_final,
        velocities_nm_per_ps=vel_final,
        temperature_K=temperature_K,
        lambda_coupling=lambda_coupling,
        omega_c_cm1=omega_c_cm1,
        include_dipole_self_energy=include_dipole_self_energy,
        finite_q=finite_q,
        coupling_start_ps=coupling_start_ps,
    )
    print(f"Final state saved to {final_path}")

    meta_path = Path(f"{output_prefix}_meta.txt")
    with open(meta_path, "w", encoding="utf-8") as meta_file:
        meta_file.write(f"seed={seed}\n")
        meta_file.write(f"temperature_K={temperature_K}\n")
        meta_file.write(f"lambda_coupling={lambda_coupling}\n")
        meta_file.write(f"coupling_start_ps={coupling_start_ps}\n")
        meta_file.write(f"runtime_ps={runtime_ps}\n")
        meta_file.write(f"finite_q={finite_q}\n")
        meta_file.write(f"include_dipole_self_energy={include_dipole_self_energy}\n")
        meta_file.write(f"enable_fkt={enable_fkt}\n")
        if enable_fkt:
            meta_file.write(f"fkt_kmag_nm_inv={fkt_kmag_nm_inv}\n")
            meta_file.write(f"fkt_start_ps={fkt_t0_ps}\n")
            meta_file.write(f"fkt_sites={fkt_sites}\n")
            meta_file.write(
                f"fkt_n_sites={NUM_MOL if fkt_sites == 'molecular_com' else n_atoms}\n"
            )
        if initial_state is not None:
            meta_file.write(f"initial_state={initial_state}\n")
            meta_file.write(f"resample_velocities={resample_velocities}\n")
    print(f"Run metadata saved to {meta_path}")
    print(f"Energies logged to {csv_path}")

    if snapshots_out is not None and snapshot_positions:
        snapshots_out = Path(snapshots_out)
        snapshots_out.parent.mkdir(parents=True, exist_ok=True)
        np.savez_compressed(
            snapshots_out,
            positions_nm=np.stack(snapshot_positions, axis=0),
            times_ps=np.asarray(snapshot_times, dtype=float),
            temperature_K=temperature_K,
            lambda_coupling=lambda_coupling,
            omega_c_cm1=omega_c_cm1,
            seed=seed,
            snapshot_interval_ps=snapshot_interval_ps,
        )
        print(f"Snapshots saved to {snapshots_out} ({len(snapshot_times)} frames)")

    return final_path


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--temperature-K", type=float, default=100.0)
    parser.add_argument("--runtime-ps", type=float, default=1000.0)
    parser.add_argument("--lambda", dest="lam", type=float, default=0.09)
    parser.add_argument("--omega-c-cm1", type=float, default=OMEGA_C_CM1,
                        help="cavity frequency in cm^-1 (default: paper value)")
    parser.add_argument("--sample-interval-ps", type=float, default=1.0)
    parser.add_argument("--dt-ps", type=float, default=0.001)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--output-prefix", required=True)
    parser.add_argument("--initial-state", type=Path, default=None)
    parser.add_argument("--calibration-file", type=Path, default=REFERENCE_CALIBRATION_FILE)
    parser.add_argument("--platform", default=None)
    parser.add_argument("--with-dse", dest="with_dse", action="store_true", default=True)
    parser.add_argument("--no-dse", dest="with_dse", action="store_false")
    parser.add_argument("--finite-q", dest="finite_q", action="store_true", default=True)
    parser.add_argument("--no-finite-q", dest="finite_q", action="store_false")
    parser.add_argument(
        "--snapshot-interval-ps",
        type=float,
        default=0.0,
        help="If > 0, save position snapshots every this many ps",
    )
    parser.add_argument(
        "--snapshots-out",
        type=Path,
        default=None,
        help="Output .npz for position snapshots (requires --snapshot-interval-ps > 0)",
    )
    parser.add_argument(
        "--coupling-start-ps",
        type=float,
        default=0.0,
        help="Step-turn on cavity coupling at this time (ps); 0 = on from t=0",
    )
    parser.add_argument(
        "--no-resample-velocities",
        action="store_true",
        help="Use velocities from --initial-state instead of resampling at temperature-K",
    )
    parser.add_argument("--enable-fkt", action="store_true")
    parser.add_argument("--fkt-kmag-nm-inv", type=float, default=None)
    parser.add_argument("--fkt-num-wavevectors", type=int, default=50)
    parser.add_argument("--fkt-ref-interval-ps", type=float, default=200.0)
    parser.add_argument("--fkt-output-period-ps", type=float, default=1.0)
    parser.add_argument("--fkt-max-refs", type=int, default=13)
    parser.add_argument(
        "--fkt-start-ps",
        type=float,
        default=None,
        help="Start F(k,t) references at this time (default: coupling-start-ps)",
    )
    parser.add_argument(
        "--fkt-sites",
        choices=("atomic", "molecular_com"),
        default="atomic",
        help="Sites summed in rho_k: all atoms (500) or molecular COM (250)",
    )
    args = parser.parse_args()

    run_cavity_equilibrium(
        temperature_K=args.temperature_K,
        runtime_ps=args.runtime_ps,
        lambda_coupling=args.lam,
        include_dipole_self_energy=args.with_dse,
        output_prefix=args.output_prefix,
        seed=args.seed,
        dt_ps=args.dt_ps,
        sample_interval_ps=args.sample_interval_ps,
        initial_state=args.initial_state,
        calibration_file=args.calibration_file,
        platform_name=args.platform,
        finite_q=args.finite_q,
        omega_c_cm1=args.omega_c_cm1,
        snapshot_interval_ps=args.snapshot_interval_ps,
        snapshots_out=args.snapshots_out,
        coupling_start_ps=args.coupling_start_ps,
        resample_velocities=not args.no_resample_velocities,
        enable_fkt=args.enable_fkt,
        fkt_kmag_nm_inv=args.fkt_kmag_nm_inv,
        fkt_num_wavevectors=args.fkt_num_wavevectors,
        fkt_ref_interval_ps=args.fkt_ref_interval_ps,
        fkt_output_period_ps=args.fkt_output_period_ps,
        fkt_max_refs=args.fkt_max_refs,
        fkt_start_ps=args.fkt_start_ps,
        fkt_sites=args.fkt_sites,
    )


if __name__ == "__main__":
    main()
