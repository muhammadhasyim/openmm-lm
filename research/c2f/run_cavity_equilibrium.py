#!/usr/bin/env python
"""NVT cavity equilibrium for the mKA system at fixed bath temperature."""

from __future__ import annotations

import argparse
import math
import sys
import time as wall_time
import zipfile
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
from openmm.cavitymd.adaptive import (
    DT_MAX_PS,
    calibrate_epsilon,
    create_adaptive_integrator,
    create_adaptive_state,
    advance_to_time_step_on,
    particle_masses_amu,
)

# Abort production trajectories when kinetic temperature exceeds this (K).
STABILITY_T_KIN_MAX_K = 5000.0

from run_c2f import (
    REFERENCE_CALIBRATION_FILE,
    OMEGA_C_CM1,
    PHOTON_MASS_AMU,
    NUM_MOL,
    FRAC_AA,
    CHARGE_MAG,
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

from checkpoint_utils import (
    archive_replica_outputs,
    archive_stale_partial_outputs,
    checkpoint_path,
    energies_csv_path,
    is_poisoned_checkpoint,
    load_checkpoint,
    read_csv_last_time_ps,
    save_checkpoint,
    trajectory_complete,
)

FKT_KMAG_NM_INV_DEFAULT = FKT_KMAG_AU / BOHR_TO_NM

_SCRIPT_DIR = Path(__file__).resolve().parent

DipoleWindow = tuple[float, float]


def _molecular_charges(n_atoms: int, num_molecules: int = NUM_MOL) -> np.ndarray:
    """Partial charges for mKA atoms (excluding cavity)."""
    n_aa = int(round(num_molecules * FRAC_AA))
    charges = np.zeros(n_atoms, dtype=np.float64)
    for mol in range(num_molecules):
        sign = 1.0 if mol < n_aa else -1.0
        atom0 = 2 * mol
        charges[atom0] = sign * CHARGE_MAG
        charges[atom0 + 1] = -sign * CHARGE_MAG
    return charges


def _dipole_nm(positions_nm: np.ndarray, charges: np.ndarray, n_atoms: int) -> np.ndarray:
    """Total dipole moment (nm * e) from atomic positions and partial charges."""
    pos_atomic = np.asarray(positions_nm[:n_atoms], dtype=np.float64)
    return np.sum(pos_atomic * charges[:, None], axis=0)


def _time_in_dipole_window(
    time_ps: float,
    windows: list[DipoleWindow],
) -> int | None:
    """Return window index if time_ps lies in [start, start+length), else None."""
    for idx, (start_ps, length_ps) in enumerate(windows):
        if start_ps - 1e-9 <= time_ps < start_ps + length_ps - 1e-9:
            return idx
    return None


def _adaptive_ramp_start_ps(
    coupling_start_ps: float,
    lambda_coupling: float,
) -> float | None:
    """Time anchor (ps) for the adaptive error-tolerance ramp.

    The SI adaptive scheme tightens the integration error tolerance at a
    lambda turn-on edge and relaxes it back over ``TAU_RAMP_PS`` (see
    ``openmm.cavitymd.adaptive.epsilon_tolerance``). This returns the edge
    time to anchor that ramp:

    - ``None`` when ``lambda_coupling == 0`` (no turn-on; tolerance stays
      relaxed at ``EPS_STAR_NM`` for the whole stable lambda=0 trajectory),
    - ``max(0, coupling_start_ps)`` otherwise, so a delayed turn-on tightens
      exactly at ``coupling_start_ps`` while an immediate coupling
      (``coupling_start_ps <= 0``) tightens from ``t = 0``.

    Parameters
    ----------
    coupling_start_ps : float
        Time at which lambda steps from 0 to ``lambda_coupling``.
    lambda_coupling : float
        Target light-matter coupling (dimensionless).

    Returns
    -------
    float or None
        The ramp anchor time in ps, or None when there is no turn-on edge.
    """
    if lambda_coupling == 0.0:
        return None
    return max(0.0, coupling_start_ps)


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
    dipole_windows: list[DipoleWindow] | None = None,
    dipole_interval_ps: float = 0.001,
    num_molecules: int = NUM_MOL,
    adaptive: bool = False,
    dt_max_ps: float = DT_MAX_PS,
    no_resume: bool = False,
) -> Path:
    """Run fixed-T NVT with cavity coupling; return final-state path.

    When *coupling_start_ps* > 0, lambda stays at 0 until that time then steps
    to *lambda_coupling* and remains on (single turn-on, no square wave).

    Parameters
    ----------
    adaptive : bool, optional
        When True, integrate with cav-hoomd max-metric adaptive timestepping
        (``openmm.cavitymd.adaptive``): plain Verlet with externally set dt from
        N·max(|F|/m), epsilon calibrated to *dt_max_ps*, and shock ramp at the
        lambda turn-on edge. Default False preserves fixed-step Verlet.
    dt_max_ps : float, optional
        Maximum step size (ps) for the adaptive integrator. Ignored when
        ``adaptive`` is False. Defaults to 1.0 fs.
    no_resume : bool, optional
        When True, ignore any existing checkpoint and start fresh from IC.
    """
    if fkt_kmag_nm_inv is None:
        fkt_kmag_nm_inv = FKT_KMAG_NM_INV_DEFAULT

    output_prefix_path = Path(output_prefix)
    ckpt_path = checkpoint_path(output_prefix)
    csv_path = energies_csv_path(output_prefix)
    final_path = Path(f"{output_prefix}_final_state.npz")
    if final_path.exists() and trajectory_complete(read_csv_last_time_ps(csv_path), runtime_ps):
        print(f"\n=== Skipping complete trajectory ({output_prefix}) ===")
        return final_path

    resuming = False
    resume_time_ps = 0.0
    checkpoint: dict = {}
    if no_resume and ckpt_path.exists():
        archived = archive_replica_outputs(
            output_prefix,
            reason="no_resume",
            runtime_ps=runtime_ps,
            lambda_coupling=lambda_coupling,
        )
        if archived is not None:
            print(f"  Archived prior outputs (--no-resume) -> {archived}")
    if ckpt_path.exists() and not no_resume:
        try:
            checkpoint = load_checkpoint(ckpt_path)
        except (OSError, ValueError, KeyError, zipfile.BadZipFile):
            archived = archive_replica_outputs(
                output_prefix,
                reason="corrupt_checkpoint",
                runtime_ps=runtime_ps,
                lambda_coupling=lambda_coupling,
            )
            if archived is not None:
                print(f"  Archived corrupt checkpoint -> {archived}")
            checkpoint = {}
            resume_time_ps = 0.0
        else:
            resume_time_ps = float(checkpoint["time_ps"])
            if trajectory_complete(resume_time_ps, runtime_ps):
                print(f"\n=== Checkpoint complete ({resume_time_ps:.1f} ps); skipping ===")
                return final_path if final_path.exists() else ckpt_path
            if is_poisoned_checkpoint(checkpoint, csv_path, t_kin_max_k=STABILITY_T_KIN_MAX_K):
                archived = archive_replica_outputs(
                    output_prefix,
                    reason="poisoned",
                    runtime_ps=runtime_ps,
                    lambda_coupling=lambda_coupling,
                )
                if archived is not None:
                    print(f"  Archived poisoned checkpoint -> {archived}")
                checkpoint = {}
                resume_time_ps = 0.0
            else:
                resuming = True
                print(f"\n=== Resuming from checkpoint at t={resume_time_ps:.3f} ps ===")
    elif not trajectory_complete(read_csv_last_time_ps(csv_path), runtime_ps):
        archived = archive_stale_partial_outputs(
            output_prefix, runtime_ps=runtime_ps, reason="no_checkpoint"
        )
        if archived is not None:
            print(f"  Archived stale partial outputs -> {archived}")

    print("\n=== Cavity NVT equilibrium ===")
    print(f"  T_bath           = {temperature_K} K")
    print(f"  runtime          = {runtime_ps} ps")
    print(f"  lambda           = {lambda_coupling}")
    print(f"  num_molecules    = {num_molecules}")
    print(f"  g = λ√N          = {lambda_coupling * (num_molecules ** 0.5):.6g}")
    print(f"  omega_c          = {omega_c_cm1} cm^-1")
    print(f"  dipole self-energy = {'ON' if include_dipole_self_energy else 'OFF'}")
    print(f"  finite-q shift   = {'ON' if finite_q else 'OFF'}")
    if coupling_start_ps > 0:
        print(f"  coupling on at   = {coupling_start_ps} ps (step, stays on)")
    if adaptive:
        ramp_desc = (
            f"ramp@{_adaptive_ramp_start_ps(coupling_start_ps, lambda_coupling)} ps"
            if lambda_coupling != 0.0
            else "relaxed (no turn-on)"
        )
        print(f"  integrator       = Verlet (max-metric adaptive, dt_max={dt_max_ps*1000:.2f} fs, {ramp_desc})")
    else:
        print(f"  integrator       = Verlet (fixed dt={dt_ps*1000:.2f} fs)")
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
    if dipole_windows:
        print(
            f"  dipole windows = {len(dipole_windows)} x interval {dipole_interval_ps} ps "
            f"for IR ({dipole_windows})"
        )
    print(f"  initial state    = {initial_state or '(fresh build, seed=' + str(seed) + ')'}")

    omegac_au = omega_c_cm1 / HARTREE_TO_CM1
    system, positions, n_atoms = build_mka_system(
        num_molecules=num_molecules,
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

    adaptive_ramp_start_ps = _adaptive_ramp_start_ps(coupling_start_ps, lambda_coupling)
    adaptive_state = None
    masses_amu: list[float] | None = None
    eps_relaxed: float | None = None
    force_max_norm_initial: float | None = None
    if adaptive:
        integrator = create_adaptive_integrator(dt_max_ps)
        resume_time_for_state = resume_time_ps if resuming else 0.0
    else:
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

    if resuming:
        checkpoint = load_checkpoint(ckpt_path)
        pos_nm = np.asarray(checkpoint["positions_nm"], dtype=float)
        vel_nm = np.asarray(checkpoint["velocities_nm_per_ps"], dtype=float)
        context.setPositions(
            [openmm.Vec3(*pos_nm[i]) * unit.nanometer for i in range(pos_nm.shape[0])]
        )
        context.setVelocities(
            [
                openmm.Vec3(*vel_nm[i]) * (unit.nanometer / unit.picosecond)
                for i in range(vel_nm.shape[0])
            ]
        )
        context.setTime(resume_time_ps * unit.picosecond)
        remove_molecular_com_velocity(context, system, n_atoms)
    elif displacer is not None and coupling_start_ps <= 0:
        displacer.displaceToEquilibrium(context, lambda_coupling)
        print(f"  Photon displaced to finite-q equilibrium (lambda={lambda_coupling})")

    if adaptive:
        masses_amu = particle_masses_amu(system)
        eps_relaxed, force_max_norm_initial = calibrate_epsilon(
            context, system, target_dt_ps=dt_max_ps
        )
        resume_time_for_state = resume_time_ps if resuming else 0.0
        adaptive_state = create_adaptive_state(
            lambda_coupling,
            coupling_start_ps,
            initial_time_ps=resume_time_for_state,
            eps_relaxed=eps_relaxed,
        )
        if resuming and adaptive_ramp_start_ps is not None:
            adaptive_state["ramp_t0"] = adaptive_ramp_start_ps
        print(
            f"  adaptive eps       = {eps_relaxed:.6e}  "
            f"(force_max_norm={force_max_norm_initial:.6e})"
        )

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
        num_molecules=num_molecules,
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

    csv_path = str(energies_csv_path(output_prefix))
    n_samples = max(1, int(round(runtime_ps / sample_interval_ps)))
    sample_dt_ps = runtime_ps / n_samples
    print(f"  {n_samples} samples over {runtime_ps} ps")

    snapshot_times: list[float] = []
    snapshot_positions: list[np.ndarray] = []
    next_snapshot_ps = snapshot_interval_ps if snapshot_interval_ps > 0 else None

    dipole_charges: np.ndarray | None = None
    dipole_recordings: list[tuple[list[float], list[np.ndarray]]] = []
    dipole_next_sample_ps: list[float] = []
    if dipole_windows:
        dipole_charges = _molecular_charges(n_atoms, num_molecules)
        dipole_recordings = [([], []) for _ in dipole_windows]
        dipole_next_sample_ps = [start_ps for start_ps, _ in dipole_windows]

    fkt_tracker = None
    fkt_t0_ps = 0.0
    next_fkt_ps = 0.0
    if enable_fkt:
        from fkt_tracker import FKTTracker, fkt_positions_nm

        fkt_t0_ps = fkt_start_ps if fkt_start_ps is not None else coupling_start_ps
        if resuming and "fkt_state" in checkpoint:
            fkt_tracker = FKTTracker.from_state_dict(
                checkpoint["fkt_state"], output_prefix=output_prefix
            )
            last_rel = fkt_tracker.last_output_time_ps
            if last_rel is not None:
                next_fkt_ps = fkt_t0_ps + last_rel + fkt_output_period_ps
            else:
                next_fkt_ps = max(fkt_t0_ps, resume_time_ps)
        else:
            fkt_tracker = FKTTracker(
                kmag_nm_inv=fkt_kmag_nm_inv,
                num_wavevectors=fkt_num_wavevectors,
                reference_interval_ps=fkt_ref_interval_ps,
                max_references=fkt_max_refs,
                output_period_ps=fkt_output_period_ps,
                output_prefix=output_prefix,
            )
            next_fkt_ps = fkt_t0_ps

    if resuming and dipole_windows and "dipole_state" in checkpoint:
        dipole_state = checkpoint["dipole_state"]
        dipole_next_sample_ps = list(dipole_state["next_sample_ps"])
        dipole_recordings = []
        for rec in dipole_state["recordings"]:
            times_out = list(rec["times_ps"])
            dipoles_out = [np.asarray(row, dtype=np.float64) for row in rec["dipoles_nm"]]
            dipole_recordings.append((times_out, dipoles_out))

    if resuming and "snapshot_times_ps" in checkpoint:
        snapshot_times = list(np.asarray(checkpoint["snapshot_times_ps"], dtype=float))
        snapshot_positions = [
            np.asarray(frame, dtype=np.float64)
            for frame in checkpoint["snapshot_positions_nm"]
        ]
        if next_snapshot_ps is not None and snapshot_times:
            next_snapshot_ps = snapshot_times[-1] + snapshot_interval_ps

    first_sample_idx = 1
    if resuming:
        first_sample_idx = int(round(resume_time_ps / sample_dt_ps)) + 1
        if first_sample_idx > n_samples:
            print(f"  Resume time {resume_time_ps:.3f} ps >= runtime; nothing to do.")
            return final_path if final_path.exists() else ckpt_path

    def _maybe_update_fkt(time_ps: float) -> None:
        nonlocal next_fkt_ps
        if fkt_tracker is None or time_ps + 1e-9 < fkt_t0_ps:
            return
        while time_ps + 1e-9 >= next_fkt_ps:
            state_fkt = context.getState(getPositions=True, enforcePeriodicBox=True)
            pos_all = state_fkt.getPositions(asNumpy=True).value_in_unit(unit.nanometer)
            pos_atomic = np.asarray(pos_all[:n_atoms], dtype=np.float64)
            pos_fkt = fkt_positions_nm(pos_atomic, num_molecules, site_mode=fkt_sites)
            fkt_tracker.update(time_ps - fkt_t0_ps, pos_fkt)
            next_fkt_ps += fkt_output_period_ps

    def _maybe_sample_dipole(time_ps: float) -> None:
        """Read-only dipole samples on the configured time grid (no velocity writes)."""
        if dipole_charges is None or not dipole_windows:
            return
        win_idx = _time_in_dipole_window(time_ps, dipole_windows)
        if win_idx is None:
            return
        start_ps, length_ps = dipole_windows[win_idx]
        end_ps = start_ps + length_ps
        times_out, dipoles_out = dipole_recordings[win_idx]
        next_ps = dipole_next_sample_ps[win_idx]
        while time_ps + 1e-9 >= next_ps and next_ps < end_ps - 1e-9:
            state_dip = context.getState(getPositions=True, enforcePeriodicBox=True)
            pos_all = state_dip.getPositions(asNumpy=True).value_in_unit(unit.nanometer)
            dipoles_out.append(_dipole_nm(pos_all, dipole_charges, n_atoms))
            times_out.append(next_ps)
            next_ps += dipole_interval_ps
            dipole_next_sample_ps[win_idx] = next_ps

    def _next_dipole_sample_time(time_ps: float) -> float | None:
        if dipole_charges is None or not dipole_windows:
            return None
        win_idx = _time_in_dipole_window(time_ps, dipole_windows)
        if win_idx is None:
            return None
        start_ps, length_ps = dipole_windows[win_idx]
        end_ps = start_ps + length_ps
        next_ps = dipole_next_sample_ps[win_idx]
        if next_ps >= end_ps - 1e-9:
            return None
        return next_ps

    def _advance_adaptive_to(target_time_ps: float) -> None:
        """Integrate to target_time_ps; FKT in on_step, dipole at grid points only."""
        while True:
            time_ps = context.getState().getTime().value_in_unit(unit.picosecond)
            if time_ps >= target_time_ps - 1e-15:
                break

            sub_target = target_time_ps
            dipole_t = _next_dipole_sample_time(time_ps)
            if dipole_t is not None and dipole_t < sub_target:
                sub_target = dipole_t

            def _on_fkt_step(step_time_ps: float, _step_dt_ps: float) -> None:
                _maybe_update_fkt(step_time_ps)

            advance_to_time_step_on(
                context,
                integrator,
                thermostat,
                system=system,
                target_time_ps=sub_target,
                lambda_coupling=lambda_coupling,
                coupling_start_ps=coupling_start_ps,
                state=adaptive_state,
                masses_amu=masses_amu,
                on_step=_on_fkt_step,
            )
            time_ps = context.getState().getTime().value_in_unit(unit.picosecond)
            if dipole_t is not None and time_ps + 1e-9 >= dipole_t:
                _maybe_sample_dipole(time_ps)

    def _dipole_state_dict() -> dict | None:
        if not dipole_windows:
            return None
        return {
            "next_sample_ps": dipole_next_sample_ps,
            "recordings": [
                {
                    "times_ps": times_out,
                    "dipoles_nm": [np.asarray(d, dtype=np.float64) for d in dipoles_out],
                }
                for times_out, dipoles_out in dipole_recordings
            ],
        }

    def _write_checkpoint(time_ps: float) -> None:
        state = context.getState(getPositions=True, getVelocities=True)
        pos_ckpt = state.getPositions(asNumpy=True).value_in_unit(unit.nanometer)
        vel_ckpt = state.getVelocities(asNumpy=True).value_in_unit(
            unit.nanometer / unit.picosecond
        )
        fkt_state = fkt_tracker.to_state_dict() if fkt_tracker is not None else None
        snap_times = np.asarray(snapshot_times, dtype=float) if snapshot_times else None
        snap_pos = (
            np.stack(snapshot_positions, axis=0)
            if snapshot_positions
            else None
        )
        save_checkpoint(
            ckpt_path,
            time_ps=time_ps,
            positions_nm=np.asarray(pos_ckpt, dtype=np.float64),
            velocities_nm_per_ps=np.asarray(vel_ckpt, dtype=np.float64),
            fkt_state=fkt_state,
            dipole_state=_dipole_state_dict(),
            snapshot_times_ps=snap_times,
            snapshot_positions_nm=snap_pos,
        )

    csv_mode = "a" if resuming else "w"
    with open(csv_path, csv_mode, encoding="utf-8") as csv_file:
        if not resuming:
            csv_file.write(
                "time_ps,T_bath_K,T_kinetic_K,T_v_fictive_K,T_s_fictive_K,"
                "E_kinetic_kjmol,E_potential_kjmol,E_mech_kjmol,"
                "E_bond_kjmol,E_nonbonded_kjmol,E_cav_harmonic_kjmol,"
                "E_cav_coupling_kjmol,E_cav_dse_kjmol\n"
            )
        t0 = wall_time.time()
        for sample_idx in range(first_sample_idx, n_samples + 1):
            target_time = sample_idx * sample_dt_ps
            if adaptive:
                _advance_adaptive_to(target_time)
            else:
                steps = max(1, int(round((target_time - context.getState().getTime().value_in_unit(unit.picosecond)) / dt_ps)))
                for _ in range(steps):
                    integrator.step(1)
                    thermostat.apply_cavity_thermostat_step(dt_ps)
                    step_time_ps = context.getState().getTime().value_in_unit(unit.picosecond)
                    _maybe_update_fkt(step_time_ps)
                    _maybe_sample_dipole(step_time_ps)

            time_ps = context.getState().getTime().value_in_unit(unit.picosecond)
            energies = energy_tracker.get_energies()
            temps = temp_tracker.get_all()
            T_kin = temps.get("kinetic", float("nan"))
            T_v = temps.get("harmonic_equipartition", float("nan"))
            T_s = temps.get("structural_fictive")
            e_kin = energies.get("total_kinetic", 0.0)
            e_pot = energies.get("total_potential", 0.0)
            if (
                not math.isfinite(T_kin)
                or T_kin > STABILITY_T_KIN_MAX_K
                or not math.isfinite(e_kin)
                or not math.isfinite(e_pot)
            ):
                _write_checkpoint(time_ps)
                dt_last = (
                    integrator.getStepSize().value_in_unit(unit.picosecond)
                    if adaptive
                    else dt_ps
                )
                raise RuntimeError(
                    f"Numerical instability at t={time_ps:.6f} ps: "
                    f"T_kin={T_kin:.4g} K (limit {STABILITY_T_KIN_MAX_K:g} K), "
                    f"dt={dt_last:.3e} ps, seed={seed}, "
                    f"lambda={lambda_coupling}, coupling_start_ps={coupling_start_ps}"
                )
            T_s_str = "" if T_s is None else f"{T_s:.4f}"
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
            _write_checkpoint(time_ps)

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
    if ckpt_path.exists():
        ckpt_path.unlink()
    print(f"Final state saved to {final_path}")

    meta_path = Path(f"{output_prefix}_meta.txt")
    with open(meta_path, "w", encoding="utf-8") as meta_file:
        meta_file.write(f"seed={seed}\n")
        meta_file.write(f"temperature_K={temperature_K}\n")
        meta_file.write(f"lambda_coupling={lambda_coupling}\n")
        meta_file.write(f"coupling_start_ps={coupling_start_ps}\n")
        meta_file.write(f"runtime_ps={runtime_ps}\n")
        meta_file.write(f"adaptive={adaptive}\n")
        if adaptive:
            meta_file.write(f"dt_max_ps={dt_max_ps}\n")
            meta_file.write("integrator_metric=max_force\n")
            if eps_relaxed is not None:
                meta_file.write(f"eps_relaxed={eps_relaxed}\n")
            if force_max_norm_initial is not None:
                meta_file.write(f"force_max_norm_initial={force_max_norm_initial}\n")
            meta_file.write(
                f"adaptive_ramp_start_ps={_adaptive_ramp_start_ps(coupling_start_ps, lambda_coupling)}\n"
            )
        else:
            meta_file.write(f"dt_ps={dt_ps}\n")
        meta_file.write(f"finite_q={finite_q}\n")
        meta_file.write(f"include_dipole_self_energy={include_dipole_self_energy}\n")
        meta_file.write(f"enable_fkt={enable_fkt}\n")
        if enable_fkt:
            meta_file.write(f"fkt_kmag_nm_inv={fkt_kmag_nm_inv}\n")
            meta_file.write(f"fkt_start_ps={fkt_t0_ps}\n")
            meta_file.write(f"fkt_sites={fkt_sites}\n")
            meta_file.write(
                f"fkt_n_sites={num_molecules if fkt_sites == 'molecular_com' else n_atoms}\n"
            )
        if initial_state is not None:
            meta_file.write(f"initial_state={initial_state}\n")
            meta_file.write(f"resample_velocities={resample_velocities}\n")
        if dipole_windows:
            meta_file.write(f"dipole_interval_ps={dipole_interval_ps}\n")
            meta_file.write(f"dipole_windows={dipole_windows}\n")
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

    if dipole_windows and dipole_recordings:
        dipole_path = Path(f"{output_prefix}_dipole.npz")
        payload: dict[str, object] = {
            "dipole_interval_ps": dipole_interval_ps,
            "window_starts_ps": np.asarray([w[0] for w in dipole_windows], dtype=float),
            "window_lengths_ps": np.asarray([w[1] for w in dipole_windows], dtype=float),
        }
        n_saved = 0
        for idx, (times_out, dipoles_out) in enumerate(dipole_recordings):
            if not times_out:
                continue
            payload[f"window_{idx}_times_ps"] = np.asarray(times_out, dtype=float)
            payload[f"window_{idx}_dipole_nm"] = np.stack(dipoles_out, axis=0)
            n_saved += 1
        if n_saved > 0:
            np.savez_compressed(dipole_path, **payload)
            print(f"Dipole windows saved to {dipole_path} ({n_saved} windows)")

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
    parser.add_argument(
        "--adaptive",
        dest="adaptive",
        action="store_true",
        default=False,
        help="Use cav-hoomd max-metric adaptive Verlet (dt from N·max|F|/m)",
    )
    parser.add_argument(
        "--no-adaptive",
        dest="adaptive",
        action="store_false",
        help="Use a fixed-dt Verlet integrator (default)",
    )
    parser.add_argument(
        "--dt-max-ps",
        type=float,
        default=DT_MAX_PS,
        help=f"Max step size (ps) for the adaptive integrator (default {DT_MAX_PS})",
    )
    parser.add_argument(
        "--no-resume",
        action="store_true",
        help="Ignore checkpoint and start fresh from IC (archives prior outputs)",
    )
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument(
        "--num-molecules",
        type=int,
        default=NUM_MOL,
        help=f"Number of mKA molecules (default {NUM_MOL})",
    )
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
    parser.add_argument(
        "--dipole-window",
        action="append",
        nargs=2,
        metavar=("START_PS", "LENGTH_PS"),
        type=float,
        default=None,
        help="Record dipole mu(t) in [START, START+LENGTH) ps; repeat for multiple windows",
    )
    parser.add_argument(
        "--dipole-interval-ps",
        type=float,
        default=0.001,
        help="Dipole sampling interval inside windows (default: 1 fs)",
    )
    args = parser.parse_args()

    dipole_windows: list[DipoleWindow] | None = None
    if args.dipole_window:
        dipole_windows = [(float(start), float(length)) for start, length in args.dipole_window]

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
        dipole_windows=dipole_windows,
        dipole_interval_ps=args.dipole_interval_ps,
        num_molecules=args.num_molecules,
        adaptive=args.adaptive,
        dt_max_ps=args.dt_max_ps,
        no_resume=args.no_resume,
    )


if __name__ == "__main__":
    main()
