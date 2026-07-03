#!/usr/bin/env python3
"""Log adaptive dt, force_max_norm, and effective epsilon around lambda turn-on."""

from __future__ import annotations

import argparse
import csv
import sys
from pathlib import Path

import numpy as np

_SCRIPT_DIR = Path(__file__).resolve().parent
if str(_SCRIPT_DIR) not in sys.path:
    sys.path.insert(0, str(_SCRIPT_DIR))

from run_c2f import (  # noqa: E402
    REFERENCE_CALIBRATION_FILE,
    OMEGA_C_CM1,
    PHOTON_MASS_AMU,
    BUSSI_TAU_PS,
    HARTREE_TO_CM1,
    build_mka_system,
    add_cavity_particle,
    remove_molecular_com_velocity,
    NUM_MOL,
)
from openmm.cavitymd import (  # noqa: E402
    DualThermostat,
    EnergyTracker,
    TemperatureTracker,
    EmpiricalTemperatureData,
    assign_force_groups,
    setup_gpu_step,
)
from openmm.cavitymd.adaptive import (  # noqa: E402
    calibrate_epsilon,
    create_adaptive_integrator,
    create_adaptive_state,
    advance_to_time_step_on,
    effective_epsilon_scaled,
    particle_masses_amu,
    DT_MAX_PS,
)

try:
    import openmm
    from openmm import unit
except ImportError:
    sys.exit("OpenMM (cavity-md) required.")


def _select_platform(name: str | None):
    if name:
        return openmm.Platform.getPlatformByName(name)
    for candidate in ("CUDA", "CPU", "Reference"):
        try:
            return openmm.Platform.getPlatformByName(candidate)
        except Exception:
            continue
    raise RuntimeError("No OpenMM platform available")


def run_diagnostic(
    *,
    seed: int,
    lambda_coupling: float,
    coupling_start_ps: float,
    window_before_ps: float,
    window_after_ps: float,
    sample_interval_ps: float,
    initial_state: Path,
    output_csv: Path,
    platform_name: str | None,
) -> None:
    np.random.seed(seed)
    temperature_K = 100.0
    omegac_au = OMEGA_C_CM1 / HARTREE_TO_CM1

    system, positions, n_atoms = build_mka_system(
        num_molecules=NUM_MOL, seed=seed
    )
    cavity_index = add_cavity_particle(system, positions)

    data = np.load(initial_state)
    pos_nm = np.asarray(data["positions_nm"], dtype=float)
    if pos_nm.shape[0] != n_atoms + 1:
        raise ValueError(
            f"Expected {n_atoms + 1} particles in {initial_state}, got {pos_nm.shape[0]}"
        )
    positions = [
        openmm.Vec3(*pos_nm[i]) * unit.nanometer for i in range(pos_nm.shape[0])
    ]

    cavity_force = openmm.CavityForce(cavity_index, omegac_au, 0.0, PHOTON_MASS_AMU)
    cavity_force.setIncludeDipoleSelfEnergy(True)
    setup_gpu_step(cavity_force, lambda_coupling, start_time_ps=coupling_start_ps)
    system.addForce(cavity_force)

    DualThermostat.setup_bussi_for_system(
        system, list(range(n_atoms)), temperature_K, BUSSI_TAU_PS
    )
    group_map = assign_force_groups(system, include_dipole_self_energy=True)

    integrator = create_adaptive_integrator(DT_MAX_PS)
    platform = _select_platform(platform_name)
    context = openmm.Context(system, integrator, platform)
    context.setPositions(positions)
    context.setVelocitiesToTemperature(temperature_K * unit.kelvin)
    remove_molecular_com_velocity(context, system, n_atoms)

    masses_amu = particle_masses_amu(system)
    eps_relaxed, fmn0 = calibrate_epsilon(context, system, target_dt_ps=DT_MAX_PS)

    energy_tracker = EnergyTracker(
        context, cavity_force, group_map, n_atoms, cavity_index
    )
    empirical_structural = EmpiricalTemperatureData(
        str(REFERENCE_CALIBRATION_FILE), energy_component="lj_coulombic"
    )
    empirical_harmonic = EmpiricalTemperatureData(
        str(REFERENCE_CALIBRATION_FILE), energy_component="harmonic"
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

    t_start = max(0.0, coupling_start_ps - window_before_ps)
    t_end = coupling_start_ps + window_after_ps
    context.setTime(t_start * unit.picosecond)

    adaptive_state = create_adaptive_state(
        lambda_coupling,
        coupling_start_ps,
        initial_time_ps=t_start,
        eps_relaxed=eps_relaxed,
    )

    output_csv.parent.mkdir(parents=True, exist_ok=True)
    rows: list[dict[str, float]] = []

    target = t_start
    while target <= t_end + 1e-12:
        target += sample_interval_ps
        if target > t_end:
            target = t_end

        advance_to_time_step_on(
            context,
            integrator,
            thermostat,
            system=system,
            target_time_ps=target,
            lambda_coupling=lambda_coupling,
            coupling_start_ps=coupling_start_ps,
            state=adaptive_state,
            masses_amu=masses_amu,
        )

        time_ps = context.getState().getTime().value_in_unit(unit.picosecond)
        dt_ps = integrator.getStepSize().value_in_unit(unit.picosecond)
        eps_eff = effective_epsilon_scaled(
            time_ps,
            adaptive_state,
            coupling_start_ps,
            lambda_coupling,
            dt_ps,
        )
        fmn = float(adaptive_state.get("last_force_max_norm") or fmn0)
        temps = temp_tracker.get_all()
        rows.append(
            {
                "time_ps": time_ps,
                "dt_ps": dt_ps,
                "dt_fs": dt_ps * 1000.0,
                "eps_effective": eps_eff,
                "eps_relaxed": eps_relaxed,
                "force_max_norm": fmn,
                "ramp_t0_ps": adaptive_state.get("ramp_t0") or -1.0,
                "T_kin_K": float(temps.get("kinetic", float("nan"))),
                "T_v_K": float(temps.get("harmonic_equipartition", float("nan"))),
            }
        )

    with output_csv.open("w", newline="", encoding="utf-8") as fh:
        writer = csv.DictWriter(fh, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)

    min_dt_fs = min(r["dt_fs"] for r in rows)
    max_t_kin = max(r["T_kin_K"] for r in rows)
    print(f"Wrote {output_csv} ({len(rows)} samples)")
    print(f"  eps_relaxed = {eps_relaxed:.6e}  force_max_norm_init = {fmn0:.6e}")
    print(f"  min dt = {min_dt_fs:.6g} fs")
    print(f"  max T_kin = {max_t_kin:.4g} K")
    print(f"  ramp_t0 = {adaptive_state.get('ramp_t0')}")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--seed", type=int, default=89)
    parser.add_argument("--lambda", dest="lam", type=float, default=0.03)
    parser.add_argument("--coupling-start-ps", type=float, default=200.0)
    parser.add_argument("--window-before-ps", type=float, default=50.0)
    parser.add_argument("--window-after-ps", type=float, default=50.0)
    parser.add_argument("--sample-interval-ps", type=float, default=0.1)
    parser.add_argument(
        "--initial-state",
        type=Path,
        default=_SCRIPT_DIR / "equilibrium_output" / "eq10ns100K_lam0_final_state.npz",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=Path("diagnostics/adaptive_switch_seed089.csv"),
    )
    parser.add_argument("--platform", default=None)
    args = parser.parse_args()

    if not args.initial_state.is_file():
        raise FileNotFoundError(args.initial_state)

    run_diagnostic(
        seed=args.seed,
        lambda_coupling=args.lam,
        coupling_start_ps=args.coupling_start_ps,
        window_before_ps=args.window_before_ps,
        window_after_ps=args.window_after_ps,
        sample_interval_ps=args.sample_interval_ps,
        initial_state=args.initial_state,
        output_csv=args.output,
        platform_name=args.platform,
    )


if __name__ == "__main__":
    main()
