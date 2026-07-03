#!/usr/bin/env python
"""
Reproduce Figure 5b — C2F cooling (300 K -> T_g ~ 32 K)
========================================================
Paper: "Non-Thermal Aging of Supercooled Liquids in Optical Cavities"
       Hasyim, Damiani, Hoffmann  (arXiv:2603.15693)

Runs the paper-scale C2F ensemble:
  1. Shared empirical calibration (vartheta(T) for structural T_s)
  2. N independent trajectories with distinct seeds
  3. Ensemble-averaged time series -> fig5_averaged.csv

Run via pixi:
    pixi run -e test python research/c2f/reproduce_figure5.py
    pixi run -e test python research/c2f/reproduce_figure5.py --quick

Note: Paper ICs come from 150 ns NVT equilibration; here distinct seeds
randomize lattice orientations and initial velocities as a practical stand-in.
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np

try:
    from scipy.interpolate import interp1d
except ImportError:
    sys.exit("scipy required (available in pixi test env: pixi run -e test ...)")

# Import protocol building blocks from the sibling module
_SCRIPT_DIR = Path(__file__).resolve().parent
if str(_SCRIPT_DIR) not in sys.path:
    sys.path.insert(0, str(_SCRIPT_DIR))

from run_c2f import (  # noqa: E402
    REFERENCE_CALIBRATION_FILE,
    build_mka_system,
    run_c2f,
    run_equilibrium_calibration,
    validate_calibration_file,
    crosscheck_calibration_against_reference,
)

# Figure 5 defaults (Methods + Fig. 5 caption + SI Section 5 + cav-hoomd square_wave_diffeq.sh)
FIG5_INITIAL_T_K = 300.0
FIG5_LAMBDA = 0.09
# Coupling + DiffEq turn on at 10 ps to match cav-hoomd Figure 5 (Finite-q: False).
FIG5_COUPLING_START_PS = 10.0
FIG5_PERIOD_PS = 10.0
FIG5_DUTY_CYCLE = 0.10
FIG5_FEEDBACK_INTERVAL_PS = 0.01  # Controller update interval (cav-hoomd diffeq default)
FIG5_CSV_INTERVAL_PS = 0.1  # CSV sample interval; bath feedback on each λ-off window (forced)
FIG5_RUNTIME_PS = 150.0
FIG5_N_TRAJ = 500
FIG5_EQUIL_PS = 200.0
FIG5_MAX_PRE_EQUIL_PS = 1000.0
FIG5_POST_CAVITY_EQUIL_PS = 0.0
FIG5_OPENMM_CALIBRATION_NAME = "fig5_openmm_calibration.txt"
FIG5_TS_BIAS_MAX_K = 10.0
FIG5_MAX_POST_EQUIL_PS = 500.0
FIG5_CALIBRATION_RUN_PS = 500.0
FIG5_DIFFEQ_TAU_PS = 1.0
FIG5_T_MIN = 0.01
# Faithful cav-hoomd Figure 5: q≈0 photon (no displacement) + every-step
# instantaneous T_s feedback (--diffeq-update-interval 0).
FIG5_FINITE_Q = False
FIG5_FEEDBACK_EVERY_STEP = True
FIG5_CALIBRATION_TEMPS = np.array(
    [30, 50, 75, 100, 125, 150, 200, 250, 300, 400, 500], dtype=float
)
FIG5_QUICK_CALIBRATION_TEMPS = np.array(
    [50, 100, 200, 300, 400, 500], dtype=float
)
FIG5_BASE_SEED = 42


def _ensemble_average(output_dir: Path, prefix: str, dt_ps: float,
                      runtime_ps: float) -> Path:
    """Interpolate per-seed CSVs onto a common grid and average."""
    csv_files = sorted(output_dir.glob(f"{prefix}_seed*_energies.csv"))
    if not csv_files:
        raise FileNotFoundError(
            f"No trajectory CSVs matching {prefix}_seed*_energies.csv in {output_dir}"
        )

    time_grid = np.arange(0.0, runtime_ps + 0.5 * dt_ps, dt_ps)
    columns = [
        "T_bath_K",
        "T_kinetic_K",
        "T_v_fictive_K",
        "T_s_fictive_K",
    ]

    stacked = {col: [] for col in columns}

    for csv_path in csv_files:
        data = np.genfromtxt(
            csv_path, delimiter=",", names=True, missing_values="", usemask=False
        )
        t = np.asarray(data["time_ps"], dtype=float)
        if t.size == 0:
            continue

        for col in columns:
            y = np.asarray(data[col], dtype=float)
            # Empty strings in T_s become NaN via genfromtxt
            y = np.where(np.isfinite(y), y, np.nan)
            if np.all(np.isnan(y)):
                interp_y = np.full_like(time_grid, np.nan)
            else:
                valid = np.isfinite(y)
                f = interp1d(
                    t[valid], y[valid],
                    kind="linear", bounds_error=False, fill_value=np.nan,
                )
                interp_y = f(time_grid)
            stacked[col].append(interp_y)

    n_traj = len(stacked["T_bath_K"])
    avg_path = output_dir / f"{prefix}_averaged.csv"
    col_pairs = [f"{c},{c}_std" for c in columns]
    with open(avg_path, "w") as out:
        out.write("time_ps," + ",".join(col_pairs) + "\n")
        for i, t in enumerate(time_grid):
            row = [f"{t:.6f}"]
            for col in columns:
                arr = np.array(stacked[col])
                vals = arr[:, i]
                valid = vals[np.isfinite(vals)]
                if valid.size == 0:
                    mean, std = float("nan"), float("nan")
                else:
                    mean = float(np.mean(valid))
                    std = float(np.std(valid))
                row.append(f"{mean:.4f}")
                row.append(f"{std:.4f}")
            out.write(",".join(row) + "\n")

    print(f"\nEnsemble average over {n_traj} trajectories -> {avg_path}")
    return avg_path


def main():
    parser = argparse.ArgumentParser(
        description="Reproduce Figure 5b C2F cooling ensemble"
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=_SCRIPT_DIR / "fig5_output",
        help="Directory for calibration, per-seed CSVs, and ensemble average",
    )
    parser.add_argument("--output-prefix", default="fig5")
    parser.add_argument("--n-traj", type=int, default=FIG5_N_TRAJ,
                        help=f"Number of independent trajectories (paper: {FIG5_N_TRAJ})")
    parser.add_argument("--runtime-ps", type=float, default=FIG5_RUNTIME_PS)
    parser.add_argument("--dt-ps", type=float, default=0.001)
    parser.add_argument("--lambda", dest="lam", type=float, default=FIG5_LAMBDA)
    parser.add_argument("--period-ps", type=float, default=FIG5_PERIOD_PS)
    parser.add_argument("--duty-cycle", type=float, default=FIG5_DUTY_CYCLE)
    parser.add_argument(
        "--coupling-start-ps", type=float, default=FIG5_COUPLING_START_PS
    )
    parser.add_argument(
        "--feedback-interval-ps", type=float, default=FIG5_FEEDBACK_INTERVAL_PS,
        help="DiffEq controller update interval (ps)",
    )
    parser.add_argument(
        "--csv-interval-ps", type=float, default=FIG5_CSV_INTERVAL_PS,
        help="Trajectory CSV sample interval (ps)",
    )
    parser.add_argument("--initial-T", type=float, default=FIG5_INITIAL_T_K)
    parser.add_argument(
        "--calibration-file", default=None,
        help="Empirical calibration for T_s/T_v (default: reference_potential_energy_vs_T.txt)",
    )
    parser.add_argument(
        "--run-self-calibration", action="store_true",
        help="Run legacy self-generated calibration and cross-check vs reference",
    )
    parser.add_argument("--calibration-run-ps", type=float, default=FIG5_CALIBRATION_RUN_PS,
                        help="Legacy short calibration duration (ps); "
                             "paper-scale: run_fictive_calibration.py")
    parser.add_argument("--equil-ps", type=float, default=FIG5_EQUIL_PS,
                        help="NVT pre-equilibration before C2F (ps)")
    parser.add_argument("--diffeq-tau-ps", type=float, default=FIG5_DIFFEQ_TAU_PS)
    parser.add_argument("--skip-calibration", action="store_true")
    parser.add_argument("--skip-simulation", action="store_true",
                        help="Only re-average existing per-seed CSVs")
    parser.add_argument("--base-seed", type=int, default=FIG5_BASE_SEED)
    parser.add_argument("--platform", default=None,
                        help="OpenMM platform (CUDA, CPU, Reference)")
    parser.add_argument(
        "--reference-traj-dir", type=Path, default=None,
        help="Directory with cav-hoomd temperature_tracker_replica_*.csv for overlay plots",
    )
    parser.add_argument(
        "--quick", action="store_true",
        help="Fast check: n-traj=3, runtime=80 ps, equil-ps=50",
    )
    args = parser.parse_args()

    max_pre_equil_ps = FIG5_MAX_PRE_EQUIL_PS
    if args.quick:
        args.n_traj = 3
        args.runtime_ps = 80.0
        args.calibration_run_ps = 300.0
        args.equil_ps = 100.0
        max_pre_equil_ps = 250.0
        print(
            f"Quick mode: n_traj=3, runtime=80 ps, equil-ps={args.equil_ps}, "
            f"OpenMM-native calibration for T_s/T_v"
        )

    output_dir = args.output_dir.resolve()
    output_dir.mkdir(parents=True, exist_ok=True)

    cal_file = (
        Path(args.calibration_file)
        if args.calibration_file
        else REFERENCE_CALIBRATION_FILE
    )
    if not cal_file.is_absolute():
        cal_file = (_SCRIPT_DIR / cal_file).resolve()

    openmm_cal = output_dir / FIG5_OPENMM_CALIBRATION_NAME

    # ---- Stage A: calibration ----
    if not args.skip_simulation:
        if args.calibration_file:
            if not cal_file.exists():
                raise FileNotFoundError(f"Calibration file not found: {cal_file}")
            print(f"Using user calibration: {cal_file}")
            validate_calibration_file(cal_file)
        else:
            if not openmm_cal.exists():
                print(f"\n=== OpenMM-native calibration -> {openmm_cal.name} ===")

                def _make_system(T):
                    return build_mka_system(
                        seed=args.base_seed, sample_bonds_at_T=T
                    )

                cal_temps = (
                    FIG5_QUICK_CALIBRATION_TEMPS if args.quick else FIG5_CALIBRATION_TEMPS
                )
                run_equilibrium_calibration(
                    _make_system,
                    cal_temps,
                    run_ps=args.calibration_run_ps,
                    dt_ps=args.dt_ps,
                    output_file=str(openmm_cal),
                    platform_name=args.platform,
                )
                validate_calibration_file(openmm_cal)
                if REFERENCE_CALIBRATION_FILE.exists():
                    crosscheck_calibration_against_reference(
                        openmm_cal, REFERENCE_CALIBRATION_FILE
                    )
            else:
                print(f"Using cached OpenMM calibration: {openmm_cal}")
            cal_file = openmm_cal

        # ---- Stage B: per-seed C2F production ----
        print(f"\n=== Running {args.n_traj} C2F trajectories ===")
        for i in range(args.n_traj):
            seed = args.base_seed + i
            prefix = f"{args.output_prefix}_seed{seed:04d}"
            csv_path = output_dir / f"{prefix}_energies.csv"
            if csv_path.exists():
                print(f"  [{i+1}/{args.n_traj}] skip existing {csv_path.name}")
                continue

            print(f"  [{i+1}/{args.n_traj}] seed={seed} -> {prefix}")
            run_c2f(
                calibration_file=str(cal_file),
                initial_temperature_K=args.initial_T,
                lambda_coupling=args.lam,
                square_wave_period_ps=args.period_ps,
                square_wave_duty_cycle=args.duty_cycle,
                coupling_start_ps=args.coupling_start_ps,
                runtime_ps=args.runtime_ps,
                dt_ps=args.dt_ps,
                feedback_interval_ps=args.feedback_interval_ps,
                sample_interval_ps=args.csv_interval_ps,
                feedback_method="diffeq",
                diffeq_tau_ps=args.diffeq_tau_ps,
                T_min=FIG5_T_MIN,
                lambda_profile="square",
                equil_ps=args.equil_ps,
                post_cavity_equil_ps=FIG5_POST_CAVITY_EQUIL_PS,
                ts_bias_max_K=FIG5_TS_BIAS_MAX_K,
                max_post_equil_ps=FIG5_MAX_POST_EQUIL_PS,
                max_pre_equil_ps=max_pre_equil_ps,
                output_prefix=str(output_dir / prefix),
                seed=seed,
                platform_name=args.platform,
                log_dt=args.quick,
                finite_q=FIG5_FINITE_Q,
                feedback_every_step=FIG5_FEEDBACK_EVERY_STEP,
            )

    # ---- Stage C: ensemble average ----
    avg_path = _ensemble_average(
        output_dir, args.output_prefix, args.csv_interval_ps, args.runtime_ps
    )

    # Write metadata for the plotting script
    meta_path = output_dir / f"{args.output_prefix}_meta.txt"
    with open(meta_path, "w") as meta:
        meta.write(f"lambda={args.lam}\n")
        meta.write(f"period_ps={args.period_ps}\n")
        meta.write(f"duty_cycle={args.duty_cycle}\n")
        meta.write(f"coupling_start_ps={args.coupling_start_ps}\n")
        meta.write(f"initial_T_K={args.initial_T}\n")
        meta.write(f"runtime_ps={args.runtime_ps}\n")
        meta.write(f"n_traj={args.n_traj}\n")
        meta.write(f"diffeq_tau_ps={args.diffeq_tau_ps}\n")
        meta.write(f"calibration_file={cal_file.name}\n")
        meta.write(f"averaged_csv={avg_path.name}\n")
        if args.reference_traj_dir:
            meta.write(f"reference_traj_dir={args.reference_traj_dir}\n")
        elif Path.home().joinpath(
            "GitRepos/cav-hoomd/examples/square_lambda0.09_diffeq_recalib"
        ).exists():
            meta.write(
                "reference_traj_dir="
                f"{Path.home() / 'GitRepos/cav-hoomd/examples/square_lambda0.09_diffeq_recalib'}\n"
            )

    print(f"Metadata written to {meta_path}")
    print("\nDone. Plot with:")
    print(f"  pixi run -e test python {_SCRIPT_DIR / 'plot_figure5.py'} "
          f"--input {avg_path}")


if __name__ == "__main__":
    main()
