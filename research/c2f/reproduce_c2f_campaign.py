#!/usr/bin/env python
"""Run a paper-aligned C²F square-wave cooling ensemble.

Uses the Fig. 5 protocol (continuous DiffEq feedback, coupling start 10 ps,
q≈0 photon, etc.) with a user-specified λ and the shared 10 ns equilibrium IC.

Example (λ=0.03, 500 replicas, 2-GPU driver launches this per replica):
    python reproduce_c2f_campaign.py \\
        --lambda 0.03 \\
        --output-dir c2f_campaign/lam0p03 \\
        --initial-state equilibrium_output/eq10ns100K_lam0_final_state.npz
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np

try:
    from scipy.interpolate import interp1d
except ImportError:
    sys.exit("scipy required (pixi run -e test ...)")

_SCRIPT_DIR = Path(__file__).resolve().parent
if str(_SCRIPT_DIR) not in sys.path:
    sys.path.insert(0, str(_SCRIPT_DIR))

from c2f_paper_protocol import (  # noqa: E402
    EQUILIBRIUM_IC,
    PAPER_BASE_SEED,
    PAPER_COUPLING_START_PS,
    PAPER_CSV_INTERVAL_PS,
    PAPER_DIFFEQ_TAU_PS,
    PAPER_DUTY_CYCLE,
    PAPER_DT_PS,
    PAPER_FEEDBACK_EVERY_STEP,
    PAPER_FEEDBACK_INTERVAL_PS,
    PAPER_FINITE_Q,
    PAPER_INITIAL_T_K,
    PAPER_MAX_WARM_EQUIL_PS,
    PAPER_N_REPLICAS,
    PAPER_PERIOD_PS,
    PAPER_RUNTIME_PS,
    PAPER_T_MIN_K,
    PAPER_TS_BIAS_MAX_K,
    PAPER_WARM_EQUIL_PS,
    WEAK_LAMBDA,
)
from run_c2f import (  # noqa: E402
    REFERENCE_CALIBRATION_FILE,
    run_c2f,
    validate_calibration_file,
)


def _ensemble_average(
    output_dir: Path, prefix: str, dt_ps: float, runtime_ps: float
) -> Path:
    csv_files = sorted(output_dir.glob(f"{prefix}_seed*_energies.csv"))
    if not csv_files:
        raise FileNotFoundError(
            f"No trajectory CSVs matching {prefix}_seed*_energies.csv in {output_dir}"
        )

    time_grid = np.arange(0.0, runtime_ps + 0.5 * dt_ps, dt_ps)
    columns = ["T_bath_K", "T_kinetic_K", "T_v_fictive_K", "T_s_fictive_K"]
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
            y = np.where(np.isfinite(y), y, np.nan)
            if np.all(np.isnan(y)):
                interp_y = np.full_like(time_grid, np.nan)
            else:
                valid = np.isfinite(y)
                f = interp1d(
                    t[valid],
                    y[valid],
                    kind="linear",
                    bounds_error=False,
                    fill_value=np.nan,
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
                row.extend([f"{mean:.4f}", f"{std:.4f}"])
            out.write(",".join(row) + "\n")

    print(f"\nEnsemble average over {n_traj} trajectories -> {avg_path}")
    return avg_path


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=_SCRIPT_DIR / "c2f_campaign" / "lam0p03",
    )
    parser.add_argument("--output-prefix", default="c2f")
    parser.add_argument("--lambda", dest="lam", type=float, default=WEAK_LAMBDA)
    parser.add_argument("--n-traj", type=int, default=PAPER_N_REPLICAS)
    parser.add_argument("--replica-start", type=int, default=0)
    parser.add_argument("--replica-end", type=int, default=None)
    parser.add_argument("--runtime-ps", type=float, default=PAPER_RUNTIME_PS)
    parser.add_argument(
        "--initial-state",
        type=Path,
        default=EQUILIBRIUM_IC,
        help="Shared equilibrated structure (positions); velocities resampled per seed",
    )
    parser.add_argument(
        "--calibration-file",
        type=Path,
        default=REFERENCE_CALIBRATION_FILE,
    )
    parser.add_argument("--base-seed", type=int, default=PAPER_BASE_SEED)
    parser.add_argument("--platform", default=None)
    parser.add_argument(
        "--skip-simulation",
        action="store_true",
        help="Only re-average existing per-seed CSVs",
    )
    parser.add_argument(
        "--quick",
        action="store_true",
        help="Smoke: 1 replica, 80 ps runtime",
    )
    args = parser.parse_args()

    if args.quick:
        args.n_traj = 1
        args.replica_start = 0
        args.replica_end = 0
        args.runtime_ps = 80.0
        print("Quick mode: 1 replica, runtime=80 ps")

    replica_end = (
        args.replica_end if args.replica_end is not None else args.n_traj - 1
    )

    output_dir = args.output_dir.resolve()
    output_dir.mkdir(parents=True, exist_ok=True)

    cal_file = args.calibration_file.resolve()
    if not cal_file.is_file():
        raise FileNotFoundError(f"Missing calibration file: {cal_file}")
    validate_calibration_file(cal_file)

    ic_path = args.initial_state.resolve()
    if not ic_path.is_file():
        raise FileNotFoundError(f"Missing equilibrium IC: {ic_path}")

    if not args.skip_simulation:
        print(f"\n=== C²F paper protocol: λ={args.lam}, replicas {args.replica_start}-{replica_end} ===")
        print(f"  IC: {ic_path}")
        print(f"  coupling start: {PAPER_COUPLING_START_PS} ps")
        print(f"  feedback: every-step DiffEq (τ={PAPER_DIFFEQ_TAU_PS} ps)")
        print(f"  warm equil: {PAPER_WARM_EQUIL_PS} ps (max {PAPER_MAX_WARM_EQUIL_PS} ps)")

        for rep in range(args.replica_start, replica_end + 1):
            seed = args.base_seed + rep
            prefix = f"{args.output_prefix}_seed{seed:04d}"
            csv_path = output_dir / f"{prefix}_energies.csv"
            final_path = output_dir / f"{prefix}_final_state.npz"
            if final_path.exists():
                print(f"  [{rep+1}/{args.n_traj}] skip complete {final_path.name}")
                continue
            if csv_path.exists() and not final_path.exists():
                print(f"  [{rep+1}/{args.n_traj}] rerunning incomplete {prefix}")

            print(f"  [{rep+1}/{args.n_traj}] seed={seed} -> {prefix}")
            run_c2f(
                calibration_file=str(cal_file),
                initial_temperature_K=PAPER_INITIAL_T_K,
                lambda_coupling=args.lam,
                square_wave_period_ps=PAPER_PERIOD_PS,
                square_wave_duty_cycle=PAPER_DUTY_CYCLE,
                coupling_start_ps=PAPER_COUPLING_START_PS,
                runtime_ps=args.runtime_ps,
                dt_ps=PAPER_DT_PS,
                feedback_interval_ps=PAPER_FEEDBACK_INTERVAL_PS,
                sample_interval_ps=PAPER_CSV_INTERVAL_PS,
                feedback_method="diffeq",
                diffeq_tau_ps=PAPER_DIFFEQ_TAU_PS,
                T_min=PAPER_T_MIN_K,
                lambda_profile="square",
                equil_ps=0.0,
                post_cavity_equil_ps=PAPER_WARM_EQUIL_PS,
                ts_bias_max_K=PAPER_TS_BIAS_MAX_K,
                max_post_equil_ps=PAPER_MAX_WARM_EQUIL_PS,
                output_prefix=str(output_dir / prefix),
                seed=seed,
                platform_name=args.platform,
                finite_q=PAPER_FINITE_Q,
                feedback_every_step=PAPER_FEEDBACK_EVERY_STEP,
                initial_state=ic_path,
                resample_velocities=True,
            )

    avg_path = _ensemble_average(
        output_dir, args.output_prefix, PAPER_CSV_INTERVAL_PS, args.runtime_ps
    )

    meta_path = output_dir / f"{args.output_prefix}_meta.txt"
    with open(meta_path, "w") as meta:
        meta.write(f"lambda={args.lam}\n")
        meta.write(f"period_ps={PAPER_PERIOD_PS}\n")
        meta.write(f"duty_cycle={PAPER_DUTY_CYCLE}\n")
        meta.write(f"coupling_start_ps={PAPER_COUPLING_START_PS}\n")
        meta.write(f"initial_T_K={PAPER_INITIAL_T_K}\n")
        meta.write(f"runtime_ps={args.runtime_ps}\n")
        meta.write(f"n_traj={args.n_traj}\n")
        meta.write(f"replica_range={args.replica_start}-{replica_end}\n")
        meta.write(f"diffeq_tau_ps={PAPER_DIFFEQ_TAU_PS}\n")
        meta.write(f"feedback_every_step={PAPER_FEEDBACK_EVERY_STEP}\n")
        meta.write(f"finite_q={PAPER_FINITE_Q}\n")
        meta.write(f"initial_state={ic_path}\n")
        meta.write(f"calibration_file={cal_file.name}\n")
        meta.write(f"averaged_csv={avg_path.name}\n")
    print(f"Metadata written to {meta_path}")


if __name__ == "__main__":
    main()
