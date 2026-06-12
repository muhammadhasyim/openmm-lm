#!/usr/bin/env python
"""Run non-thermal cavity aging from the 10 ns lambda=0 equilibrium endpoint.

Starts each trajectory from eq10ns100K_lam0_final_state.npz at 100 K, holds
lambda=0 until coupling_start_ps, then runs the Fig 5 square-wave + DiffEq
protocol in q≈0 mode (finite_q=False).
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np

_SCRIPT_DIR = Path(__file__).resolve().parent
if str(_SCRIPT_DIR) not in sys.path:
    sys.path.insert(0, str(_SCRIPT_DIR))

from run_c2f import (  # noqa: E402
    REFERENCE_CALIBRATION_FILE,
    run_c2f,
    validate_calibration_file,
)

DEFAULT_IC = _SCRIPT_DIR / "equilibrium_output" / "eq10ns100K_lam0_final_state.npz"
LAMBDAS = [0.01, 0.03, 0.042, 0.07, 0.09, 0.141]

INITIAL_T_K = 100.0
COUPLING_START_PS = 10.0
PERIOD_PS = 10.0
DUTY_CYCLE = 0.10
FEEDBACK_INTERVAL_PS = 0.01
CSV_INTERVAL_PS = 0.1
RUNTIME_PS = 150.0
N_TRAJ = 10
DIFFEQ_TAU_PS = 1.0
T_MIN = 0.01
FINITE_Q = False
FEEDBACK_EVERY_STEP = True
BASE_SEED = 42


def _lam_tag(lam: float) -> str:
    return f"{lam:g}".replace(".", "p")


def main() -> None:
    parser = argparse.ArgumentParser(
        description="C2F aging ensemble from lambda=0 equilibrium endpoint"
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=_SCRIPT_DIR / "aging_from_lam0",
        help="Directory for per-replica CSVs and run metadata",
    )
    parser.add_argument(
        "--initial-state",
        type=Path,
        default=DEFAULT_IC,
        help="Final-state npz from 10 ns lambda=0 equilibration",
    )
    parser.add_argument(
        "--calibration-file",
        type=Path,
        default=REFERENCE_CALIBRATION_FILE,
        help="Empirical calibration for structural T_s",
    )
    parser.add_argument(
        "--lambdas",
        type=float,
        nargs="+",
        default=LAMBDAS,
        help="Coupling strengths to run",
    )
    parser.add_argument("--lambda", dest="single_lam", type=float, default=None,
                        help="Run a single coupling (overrides --lambdas)")
    parser.add_argument("--n-traj", type=int, default=N_TRAJ)
    parser.add_argument("--runtime-ps", type=float, default=RUNTIME_PS)
    parser.add_argument("--coupling-start-ps", type=float, default=COUPLING_START_PS)
    parser.add_argument("--period-ps", type=float, default=PERIOD_PS)
    parser.add_argument("--duty-cycle", type=float, default=DUTY_CYCLE)
    parser.add_argument("--initial-T", type=float, default=INITIAL_T_K)
    parser.add_argument("--csv-interval-ps", type=float, default=CSV_INTERVAL_PS)
    parser.add_argument("--feedback-interval-ps", type=float, default=FEEDBACK_INTERVAL_PS)
    parser.add_argument("--diffeq-tau-ps", type=float, default=DIFFEQ_TAU_PS)
    parser.add_argument("--base-seed", type=int, default=BASE_SEED)
    parser.add_argument("--platform", default=None)
    parser.add_argument("--skip-simulation", action="store_true")
    parser.add_argument(
        "--quick",
        action="store_true",
        help="Smoke test: 1 lambda, 2 replicas, 30 ps runtime",
    )
    args = parser.parse_args()

    if args.quick:
        args.lambdas = [args.lambdas[0]]
        args.n_traj = 2
        args.runtime_ps = 30.0
        print(
            f"Quick mode: lambda={args.lambdas[0]}, n_traj=2, runtime=30 ps"
        )

    if args.single_lam is not None:
        args.lambdas = [args.single_lam]

    initial_state = args.initial_state.resolve()
    if not initial_state.exists():
        raise FileNotFoundError(f"Initial state not found: {initial_state}")

    cal_file = args.calibration_file.resolve()
    if not cal_file.exists():
        raise FileNotFoundError(f"Calibration file not found: {cal_file}")
    validate_calibration_file(cal_file)

    output_dir = args.output_dir.resolve()
    output_dir.mkdir(parents=True, exist_ok=True)

    meta_path = output_dir / "run_meta.txt"
    if not args.skip_simulation:
        with open(meta_path, "w") as meta:
            meta.write(f"initial_state={initial_state}\n")
            meta.write(f"initial_T_K={args.initial_T}\n")
            meta.write(f"coupling_start_ps={args.coupling_start_ps}\n")
            meta.write(f"runtime_ps={args.runtime_ps}\n")
            meta.write(f"period_ps={args.period_ps}\n")
            meta.write(f"duty_cycle={args.duty_cycle}\n")
            meta.write(f"n_traj={args.n_traj}\n")
            meta.write(f"finite_q={FINITE_Q}\n")
            meta.write(f"feedback_every_step={FEEDBACK_EVERY_STEP}\n")
            meta.write(f"lambdas={','.join(str(l) for l in args.lambdas)}\n")
            meta.write(f"calibration_file={cal_file}\n")

        total_runs = len(args.lambdas) * args.n_traj
        run_idx = 0
        for lam in args.lambdas:
            lam_tag = _lam_tag(lam)
            print(f"\n=== lambda={lam} ({lam_tag}) ===")
            for i in range(args.n_traj):
                run_idx += 1
                seed = args.base_seed + i
                prefix = f"lam{lam_tag}_seed{seed:04d}"
                csv_path = output_dir / f"{prefix}_energies.csv"
                if csv_path.exists():
                    print(
                        f"  [{run_idx}/{total_runs}] skip existing {csv_path.name}"
                    )
                    continue

                print(f"  [{run_idx}/{total_runs}] seed={seed} -> {prefix}")
                run_c2f(
                    calibration_file=str(cal_file),
                    initial_temperature_K=args.initial_T,
                    cavity_freq_cm1=1560.0,
                    lambda_coupling=lam,
                    square_wave_period_ps=args.period_ps,
                    square_wave_duty_cycle=args.duty_cycle,
                    coupling_start_ps=args.coupling_start_ps,
                    runtime_ps=args.runtime_ps,
                    dt_ps=0.001,
                    feedback_interval_ps=args.feedback_interval_ps,
                    sample_interval_ps=args.csv_interval_ps,
                    feedback_method="diffeq",
                    diffeq_tau_ps=args.diffeq_tau_ps,
                    T_min=T_MIN,
                    lambda_profile="square",
                    equil_ps=0.0,
                    post_cavity_equil_ps=0.0,
                    output_prefix=str(output_dir / prefix),
                    seed=seed,
                    include_dipole_self_energy=True,
                    platform_name=args.platform,
                    finite_q=FINITE_Q,
                    feedback_every_step=FEEDBACK_EVERY_STEP,
                    initial_state=initial_state,
                    resample_velocities=True,
                )

    print(f"\nDone. Analyze with:")
    print(
        f"  pixi run -e test python {_SCRIPT_DIR / 'analyze_aging_energies.py'} "
        f"--input-dir {output_dir}"
    )


if __name__ == "__main__":
    main()
