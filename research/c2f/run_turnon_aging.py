#!/usr/bin/env python
"""Step turn-on cavity aging from the 10 ns lambda=0 equilibrium endpoint.

Protocol (paper non-thermal aging):
  1. Load lambda=0 equilibrated IC at 100 K
  2. Hold lambda=0 for coupling_start_ps
  3. Step-turn on coupling; leave it on for the production window
  4. Fixed-T NVT (Bussi + cavity Langevin), no C2F / no DiffEq feedback
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

_SCRIPT_DIR = Path(__file__).resolve().parent
if str(_SCRIPT_DIR) not in sys.path:
    sys.path.insert(0, str(_SCRIPT_DIR))

from run_cavity_equilibrium import run_cavity_equilibrium  # noqa: E402

DEFAULT_IC = _SCRIPT_DIR / "equilibrium_output" / "eq10ns100K_lam0_final_state.npz"
LAMBDAS = [0.01, 0.03, 0.042, 0.07, 0.09, 0.141]

INITIAL_T_K = 100.0
COUPLING_START_PS = 10.0
POST_TURNON_PS = 150.0
RUNTIME_PS = COUPLING_START_PS + POST_TURNON_PS
CSV_INTERVAL_PS = 0.1
N_TRAJ = 10
FINITE_Q = False
BASE_SEED = 42


def _lam_tag(lam: float) -> str:
    return f"{lam:g}".replace(".", "p")


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Step turn-on aging ensemble from lambda=0 IC"
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=_SCRIPT_DIR / "turnon_aging",
    )
    parser.add_argument("--initial-state", type=Path, default=DEFAULT_IC)
    parser.add_argument("--lambdas", type=float, nargs="+", default=LAMBDAS)
    parser.add_argument("--lambda", dest="single_lam", type=float, default=None)
    parser.add_argument("--n-traj", type=int, default=N_TRAJ)
    parser.add_argument("--coupling-start-ps", type=float, default=COUPLING_START_PS)
    parser.add_argument("--post-turnon-ps", type=float, default=POST_TURNON_PS)
    parser.add_argument("--initial-T", type=float, default=INITIAL_T_K)
    parser.add_argument("--csv-interval-ps", type=float, default=CSV_INTERVAL_PS)
    parser.add_argument("--base-seed", type=int, default=BASE_SEED)
    parser.add_argument("--platform", default=None)
    parser.add_argument("--skip-simulation", action="store_true")
    parser.add_argument(
        "--quick",
        action="store_true",
        help="Smoke test: 1 lambda, 2 replicas, 30 ps total runtime",
    )
    args = parser.parse_args()

    if args.quick:
        args.lambdas = [args.lambdas[0]]
        args.n_traj = 2
        args.post_turnon_ps = 20.0
        print(
            f"Quick mode: lambda={args.lambdas[0]}, n_traj=2, "
            f"runtime={args.coupling_start_ps + args.post_turnon_ps} ps"
        )

    if args.single_lam is not None:
        args.lambdas = [args.single_lam]

    runtime_ps = args.coupling_start_ps + args.post_turnon_ps
    initial_state = args.initial_state.resolve()
    if not initial_state.exists():
        raise FileNotFoundError(f"Initial state not found: {initial_state}")

    output_dir = args.output_dir.resolve()
    output_dir.mkdir(parents=True, exist_ok=True)

    if not args.skip_simulation:
        meta_path = output_dir / "run_meta.txt"
        with open(meta_path, "w", encoding="utf-8") as meta:
            meta.write(f"initial_state={initial_state}\n")
            meta.write(f"initial_T_K={args.initial_T}\n")
            meta.write(f"coupling_start_ps={args.coupling_start_ps}\n")
            meta.write(f"post_turnon_ps={args.post_turnon_ps}\n")
            meta.write(f"runtime_ps={runtime_ps}\n")
            meta.write(f"n_traj={args.n_traj}\n")
            meta.write(f"finite_q={FINITE_Q}\n")
            meta.write(f"lambdas={','.join(str(l) for l in args.lambdas)}\n")

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
                    print(f"  [{run_idx}/{total_runs}] skip existing {csv_path.name}")
                    continue

                print(f"  [{run_idx}/{total_runs}] seed={seed} -> {prefix}")
                run_cavity_equilibrium(
                    temperature_K=args.initial_T,
                    runtime_ps=runtime_ps,
                    lambda_coupling=lam,
                    include_dipole_self_energy=True,
                    output_prefix=str(output_dir / prefix),
                    seed=seed,
                    dt_ps=0.001,
                    sample_interval_ps=args.csv_interval_ps,
                    initial_state=initial_state,
                    platform_name=args.platform,
                    finite_q=FINITE_Q,
                    coupling_start_ps=args.coupling_start_ps,
                    resample_velocities=True,
                )

    print("\nDone. Analyze with:")
    print(
        f"  {_SCRIPT_DIR / '.pixi/envs/test/bin/python'} "
        f"{_SCRIPT_DIR / 'analyze_aging_energies.py'} --input-dir {output_dir}"
    )


if __name__ == "__main__":
    main()
