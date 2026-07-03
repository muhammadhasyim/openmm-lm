#!/usr/bin/env python3
"""Check replica energies CSV for numerical stability (T_kin blow-up)."""

from __future__ import annotations

import argparse
import csv
import sys
from pathlib import Path

_SCRIPT_DIR = Path(__file__).resolve().parent
_C2F_ROOT = _SCRIPT_DIR.parent
if str(_C2F_ROOT) not in sys.path:
    sys.path.insert(0, str(_C2F_ROOT))
if str(_SCRIPT_DIR) not in sys.path:
    sys.path.insert(0, str(_SCRIPT_DIR))

from config import LAMBDAS, N_REPLICAS, RUNTIME_PS, job_dir_path, run_prefix  # noqa: E402
from checkpoint_utils import energies_csv_path  # noqa: E402
from fkt_utils import replica_complete  # noqa: E402

STABILITY_T_KIN_MAX_K = 5000.0


def check_replica(
    lam: float,
    replica: int,
    *,
    runtime_ps: float,
    t_kin_max_k: float,
) -> list[str]:
    """Return list of failure messages (empty if OK)."""
    prefix = job_dir_path(lam) / run_prefix(lam, replica)
    csv_path = energies_csv_path(prefix)
    failures: list[str] = []

    if not replica_complete(job_dir_path(lam), lam, replica, runtime_ps):
        failures.append(f"incomplete trajectory (missing CSV/FKT or short runtime)")
        return failures

    try:
        with csv_path.open(encoding="utf-8") as fh:
            rows = list(csv.DictReader(fh))
    except OSError as exc:
        failures.append(f"cannot read CSV: {exc}")
        return failures

    if not rows:
        failures.append("empty energies CSV")
        return failures

    max_t_kin = max(float(r["T_kinetic_K"]) for r in rows)
    if max_t_kin > t_kin_max_k:
        failures.append(f"T_kin max={max_t_kin:.4g} K > {t_kin_max_k:g} K")

    meta_path = Path(f"{prefix}_meta.txt")
    if meta_path.exists():
        meta = meta_path.read_text(encoding="utf-8")
        if "integrator_metric=max_force" not in meta and "adaptive=True" in meta:
            failures.append("meta missing integrator_metric=max_force")

    return failures


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--lambdas", type=float, nargs="+", default=LAMBDAS)
    parser.add_argument("--replica-start", type=int, default=0)
    parser.add_argument("--replica-end", type=int, default=min(9, N_REPLICAS - 1))
    parser.add_argument("--runtime-ps", type=float, default=RUNTIME_PS)
    parser.add_argument("--t-kin-max-k", type=float, default=STABILITY_T_KIN_MAX_K)
    args = parser.parse_args()

    failed = 0
    for lam in args.lambdas:
        for replica in range(args.replica_start, args.replica_end + 1):
            issues = check_replica(
                lam,
                replica,
                runtime_ps=args.runtime_ps,
                t_kin_max_k=args.t_kin_max_k,
            )
            if issues:
                failed += 1
                print(f"FAIL lam={lam:g} rep={replica}: {'; '.join(issues)}")
            else:
                print(f"OK   lam={lam:g} rep={replica}")

    if failed:
        print(f"\n{failed} replica-λ pairs failed stability check", file=sys.stderr)
        sys.exit(1)
    print("\nAll checked replicas passed")


if __name__ == "__main__":
    main()
