#!/usr/bin/env python3
"""Validate completed N=10k aging replicas (finite T_k, FKT, adaptive meta)."""

from __future__ import annotations

import argparse
import csv
import sys
from pathlib import Path

_SCRIPT_DIR = Path(__file__).resolve().parent
if str(_SCRIPT_DIR) not in sys.path:
    sys.path.insert(0, str(_SCRIPT_DIR))

from config import (  # noqa: E402
    N10K_CAMPAIGN_DIR,
    N10K_LAMBDA_REF,
    N10K_RUNTIME_PS,
    job_dir_path,
    run_prefix,
)
from checkpoint_utils import energies_csv_path  # noqa: E402
from fkt_utils import replica_complete  # noqa: E402

STABILITY_T_KIN_MAX_K = 5000.0


def check_replica(replica: int, *, runtime_ps: float, t_kin_max_k: float) -> list[str]:
    lam = N10K_LAMBDA_REF
    job_dir = job_dir_path(lam, campaign_root=N10K_CAMPAIGN_DIR)
    prefix = job_dir / run_prefix(lam, replica)
    csv_path = energies_csv_path(prefix)
    failures: list[str] = []

    if not replica_complete(job_dir, lam, replica, runtime_ps):
        failures.append("incomplete trajectory (missing CSV/FKT or short runtime)")
        return failures

    with csv_path.open(encoding="utf-8") as fh:
        rows = list(csv.DictReader(fh))
    if not rows:
        failures.append("empty energies CSV")
        return failures

    t_kin_vals = [float(r["T_kinetic_K"]) for r in rows]
    max_t_kin = max(t_kin_vals)
    if not all(map(lambda x: x == x, t_kin_vals)):  # NaN check
        failures.append("T_kin contains NaN")
    if max_t_kin > t_kin_max_k:
        failures.append(f"T_kin max={max_t_kin:.4g} K > {t_kin_max_k:g} K")

    fkt_path = job_dir / f"{run_prefix(lam, replica)}_fkt_ref_000.txt"
    if not fkt_path.exists():
        failures.append(f"missing {fkt_path.name}")

    meta_path = Path(f"{prefix}_meta.txt")
    if not meta_path.exists():
        failures.append("missing _meta.txt")
    else:
        meta = meta_path.read_text(encoding="utf-8")
        if "adaptive=True" not in meta:
            failures.append("meta missing adaptive=True")
        if "integrator_metric=max_force" not in meta:
            failures.append("meta missing integrator_metric=max_force")

    return failures


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--replica-start", type=int, default=0)
    parser.add_argument("--replica-end", type=int, default=9)
    parser.add_argument("--runtime-ps", type=float, default=N10K_RUNTIME_PS)
    parser.add_argument("--t-kin-max-k", type=float, default=STABILITY_T_KIN_MAX_K)
    args = parser.parse_args()

    failed = 0
    checked = 0
    for replica in range(args.replica_start, args.replica_end + 1):
        issues = check_replica(
            replica,
            runtime_ps=args.runtime_ps,
            t_kin_max_k=args.t_kin_max_k,
        )
        if issues == ["incomplete trajectory (missing CSV/FKT or short runtime)"]:
            continue
        checked += 1
        if issues:
            failed += 1
            print(f"FAIL replica {replica}: {'; '.join(issues)}")
        else:
            print(f"OK   replica {replica}")

    if checked == 0:
        print("No completed replicas in range yet.")
        return
    if failed:
        sys.exit(1)
    print(f"All {checked} completed replica(s) passed stability checks.")


if __name__ == "__main__":
    main()
