#!/usr/bin/env python3
"""Archive campaign replica outputs before full rerun or poisoned-resubmit."""

from __future__ import annotations

import argparse
import csv
import sys
import zipfile
from pathlib import Path

_SCRIPT_DIR = Path(__file__).resolve().parent
_C2F_ROOT = _SCRIPT_DIR.parent
if str(_C2F_ROOT) not in sys.path:
    sys.path.insert(0, str(_C2F_ROOT))
if str(_SCRIPT_DIR) not in sys.path:
    sys.path.insert(0, str(_SCRIPT_DIR))

from checkpoint_utils import (  # noqa: E402
    archive_replica_outputs,
    checkpoint_path,
    energies_csv_path,
    is_poisoned_checkpoint,
    load_checkpoint,
    read_csv_last_time_ps,
    trajectory_complete,
)
from config import LAMBDAS, N_REPLICAS, RUNTIME_PS, job_dir_path, run_prefix  # noqa: E402
from fkt_utils import replica_complete  # noqa: E402

STABILITY_T_KIN_MAX_K = 5000.0


def _has_outputs(prefix: Path) -> bool:
    patterns = [
        f"{prefix.name}_energies.csv",
        f"{prefix.name}_final_state.npz",
        f"{prefix.name}_checkpoint.npz",
        f"{prefix.name}_fkt_ref_*.txt",
    ]
    parent = prefix.parent
    return any(parent.glob(pat) for pat in patterns)


def _csv_max_t_kin(csv_path: Path) -> float | None:
    if not csv_path.exists():
        return None
    try:
        with csv_path.open(encoding="utf-8") as fh:
            rows = list(csv.DictReader(fh))
        if not rows:
            return None
        return max(float(r["T_kinetic_K"]) for r in rows)
    except (ValueError, KeyError, OSError):
        return None


def _archive_reason(
    lam: float,
    replica: int,
    *,
    runtime_ps: float,
    full_rerun: bool,
) -> str | None:
    """Return archive reason string, or None if replica should be left alone."""
    job_dir = job_dir_path(lam)
    prefix = job_dir / run_prefix(lam, replica)
    if not _has_outputs(prefix):
        return None

    if full_rerun:
        return "full_rerun"

    csv_path = energies_csv_path(prefix)
    ckpt = checkpoint_path(prefix)

    if replica_complete(job_dir, lam, replica, runtime_ps):
        max_t = _csv_max_t_kin(csv_path)
        if max_t is not None and max_t > STABILITY_T_KIN_MAX_K:
            return "complete_but_blown_up"
        return None

    if ckpt.exists():
        try:
            checkpoint = load_checkpoint(ckpt)
        except (OSError, ValueError, KeyError, zipfile.BadZipFile):
            return "corrupt_checkpoint"
        if is_poisoned_checkpoint(checkpoint, csv_path, t_kin_max_k=STABILITY_T_KIN_MAX_K):
            return "poisoned_checkpoint"
        return None

    max_t = _csv_max_t_kin(csv_path)
    last_t = read_csv_last_time_ps(csv_path)
    if max_t is not None and max_t > STABILITY_T_KIN_MAX_K:
        return "partial_blown_up"
    if last_t is not None and not trajectory_complete(last_t, runtime_ps):
        if max_t is not None and max_t > STABILITY_T_KIN_MAX_K:
            return "partial_blown_up"
    return None


def archive_campaign(
    *,
    lambdas: list[float],
    replica_start: int,
    replica_end: int,
    runtime_ps: float,
    full_rerun: bool,
    dry_run: bool,
) -> dict[str, list[int]]:
    """Archive replicas across all requested λ values."""
    archived_by_lam: dict[str, list[int]] = {}

    for lam in lambdas:
        job_dir_name = job_dir_path(lam).name
        archived: list[int] = []
        for replica in range(replica_start, replica_end + 1):
            reason = _archive_reason(
                lam, replica, runtime_ps=runtime_ps, full_rerun=full_rerun
            )
            if reason is None:
                continue

            prefix = job_dir_path(lam) / run_prefix(lam, replica)
            print(f"{job_dir_name} replica {replica}: archive ({reason})")
            if dry_run:
                archived.append(replica)
                continue

            out = archive_replica_outputs(
                prefix,
                reason=reason,
                runtime_ps=runtime_ps,
                lambda_coupling=lam,
                replica=replica,
            )
            if out is not None:
                archived.append(replica)
                print(f"  -> {out}")

        archived_by_lam[job_dir_name] = archived

    return archived_by_lam


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--lambdas",
        type=float,
        nargs="+",
        default=LAMBDAS,
        help="Coupling values to scan (default: all campaign λ)",
    )
    parser.add_argument("--replica-start", type=int, default=0)
    parser.add_argument("--replica-end", type=int, default=N_REPLICAS - 1)
    parser.add_argument("--runtime-ps", type=float, default=RUNTIME_PS)
    parser.add_argument(
        "--full-rerun",
        action="store_true",
        help="Archive every replica that has any output files",
    )
    parser.add_argument("--dry-run", action="store_true")
    args = parser.parse_args()

    archived_by_lam = archive_campaign(
        lambdas=args.lambdas,
        replica_start=args.replica_start,
        replica_end=args.replica_end,
        runtime_ps=args.runtime_ps,
        full_rerun=args.full_rerun,
        dry_run=args.dry_run,
    )
    total = sum(len(v) for v in archived_by_lam.values())
    print(f"Archived {total} replica-λ pairs across {len(args.lambdas)} λ values")
    for lam_name, reps in archived_by_lam.items():
        if reps:
            print(f"  {lam_name}: {len(reps)}")


if __name__ == "__main__":
    main()
