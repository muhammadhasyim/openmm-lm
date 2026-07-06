#!/usr/bin/env python3
"""Archive poisoned or blown-up lambda=0.03 replica outputs before resubmit."""

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
from config import RUNTIME_PS, job_dir_path, run_prefix  # noqa: E402
from fkt_utils import replica_complete  # noqa: E402

STABILITY_T_KIN_MAX_K = 5000.0


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


def archive_poisoned_replicas(
    lam: float,
    *,
    replica_start: int = 0,
    replica_end: int = 499,
    runtime_ps: float = RUNTIME_PS,
    dry_run: bool = False,
) -> list[int]:
    """Archive replicas with poisoned checkpoints or T_kin blow-up."""
    job_dir = job_dir_path(lam)
    archived: list[int] = []

    for replica in range(replica_start, replica_end + 1):
        prefix = job_dir / run_prefix(lam, replica)
        csv_path = energies_csv_path(prefix)
        ckpt = checkpoint_path(prefix)

        should_archive = False
        reason = ""

        if replica_complete(job_dir, lam, replica, runtime_ps):
            max_t = _csv_max_t_kin(csv_path)
            if max_t is not None and max_t > STABILITY_T_KIN_MAX_K:
                should_archive = True
                reason = "complete_but_blown_up"
        elif ckpt.exists():
            try:
                checkpoint = load_checkpoint(ckpt)
            except (OSError, ValueError, KeyError, zipfile.BadZipFile):
                should_archive = True
                reason = "corrupt_checkpoint"
            else:
                if is_poisoned_checkpoint(
                    checkpoint, csv_path, t_kin_max_k=STABILITY_T_KIN_MAX_K
                ):
                    should_archive = True
                    reason = "poisoned_checkpoint"
        else:
            max_t = _csv_max_t_kin(csv_path)
            last_t = read_csv_last_time_ps(csv_path)
            if max_t is not None and max_t > STABILITY_T_KIN_MAX_K:
                should_archive = True
                reason = "partial_blown_up"
            elif last_t is not None and not trajectory_complete(last_t, runtime_ps):
                max_t_partial = max_t
                if max_t_partial is not None and max_t_partial > STABILITY_T_KIN_MAX_K:
                    should_archive = True
                    reason = "partial_blown_up"

        if not should_archive:
            continue

        print(f"replica {replica}: archive ({reason})")
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

    return archived


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--lambda", dest="lam", type=float, default=0.03)
    parser.add_argument("--replica-start", type=int, default=0)
    parser.add_argument("--replica-end", type=int, default=499)
    parser.add_argument("--runtime-ps", type=float, default=RUNTIME_PS)
    parser.add_argument("--dry-run", action="store_true")
    args = parser.parse_args()

    archived = archive_poisoned_replicas(
        args.lam,
        replica_start=args.replica_start,
        replica_end=args.replica_end,
        runtime_ps=args.runtime_ps,
        dry_run=args.dry_run,
    )
    print(f"Archived {len(archived)} replicas")


if __name__ == "__main__":
    main()
