#!/usr/bin/env python3
"""Batch launcher for OpenMM weak-coupling aging campaign."""

from __future__ import annotations

import argparse
import json
import subprocess
import sys
import time
from concurrent.futures import ProcessPoolExecutor, as_completed
from datetime import datetime
from pathlib import Path

from config import (
    CAMPAIGN_DIR,
    CAMPAIGN_LOG,
    DEFAULT_JOBS,
    LAMBDAS,
    REPLICA_END,
    REPLICA_START,
    RUNTIME_PS,
    SWITCH_TIME_PS,
    job_dir_name,
    job_dir_path,
)
from fkt_utils import replica_complete


def _build_pending(
    lambdas: list[float],
    replica_start: int,
    replica_end: int,
    runtime_ps: float,
    schedule: str,
    no_skip: bool,
) -> tuple[list[tuple[float, int]], int]:
    """Return (pending runs, skipped count).

    replica_round: for each replica, queue all λ (5 parallel jobs per round).
    lambda_sweep: finish all replicas at λ before next λ (pilot default).
    """
    pending: list[tuple[float, int]] = []
    skipped = 0

    if schedule == "replica_round":
        for replica in range(replica_start, replica_end + 1):
            for lam in lambdas:
                job_dir = job_dir_path(lam)
                if not no_skip and replica_complete(job_dir, lam, replica, runtime_ps):
                    skipped += 1
                    continue
                pending.append((lam, replica))
    elif schedule == "lambda_sweep":
        for lam in lambdas:
            job_dir = job_dir_path(lam)
            for replica in range(replica_start, replica_end + 1):
                if not no_skip and replica_complete(job_dir, lam, replica, runtime_ps):
                    skipped += 1
                    continue
                pending.append((lam, replica))
    else:
        raise ValueError(f"Unknown schedule: {schedule}")

    return pending, skipped


def _run_one(
    lam: float,
    replica: int,
    runtime_ps: float,
    switch_time_ps: float,
    smoke: bool,
    dry_run: bool,
) -> dict:
    run_script = CAMPAIGN_DIR / "run_single.py"
    label = f"lam={lam:g} rep={replica}"
    cmd = [
        sys.executable,
        str(run_script),
        "--lambda",
        str(lam),
        "--replica",
        str(replica),
        "--runtime-ps",
        str(runtime_ps),
        "--switch-time-ps",
        str(switch_time_ps),
    ]
    if smoke:
        cmd.append("--smoke")

    t0 = time.time()
    if dry_run:
        print(f"DRY-RUN would execute: {' '.join(cmd)}", flush=True)
        rc = 0
    else:
        rc = subprocess.run(cmd, cwd=CAMPAIGN_DIR, check=False).returncode
    elapsed = time.time() - t0

    return {
        "timestamp": datetime.utcnow().isoformat() + "Z",
        "lambda": lam,
        "replica": replica,
        "job_dir": job_dir_name(lam),
        "returncode": rc,
        "elapsed_s": elapsed,
        "label": label,
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--lambdas", type=float, nargs="+", default=LAMBDAS)
    parser.add_argument("--replica-start", type=int, default=REPLICA_START)
    parser.add_argument("--replica-end", type=int, default=REPLICA_END)
    parser.add_argument("--runtime-ps", type=float, default=RUNTIME_PS)
    parser.add_argument("--switch-time-ps", type=float, default=SWITCH_TIME_PS)
    parser.add_argument(
        "--schedule",
        choices=("replica_round", "lambda_sweep"),
        default="replica_round",
        help="replica_round: all λ in parallel, then next replica (default for N=500)",
    )
    parser.add_argument("--jobs", type=int, default=DEFAULT_JOBS)
    parser.add_argument("--log", type=Path, default=CAMPAIGN_LOG)
    parser.add_argument("--smoke", action="store_true")
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument(
        "--no-skip",
        action="store_true",
        help="Re-run even if outputs look complete (use for FKT F(0) fix / fresh start)",
    )
    args = parser.parse_args()

    n_lam = len(args.lambdas)
    if args.schedule == "replica_round" and args.jobs % n_lam != 0:
        print(
            f"Warning: jobs={args.jobs} is not a multiple of {n_lam} lambdas; "
            "replica rounds may overlap.",
            flush=True,
        )

    pending, skipped = _build_pending(
        args.lambdas,
        args.replica_start,
        args.replica_end,
        args.runtime_ps,
        args.schedule,
        args.no_skip,
    )

    total = len(pending)
    n_rounds = (args.replica_end - args.replica_start + 1) if args.schedule == "replica_round" else None
    print(
        f"Campaign: {total} runs queued, {skipped} skipped "
        f"(schedule={args.schedule}, jobs={args.jobs}, "
        f"replicas {args.replica_start}-{args.replica_end}, log={args.log.name})",
        flush=True,
    )
    if n_rounds is not None:
        print(f"  Replica rounds: {n_rounds} × {n_lam} λ = {n_rounds * n_lam} trajectories", flush=True)
    if total == 0:
        print("Nothing to do.")
        return

    failed: list[str] = []
    done = 0

    def _handle(record: dict) -> None:
        nonlocal done
        done += 1
        with open(args.log, "a", encoding="utf-8") as fh:
            fh.write(json.dumps({k: v for k, v in record.items() if k != "label"}) + "\n")
        label = record["label"]
        print(f"\n[{done}/{total}] {label}", flush=True)
        if record["returncode"] != 0:
            failed.append(label)
            print(f"FAILED {label} (rc={record['returncode']})", flush=True)
        else:
            print(f"OK {label} in {record['elapsed_s']:.1f}s", flush=True)

    if args.jobs == 1:
        for lam, replica in pending:
            _handle(
                _run_one(
                    lam, replica, args.runtime_ps, args.switch_time_ps, args.smoke, args.dry_run
                )
            )
    else:
        with ProcessPoolExecutor(max_workers=args.jobs) as pool:
            futures = {
                pool.submit(
                    _run_one,
                    lam,
                    replica,
                    args.runtime_ps,
                    args.switch_time_ps,
                    args.smoke,
                    args.dry_run,
                ): (lam, replica)
                for lam, replica in pending
            }
            for fut in as_completed(futures):
                _handle(fut.result())

    if failed:
        print(f"\nCampaign finished with {len(failed)} failures:")
        for item in failed:
            print(f"  - {item}")
        sys.exit(1)

    print("\nCampaign finished successfully.")


if __name__ == "__main__":
    main()
