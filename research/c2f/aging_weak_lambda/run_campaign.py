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

from openmm.cavitymd.adaptive import DT_MAX_PS

from config import (
    CAMPAIGN_DIR,
    CAMPAIGN_LOG,
    DEFAULT_JOBS,
    INITIAL_STATE,
    LAMBDAS,
    REFERENCE_NUM_MOL,
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
    campaign_root: Path | None = None,
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
                job_dir = job_dir_path(lam, campaign_root=campaign_root)
                if not no_skip and replica_complete(job_dir, lam, replica, runtime_ps):
                    skipped += 1
                    continue
                pending.append((lam, replica))
    elif schedule == "lambda_sweep":
        for lam in lambdas:
            job_dir = job_dir_path(lam, campaign_root=campaign_root)
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
    adaptive: bool,
    allow_fallback: bool = False,
    retry_fixed_dt: bool = False,
    no_resume: bool = False,
    num_molecules: int = REFERENCE_NUM_MOL,
    initial_state: Path = INITIAL_STATE,
    platform: str | None = None,
    campaign_root: Path | None = None,
    ir_windows: list[tuple[float, float]] | None = None,
    dt_max_ps: float = DT_MAX_PS,
    dt_ps: float = 0.001,
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
    if adaptive and not retry_fixed_dt:
        cmd.append("--adaptive")
        cmd.extend(["--dt-max-ps", str(dt_max_ps)])
    else:
        cmd.append("--no-adaptive")
        cmd.extend(["--dt-ps", str(dt_ps)])
    if no_resume:
        cmd.append("--no-resume")
    if num_molecules != REFERENCE_NUM_MOL:
        cmd.extend(["--num-molecules", str(num_molecules)])
    if initial_state != INITIAL_STATE:
        cmd.extend(["--initial-state", str(initial_state)])
    if platform is not None:
        cmd.extend(["--platform", platform])
    if campaign_root is not None:
        cmd.extend(["--campaign-dir", str(campaign_root)])
    if ir_windows:
        for start_ps, length_ps in ir_windows:
            cmd.extend(["--ir-windows", str(start_ps), str(length_ps)])

    t0 = time.time()
    if dry_run:
        print(f"DRY-RUN would execute: {' '.join(cmd)}", flush=True)
        rc = 0
    else:
        rc = subprocess.run(cmd, cwd=CAMPAIGN_DIR, check=False).returncode
    elapsed = time.time() - t0

    record = {
        "timestamp": datetime.utcnow().isoformat() + "Z",
        "lambda": lam,
        "replica": replica,
        "job_dir": job_dir_name(lam),
        "returncode": rc,
        "elapsed_s": elapsed,
        "label": label,
        "adaptive": adaptive and not retry_fixed_dt,
        "no_resume": no_resume,
        "retry_fixed_dt": retry_fixed_dt,
        "integrator": (
            "max_metric_adaptive" if adaptive and not retry_fixed_dt else "fixed_verlet"
        ),
        "integrator_metric": (
            "max_force" if adaptive and not retry_fixed_dt else "fixed_dt"
        ),
        "dt_max_ps": dt_max_ps if adaptive and not retry_fixed_dt else None,
        "dt_ps": dt_ps if not (adaptive and not retry_fixed_dt) else None,
    }

    if (
        rc != 0
        and adaptive
        and allow_fallback
        and retry_fixed_dt is False
        and not dry_run
        and not smoke
    ):
        print(
            f"Adaptive run failed for {label}; retrying with fixed dt=1 fs",
            flush=True,
        )
        fallback = _run_one(
            lam,
            replica,
            runtime_ps,
            switch_time_ps,
            smoke,
            dry_run,
            adaptive=False,
            retry_fixed_dt=True,
            no_resume=no_resume,
            num_molecules=num_molecules,
            initial_state=initial_state,
            platform=platform,
            campaign_root=campaign_root,
            ir_windows=ir_windows,
            dt_max_ps=dt_max_ps,
        )
        record["fallback"] = fallback
        record["returncode"] = fallback["returncode"]
        record["elapsed_s"] += fallback["elapsed_s"]

    return record


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
        help="replica_round: all λ in parallel, then next replica (default for N=1000)",
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
    parser.add_argument(
        "--adaptive",
        action="store_true",
        default=True,
        help="Use cav-hoomd max-metric adaptive Verlet integrator (default: on)",
    )
    parser.add_argument(
        "--no-adaptive",
        action="store_false",
        dest="adaptive",
        help="Use fixed dt=1 fs Verlet instead of adaptive integrator",
    )
    parser.add_argument(
        "--allow-fallback",
        action="store_true",
        help="On adaptive failure, retry with fixed dt=1 fs (disabled by default)",
    )
    parser.add_argument(
        "--no-resume",
        action="store_true",
        help="Ignore checkpoints and start fresh from IC (archives prior outputs)",
    )
    parser.add_argument(
        "--campaign-dir",
        type=Path,
        default=None,
        help="Root directory for lambda job outputs (default: aging_weak_lambda/)",
    )
    parser.add_argument(
        "--num-molecules",
        type=int,
        default=REFERENCE_NUM_MOL,
        help=f"System size N (default {REFERENCE_NUM_MOL})",
    )
    parser.add_argument(
        "--initial-state",
        type=Path,
        default=INITIAL_STATE,
        help="Equilibrated IC npz for production runs",
    )
    parser.add_argument(
        "--platform",
        default=None,
        help="OpenMM platform name (e.g. CUDA, Reference)",
    )
    parser.add_argument(
        "--ir-windows",
        action="append",
        nargs=2,
        metavar=("START_PS", "LENGTH_PS"),
        type=float,
        default=None,
        help="IR dipole windows forwarded to run_single.py",
    )
    parser.add_argument(
        "--dt-max-ps",
        type=float,
        default=DT_MAX_PS,
        help=f"Max adaptive step size in ps (default {DT_MAX_PS} = 1.0 fs)",
    )
    parser.add_argument(
        "--dt-ps",
        type=float,
        default=0.001,
        help="Fixed integrator step size in ps when --no-adaptive (default 0.001 = 1.0 fs)",
    )
    args = parser.parse_args()

    campaign_root = args.campaign_dir
    ir_windows: list[tuple[float, float]] | None = None
    if args.ir_windows:
        ir_windows = [(float(s), float(l)) for s, l in args.ir_windows]

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
        campaign_root=campaign_root,
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

    run_kwargs = dict(
        num_molecules=args.num_molecules,
        initial_state=args.initial_state,
        platform=args.platform,
        campaign_root=campaign_root,
        ir_windows=ir_windows,
        dt_max_ps=args.dt_max_ps,
        dt_ps=args.dt_ps,
    )

    if args.jobs == 1:
        for lam, replica in pending:
            _handle(
                _run_one(
                    lam,
                    replica,
                    args.runtime_ps,
                    args.switch_time_ps,
                    args.smoke,
                    args.dry_run,
                    args.adaptive,
                    args.allow_fallback,
                    False,
                    args.no_resume,
                    **run_kwargs,
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
                    args.adaptive,
                    args.allow_fallback,
                    False,
                    args.no_resume,
                    **run_kwargs,
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
