#!/usr/bin/env python3
"""Saturate GPU 0 with λ=0.03 and GPU 1 with λ=0.023333 (N=250 aging resubmit).

Each trajectory uses ~844 MiB on an A100-80GB; this script fills both GPUs with
independent worker pools and skips replicas that already have complete outputs
unless --rerun-complete is set.
"""

from __future__ import annotations

import argparse
import json
import os
import subprocess
import sys
import time
from datetime import datetime, timezone
from pathlib import Path

from config import (
    CAMPAIGN_DIR,
    RUNTIME_PS,
    SWITCH_TIME_PS,
    job_dir_path,
    run_prefix,
)
from fkt_utils import build_energy_csv_index, build_replica_root_index, resolve_energy_csv

GPU_LAMBDAS: dict[int, float] = {0: 0.03, 1: 0.023333}
DEFAULT_JOB_MEM_MIB = 844.0
GPU_TOTAL_MIB = 81920.0
# 844 MiB/job on A100-80GB: floor((81920 - 4096) / 844) = 92 concurrent jobs/GPU
GPU_HEADROOM_MIB = 4096.0
DEFAULT_MAX_JOBS_PER_GPU = int((GPU_TOTAL_MIB - GPU_HEADROOM_MIB) // DEFAULT_JOB_MEM_MIB)


def _utcnow() -> str:
    return datetime.now(timezone.utc).isoformat().replace("+00:00", "Z")


def replica_is_complete(job_dir: Path, lam: float, replica: int, runtime_ps: float) -> bool:
    index = build_energy_csv_index(job_dir, lam)
    csv_path = resolve_energy_csv(job_dir, lam, replica, index)
    if csv_path is None:
        return False

    roots = build_replica_root_index(job_dir, lam)
    root = roots.get(replica, job_dir)
    fkt_path = root / f"{run_prefix(lam, replica)}_fkt_ref_000.txt"
    if not fkt_path.is_file():
        fkt_path = job_dir / f"{run_prefix(lam, replica)}_fkt_ref_000.txt"
    if not fkt_path.is_file():
        return False

    try:
        last_time = float(csv_path.read_text().strip().splitlines()[-1].split(",")[0])
    except (OSError, ValueError, IndexError):
        return False
    return last_time >= 0.98 * runtime_ps


def build_pending(
    lam: float,
    replica_start: int,
    replica_end: int,
    runtime_ps: float,
    *,
    rerun_complete: bool,
) -> list[int]:
    job_dir = job_dir_path(lam)
    index = build_energy_csv_index(job_dir, lam)
    roots = build_replica_root_index(job_dir, lam)
    pending: list[int] = []
    for replica in range(replica_start, replica_end + 1):
        if rerun_complete:
            pending.append(replica)
            continue
        csv_path = resolve_energy_csv(job_dir, lam, replica, index)
        if csv_path is None:
            pending.append(replica)
            continue
        root = roots.get(replica, job_dir)
        fkt_path = root / f"{run_prefix(lam, replica)}_fkt_ref_000.txt"
        if not fkt_path.is_file():
            fkt_path = job_dir / f"{run_prefix(lam, replica)}_fkt_ref_000.txt"
        if not fkt_path.is_file():
            pending.append(replica)
            continue
        try:
            last_time = float(csv_path.read_text().strip().splitlines()[-1].split(",")[0])
        except (OSError, ValueError, IndexError):
            pending.append(replica)
            continue
        if last_time < 0.98 * runtime_ps:
            pending.append(replica)
    return pending


def gpu_used_mib(gpu_id: int) -> float:
    out = subprocess.check_output(
        [
            "nvidia-smi",
            "--query-gpu=index,memory.used",
            "--format=csv,noheader,nounits",
        ],
        text=True,
    )
    for line in out.strip().splitlines():
        idx, used = [part.strip() for part in line.split(",")]
        if int(idx) == gpu_id:
            return float(used)
    return 0.0


def jobs_for_gpu(gpu_id: int, job_mem_mib: float) -> int:
    used = gpu_used_mib(gpu_id)
    available = max(0.0, GPU_TOTAL_MIB - used - GPU_HEADROOM_MIB)
    return max(1, int(available // job_mem_mib))

def run_single(
    lam: float,
    replica: int,
    gpu: int,
    *,
    runtime_ps: float,
    switch_time_ps: float,
    adaptive: bool,
    no_resume: bool,
    log_path: Path,
) -> int:
    env = os.environ.copy()
    env["CUDA_VISIBLE_DEVICES"] = str(gpu)
    base_cmd = [
        sys.executable,
        str(CAMPAIGN_DIR / "run_single.py"),
        "--lambda",
        str(lam),
        "--replica",
        str(replica),
        "--runtime-ps",
        str(runtime_ps),
        "--switch-time-ps",
        str(switch_time_ps),
        "--platform",
        "CUDA",
    ]
    if no_resume:
        base_cmd.append("--no-resume")

    def _invoke(use_adaptive: bool) -> int:
        cmd = list(base_cmd)
        if use_adaptive:
            cmd.append("--adaptive")
        log_path.parent.mkdir(parents=True, exist_ok=True)
        with open(log_path, "w", encoding="utf-8") as fh:
            fh.write(f"# start {_utcnow()} gpu={gpu} lam={lam:g} rep={replica}\n")
            fh.write(f"# cmd: {' '.join(cmd)}\n\n")
            fh.flush()
            return subprocess.run(
                cmd,
                cwd=CAMPAIGN_DIR,
                env=env,
                stdout=fh,
                stderr=subprocess.STDOUT,
                check=False,
            ).returncode

    rc = _invoke(adaptive)
    if rc != 0 and adaptive:
        with open(log_path, "a", encoding="utf-8") as fh:
            fh.write(f"\n# adaptive failed rc={rc}; retrying fixed dt {_utcnow()}\n")
        rc = _invoke(False)
    return rc


def run_gpu_pool(
    gpu: int,
    lam: float,
    pending: list[int],
    *,
    max_jobs: int,
    runtime_ps: float,
    switch_time_ps: float,
    adaptive: bool,
    no_resume: bool,
    log_dir: Path,
    campaign_log: Path,
) -> tuple[int, int]:
    queue = list(pending)
    active: dict[int, tuple[int, subprocess.Popen[int]]] = {}
    ok = fail = 0
    total = len(queue)

    print(
        f"[GPU {gpu}] λ={lam:g}: {total} replicas queued, max_jobs={max_jobs}",
        flush=True,
    )

    while queue or active:
        while queue and len(active) < max_jobs:
            replica = queue.pop(0)
            log_path = log_dir / f"gpu{gpu}_lam{lam:g}_rep{replica:04d}.log"
            cmd = [
                sys.executable,
                str(Path(__file__).resolve()),
                "--worker",
                "--gpu",
                str(gpu),
                "--lambda",
                str(lam),
                "--replica",
                str(replica),
                "--runtime-ps",
                str(runtime_ps),
                "--switch-time-ps",
                str(switch_time_ps),
                "--log-path",
                str(log_path),
            ]
            if adaptive:
                cmd.append("--adaptive")
            if no_resume:
                cmd.append("--no-resume")
            proc = subprocess.Popen(cmd, cwd=CAMPAIGN_DIR)
            active[proc.pid] = (replica, proc)
            print(
                f"[GPU {gpu}] started rep={replica} "
                f"(running={len(active)}/{max_jobs}, left={len(queue)})",
                flush=True,
            )

        finished: list[int] = []
        for pid, (replica, proc) in active.items():
            rc = proc.poll()
            if rc is None:
                continue
            finished.append(pid)
            record = {
                "timestamp": _utcnow(),
                "gpu": gpu,
                "lambda": lam,
                "replica": replica,
                "returncode": rc,
                "log": str(log_dir / f"gpu{gpu}_lam{lam:g}_rep{replica:04d}.log"),
            }
            with open(campaign_log, "a", encoding="utf-8") as fh:
                fh.write(json.dumps(record) + "\n")
            if rc == 0:
                ok += 1
                print(f"[GPU {gpu}] OK rep={replica} ({ok + fail}/{total})", flush=True)
            else:
                fail += 1
                print(
                    f"[GPU {gpu}] FAIL rep={replica} rc={rc} ({ok + fail}/{total})",
                    flush=True,
                )

        for pid in finished:
            del active[pid]

        if active:
            time.sleep(5.0)

    print(f"[GPU {gpu}] λ={lam:g} finished ok={ok} fail={fail}", flush=True)
    return ok, fail


def _worker_main(args: argparse.Namespace) -> int:
    rc = run_single(
        args.lam,
        args.replica,
        args.gpu,
        runtime_ps=args.runtime_ps,
        switch_time_ps=args.switch_time_ps,
        adaptive=args.adaptive,
        no_resume=args.no_resume,
        log_path=args.log_path,
    )
    return rc


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--worker", action="store_true", help=argparse.SUPPRESS)
    parser.add_argument("--gpu", type=int, default=None)
    parser.add_argument("--lambda", dest="lam", type=float, default=None)
    parser.add_argument("--replica", type=int, default=None)
    parser.add_argument("--runtime-ps", type=float, default=RUNTIME_PS)
    parser.add_argument("--switch-time-ps", type=float, default=SWITCH_TIME_PS)
    parser.add_argument("--log-path", type=Path, default=None)
    parser.add_argument("--replica-start", type=int, default=0)
    parser.add_argument("--replica-end", type=int, default=999)
    parser.add_argument("--job-mem-mib", type=float, default=DEFAULT_JOB_MEM_MIB)
    parser.add_argument(
        "--max-jobs-gpu0",
        type=int,
        default=None,
        help="Override concurrent jobs on GPU 0 (default: auto from memory probe)",
    )
    parser.add_argument(
        "--max-jobs-gpu1",
        type=int,
        default=None,
        help="Override concurrent jobs on GPU 1 (default: auto from memory probe)",
    )
    parser.add_argument("--adaptive", action="store_true", default=True)
    parser.add_argument("--no-adaptive", action="store_false", dest="adaptive")
    parser.add_argument(
        "--no-resume",
        action="store_true",
        help="Archive partial outputs and restart from IC (recommended for FKT resubmit)",
    )
    parser.add_argument(
        "--rerun-complete",
        action="store_true",
        help="Re-run even replicas that already look complete",
    )
    parser.add_argument(
        "--log-dir",
        type=Path,
        default=CAMPAIGN_DIR / "logs" / "dual_gpu_fkt_resubmit",
    )
    parser.add_argument(
        "--campaign-log",
        type=Path,
        default=CAMPAIGN_DIR / "logs" / "dual_gpu_fkt_resubmit" / "campaign.jsonl",
    )
    args = parser.parse_args()

    if args.worker:
        if None in (args.gpu, args.lam, args.replica, args.log_path):
            raise SystemExit("--worker requires --gpu, --lambda, --replica, --log-path")
        sys.exit(_worker_main(args))

    args.log_dir.mkdir(parents=True, exist_ok=True)
    args.campaign_log.parent.mkdir(parents=True, exist_ok=True)

    max_jobs = {
        0: args.max_jobs_gpu0 or DEFAULT_MAX_JOBS_PER_GPU,
        1: args.max_jobs_gpu1 or DEFAULT_MAX_JOBS_PER_GPU,
    }
    print(
        f"Dual-GPU resubmit starting {_utcnow()} "
        f"(job_mem={args.job_mem_mib:.0f} MiB, "
        f"max_jobs gpu0={max_jobs[0]}, gpu1={max_jobs[1]} "
        f"[844 MiB/job -> up to {DEFAULT_MAX_JOBS_PER_GPU}/GPU on 80GB])",
        flush=True,
    )

    if not args.rerun_complete and not args.max_jobs_gpu0 and not args.max_jobs_gpu1:
        # Respect current GPU occupancy when auto-sizing (e.g. stray jobs on one GPU).
        for gpu in GPU_LAMBDAS:
            auto = jobs_for_gpu(gpu, args.job_mem_mib)
            if auto < max_jobs[gpu]:
                print(
                    f"GPU {gpu} already using {gpu_used_mib(gpu):.0f} MiB; "
                    f"capping to {auto} concurrent jobs",
                    flush=True,
                )
                max_jobs[gpu] = auto

    import threading

    results: dict[int, tuple[int, int]] = {}
    errors: list[str] = []
    pending_by_gpu: dict[int, list[int]] = {}

    def _target(gpu: int) -> None:
        lam = GPU_LAMBDAS[gpu]
        try:
            pending = build_pending(
                lam,
                args.replica_start,
                args.replica_end,
                args.runtime_ps,
                rerun_complete=args.rerun_complete,
            )
            pending_by_gpu[gpu] = pending
            print(
                f"λ={lam:g} on GPU {gpu}: {len(pending)} pending replicas "
                f"[{args.replica_start}, {args.replica_end}]",
                flush=True,
            )
            if not pending:
                results[gpu] = (0, 0)
                return
            results[gpu] = run_gpu_pool(
                gpu,
                lam,
                pending,
                max_jobs=max_jobs[gpu],
                runtime_ps=args.runtime_ps,
                switch_time_ps=args.switch_time_ps,
                adaptive=args.adaptive,
                no_resume=args.no_resume,
                log_dir=args.log_dir,
                campaign_log=args.campaign_log,
            )
        except Exception as exc:
            errors.append(f"GPU {gpu}: {exc}")
            raise

    threads = [
        threading.Thread(target=_target, args=(gpu,), name=f"gpu{gpu}", daemon=False)
        for gpu in sorted(GPU_LAMBDAS)
    ]
    for thread in threads:
        thread.start()
    for thread in threads:
        thread.join()

    if not any(pending_by_gpu.get(gpu) for gpu in GPU_LAMBDAS):
        print("Nothing to do.")
        return

    total_ok = sum(r[0] for r in results.values())
    total_fail = sum(r[1] for r in results.values())
    print(
        f"\nDual-GPU resubmit finished {_utcnow()}: ok={total_ok} fail={total_fail}",
        flush=True,
    )
    if errors:
        raise SystemExit("\n".join(errors))
    if total_fail:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
