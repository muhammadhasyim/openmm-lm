#!/usr/bin/env python3
"""Print OpenMM aging campaign progress."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from config import CAMPAIGN_LOG, LAMBDAS, N_REPLICAS, job_dir_path
from fkt_utils import list_available_replicas


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--log", type=Path, default=CAMPAIGN_LOG)
    args = parser.parse_args()

    log_path = args.log
    completed = failed = 0
    if log_path.exists():
        for line in log_path.read_text().splitlines():
            if not line.strip():
                continue
            rec = json.loads(line)
            if rec.get("returncode") == 0:
                completed += 1
            else:
                failed += 1
    print(f"Logged runs: {completed} ok, {failed} failed")
    for lam in LAMBDAS:
        job_dir = job_dir_path(lam)
        n = len(list_available_replicas(job_dir, lam)) if job_dir.exists() else 0
        print(f"  lambda={lam:g}: {n}/{N_REPLICAS} replicas in {job_dir.name}")


if __name__ == "__main__":
    main()
