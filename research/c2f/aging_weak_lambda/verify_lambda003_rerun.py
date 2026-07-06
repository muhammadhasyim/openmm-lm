#!/usr/bin/env python3
"""Post-rerun verification for lambda=0.03-only production rerun."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

from config import LAMBDAS, N_REPLICAS, RUNTIME_PS, job_dir_path, run_prefix
from fkt_utils import replica_complete

try:
    from checkpoint_utils import read_csv_last_time_ps, energies_csv_path
except ImportError:
    c2f = Path(__file__).resolve().parents[1]
    sys.path.insert(0, str(c2f))
    from checkpoint_utils import read_csv_last_time_ps, energies_csv_path  # noqa: E402


def verify_lambda003_rerun(*, baseline_counts: dict[str, int] | None = None) -> dict:
    """Return verification report; raise SystemExit(1) on failure when run as CLI."""
    lam003 = 0.03
    job_dir = job_dir_path(lam003)
    other_lams = [lam for lam in LAMBDAS if lam != lam003]

    complete = [
        r for r in range(N_REPLICAS) if replica_complete(job_dir, lam003, r, RUNTIME_PS)
    ]
    incomplete = [r for r in range(N_REPLICAS) if r not in complete]

    last_times: list[float] = []
    for rep in complete[:20]:
        prefix = job_dir / run_prefix(lam003, rep)
        last = read_csv_last_time_ps(energies_csv_path(prefix))
        if last is not None:
            last_times.append(last)

    other_ok = True
    other_counts: dict[str, dict[str, int]] = {}
    for lam in other_lams:
        jd = job_dir_path(lam)
        counts = {
            "fkt_ref_000": len(list(jd.glob("*_fkt_ref_000.txt"))),
            "complete": sum(
                1 for r in range(N_REPLICAS) if replica_complete(jd, lam, r, RUNTIME_PS)
            ),
            "final_state": len(list(jd.glob("*_final_state.npz"))),
        }
        other_counts[jd.name] = counts
        expected = baseline_counts.get(jd.name) if baseline_counts else N_REPLICAS
        if expected is not None and counts["complete"] != expected:
            other_ok = False

    report = {
        "lambda003": {
            "complete": len(complete),
            "incomplete_replicas": incomplete[:20],
            "fkt_ref_000": len(list(job_dir.glob("*_fkt_ref_000.txt"))),
            "final_state": len(list(job_dir.glob("*_final_state.npz"))),
            "sample_last_time_ps": last_times,
        },
        "other_lambdas_unchanged": other_ok,
        "other_counts": other_counts,
        "passed": len(complete) == N_REPLICAS and other_ok,
    }
    return report


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--baseline-json",
        type=Path,
        help="Optional JSON with expected complete counts per job dir",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=Path("results/lambda003_rerun_verification.json"),
    )
    args = parser.parse_args()

    baseline = None
    if args.baseline_json and args.baseline_json.is_file():
        baseline = json.loads(args.baseline_json.read_text(encoding="utf-8"))

    report = verify_lambda003_rerun(baseline_counts=baseline)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")

    print(json.dumps(report, indent=2))
    if not report["passed"]:
        print("VERIFICATION FAILED", file=sys.stderr)
        sys.exit(1)
    print("VERIFICATION PASSED")


if __name__ == "__main__":
    main()
