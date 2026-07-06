#!/usr/bin/env python
"""Write provenance metadata for OpenMM fictive calibration."""

from __future__ import annotations

import argparse
import json
import subprocess
import sys
from datetime import datetime, timezone
from pathlib import Path

_SCRIPT_DIR = Path(__file__).resolve().parent
DEFAULT_OUT = _SCRIPT_DIR / "calibration_output" / "provenance.json"


def _git_head() -> str | None:
    try:
        return (
            subprocess.check_output(
                ["git", "rev-parse", "HEAD"],
                cwd=_SCRIPT_DIR.parents[2],
                stderr=subprocess.DEVNULL,
                text=True,
            )
            .strip()
        )
    except (subprocess.CalledProcessError, FileNotFoundError):
        return None


def main() -> int:
    parser = argparse.ArgumentParser(description="Write calibration provenance JSON")
    parser.add_argument("--output", type=Path, default=DEFAULT_OUT)
    parser.add_argument("--plan", type=Path, default=_SCRIPT_DIR / "calibration_output" / "plan.json")
    parser.add_argument("--benchmark", type=Path, default=_SCRIPT_DIR / "calibration_output" / "benchmark_100k.json")
    args = parser.parse_args()

    payload = {
        "created_utc": datetime.now(timezone.utc).isoformat(),
        "git_commit": _git_head(),
        "openmm_branch": "cavity-md",
        "relaxation_file": str(
            _SCRIPT_DIR.parents[3] / "third_party/cav-hoomd" / "relaxation_times_vs_temperature.txt"
        ),
        "plan_manifest": str(args.plan) if args.plan.exists() else None,
        "benchmark_report": str(args.benchmark) if args.benchmark.exists() else None,
        "calibration_slim": str(_SCRIPT_DIR / "calibration_output" / "calibration_data.txt"),
        "calibration_full": str(
            _SCRIPT_DIR / "calibration_output" / "potential_energy_components_vs_temperature.txt"
        ),
        "n_eff_report": str(_SCRIPT_DIR / "calibration_output" / "n_eff_report.json"),
        "sampling": {
            "prod_tau_factor": 100,
            "equil_tau_factor": 10,
            "n_samples": 1000,
            "error_method": "pyblock reblocking (pymbar cross-check)",
        },
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(payload, indent=2) + "\n")
    print(f"Wrote {args.output}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
