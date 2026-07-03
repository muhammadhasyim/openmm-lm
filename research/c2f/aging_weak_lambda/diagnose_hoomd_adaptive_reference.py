#!/usr/bin/env python3
"""Compare OpenMM adaptive diagnostics with cav-hoomd log references."""

from __future__ import annotations

import argparse
import csv
import json
import re
import subprocess
import sys
from pathlib import Path

_SCRIPT_DIR = Path(__file__).resolve().parent
_C2F_ROOT = _SCRIPT_DIR.parent
_REPO_ROOT = _C2F_ROOT.parent.parent
_CAV_HOOMD = Path("/scratch/mh7373/projects/cav-hoomd")


def _run_openmm_diagnostic(
    *,
    seed: int,
    lam: float,
    output_csv: Path,
    window_before_ps: float,
    window_after_ps: float,
    sample_interval_ps: float,
    python_exe: Path,
) -> None:
    script = _C2F_ROOT / "diagnose_adaptive_switch.py"
    ic = _C2F_ROOT / "equilibrium_output" / "eq10ns100K_lam0_final_state.npz"
    cmd = [
        str(python_exe),
        str(script),
        "--seed",
        str(seed),
        "--lambda",
        str(lam),
        "--window-before-ps",
        str(window_before_ps),
        "--window-after-ps",
        str(window_after_ps),
        "--sample-interval-ps",
        str(sample_interval_ps),
        "--initial-state",
        str(ic),
        "--output",
        str(output_csv),
        "--platform",
        "CUDA",
    ]
    subprocess.run(cmd, check=True, cwd=_C2F_ROOT)


def _parse_hoomd_floquet_log(log_path: Path) -> list[dict[str, float]]:
    """Extract timestep and error_tolerance rows from cav-hoomd console logs."""
    rows: list[dict[str, float]] = []
    if not log_path.is_file():
        return rows
    pattern = re.compile(
        r"^\s*(\d+)\s+[\d.]+\s+([\d.]+)\s+[\d.]+\s+[\d:]+\s+[\d.eE+-]+\s+([\d.eE+-]+)\s+([\d.eE+-]+)"
    )
    for line in log_path.read_text(encoding="utf-8", errors="replace").splitlines():
        m = pattern.match(line)
        if not m:
            continue
        rows.append(
            {
                "timestep": float(m.group(1)),
                "elapsed_ps": float(m.group(2)),
                "lambda": float(m.group(3)),
                "error_tolerance": float(m.group(4)),
            }
        )
    return rows


def _load_openmm_csv(path: Path) -> list[dict[str, float]]:
    with path.open(encoding="utf-8") as fh:
        return list(csv.DictReader(fh))


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--python", type=Path, default=_REPO_ROOT / ".pixi/envs/test/bin/python")
    parser.add_argument("--output-dir", type=Path, default=_SCRIPT_DIR / "diagnose_fkt" / "parity_runs" / "phase3")
    args = parser.parse_args()

    args.output_dir.mkdir(parents=True, exist_ok=True)
    report: dict = {
        "openmm_runs": [],
        "hoomd_log_references": [],
        "notes": (
            "HOOMD Python module not available in OpenMM pixi env; "
            "using archived cav-hoomd floquet logs for runtime adaptive semantics."
        ),
    }

    scenarios = [
        ("lam003_turnon", 0.03, 110.0, 60.0, 0.1),
        ("lam001_aged", 0.01, 10.0, 560.0, 1.0),
    ]
    for tag, lam, before, after, interval in scenarios:
        out_csv = args.output_dir / f"openmm_{tag}_seed{args.seed:04d}.csv"
        _run_openmm_diagnostic(
            seed=args.seed,
            lam=lam,
            output_csv=out_csv,
            window_before_ps=before,
            window_after_ps=after,
            sample_interval_ps=interval,
            python_exe=args.python,
        )
        rows = _load_openmm_csv(out_csv)
        report["openmm_runs"].append(
            {
                "tag": tag,
                "lambda": lam,
                "csv": str(out_csv),
                "n_samples": len(rows),
                "min_dt_fs": min(float(r["dt_fs"]) for r in rows),
                "max_T_kin_K": max(float(r["T_kin_K"]) for r in rows),
            }
        )

    for log_path in sorted((_CAV_HOOMD / "examples/slurm_logs").glob("floquet_*.out"))[:2]:
        parsed = _parse_hoomd_floquet_log(log_path)
        if parsed:
            report["hoomd_log_references"].append(
                {
                    "log": str(log_path),
                    "n_rows": len(parsed),
                    "error_tolerance_range": [
                        min(r["error_tolerance"] for r in parsed),
                        max(r["error_tolerance"] for r in parsed),
                    ],
                    "first_rows": parsed[:5],
                }
            )

    out_json = args.output_dir / "hoomd_openmm_comparison.json"
    out_json.write_text(json.dumps(report, indent=2), encoding="utf-8")
    print(f"Wrote {out_json}")


if __name__ == "__main__":
    main()
