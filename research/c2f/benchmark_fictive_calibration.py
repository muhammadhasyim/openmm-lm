#!/usr/bin/env python
"""Benchmark fictive calibration throughput at the plan point nearest 100 K."""

from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

_SCRIPT_DIR = Path(__file__).resolve().parent
if str(_SCRIPT_DIR) not in sys.path:
    sys.path.insert(0, str(_SCRIPT_DIR))

from run_fictive_calibration import main as _unused  # noqa: F401

DEFAULT_PLAN = _SCRIPT_DIR / "calibration_output" / "plan.json"
DEFAULT_OUT = _SCRIPT_DIR / "calibration_output" / "benchmark_100k.json"


def _nearest_entry(plan_path: Path, target_T: float) -> dict:
    payload = json.loads(plan_path.read_text())
    return min(payload["entries"], key=lambda e: abs(e["temperature_K"] - target_T))


def main() -> int:
    parser = argparse.ArgumentParser(description="Benchmark ~100 K calibration point")
    parser.add_argument("--plan", type=Path, default=DEFAULT_PLAN)
    parser.add_argument("--target-T", type=float, default=100.0)
    parser.add_argument("--platform", default="CUDA")
    parser.add_argument("--output", type=Path, default=DEFAULT_OUT)
    args = parser.parse_args()

    entry = _nearest_entry(args.plan, args.target_T)
    out_dir = _SCRIPT_DIR / "calibration_output"
    out_dir.mkdir(parents=True, exist_ok=True)
    full_out = out_dir / "benchmark_100k_full.txt"
    slim_out = out_dir / "benchmark_100k_slim.txt"
    ts_dir = out_dir / "benchmark_timeseries"

    from run_c2f import build_mka_system
    from openmm.cavitymd.calibration import run_nvt_energy_calibration, CalibrationRunSpec

    spec = CalibrationRunSpec(
        temperature_K=entry["temperature_K"],
        equil_ps=entry["equil_ps"],
        prod_ps=entry["prod_ps"],
        n_samples=entry["n_samples"],
    )

    t0 = time.perf_counter()
    run_nvt_energy_calibration(
        lambda T: build_mka_system(seed=42, sample_bonds_at_T=T),
        [spec.temperature_K],
        run_specs=[spec],
        platform_name=args.platform,
        output_file=full_out,
        slim_output_file=slim_out,
        timeseries_dir=ts_dir,
        n_eff_report_file=out_dir / "benchmark_100k_n_eff.json",
    )
    elapsed = time.perf_counter() - t0
    total_ps = entry["equil_ps"] + entry["prod_ps"]
    ps_per_s = total_ps / elapsed

    report = {
        "entry": entry,
        "elapsed_s": elapsed,
        "ps_per_s": ps_per_s,
        "platform": args.platform,
        "full_output": str(full_out),
        "slim_output": str(slim_out),
    }
    args.output.write_text(json.dumps(report, indent=2) + "\n")
    print(json.dumps(report, indent=2))
    return 0


if __name__ == "__main__":
    sys.exit(main())
