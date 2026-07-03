#!/usr/bin/env python3
"""Compare tracked T_s_fictive_K vs unified energy-inferred T_s across replicas."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np

from config import ANALYSIS_LAMBDAS, POTENTIAL_ENERGY_VS_T, RESULTS_DIR, job_dir_path
from fkt_utils import build_energy_csv_index, list_available_replicas, resolve_energy_csv

# Reuse Fig-4 structural T_s helper after local import path setup.
from analyze_material_time_aging import _csv_is_sane, _structural_Ts_from_csv


def _load_calibrator():
    from openmm.cavitymd.empirical import EmpiricalTemperatureData

    return EmpiricalTemperatureData(str(POTENTIAL_ENERGY_VS_T), energy_component="lj_coulombic")


def _tracked_Ts(data: np.ndarray) -> np.ndarray:
    raw = np.asarray(data["T_s_fictive_K"], dtype=float)
    valid = np.isfinite(raw) & (raw > 0.0) & (raw < 600.0)
    out = np.full_like(raw, np.nan, dtype=float)
    out[valid] = raw[valid]
    return out


def compare_lambda(lam: float, max_replicas: int | None) -> dict:
    job_dir = job_dir_path(lam)
    replicas = list_available_replicas(job_dir, lam)
    if max_replicas is not None:
        replicas = replicas[:max_replicas]
    index = build_energy_csv_index(job_dir, lam)

    deltas: list[float] = []
    n_with_tracked = 0
    n_compared = 0
    n_total = 0

    for replica in replicas:
        csv_path = resolve_energy_csv(job_dir, lam, replica, index)
        if csv_path is None:
            continue
        data = np.genfromtxt(csv_path, delimiter=",", names=True, missing_values="", usemask=False)
        if not _csv_is_sane(data):
            continue
        n_total += 1
        tracked = _tracked_Ts(data)
        inferred = _structural_Ts_from_csv(data)
        mask = np.isfinite(tracked)
        if not np.any(mask):
            continue
        n_with_tracked += 1
        delta = inferred[mask] - tracked[mask]
        deltas.extend(delta.tolist())
        n_compared += int(mask.sum())

    arr = np.asarray(deltas, dtype=float)
    summary = {
        "lambda": lam,
        "replicas_scanned": n_total,
        "replicas_with_tracked_Ts": n_with_tracked,
        "points_compared": n_compared,
    }
    if arr.size:
        summary.update(
            {
                "mean_delta_K": float(np.mean(arr)),
                "std_delta_K": float(np.std(arr)),
                "median_abs_delta_K": float(np.median(np.abs(arr))),
                "max_abs_delta_K": float(np.max(np.abs(arr))),
            }
        )
    return summary


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--lambdas", type=float, nargs="+", default=ANALYSIS_LAMBDAS)
    parser.add_argument("--max-replicas", type=int, default=200)
    parser.add_argument("--output", type=Path, default=RESULTS_DIR / "ts_inversion_validation.json")
    args = parser.parse_args()

    calibrator = _load_calibrator()
    print(
        f"calibrator: t35 fit R²={calibrator.extended_t35_fit.get('r2', float('nan')):.4f}",
        flush=True,
    )

    results = [compare_lambda(lam, args.max_replicas) for lam in args.lambdas]
    args.output.parent.mkdir(parents=True, exist_ok=True)
    with open(args.output, "w", encoding="utf-8") as fh:
        json.dump({"comparisons": results}, fh, indent=2)

    for row in results:
        print(json.dumps(row), flush=True)
    print(f"Wrote {args.output}", flush=True)


if __name__ == "__main__":
    main()
