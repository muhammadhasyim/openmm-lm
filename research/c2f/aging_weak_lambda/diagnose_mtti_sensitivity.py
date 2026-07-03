#!/usr/bin/env python3
"""Sweep MTTI smoothness/grid settings and report h(t_w,end) stability."""

from __future__ import annotations

import argparse
import importlib.util
import json
from pathlib import Path

import numpy as np

from analyze_material_time_aging import _load_cavitymd_analysis, _tau_tw_from_summary
from config import ANALYSIS_LAMBDAS, RELAXATION_TIMES_VS_T, RESULTS_DIR, SWITCH_TIME_PS, job_dir_path
from fkt_utils import list_available_replicas

RelaxationTimeModel = _load_cavitymd_analysis().RelaxationTimeModel
ToolNarayanaswamy = _load_cavitymd_analysis().ToolNarayanaswamy


def _tau_tw_for_lambda(lam: float, relax_summary: dict, job_dir: Path, replicas: list[int]):
    from analyze_material_time_aging import _tau_tw_table

    tw, tau = _tau_tw_from_summary(relax_summary, lam)
    if tw.size:
        return tw, tau
    return _tau_tw_table(job_dir, lam, replicas)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--lambdas", type=float, nargs="+", default=ANALYSIS_LAMBDAS)
    parser.add_argument(
        "--smoothness-alphas",
        type=float,
        nargs="+",
        default=[0.5, 1.0, 2.0],
    )
    parser.add_argument("--output", type=Path, default=RESULTS_DIR / "mtti_sensitivity.json")
    args = parser.parse_args()

    relax_summary_path = RESULTS_DIR / "relaxation_summary.json"
    relax_summary: dict = {}
    if relax_summary_path.is_file():
        with open(relax_summary_path, encoding="utf-8") as fh:
            relax_summary = json.load(fh)

    relax_model = RelaxationTimeModel(str(RELAXATION_TIMES_VS_T))
    rows: list[dict] = []
    h_end_by_lam: dict[str, list[float]] = {}

    for lam in args.lambdas:
        job_dir = job_dir_path(lam)
        replicas = list_available_replicas(job_dir, lam)
        tw, tau = _tau_tw_for_lambda(lam, relax_summary, job_dir, replicas)
        if tw.size < 2:
            continue
        abs_times = SWITCH_TIME_PS + tw
        n_constraints = tw.size
        for alpha in args.smoothness_alphas:
            smooth_alpha = alpha * max(n_constraints, 3) / 13.0
            tn = ToolNarayanaswamy(
                relaxation_model=relax_model,
                beta=0.55,
                smoothness_alpha=smooth_alpha,
            )
            _t, h = tn.reconstruct_material_time(
                abs_times,
                tau,
                origin_time_ps=SWITCH_TIME_PS,
            )
            h_end = float(h[-1]) if h.size else float("nan")
            rows.append(
                {
                    "lambda": lam,
                    "smoothness_alpha": alpha,
                    "n_constraints": int(n_constraints),
                    "h_end": h_end,
                }
            )
            h_end_by_lam.setdefault(str(lam), []).append(h_end)

    ordering_stable = True
    lam_keys = sorted(h_end_by_lam, key=float)
    for alpha_idx, alpha in enumerate(args.smoothness_alphas):
        ends = []
        for lam_key in lam_keys:
            match = [r for r in rows if str(r["lambda"]) == lam_key and r["smoothness_alpha"] == alpha]
            if match:
                ends.append((float(lam_key), match[0]["h_end"]))
        ends.sort(key=lambda item: item[0])
        if len(ends) >= 2:
            # Expect higher lambda to have >= h_end than lower at late time (qualitative check).
            for (_l0, h0), (l1, h1) in zip(ends, ends[1:]):
                if h1 + 0.5 < h0:
                    ordering_stable = False

    summary = {
        "rows": rows,
        "ordering_stable_across_alphas": ordering_stable,
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    with open(args.output, "w", encoding="utf-8") as fh:
        json.dump(summary, fh, indent=2)
    print(json.dumps(summary, indent=2), flush=True)
    print(f"Wrote {args.output}", flush=True)


if __name__ == "__main__":
    main()
