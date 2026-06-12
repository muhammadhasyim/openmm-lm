#!/usr/bin/env python3
"""Extract tau_s and tau_tilde vs lambda and t_w (Fig 2b/c)."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

from config import FIGURES_DIR, LAMBDAS, N_REPLICAS, RESULTS_DIR, job_dir_path
from fkt_utils import (
    average_fkt_over_replicas,
    average_phi_over_replicas,
    collect_replica_fkt_files,
    extract_tau_s,
    fit_kww_tau,
    list_available_replicas,
    waiting_time_ps,
)


def analyze_lambda(lam: float, replicas: list[int]) -> dict:
    job_dir = job_dir_path(lam)
    results: dict[str, object] = {"lambda": lam, "job_dir": str(job_dir), "by_tw": {}}
    if not job_dir.exists():
        results["error"] = "missing job_dir"
        return results

    available = [r for r in replicas if r in list_available_replicas(job_dir, lam)]
    results["n_replicas"] = len(available)
    ref_indices = sorted(
        {
            idx
            for replica in available
            for idx in collect_replica_fkt_files(job_dir, lam, replica)
        }
    )
    for ref_idx in ref_indices:
        ref_time, lags, mean_fkt, n_used = average_fkt_over_replicas(
            job_dir, lam, available, ref_idx
        )
        if lags.size == 0:
            continue
        tau_s = extract_tau_s(lags, mean_fkt, threshold=0.1, min_lag_ps=10.0)
        tau_kww = fit_kww_tau(lags, mean_fkt, min_lag_ps=10.0)
        t_w = waiting_time_ps(ref_time, ref_idx)
        results["by_tw"][str(t_w)] = {
            "ref_idx": ref_idx,
            "ref_time_ps": ref_time,
            "tau_s_ps": tau_s,
            "tau_kww_ps": tau_kww,
            "n_replicas": n_used,
        }
    return results


def plot_tau_tilde(
    all_results: dict[float, dict],
    baseline: dict,
    out_dir: Path,
) -> None:
    out_dir.mkdir(parents=True, exist_ok=True)
    baseline_tw = {
        float(k): v["tau_s_ps"]
        for k, v in baseline.get("by_tw", {}).items()
        if v.get("tau_s_ps") is not None
    }

    fig, ax = plt.subplots(figsize=(8, 5))
    tw_values = sorted({float(k) for r in all_results.values() for k in r.get("by_tw", {})})
    colors = plt.cm.plasma(np.linspace(0.1, 0.9, max(len(tw_values), 1)))
    for color, t_w in zip(colors, tw_values):
        xs, ys = [], []
        for lam in sorted(all_results):
            if lam == 0.0:
                continue
            entry = all_results[lam]["by_tw"].get(str(t_w))
            if not entry or entry.get("tau_s_ps") is None:
                continue
            base_tau = baseline_tw.get(t_w)
            if not base_tau:
                continue
            xs.append(lam)
            ys.append(entry["tau_s_ps"] / base_tau)
        if xs:
            ax.plot(xs, ys, "o-", color=color, label=f"$t_w$={t_w:.0f} ps")
    ax.axhline(1.0, color="gray", ls=":", lw=1.0)
    ax.set_xlabel("$\\lambda$ (a.u.)")
    ax.set_ylabel("$\\tilde{\\tau}_s = \\tau_s/\\tau_{s,\\lambda=0}$")
    ax.set_title("Cavity-induced structural slowdown (weak coupling)")
    ax.legend(fontsize=8, ncol=2)
    fig.tight_layout()
    fig.savefig(out_dir / "fig2b_tau_tilde_vs_lambda.png", dpi=150)
    plt.close(fig)

    fig, ax = plt.subplots(figsize=(8, 5))
    lam_colors = plt.cm.viridis(np.linspace(0.15, 0.9, max(len(all_results), 1)))
    for (lam, res), color in zip(sorted(all_results.items()), lam_colors):
        if lam == 0.0:
            continue
        xs, ys = [], []
        for t_w_str, entry in sorted(res.get("by_tw", {}).items(), key=lambda x: float(x[0])):
            tau = entry.get("tau_s_ps")
            if tau is None:
                continue
            t_w = float(t_w_str)
            base_tau = baseline_tw.get(t_w)
            if not base_tau:
                continue
            xs.append(t_w)
            ys.append(tau / base_tau)
        if xs:
            ax.plot(xs, ys, "o-", color=color, label=f"$\\lambda$={lam:g}")
    ax.axhline(1.0, color="gray", ls=":", lw=1.0)
    ax.set_xlabel("$t_w$ (ps)")
    ax.set_ylabel("$\\tilde{\\tau}_s$")
    ax.set_title("Memory of cavity perturbation vs waiting time")
    ax.legend(fontsize=8)
    fig.tight_layout()
    fig.savefig(out_dir / "fig2c_tau_tilde_vs_tw.png", dpi=150)
    plt.close(fig)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-dir", type=Path, default=FIGURES_DIR)
    parser.add_argument("--results-dir", type=Path, default=RESULTS_DIR)
    parser.add_argument("--lambdas", type=float, nargs="+", default=LAMBDAS)
    parser.add_argument("--replicas", type=int, nargs="+", default=list(range(N_REPLICAS)))
    args = parser.parse_args()

    all_results: dict[float, dict] = {}
    for lam in args.lambdas:
        all_results[lam] = analyze_lambda(lam, args.replicas)
        print(f"lambda={lam:g}: n_rep={all_results[lam].get('n_replicas', 0)}")

    baseline = all_results.get(0.0, {})
    plot_tau_tilde(all_results, baseline, args.output_dir)

    args.results_dir.mkdir(parents=True, exist_ok=True)
    summary_path = args.results_dir / "relaxation_summary.json"
    with open(summary_path, "w", encoding="utf-8") as fh:
        json.dump({str(k): v for k, v in all_results.items()}, fh, indent=2)
    print(f"Summary -> {summary_path}")


if __name__ == "__main__":
    main()
