#!/usr/bin/env python3
"""Fig 4: material time, ISF collapse, TN overlays."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

from config import (
    FIGURES_DIR,
    LAMBDAS,
    N_REPLICAS,
    RELAXATION_TIMES_VS_T,
    RESULTS_DIR,
    SWITCH_TIME_PS,
    job_dir_path,
    run_prefix,
)
from fkt_utils import (
    average_fkt_over_replicas,
    average_phi_over_replicas,
    collect_replica_fkt_files,
    extract_tau_s,
    list_available_replicas,
    waiting_time_ps,
)

import importlib.util

def _load_cavitymd_analysis():
    analysis_path = (
        Path(__file__).resolve().parents[4]
        / "wrappers"
        / "python"
        / "openmm"
        / "cavitymd"
        / "analysis.py"
    )
    spec = importlib.util.spec_from_file_location("cavitymd_analysis", analysis_path)
    if spec is None or spec.loader is None:
        raise ImportError(f"Cannot load {analysis_path}")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


_cavitymd_analysis = _load_cavitymd_analysis()
RelaxationTimeModel = _cavitymd_analysis.RelaxationTimeModel
ToolNarayanaswamy = _cavitymd_analysis.ToolNarayanaswamy


def _load_Ts_timeseries(
    job_dir: Path, lam: float, replicas: list[int]
) -> tuple[np.ndarray, np.ndarray]:
    times_list: list[np.ndarray] = []
    ts_list: list[np.ndarray] = []
    for replica in replicas:
        csv_path = job_dir / f"{run_prefix(lam, replica)}_energies.csv"
        if not csv_path.exists():
            continue
        data = np.genfromtxt(
            csv_path, delimiter=",", names=True, missing_values="", usemask=False
        )
        t = np.asarray(data["time_ps"], dtype=float)
        ts_raw = np.asarray(data["T_s_fictive_K"], dtype=float)
        ts = np.where(np.isfinite(ts_raw), ts_raw, np.nan)
        times_list.append(t)
        ts_list.append(ts)
    if not times_list:
        return np.array([]), np.array([])
    t_ref = times_list[0]
    stack = np.vstack([np.interp(t_ref, times_list[i], ts_list[i]) for i in range(len(times_list))])
    return t_ref, np.nanmean(stack, axis=0)


def _tau_tw_table(job_dir: Path, lam: float, replicas: list[int]) -> tuple[np.ndarray, np.ndarray]:
    tw_list: list[float] = []
    tau_list: list[float] = []
    ref_indices = sorted(
        {idx for r in replicas for idx in collect_replica_fkt_files(job_dir, lam, r)}
    )
    for ref_idx in ref_indices:
        ref_time, lags, mean_fkt, _ = average_fkt_over_replicas(job_dir, lam, replicas, ref_idx)
        tau = extract_tau_s(lags, mean_fkt, threshold=0.1, min_lag_ps=10.0)
        if tau is None:
            continue
        tw_list.append(waiting_time_ps(ref_time, ref_idx))
        tau_list.append(tau)
    order = np.argsort(tw_list)
    return np.asarray(tw_list, dtype=float)[order], np.asarray(tau_list, dtype=float)[order]


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-dir", type=Path, default=FIGURES_DIR)
    parser.add_argument("--results-dir", type=Path, default=RESULTS_DIR)
    parser.add_argument("--lambdas", type=float, nargs="+", default=[lam for lam in LAMBDAS if lam > 0])
    parser.add_argument("--replicas", type=int, nargs="+", default=list(range(N_REPLICAS)))
    parser.add_argument("--smoothness-alpha", type=float, default=0.1)
    args = parser.parse_args()

    args.output_dir.mkdir(parents=True, exist_ok=True)
    relax_model = RelaxationTimeModel(str(RELAXATION_TIMES_VS_T))
    tn = ToolNarayanaswamy(
        relaxation_model=relax_model, beta=0.55, smoothness_alpha=args.smoothness_alpha
    )

    summary: dict[str, object] = {"lambdas": {}}
    fig_a, ax_a = plt.subplots(figsize=(8, 5))
    collapse_h: list[np.ndarray] = []
    collapse_phi: list[np.ndarray] = []

    for lam in args.lambdas:
        job_dir = job_dir_path(lam)
        replicas = [r for r in args.replicas if r in list_available_replicas(job_dir, lam)]
        if not replicas:
            continue

        tw, tau = _tau_tw_table(job_dir, lam, replicas)
        if tw.size < 2:
            continue
        abs_times = tw + SWITCH_TIME_PS
        t_grid, h_meas = tn.reconstruct_material_time(abs_times, tau + tw)
        t_ts, T_s = _load_Ts_timeseries(job_dir, lam, replicas)
        h_tn = tn.integrate_tn(t_ts, T_s) if t_ts.size else np.zeros_like(t_grid)

        ax_a.plot(t_grid, h_meas, lw=1.5, label=f"$\\lambda$={lam:g}")
        if t_ts.size:
            ax_a.plot(t_ts, h_tn, ls="--", lw=1.0, alpha=0.8)

        ref_files = collect_replica_fkt_files(job_dir, lam, replicas[0])
        if ref_files:
            ref_idx = min(ref_files.keys())
            _, lags, mean_phi, _, _ = average_phi_over_replicas(
                job_dir, lam, replicas, ref_idx
            )
            if lags.size == 0:
                continue
            phi = mean_phi
            if lags.size:
                h_diff, _ = tn.collapse_isf(
                    lags, np.array([waiting_time_ps(None, ref_idx)]), h_meas, t_grid
                )
                collapse_h.append(h_diff)
                collapse_phi.append(phi[: h_diff.size])

        summary["lambdas"][str(lam)] = {
            "waiting_times_ps": tw.tolist(),
            "tau_s_ps": tau.tolist(),
            "h_end": float(h_meas[-1]),
        }

    ax_a.set_xlabel("$t$ (ps)")
    ax_a.set_ylabel("$h_\\lambda(t)$")
    ax_a.set_title("Material time: measured (solid) vs TN (dashed)")
    ax_a.legend(fontsize=8)
    fig_a.tight_layout()
    fig_a.savefig(args.output_dir / "fig4a_material_time.png", dpi=150)
    plt.close(fig_a)

    fig_b, ax_b = plt.subplots(figsize=(6, 5))
    h_master = np.linspace(0.0, 3.0, 200)
    ax_b.plot(h_master, tn.stretched_exponential(h_master), "k--", lw=1.5, label=r"$\Phi(h)=e^{-h^{0.55}}$")
    for h_diff, phi in zip(collapse_h, collapse_phi):
        n = min(h_diff.size, phi.size)
        if n > 2:
            ax_b.plot(h_diff[:n], phi[:n], "o", ms=3, alpha=0.5)
    ax_b.set_xlabel("$h$")
    ax_b.set_ylabel("$\\Phi_k(h)$")
    ax_b.set_title("ISF collapse onto stretched exponential")
    ax_b.legend(fontsize=9)
    fig_b.tight_layout()
    fig_b.savefig(args.output_dir / "fig4b_isf_collapse.png", dpi=150)
    plt.close(fig_b)

    relax_json = args.results_dir / "relaxation_summary.json"
    if relax_json.exists():
        with open(relax_json, encoding="utf-8") as fh:
            relax_data = json.load(fh)
        baseline = relax_data.get("0", relax_data.get("0.0", {}))
        baseline_tw = {
            float(k): v["tau_s_ps"]
            for k, v in baseline.get("by_tw", {}).items()
            if v.get("tau_s_ps") is not None
        }

        fig_c, ax_c = plt.subplots(figsize=(8, 5))
        for lam in args.lambdas:
            job_dir = job_dir_path(lam)
            replicas = [r for r in args.replicas if r in list_available_replicas(job_dir, lam)]
            tw, tau = _tau_tw_table(job_dir, lam, replicas)
            if tw.size == 0:
                continue
            tau_tilde = [
                tau[i] / baseline_tw[float(tw[i])]
                for i in range(tw.size)
                if float(tw[i]) in baseline_tw
            ]
            if tau_tilde:
                ax_c.plot([lam] * len(tau_tilde), tau_tilde, "o", alpha=0.6)
        ax_c.set_xlabel("$\\lambda$ (a.u.)")
        ax_c.set_ylabel("$\\tilde{\\tau}_s$ (TN approx)")
        ax_c.set_title("TN-predicted slowdown vs coupling")
        fig_c.tight_layout()
        fig_c.savefig(args.output_dir / "fig4c_tn_tau_tilde_vs_lambda.png", dpi=150)
        plt.close(fig_c)

        fig_d, ax_d = plt.subplots(figsize=(8, 5))
        for lam in args.lambdas:
            job_dir = job_dir_path(lam)
            replicas = [r for r in args.replicas if r in list_available_replicas(job_dir, lam)]
            tw, tau = _tau_tw_table(job_dir, lam, replicas)
            if tw.size == 0:
                continue
            ys = [tau[i] / baseline_tw[float(tw[i])] for i in range(tw.size) if float(tw[i]) in baseline_tw]
            xs = [tw[i] for i in range(tw.size) if float(tw[i]) in baseline_tw]
            if ys:
                ax_d.plot(xs, ys, "o-", label=f"$\\lambda$={lam:g}")
        ax_d.set_xlabel("$t_w$ (ps)")
        ax_d.set_ylabel("$\\tilde{\\tau}_s$")
        ax_d.set_title("TN comparison vs waiting time")
        ax_d.legend(fontsize=8)
        fig_d.tight_layout()
        fig_d.savefig(args.output_dir / "fig4d_tn_tau_tilde_vs_tw.png", dpi=150)
        plt.close(fig_d)

    args.results_dir.mkdir(parents=True, exist_ok=True)
    with open(args.results_dir / "material_time_summary.json", "w", encoding="utf-8") as fh:
        json.dump(summary, fh, indent=2)
    print(f"Wrote Fig 4 panels to {args.output_dir}")


if __name__ == "__main__":
    main()
