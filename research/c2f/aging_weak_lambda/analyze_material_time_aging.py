#!/usr/bin/env python3
"""Fig 4: material time, ISF collapse, TN overlays."""

from __future__ import annotations

import argparse
import importlib.util
import json
import sys
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

from config import (
    ANALYSIS_LAMBDAS,
    FIGURES_DIR,
    N_REPLICAS,
    POTENTIAL_ENERGY_VS_T,
    RELAXATION_TIMES_VS_T,
    RESULTS_DIR,
    RUNTIME_PS,
    SWITCH_TIME_PS,
    TEMPERATURE_K,
    job_dir_path,
    run_prefix,
)
from fkt_utils import (
    average_fkt_over_replicas,
    average_phi_over_replicas,
    build_energy_csv_index,
    collect_replica_fkt_files,
    extract_tau_s,
    list_available_replicas,
    resolve_energy_csv,
    waiting_time_ps,
)
from paper_style import apply_paper_style, paper_legend, save_figure, style_axes


def _load_cavitymd_analysis():
    analysis_path = (
        Path(__file__).resolve().parents[3]
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

_STRUCTURAL_CALIBRATOR: object | None = None
_UNITS: object | None = None


def _structural_calibrator():
    global _STRUCTURAL_CALIBRATOR, _UNITS
    if _STRUCTURAL_CALIBRATOR is None:
        from openmm.cavitymd.constants import Units
        from openmm.cavitymd.empirical import EmpiricalTemperatureData

        if not POTENTIAL_ENERGY_VS_T.is_file():
            raise FileNotFoundError(
                f"structural calibration table missing: {POTENTIAL_ENERGY_VS_T}"
            )
        _STRUCTURAL_CALIBRATOR = EmpiricalTemperatureData(
            str(POTENTIAL_ENERGY_VS_T), energy_component="lj_coulombic"
        )
        _UNITS = Units
    return _STRUCTURAL_CALIBRATOR, _UNITS


def _structural_Ts_from_csv(data: np.ndarray) -> np.ndarray:
    """Infer T_s uniformly from nonbonded energy via the fitted calibration model."""
    calibrator, units = _structural_calibrator()
    E_nb = np.asarray(data["E_nonbonded_kjmol"], dtype=float)
    order = np.argsort(calibrator.energies)
    e_sorted = calibrator.energies[order]
    e_lo, e_hi = float(e_sorted[0]), float(e_sorted[-1])

    E_hartree = E_nb * units.KJMOL_TO_HARTREE
    physical = np.isfinite(E_hartree) & (E_hartree >= e_lo) & (E_hartree <= e_hi)
    T_s = np.full(E_nb.shape, TEMPERATURE_K, dtype=float)
    if np.any(physical):
        T_s[physical] = calibrator.calculate_temperature_array(E_hartree[physical])
    invalid = ~np.isfinite(T_s) | (T_s <= 0.0) | (T_s > 600.0)
    T_s[invalid] = TEMPERATURE_K
    return T_s


def _csv_is_sane(data: np.ndarray) -> bool:
    """Reject trajectories with non-physical energy spikes in the CSV."""
    if "E_nonbonded_kjmol" not in (data.dtype.names or ()):
        return False
    e_nb = np.asarray(data["E_nonbonded_kjmol"], dtype=float)
    if not np.all(np.isfinite(e_nb)):
        return False
    if np.max(e_nb) > 0.0:
        return False
    if np.min(e_nb) < -8000.0:
        return False
    return True


def _load_Ts_timeseries(
    job_dir: Path, lam: float, replicas: list[int]
) -> tuple[np.ndarray, np.ndarray]:
    """Return (times, mean T_s) for diagnostics / Fig 4c–d overlays."""
    index = build_energy_csv_index(job_dir, lam)
    times_list: list[np.ndarray] = []
    ts_list: list[np.ndarray] = []
    for replica in replicas:
        csv_path = resolve_energy_csv(job_dir, lam, replica, index)
        if csv_path is None:
            continue
        data = np.genfromtxt(
            csv_path, delimiter=",", names=True, missing_values="", usemask=False
        )
        if not _csv_is_sane(data):
            continue
        t = np.asarray(data["time_ps"], dtype=float)
        ts = _structural_Ts_from_csv(data)
        times_list.append(t)
        ts_list.append(ts)
    if not times_list:
        return np.array([]), np.array([])
    t_ref = times_list[0]
    stack = np.vstack(
        [np.interp(t_ref, times_list[i], ts_list[i]) for i in range(len(times_list))]
    )
    return t_ref, np.nanmean(stack, axis=0)


def _load_h_tn_timeseries(
    job_dir: Path,
    lam: float,
    replicas: list[int],
    tn: ToolNarayanaswamy,
    *,
    rate_scale: float = 1.0,
) -> tuple[np.ndarray, np.ndarray]:
    """Integrate TN material time per replica, then ensemble-average (avoids Jensen bias)."""
    index = build_energy_csv_index(job_dir, lam)
    t_ref: np.ndarray | None = None
    h_stack: list[np.ndarray] = []

    for replica in replicas:
        csv_path = resolve_energy_csv(job_dir, lam, replica, index)
        if csv_path is None:
            continue
        data = np.genfromtxt(
            csv_path, delimiter=",", names=True, missing_values="", usemask=False
        )
        if not _csv_is_sane(data):
            continue
        t = np.asarray(data["time_ps"], dtype=float)
        ts = _structural_Ts_from_csv(data)
        h = tn.integrate_tn(t, ts, switch_time_ps=SWITCH_TIME_PS, rate_scale=rate_scale)
        if t_ref is None:
            t_ref = t
        h_stack.append(np.interp(t_ref, t, h))

    if t_ref is None or not h_stack:
        return np.array([]), np.array([])
    return t_ref, np.nanmean(np.vstack(h_stack), axis=0)


def _tau_tw_table(job_dir: Path, lam: float, replicas: list[int]) -> tuple[np.ndarray, np.ndarray]:
    tw_list: list[float] = []
    tau_list: list[float] = []
    ref_indices = sorted(
        {idx for r in replicas for idx in collect_replica_fkt_files(job_dir, lam, r)}
    )
    for ref_idx in ref_indices:
        ref_time, lags, mean_fkt, _ = average_fkt_over_replicas(
            job_dir, lam, replicas, ref_idx
        )
        tau = extract_tau_s(lags, mean_fkt, threshold=0.1, min_lag_ps=10.0)
        if tau is None:
            continue
        tw_list.append(waiting_time_ps(ref_time, ref_idx))
        tau_list.append(tau)
    order = np.argsort(tw_list)
    return np.asarray(tw_list, dtype=float)[order], np.asarray(tau_list, dtype=float)[order]


def _tau_tw_from_summary(relax_data: dict, lam: float) -> tuple[np.ndarray, np.ndarray]:
    """Load precomputed tau_s(t_w) from relaxation_summary.json (full ensemble)."""
    entry = relax_data.get(str(lam), relax_data.get(f"{lam:g}", {}))
    by_tw = entry.get("by_tw", {})
    tw_list: list[float] = []
    tau_list: list[float] = []
    for tw_str, row in sorted(by_tw.items(), key=lambda item: float(item[0])):
        tau = row.get("tau_s_ps")
        if tau is None:
            continue
        tw_list.append(float(tw_str))
        tau_list.append(float(tau))
    return np.asarray(tw_list, dtype=float), np.asarray(tau_list, dtype=float)


def _lab_time_grid() -> np.ndarray:
    return np.linspace(SWITCH_TIME_PS, RUNTIME_PS, 500)


def _aging_time(t_lab: np.ndarray) -> np.ndarray:
    return np.asarray(t_lab, dtype=float) - SWITCH_TIME_PS


def _equilibrium_h(t_grid: np.ndarray, relax_model: RelaxationTimeModel) -> np.ndarray:
    """Paper Eq. 12 with h=0 at coupling turn-on (SWITCH_TIME_PS)."""
    tau_eq = relax_model.get_relaxation_time(TEMPERATURE_K)
    aging = np.maximum(np.asarray(t_grid, dtype=float) - SWITCH_TIME_PS, 0.0)
    return aging / max(tau_eq, 1e-12)


def _measured_baseline_arrays(
    relax_summary: dict,
    job_dir: Path,
    replicas: list[int],
) -> tuple[np.ndarray, np.ndarray]:
    """Sorted measured tau_s(λ=0, t_w) arrays for interpolated tau_tilde normalization."""
    tw, tau = _tau_tw_from_summary(relax_summary, 0.0)
    if tw.size == 0:
        tw, tau = _tau_tw_table(job_dir, 0.0, replicas)
    order = np.argsort(tw)
    tw = tw[order]
    tau = tau[order]
    valid = tau > 0.0
    return tw[valid], tau[valid]


def _baseline_tau_at(t_w: float, tw0: np.ndarray, tau0: np.ndarray) -> float | None:
    if tw0.size == 0:
        return None
    base = float(np.interp(t_w, tw0, tau0))
    if base <= 0.0 or not np.isfinite(base):
        return None
    return base


def _ensure_phi_starts_at_one(
    lags: np.ndarray, phi: np.ndarray
) -> tuple[np.ndarray, np.ndarray]:
    """Ensure ISF normalization gives φ(τ=0)=1 for collapse plots."""
    if lags.size == 0:
        return lags, phi
    if lags[0] > 1e-12:
        return np.concatenate([[0.0], lags]), np.concatenate([[1.0], phi])
    phi_out = phi.copy()
    phi_out[0] = 1.0
    return lags, phi_out


def _tn_baseline_arrays(
    tn: ToolNarayanaswamy,
    t_lab: np.ndarray,
    h_tn: np.ndarray,
    tw_grid: np.ndarray,
) -> tuple[np.ndarray, np.ndarray]:
    """TN-predicted tau_s(λ=0, t_w) for normalizing TN slowdown curves."""
    tw_list: list[float] = []
    tau_list: list[float] = []
    for t_w in tw_grid:
        tau = tn.tau_s_tn_from_h(t_lab, h_tn, float(t_w), SWITCH_TIME_PS)
        if tau is not None and tau > 0.0:
            tw_list.append(float(t_w))
            tau_list.append(float(tau))
    if not tw_list:
        return np.array([]), np.array([])
    order = np.argsort(tw_list)
    return np.asarray(tw_list, dtype=float)[order], np.asarray(tau_list, dtype=float)[order]


def _tn_tau_tilde_table(
    tn: ToolNarayanaswamy,
    t_lab: np.ndarray,
    h_tn: np.ndarray,
    tw_values: np.ndarray,
    baseline_tw: np.ndarray,
    baseline_tau: np.ndarray,
) -> tuple[list[float], list[float]]:
    xs: list[float] = []
    ys: list[float] = []
    for t_w in tw_values:
        tau_tn = tn.tau_s_tn_from_h(t_lab, h_tn, float(t_w), SWITCH_TIME_PS)
        base = _baseline_tau_at(float(t_w), baseline_tw, baseline_tau)
        if tau_tn is None or base is None:
            continue
        xs.append(float(t_w))
        ys.append(float(tau_tn / base))
    return xs, ys


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-dir", type=Path, default=FIGURES_DIR)
    parser.add_argument("--results-dir", type=Path, default=RESULTS_DIR)
    parser.add_argument("--lambdas", type=float, nargs="+", default=ANALYSIS_LAMBDAS)
    parser.add_argument("--replicas", type=int, nargs="+", default=list(range(N_REPLICAS)))
    parser.add_argument("--smoothness-alpha", type=float, default=1.0)
    parser.add_argument("--block-window-ps", type=float, default=10.0)
    parser.add_argument(
        "--collapse-max-replicas",
        type=int,
        default=100,
        help="Max replicas for Fig 4b ISF collapse (full ensemble used for tau if no summary JSON).",
    )
    args = parser.parse_args()

    apply_paper_style(grid=False)

    relax_summary_path = args.results_dir / "relaxation_summary.json"
    relax_summary: dict = {}
    if relax_summary_path.is_file():
        with open(relax_summary_path, encoding="utf-8") as fh:
            relax_summary = json.load(fh)
        print(f"[fig4] using tau_s from {relax_summary_path}", flush=True)

    if not RELAXATION_TIMES_VS_T.is_file():
        raise SystemExit(f"relaxation table not found: {RELAXATION_TIMES_VS_T}")

    args.output_dir.mkdir(parents=True, exist_ok=True)
    relax_model = RelaxationTimeModel(str(RELAXATION_TIMES_VS_T))
    if not relax_model.is_fitted:
        raise SystemExit(f"RelaxationTimeModel failed to load: {RELAXATION_TIMES_VS_T}")

    tn = ToolNarayanaswamy(
        relaxation_model=relax_model, beta=0.55, smoothness_alpha=args.smoothness_alpha
    )

    plot_lams = sorted(set(args.lambdas))
    t_plot = _lab_time_grid()
    summary: dict[str, object] = {"lambdas": {}}
    h_curves: dict[float, tuple[np.ndarray, np.ndarray]] = {}
    ts_cache: dict[float, tuple[np.ndarray, np.ndarray]] = {}
    h_tn_cache: dict[float, np.ndarray] = {}
    h_tn_unc_lo: dict[float, np.ndarray] = {}
    h_tn_unc_hi: dict[float, np.ndarray] = {}

    fig_a, ax_a = plt.subplots(figsize=(8, 5))
    collapse_h: list[np.ndarray] = []
    collapse_phi: list[np.ndarray] = []

    def tau_tw_for_lambda(lam: float, job_dir: Path, replicas: list[int]) -> tuple[np.ndarray, np.ndarray]:
        if relax_summary:
            tw, tau = _tau_tw_from_summary(relax_summary, lam)
            if tw.size:
                return tw, tau
        return _tau_tw_table(job_dir, lam, replicas)

    job_dir_0 = job_dir_path(0.0)
    replicas_0 = [r for r in args.replicas if r in list_available_replicas(job_dir_0, 0.0)]
    baseline_tw, baseline_tau = _measured_baseline_arrays(relax_summary, job_dir_0, replicas_0)

    tau_eq_ref = relax_model.get_relaxation_time(TEMPERATURE_K)
    tau_std_ref = relax_model.get_relaxation_time_std(TEMPERATURE_K)
    tau_rel_unc = tau_std_ref / max(tau_eq_ref, 1e-12)
    if tau_std_ref > 0.0:
        print(
            f"[fig4] tau_s,eq({TEMPERATURE_K:.0f}K)={tau_eq_ref:.2f} ps "
            f"(bootstrap σ={tau_std_ref:.2f} ps)",
            flush=True,
        )

    for lam in plot_lams:
        job_dir = job_dir_path(lam)
        replicas = [r for r in args.replicas if r in list_available_replicas(job_dir, lam)]
        if not replicas:
            continue
        print(f"[fig4] lambda={lam:g}: loading T_s ({len(replicas)} replicas)...", flush=True)

        t_ts, T_s = _load_Ts_timeseries(job_dir, lam, replicas)
        ts_cache[lam] = (t_ts, T_s)
        if t_ts.size:
            h_tn_cache[lam] = _load_h_tn_timeseries(job_dir, lam, replicas, tn)[1]
            if tau_rel_unc > 0.0:
                h_tn_unc_lo[lam] = _load_h_tn_timeseries(
                    job_dir, lam, replicas, tn, rate_scale=1.0 / (1.0 + tau_rel_unc)
                )[1]
                h_tn_unc_hi[lam] = _load_h_tn_timeseries(
                    job_dir,
                    lam,
                    replicas,
                    tn,
                    rate_scale=1.0 / max(1.0 - tau_rel_unc, 0.05),
                )[1]
        print(f"[fig4] lambda={lam:g}: MTTI + ISF collapse...", flush=True)

        if lam == 0.0:
            h_eq = _equilibrium_h(t_plot, relax_model)
            if tau_std_ref > 0.0:
                aging = _aging_time(t_plot)
                ax_a.fill_between(
                    aging,
                    aging / (tau_eq_ref + tau_std_ref),
                    aging / max(tau_eq_ref - tau_std_ref, 1e-12),
                    color="C0",
                    alpha=0.15,
                    zorder=2,
                    label=rf"$\lambda=0$ Eq. 12 $\pm\sigma_{{\tau}}$",
                )
            ax_a.plot(
                _aging_time(t_plot),
                h_eq,
                color="C0",
                lw=2.5,
                zorder=5,
                label=r"$\lambda=0$ (Eq. 12)",
            )
            h_curves[lam] = (t_plot, h_eq)
            tw, tau = tau_tw_for_lambda(lam, job_dir, replicas)
            if tw.size >= 2:
                abs_times = SWITCH_TIME_PS + tw
                n_constraints = tw.size
                smooth_alpha = args.smoothness_alpha * max(n_constraints, 3) / 13.0
                tn_lam = ToolNarayanaswamy(
                    relaxation_model=relax_model,
                    beta=0.55,
                    smoothness_alpha=smooth_alpha,
                )
                t_grid, h_meas = tn_lam.reconstruct_material_time(
                    abs_times,
                    tau,
                    time_grid_ps=t_plot,
                    origin_time_ps=SWITCH_TIME_PS,
                )
                ax_a.plot(
                    _aging_time(t_grid),
                    h_meas,
                    color="C0",
                    lw=1.5,
                    ls=":",
                    zorder=4,
                    label=r"$\lambda=0$ MTTI",
                )
            summary["lambdas"]["0"] = {"h_end": float(h_eq[-1])}
            continue

        tw, tau = tau_tw_for_lambda(lam, job_dir, replicas)
        if tw.size < 2:
            continue
        abs_times = SWITCH_TIME_PS + tw
        n_constraints = tw.size
        smooth_alpha = args.smoothness_alpha * max(n_constraints, 3) / 13.0
        tn_lam = ToolNarayanaswamy(
            relaxation_model=relax_model,
            beta=0.55,
            smoothness_alpha=smooth_alpha,
        )
        t_grid, h_meas = tn_lam.reconstruct_material_time(
            abs_times,
            tau,
            time_grid_ps=t_plot,
            origin_time_ps=SWITCH_TIME_PS,
        )
        h_curves[lam] = (t_grid, h_meas)

        h_tn = h_tn_cache.get(lam, np.zeros_like(t_grid))

        ax_a.plot(_aging_time(t_grid), h_meas, lw=1.5, label=f"$\\lambda$={lam:g}")
        if t_ts.size:
            line_color = ax_a.lines[-1].get_color()
            ax_a.plot(
                _aging_time(t_ts),
                h_tn,
                ls="--",
                lw=0.9,
                alpha=0.35,
                color=line_color,
                zorder=1,
            )
            h_lo = h_tn_unc_lo.get(lam)
            h_hi = h_tn_unc_hi.get(lam)
            if h_lo is not None and h_hi is not None and h_lo.size == h_hi.size:
                ax_a.fill_between(
                    _aging_time(t_ts),
                    h_lo,
                    h_hi,
                    color=line_color,
                    alpha=0.08,
                    zorder=0,
                )

        ref_indices = sorted(
            {idx for r in replicas for idx in collect_replica_fkt_files(job_dir, lam, r)}
        )
        collapse_replicas = replicas[: max(1, args.collapse_max_replicas)]
        for ref_idx in ref_indices:
            ref_time, lags, mean_phi, _, _ = average_phi_over_replicas(
                job_dir,
                lam,
                collapse_replicas,
                ref_idx,
                block_window_ps=args.block_window_ps,
            )
            if lags.size == 0:
                continue
            lags, mean_phi = _ensure_phi_starts_at_one(lags, mean_phi)
            t_w = waiting_time_ps(ref_time, ref_idx)
            lab_t_w = SWITCH_TIME_PS + t_w
            h_diff, _ = tn.collapse_isf(
                lags,
                np.array([lab_t_w]),
                h_meas,
                t_grid,
            )
            collapse_h.append(h_diff)
            collapse_phi.append(mean_phi[: h_diff.size])

        summary["lambdas"][str(lam)] = {
            "waiting_times_ps": tw.tolist(),
            "tau_s_ps": tau.tolist(),
            "h_end": float(h_meas[-1]),
        }

    ax_a.set_xlabel("$t_w$ (ps)")
    ax_a.set_ylabel("$h_\\lambda(t)$")
    ax_a.set_title("Material time: measured (solid) vs TN (dashed)")
    ax_a.set_ylim(bottom=0.0)
    style_axes(ax_a, grid=False)
    paper_legend(ax_a, loc="best", fontsize=9)
    fig_a.tight_layout()
    save_figure(fig_a, args.output_dir / "fig4a_material_time")
    plt.close(fig_a)

    fig_b, ax_b = plt.subplots(figsize=(6, 5))
    collapse_beta = 0.55
    h_master = np.linspace(0.0, 3.0, 200)
    ax_b.plot(
        h_master,
        tn.stretched_exponential(h_master, beta=collapse_beta),
        "k--",
        lw=1.5,
        label=rf"$\Phi_k(h)=e^{{-h^\beta}}$, $\beta={collapse_beta}$",
    )
    for h_diff, phi in zip(collapse_h, collapse_phi):
        n = min(h_diff.size, phi.size)
        if n > 2:
            ax_b.plot(h_diff[:n], phi[:n], "o", ms=2, alpha=0.25)
    ax_b.set_xlabel("$h$")
    ax_b.set_ylabel("$\\Phi_k(h)$")
    ax_b.set_title("ISF collapse onto stretched exponential")
    ax_b.set_xlim(0.0, 3.0)
    ax_b.set_ylim(0.0, 1.05)
    style_axes(ax_b, grid=False)
    paper_legend(ax_b, loc="best", fontsize=10)
    fig_b.tight_layout()
    save_figure(fig_b, args.output_dir / "fig4b_isf_collapse")
    plt.close(fig_b)

    baseline_ts = ts_cache.get(0.0, (np.array([]), np.array([])))
    t_base, _T_base = baseline_ts
    h_tn_0 = h_tn_cache.get(0.0, np.array([]))
    tw_values = sorted(
        {
            float(tw)
            for lam in plot_lams
            if lam != 0.0
            for tw in summary.get("lambdas", {})
            .get(str(lam), {})
            .get("waiting_times_ps", [])
        }
    )
    if not tw_values and t_base.size:
        tw_values = sorted(
            {
                waiting_time_ps(None, idx)
                for idx in range(13)
            }
        )
    baseline_tn_tw, baseline_tn_tau = _tn_baseline_arrays(
        tn, t_base, h_tn_0, np.asarray(tw_values, dtype=float)
    )
    if baseline_tn_tw.size == 0:
        baseline_tn_tw, baseline_tn_tau = baseline_tw, baseline_tau

    fig_c, ax_c = plt.subplots(figsize=(8, 5))
    tw_colors = plt.cm.plasma(np.linspace(0.1, 0.9, max(len(tw_values), 1)))
    for color, t_w in zip(tw_colors, tw_values):
        base = _baseline_tau_at(float(t_w), baseline_tn_tw, baseline_tn_tau)
        if base is None:
            continue
        xs, ys = [0.0], [1.0]
        for lam in plot_lams:
            if lam == 0.0:
                continue
            t_lab, _T_s = ts_cache.get(lam, (np.array([]), np.array([])))
            h_lam = h_tn_cache.get(lam)
            if t_lab.size == 0 or h_lam is None:
                continue
            tau_tn = tn.tau_s_tn_from_h(t_lab, h_lam, t_w, SWITCH_TIME_PS)
            if tau_tn is None:
                continue
            xs.append(lam)
            ys.append(tau_tn / base)
        if len(xs) > 1:
            ax_c.plot(xs, ys, "o-", color=color, label=f"$t_w$={t_w:.0f} ps")
    ax_c.axhline(1.0, color="gray", ls=":", lw=1.0)
    ax_c.set_xlabel("$\\lambda$ (a.u.)")
    ax_c.set_ylabel("$\\tilde{\\tau}_{s,\\mathrm{TN}}$")
    ax_c.set_title("TN-predicted slowdown vs coupling")
    style_axes(ax_c, grid=False)
    paper_legend(ax_c, loc="best", fontsize=8, ncol=2)
    fig_c.tight_layout()
    save_figure(fig_c, args.output_dir / "fig4c_tn_tau_tilde_vs_lambda")
    plt.close(fig_c)

    fig_d, ax_d = plt.subplots(figsize=(8, 5))
    lam_colors = plt.cm.viridis(np.linspace(0.15, 0.9, max(len(plot_lams), 1)))
    if baseline_tn_tw.size:
        ax_d.plot(
            baseline_tn_tw,
            [1.0] * baseline_tn_tw.size,
            "k-o",
            lw=1.5,
            label=r"$\lambda=0$ (TN)",
        )
    for (lam, color) in zip(sorted(l for l in plot_lams if l != 0.0), lam_colors):
        t_lab, _T_s = ts_cache.get(lam, (np.array([]), np.array([])))
        h_lam = h_tn_cache.get(lam)
        if t_lab.size == 0 or h_lam is None:
            continue
        xs, ys = _tn_tau_tilde_table(
            tn, t_lab, h_lam, np.asarray(tw_values), baseline_tn_tw, baseline_tn_tau
        )
        if xs:
            ax_d.plot(xs, ys, "o-", color=color, label=f"$\\lambda$={lam:g}")
    ax_d.axhline(1.0, color="gray", ls=":", lw=1.0)
    ax_d.set_xlabel("$t_w$ (ps)")
    ax_d.set_ylabel("$\\tilde{\\tau}_{s,\\mathrm{TN}}$")
    ax_d.set_title("TN-predicted memory vs waiting time")
    style_axes(ax_d, grid=False)
    paper_legend(ax_d, loc="best", fontsize=9)
    fig_d.tight_layout()
    save_figure(fig_d, args.output_dir / "fig4d_tn_tau_tilde_vs_tw")
    plt.close(fig_d)

    args.results_dir.mkdir(parents=True, exist_ok=True)
    with open(args.results_dir / "material_time_summary.json", "w", encoding="utf-8") as fh:
        json.dump(summary, fh, indent=2)
    print(f"Wrote Fig 4 panels to {args.output_dir}")


if __name__ == "__main__":
    main()
