#!/usr/bin/env python3
"""Extract tau_s and tau_tilde vs lambda and t_w (Fig 2b/c).

tau_s uses the archive F/F_ref0 = 0.1 crossing on replica-averaged F(k,t).
tau_tilde divides by a monotone plateau fit to lambda=0 (median per-replica tau),
which stabilizes the baseline against pointwise noise and non-monotonic mean ISFs
at larger lambda.
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import sys
from collections.abc import Callable
from dataclasses import dataclass
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.colors import Normalize
from scipy.interpolate import PchipInterpolator
from scipy.optimize import curve_fit

from config import ANALYSIS_LAMBDAS, FIGURES_DIR, N_REPLICAS, RESULTS_DIR, job_dir_path
from fkt_utils import (
    collect_replica_fkt_files,
    extract_tau_s,
    fit_kww_tau,
    list_analysis_replicas,
    load_lambda_fkt_data,
    parse_fkt_file,
    waiting_time_ps,
)
from replica_qc import load_exclusion_report

# Archive LaTeX helpers (latex_config_adobe.py)
_ARCHIVE_SCRIPTS = (
    Path(__file__).resolve().parents[3]
    / "third_party/cavity_supercooled_archive/final_production_run/scripts/2026-01-29"
)
if str(_ARCHIVE_SCRIPTS) not in sys.path:
    sys.path.insert(0, str(_ARCHIVE_SCRIPTS))
from latex_config_adobe import latex_safe, setup_latex_fonts  # noqa: E402

COUPLING_ON_PS = 200.0


@dataclass(frozen=True)
class BaselineFit:
    """Monotone lambda=0 baseline tau_s(t_w) used for tilde normalization."""

    tw_lab_ps: np.ndarray
    tau_ps: np.ndarray
    model: str
    params: dict[str, float]

    def __call__(self, tw_lab: float) -> float:
        tw_shift = float(tw_lab) - COUPLING_ON_PS
        if self.model == "pchip":
            return float(self._pchip(tw_shift))
        return float(_exp_plateau(tw_shift, **self.params))

    @property
    def _pchip(self) -> PchipInterpolator:
        tw_shift = self.tw_lab_ps - COUPLING_ON_PS
        return PchipInterpolator(tw_shift, self.tau_ps)


def _exp_plateau(tw_shift: np.ndarray | float, tau_inf: float, delta: float, tau_w: float) -> np.ndarray | float:
    t = np.asarray(tw_shift, dtype=float)
    return tau_inf + delta * np.exp(-np.maximum(t, 0.0) / max(tau_w, 1.0))


def _baseline_points_from_results(baseline: dict) -> tuple[np.ndarray, np.ndarray]:
    tw_vals: list[float] = []
    tau_vals: list[float] = []
    for tw_str, entry in sorted(baseline.get("by_tw", {}).items(), key=lambda x: float(x[0])):
        tw = float(tw_str)
        if tw <= 0.0:
            continue
        tau = entry.get("tau_s_ps")
        if tau is None or not np.isfinite(tau) or tau <= 0.0:
            continue
        tw_vals.append(tw)
        tau_vals.append(float(tau))
    if not tw_vals:
        raise ValueError("No lambda=0 baseline points with tw > 0")
    return np.asarray(tw_vals, dtype=float), np.asarray(tau_vals, dtype=float)


def _median_tau_table(lam: float, replicas: list[int]) -> dict[float, float]:
    """Per-replica tau_s (0.1 crossing) aggregated by median at each t_w."""
    job_dir = job_dir_path(lam)
    qc_replicas = list_analysis_replicas(job_dir, lam, replicas)
    data, norm_value, _ = load_lambda_fkt_data(lam, qc_replicas, job_dir=job_dir)
    if norm_value is None or norm_value <= 0.0:
        return {}
    table: dict[float, float] = {}
    for ref_idx, (ref_time, _lags, _mean_fkt) in data.items():
        tw = waiting_time_ps(ref_time, ref_idx)
        taus: list[float] = []
        for replica in qc_replicas:
            files = collect_replica_fkt_files(job_dir, lam, replica)
            if ref_idx not in files:
                continue
            _rt, lags_r, vals_r = parse_fkt_file(files[ref_idx])
            tau = extract_tau_s(
                lags_r,
                vals_r,
                threshold=0.1,
                normalization_value=norm_value,
            )
            if tau is not None and np.isfinite(tau) and tau > 0.0:
                taus.append(float(tau))
        if taus:
            table[tw] = float(np.median(taus))
    return table


def apply_median_tau_tables(
    all_results: dict[float, dict],
    replicas: list[int],
    *,
    results_dir: Path,
    exclusion_path: Path | None = None,
) -> dict[float, dict]:
    """Return a copy of all_results with tau_s_ps replaced by median-per-replica values."""
    cache_path = results_dir / "median_tau_tables.json"
    cache_key = _median_cache_key(replicas, exclusion_path)
    if cache_path.is_file():
        cached = json.loads(cache_path.read_text(encoding="utf-8"))
        if cached.get("cache_key") == cache_key:
            tables = {float(k): v for k, v in cached["tables"].items()}
            return _merge_median_tables(all_results, tables)

    tables: dict[float, dict[str, float]] = {}
    for lam in all_results:
        median_table = _median_tau_table(lam, replicas)
        tables[lam] = {str(tw): tau for tw, tau in median_table.items()}

    results_dir.mkdir(parents=True, exist_ok=True)
    cache_path.write_text(
        json.dumps({"cache_key": cache_key, "tables": tables}, indent=2),
        encoding="utf-8",
    )
    float_tables = {
        lam: {float(tw): tau for tw, tau in table.items()} for lam, table in tables.items()
    }
    return _merge_median_tables(all_results, float_tables)


def _median_cache_key(replicas: list[int], exclusion_path: Path | None) -> str:
    payload = f"replicas={replicas}"
    if exclusion_path and exclusion_path.is_file():
        payload += exclusion_path.read_text(encoding="utf-8")
    return hashlib.sha256(payload.encode()).hexdigest()[:16]


def _merge_median_tables(
    all_results: dict[float, dict],
    tables: dict[float, dict[float, float]],
) -> dict[float, dict]:
    updated: dict[float, dict] = {}
    for lam, res in all_results.items():
        median_table = tables.get(lam, {})
        by_tw = {}
        for tw_str, entry in res.get("by_tw", {}).items():
            tw = float(tw_str)
            new_entry = dict(entry)
            if tw in median_table:
                new_entry["tau_s_ps"] = median_table[tw]
                new_entry["tau_statistic"] = "median_replica"
            by_tw[tw_str] = new_entry
        updated[lam] = {**res, "by_tw": by_tw}
    return updated


def fit_baseline_tau(baseline: dict) -> BaselineFit:
    """
    Fit a smooth monotone plateau to lambda=0 tau_s(t_w).

    Prefer a two-parameter exponential plateau; fall back to PCHIP if the fit fails.
    """
    tw_lab, tau = _baseline_points_from_results(baseline)
    tw_shift = tw_lab - COUPLING_ON_PS
    tau_inf0 = float(np.min(tau[-max(3, len(tau) // 3) :]))
    delta0 = max(float(tau[0] - tau_inf0), 1.0)
    lower = np.array([80.0, 0.0, 100.0])
    upper = np.array([220.0, 120.0, 12000.0])
    p0 = np.clip(np.array([tau_inf0, delta0, 2000.0]), lower, upper)
    try:
        popt, _ = curve_fit(
            _exp_plateau,
            tw_shift,
            tau,
            p0=p0,
            bounds=(lower, upper),
            maxfev=20000,
        )
        tau_inf, delta, tau_w = (float(popt[0]), float(popt[1]), float(popt[2]))
        fitted = _exp_plateau(tw_shift, tau_inf, delta, tau_w)
        print(
            "  Baseline fit: exp plateau "
            f"tau_inf={tau_inf:.1f} ps, delta={delta:.1f} ps, tau_w={tau_w:.0f} ps"
        )
        return BaselineFit(
            tw_lab_ps=tw_lab,
            tau_ps=tau,
            model="exp_plateau",
            params={"tau_inf": tau_inf, "delta": delta, "tau_w": tau_w},
        )
    except (RuntimeError, ValueError) as exc:
        print(f"  Warning: exponential baseline fit failed ({exc}); using PCHIP")
        return BaselineFit(
            tw_lab_ps=tw_lab,
            tau_ps=tau,
            model="pchip",
            params={},
        )


def analyze_lambda(lam: float, replicas: list[int]) -> dict:
    job_dir = job_dir_path(lam)
    results: dict[str, object] = {"lambda": lam, "job_dir": str(job_dir), "by_tw": {}}
    if not job_dir.exists():
        results["error"] = "missing job_dir"
        return results

    qc_replicas = list_analysis_replicas(job_dir, lam, replicas)
    data, norm_value, n_used = load_lambda_fkt_data(lam, qc_replicas, job_dir=job_dir)
    results["n_replicas"] = n_used
    results["n_replicas_qc_passed"] = len(qc_replicas)
    results["ref0_normalization"] = norm_value
    for ref_idx, (ref_time, lags, mean_fkt) in sorted(data.items()):
        tau_s = extract_tau_s(
            lags,
            mean_fkt,
            threshold=0.1,
            normalization_value=norm_value,
        )
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


def _entry_at_tw(by_tw: dict, t_w: float, atol_ps: float = 5.0) -> dict | None:
    """Match relaxation_summary row to nominal waiting time (handles 200.001 ps keys)."""
    best_key: str | None = None
    best_dist = float("inf")
    for key in by_tw:
        dist = abs(float(key) - t_w)
        if dist < best_dist:
            best_dist = dist
            best_key = key
    if best_key is None or best_dist > atol_ps:
        return None
    return by_tw[best_key]


def build_normalization_data(
    all_results: dict[float, dict],
    baseline_fit: Callable[[float], float],
) -> dict[float, dict[str, np.ndarray]]:
    """Per t_w: λ array and tau_tilde = tau / fitted lambda=0 baseline."""
    tw_values = sorted(
        {
            float(k)
            for res in all_results.values()
            for k in res.get("by_tw", {})
            if float(k) > 0.0
        }
    )
    normalization_data: dict[float, dict[str, np.ndarray]] = {}
    for tw in tw_values:
        base_tau = baseline_fit(tw)
        if base_tau <= 0.0 or not np.isfinite(base_tau):
            continue
        lambdas: list[float] = []
        taus: list[float] = []
        for lam in sorted(all_results):
            if lam == 0.0:
                entry = _entry_at_tw(all_results[0.0].get("by_tw", {}), tw)
            else:
                entry = _entry_at_tw(all_results[lam].get("by_tw", {}), tw)
            if not entry:
                continue
            tau = entry.get("tau_s_ps")
            if tau is None or not np.isfinite(tau):
                continue
            lambdas.append(lam)
            taus.append(float(tau) / base_tau)
        if lambdas:
            normalization_data[tw] = {
                "coupling_strengths": np.asarray(lambdas, dtype=float),
                "normalization_times": np.asarray(taus, dtype=float),
            }
    return normalization_data


def build_relaxation_data(all_results: dict[float, dict]) -> dict[float, dict]:
    """Archive relaxation_data keyed by λ."""
    relaxation_data: dict[float, dict] = {}
    for lam, res in sorted(all_results.items()):
        waiting_times: list[float] = []
        relaxation_times: list[float] = []
        for tw_str, entry in sorted(res.get("by_tw", {}).items(), key=lambda x: float(x[0])):
            tau = entry.get("tau_s_ps")
            if tau is None or not np.isfinite(tau):
                continue
            waiting_times.append(float(tw_str))
            relaxation_times.append(float(tau))
        if waiting_times:
            relaxation_data[lam] = {
                "waiting_times": waiting_times,
                "relaxation_times": relaxation_times,
            }
    return relaxation_data


def _format_lambda_axis_label(use_latex: bool) -> str:
    return latex_safe(r"$\lambda$ (a.u.)", "λ (a.u.)", use_latex)


def _format_lambda_label(lam: float, use_latex: bool) -> str:
    if use_latex:
        return f"$\\lambda = {lam:.3f}$"
    return f"λ = {lam:.3f}"


def _panel_tag(ax: plt.Axes, label: str) -> None:
    ax.text(
        -0.12,
        1.03,
        label,
        transform=ax.transAxes,
        fontsize=14,
        fontweight="bold",
        va="bottom",
        ha="left",
    )


def plot_normalization_panel_styled(
    ax: plt.Axes,
    normalization_data: dict[float, dict[str, np.ndarray]],
    *,
    use_latex: bool,
) -> None:
    """Literal port of FictiveThreePanelAnalyzer._plot_normalization_panel_styled."""
    all_waiting_times = sorted([tw for tw in normalization_data if tw > 0])
    if not all_waiting_times:
        print("    Warning: No valid waiting times found!")
        return

    waiting_times_shifted = [tw - COUPLING_ON_PS for tw in all_waiting_times]
    waiting_time_norm = Normalize(vmin=0, vmax=max(waiting_times_shifted))
    waiting_time_cmap = plt.colormaps.get_cmap("viridis")

    panel1_lines: list = []

    for tw in all_waiting_times:
        data = normalization_data[tw]
        coupling_values_lambda = data["coupling_strengths"]
        relaxation_times = data["normalization_times"]

        if relaxation_times.size == 0:
            continue

        tw_shifted = tw - COUPLING_ON_PS
        color = waiting_time_cmap(waiting_time_norm(tw_shifted))
        line = ax.plot(
            coupling_values_lambda,
            relaxation_times,
            "-o",
            color=color,
            linewidth=3,
            markersize=8,
            markerfacecolor="white",
            markeredgecolor=color,
            markeredgewidth=2.0,
        )
        panel1_lines.append(line[0])
        print(
            f"    tw={tw_shifted:.0f} ps: {len(relaxation_times)} valid coupling points"
        )

    ax.set_xlabel(_format_lambda_axis_label(use_latex), fontsize=16)
    ax.set_ylabel(
        latex_safe(r"$\tilde{\tau}_{\mathrm{s}}$", "τ̃_s", use_latex),
        fontsize=16,
    )
    ax.grid(True, alpha=0.3, linestyle="--")
    ax.tick_params(axis="both", which="major", labelsize=12)
    ax.axhline(y=1, color="gray", linestyle="--", alpha=0.7, linewidth=1.5)

    lam_vals = sorted(
        {
            float(x)
            for d in normalization_data.values()
            for x in d["coupling_strengths"]
        }
    )
    if lam_vals:
        ax.set_xlim(left=0, right=max(lam_vals) * 1.15)
        ax.set_xticks(lam_vals)
    ax.set_ylim(bottom=0.7)
    _panel_tag(ax, "(b)")

    if panel1_lines:
        import matplotlib.cm as cm

        sm1 = cm.ScalarMappable(norm=waiting_time_norm, cmap=waiting_time_cmap)
        sm1.set_array([])
        cbar1 = plt.colorbar(sm1, ax=ax, location="top", shrink=0.8, pad=0.1)
        cbar1.set_label(
            latex_safe(r"$t_{\mathrm{w}}$ (ps)", "t_w (ps)", use_latex),
            fontsize=12,
            labelpad=10,
        )
        cbar1.ax.tick_params(labelsize=10)


def plot_relaxation_panel_styled(
    ax: plt.Axes,
    relaxation_data: dict[float, dict],
    coupling_values: list[float],
    baseline_fit: Callable[[float], float],
    *,
    use_latex: bool,
) -> None:
    """Literal port of FictiveThreePanelAnalyzer._plot_relaxation_panel_styled."""
    coupling_values_sorted = sorted(coupling_values)
    coupling_values_sorted_lambda = coupling_values_sorted
    coupling_norm = Normalize(vmin=0, vmax=max(coupling_values_sorted_lambda))
    coupling_cmap = plt.colormaps.get_cmap("coolwarm")

    for coupling_val in coupling_values_sorted:
        if coupling_val not in relaxation_data:
            continue
        data = relaxation_data[coupling_val]
        waiting_times = data["waiting_times"]
        relaxation_times = data["relaxation_times"]

        valid_indices = [i for i, tw in enumerate(waiting_times) if tw > 0]
        if not valid_indices:
            continue

        filtered_waiting_times = [waiting_times[i] - COUPLING_ON_PS for i in valid_indices]
        filtered_relaxation_times = [relaxation_times[i] for i in valid_indices]

        normalized_relaxation_times: list[float] = []
        final_waiting_times: list[float] = []

        for i, tw_shift in enumerate(filtered_waiting_times):
            original_tw = tw_shift + COUPLING_ON_PS
            ref_value = baseline_fit(original_tw)
            if ref_value > 0 and not np.isnan(filtered_relaxation_times[i]):
                normalized_relaxation_times.append(
                    filtered_relaxation_times[i] / ref_value
                )
                final_waiting_times.append(tw_shift)

        if not normalized_relaxation_times:
            continue

        color = coupling_cmap(coupling_norm(coupling_val))
        ax.plot(
            final_waiting_times,
            normalized_relaxation_times,
            "-o",
            color=color,
            linewidth=3,
            markersize=8,
            markerfacecolor="white",
            markeredgecolor=color,
            markeredgewidth=2.0,
            label=_format_lambda_label(coupling_val, use_latex),
        )
        print(
            f"    λ={coupling_val:.3f}: {len(normalized_relaxation_times)} valid waiting time points"
        )

    ax.set_xlabel(
        latex_safe(r"$t_{\mathrm{w}}$ (ps)", "t_w (ps)", use_latex),
        fontsize=16,
    )
    ax.set_ylabel(
        latex_safe(r"$\tilde{\tau}_{\mathrm{s}}$", "τ̃_s", use_latex),
        fontsize=16,
    )
    ax.grid(True, alpha=0.3, linestyle="--")
    ax.tick_params(axis="both", which="major", labelsize=12)
    ax.legend(fontsize=10, loc="best")
    ax.axhline(y=1, color="gray", linestyle="--", alpha=0.7, linewidth=1.5)
    ax.set_xlim(left=0, right=1800)
    ax.set_ylim(bottom=0.7)
    _panel_tag(ax, "(c)")


def _save_panel(fig: plt.Figure, stem: Path) -> None:
    stem.parent.mkdir(parents=True, exist_ok=True)
    pdf_path = stem.with_suffix(".pdf")
    png_path = stem.with_suffix(".png")
    fig.savefig(pdf_path, dpi=300, bbox_inches="tight")
    fig.savefig(png_path, dpi=300, bbox_inches="tight")
    print(f"Wrote {pdf_path}")
    print(f"Wrote {png_path}")


def plot_tau_tilde(
    all_results: dict[float, dict],
    baseline_fit: BaselineFit,
    out_dir: Path,
) -> None:
    use_latex = setup_latex_fonts()
    plt.style.use("classic")
    if use_latex:
        print("  Using LaTeX Computer Modern fonts for Fig 2b/c")
    else:
        print("  Using fallback fonts for Fig 2b/c")

    normalization_data = build_normalization_data(all_results, baseline_fit)
    relaxation_data = build_relaxation_data(all_results)
    coupling_values = sorted(all_results.keys())

    print("  Creating Fig 2b: τ̃_s vs λ...")
    fig_b, ax_b = plt.subplots(figsize=(5, 4))
    plot_normalization_panel_styled(ax_b, normalization_data, use_latex=use_latex)
    fig_b.tight_layout(pad=2.0)
    _save_panel(fig_b, out_dir / "fig2b_tau_tilde_vs_lambda")
    plt.close(fig_b)

    print("  Creating Fig 2c: τ̃_s vs t_w...")
    fig_c, ax_c = plt.subplots(figsize=(5, 4))
    plot_relaxation_panel_styled(
        ax_c,
        relaxation_data,
        coupling_values,
        baseline_fit,
        use_latex=use_latex,
    )
    fig_c.tight_layout(pad=2.0)
    _save_panel(fig_c, out_dir / "fig2c_tau_tilde_vs_tw")
    plt.close(fig_c)

    plt.rcParams.update(plt.rcParamsDefault)


def plot_tau_tilde_on_axes(
    ax_b: plt.Axes,
    ax_c: plt.Axes,
    all_results: dict[float, dict],
    baseline_fit: BaselineFit,
    *,
    use_latex: bool,
) -> None:
    normalization_data = build_normalization_data(all_results, baseline_fit)
    relaxation_data = build_relaxation_data(all_results)
    plot_normalization_panel_styled(ax_b, normalization_data, use_latex=use_latex)
    plot_relaxation_panel_styled(
        ax_c,
        relaxation_data,
        sorted(all_results.keys()),
        baseline_fit,
        use_latex=use_latex,
    )


def _export_fig2_csvs(
    all_results: dict[float, dict],
    baseline_fit: BaselineFit,
    results_dir: Path,
) -> None:
    """Write figure2b/c CSVs mirroring export_fig2_minimal_csvs.py columns."""
    rows_b: list[list[float]] = []
    rows_c: list[list[float]] = []
    tw_values = sorted(
        {
            float(k)
            for res in all_results.values()
            for k in res.get("by_tw", {})
            if float(k) > 0.0
        }
    )
    for tw in tw_values:
        base = baseline_fit(tw)
        if base <= 0 or not np.isfinite(base):
            continue
        tw_shift = tw - COUPLING_ON_PS
        for lam in sorted(all_results):
            entry = _entry_at_tw(all_results[lam].get("by_tw", {}), tw)
            if not entry:
                continue
            tau = entry.get("tau_s_ps")
            if tau is None or not np.isfinite(tau):
                continue
            tau_tilde = float(tau) / base
            rows_b.append([lam, tw_shift, tau_tilde])
            rows_c.append([lam, tw_shift, tau_tilde])
    rows_b.sort(key=lambda r: (r[1], r[0]))
    rows_c.sort(key=lambda r: (r[0], r[1]))

    header = [
        "Normalized relaxation time tau_s_tilde = tau_s(lambda, t_w) / baseline_fit(t_w)",
        "QC-filtered median per-replica tau_s; baseline = fitted lambda=0 plateau",
    ]
    results_dir.mkdir(parents=True, exist_ok=True)
    for name, rows in [
        ("figure2b_relaxation_vs_lambda.csv", rows_b),
        ("figure2c_relaxation_vs_tw.csv", rows_c),
    ]:
        path = results_dir / name
        with path.open("w", newline="", encoding="utf-8") as fh:
            for line in header:
                fh.write(f"# {line}\n")
            writer = csv.writer(fh)
            writer.writerow(["lambda_au", "t_w_ps", "tau_s_tilde"])
            writer.writerows(rows)
        print(f"Wrote {path} ({len(rows)} rows)")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-dir", type=Path, default=FIGURES_DIR)
    parser.add_argument("--results-dir", type=Path, default=RESULTS_DIR)
    parser.add_argument("--lambdas", type=float, nargs="+", default=ANALYSIS_LAMBDAS)
    parser.add_argument("--replicas", type=int, nargs="+", default=list(range(N_REPLICAS)))
    parser.add_argument(
        "--exclusion-json",
        type=Path,
        default=RESULTS_DIR / "replica_exclusion.json",
        help="QC exclusion report from replica_qc.py",
    )
    parser.add_argument(
        "--skip-median-tau",
        action="store_true",
        help="Use QC-filtered ensemble tau from master F(k,t) (fast; ~1 min)",
    )
    parser.add_argument(
        "--median-tau",
        action="store_true",
        help="Use median per-replica tau (slow; ~30+ min)",
    )
    args = parser.parse_args()

    exclusion_report = load_exclusion_report(args.exclusion_json)
    if exclusion_report is None:
        print(f"Warning: no QC report at {args.exclusion_json}; running inline QC filter")

    all_results: dict[float, dict] = {}
    qc_stats: dict[str, dict] = {}
    for lam in args.lambdas:
        all_results[lam] = analyze_lambda(lam, args.replicas)
        n_qc = all_results[lam].get("n_replicas_qc_passed", 0)
        print(f"lambda={lam:g}: n_rep_qc={n_qc}")
        if exclusion_report:
            entry = exclusion_report.get("by_lambda", {}).get(str(lam), {})
            qc_stats[str(lam)] = {
                "n_available": entry.get("n_available"),
                "n_qc_passed": entry.get("n_qc_passed"),
                "n_excluded": entry.get("n_excluded"),
            }

    if 0.0 not in all_results:
        raise SystemExit("λ=0 baseline missing from --lambdas; Fig 2b/c normalization requires it")

    if args.median_tau:
        plot_results = apply_median_tau_tables(
            all_results,
            args.replicas,
            results_dir=args.results_dir,
            exclusion_path=args.exclusion_json,
        )
        tau_stat = "median_replica"
    else:
        plot_results = all_results
        tau_stat = "ensemble_master_qc"
        print("  Using QC-filtered ensemble tau from master F(k,t) (--median-tau for per-replica median)")
    baseline_fit = fit_baseline_tau(plot_results[0.0])
    plot_tau_tilde(plot_results, baseline_fit, args.output_dir)
    _export_fig2_csvs(plot_results, baseline_fit, args.results_dir)

    args.results_dir.mkdir(parents=True, exist_ok=True)
    summary_path = args.results_dir / "relaxation_summary.json"
    with open(summary_path, "w", encoding="utf-8") as fh:
        payload = {str(k): v for k, v in all_results.items()}
        payload["_baseline_fit"] = {
            "model": baseline_fit.model,
            "params": baseline_fit.params,
            "tw_lab_ps": baseline_fit.tw_lab_ps.tolist(),
            "tau_ps": baseline_fit.tau_ps.tolist(),
            "tau_statistic_for_plot": tau_stat,
        }
        payload["_qc"] = qc_stats
        json.dump(payload, fh, indent=2)
    print(f"Summary -> {summary_path}")


if __name__ == "__main__":
    main()
