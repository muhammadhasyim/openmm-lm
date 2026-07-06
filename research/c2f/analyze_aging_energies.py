#!/usr/bin/env python
"""Ensemble-average cavity coupling and DSE energies from step turn-on aging."""

from __future__ import annotations

import argparse
import json
import re
import sys
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

try:
    from scipy.interpolate import interp1d
except ImportError:
    sys.exit("scipy required (pixi run -e test ...)")

_SCRIPT_DIR = Path(__file__).resolve().parent
DEFAULT_INPUT_DIR = _SCRIPT_DIR / "turnon_aging"
DEFAULT_OUT_DIR = _SCRIPT_DIR / "reviewer_response"
TURNON_SUMMARY = DEFAULT_OUT_DIR / "turnon_energy_vs_lambda.txt"
LAMBDAS = [0.01, 0.03, 0.042, 0.07, 0.09, 0.141]

COLUMNS = [
    "T_bath_K",
    "T_kinetic_K",
    "T_v_fictive_K",
    "T_s_fictive_K",
    "E_bond_kjmol",
    "E_nonbonded_kjmol",
    "E_cav_harmonic_kjmol",
    "E_cav_coupling_kjmol",
    "E_cav_dse_kjmol",
]


def _lam_tag(lam: float) -> str:
    return f"{lam:g}".replace(".", "p")


def _read_run_meta(input_dir: Path) -> dict[str, str]:
    meta_path = input_dir / "run_meta.txt"
    if not meta_path.exists():
        return {}
    meta: dict[str, str] = {}
    for line in meta_path.read_text().splitlines():
        if "=" in line:
            key, value = line.split("=", 1)
            meta[key.strip()] = value.strip()
    return meta


def _ensemble_average_lambda(
    input_dir: Path,
    lam: float,
    dt_ps: float,
    runtime_ps: float,
) -> tuple[np.ndarray, dict[str, np.ndarray], dict[str, np.ndarray], int]:
    """Interpolate replica CSVs onto a common grid and return mean/std arrays."""
    lam_tag = _lam_tag(lam)
    pattern = f"lam{lam_tag}_seed*_energies.csv"
    csv_files = sorted(input_dir.glob(pattern))
    if not csv_files:
        raise FileNotFoundError(
            f"No trajectories matching {pattern} in {input_dir}"
        )

    time_grid = np.arange(0.0, runtime_ps + 0.5 * dt_ps, dt_ps)
    stacked_mean: dict[str, list[np.ndarray]] = {col: [] for col in COLUMNS}
    stacked_std: dict[str, list[np.ndarray]] = {col: [] for col in COLUMNS}

    for csv_path in csv_files:
        data = np.genfromtxt(
            csv_path, delimiter=",", names=True, missing_values="", usemask=False
        )
        t = np.asarray(data["time_ps"], dtype=float)
        if t.size == 0:
            continue

        for col in COLUMNS:
            y = np.asarray(data[col], dtype=float)
            y = np.where(np.isfinite(y), y, np.nan)
            if np.all(np.isnan(y)):
                interp_y = np.full_like(time_grid, np.nan)
            else:
                valid = np.isfinite(y)
                f = interp1d(
                    t[valid],
                    y[valid],
                    kind="linear",
                    bounds_error=False,
                    fill_value=np.nan,
                )
                interp_y = f(time_grid)
            stacked_mean[col].append(interp_y)

    n_traj = len(stacked_mean[COLUMNS[0]])
    means: dict[str, np.ndarray] = {}
    stds: dict[str, np.ndarray] = {}
    for col in COLUMNS:
        arr = np.array(stacked_mean[col])
        means[col] = np.nanmean(arr, axis=0)
        stds[col] = np.nanstd(arr, axis=0)

    return time_grid, means, stds, n_traj


def _write_averaged_csv(
    out_path: Path,
    time_grid: np.ndarray,
    means: dict[str, np.ndarray],
    stds: dict[str, np.ndarray],
) -> None:
    header = ["time_ps"] + [f"{c},{c}_std" for c in COLUMNS]
    with open(out_path, "w") as out:
        out.write(",".join(header) + "\n")
        for i, t in enumerate(time_grid):
            row = [f"{t:.6f}"]
            for col in COLUMNS:
                row.append(f"{means[col][i]:.4f}")
                row.append(f"{stds[col][i]:.4f}")
            out.write(",".join(row) + "\n")


def _load_turnon_reference(path: Path) -> dict[float, dict[str, float]]:
    """Parse turnon_energy_vs_lambda.txt if present."""
    if not path.exists():
        return {}

    text = path.read_text()
    ref: dict[float, dict[str, float]] = {}
    row_re = re.compile(
        r"^\s*(\d+\.\d+)\s+"
        r"([-+eE0-9.]+)\s+"
        r"([-+eE0-9.]+)\s+"
        r"([-+eE0-9.]+)\s+"
        r"([-+eE0-9.]+)"
    )
    for line in text.splitlines():
        match = row_re.match(line)
        if match:
            lam = float(match.group(1))
            ref[lam] = {
                "mean_turnon": float(match.group(2)),
                "rms_turnon": float(match.group(3)),
                "mean_relaxed": float(match.group(4)),
                "mean_dse": float(match.group(5)),
            }
    return ref


def _summarize_lambda(
    time_grid: np.ndarray,
    means: dict[str, np.ndarray],
    coupling_start_ps: float,
) -> dict[str, float]:
    mask_on = time_grid >= coupling_start_ps
    e_coup = means["E_cav_coupling_kjmol"][mask_on]
    e_dse = means["E_cav_dse_kjmol"][mask_on]
    abs_coup = np.abs(e_coup)
    abs_dse = np.abs(e_dse)

    ratio = np.where(abs_dse > 1e-6, abs_coup / abs_dse, np.nan)
    e_bond = means["E_bond_kjmol"]
    e_harm = means["E_cav_harmonic_kjmol"]
    net_cav = (
        means["E_cav_coupling_kjmol"]
        + means["E_cav_dse_kjmol"]
        + means["E_cav_harmonic_kjmol"]
    )
    pre_mask = time_grid < coupling_start_ps
    tail = max(1, int(0.2 * mask_on.sum()))
    e_bond_pre = float(np.nanmean(e_bond[pre_mask])) if pre_mask.any() else float("nan")
    e_bond_post = float(np.nanmean(e_bond[mask_on]))
    return {
        "mean_E_bond_pre_turnon": e_bond_pre,
        "mean_E_bond_post_turnon": e_bond_post,
        "delta_E_bond_post_minus_pre": e_bond_post - e_bond_pre,
        "mean_E_cav_harmonic_post_turnon": float(np.nanmean(e_harm[mask_on])),
        "mean_net_cavity_post_turnon": float(np.nanmean(net_cav[mask_on])),
        "steady_net_cavity": float(np.nanmean(net_cav[mask_on][-tail:])),
        "mean_E_coup_post_turnon": float(np.nanmean(e_coup)),
        "mean_abs_E_coup_post_turnon": float(np.nanmean(abs_coup)),
        "peak_abs_E_coup": float(np.nanmax(abs_coup)) if abs_coup.size else float("nan"),
        "mean_E_dse_post_turnon": float(np.nanmean(e_dse)),
        "mean_abs_E_dse_post_turnon": float(np.nanmean(abs_dse)),
        "steady_abs_E_coup": float(np.nanmean(abs_coup[-tail:])),
        "steady_abs_E_dse": float(np.nanmean(abs_dse[-tail:])),
        "steady_ratio_abs_coup_over_dse": float(np.nanmean(ratio[-tail:])),
    }


def _plot_energy_vs_time(
    all_data: dict[float, tuple[np.ndarray, dict[str, np.ndarray], dict[str, np.ndarray]]],
    column: str,
    ylabel: str,
    out_path: Path,
    coupling_start_ps: float,
    turnon_ref: dict[float, dict[str, float]] | None = None,
    ref_key: str | None = None,
    xmax_ps: float | None = None,
) -> None:
    fig, ax = plt.subplots(figsize=(9, 5))
    colors = plt.cm.viridis(np.linspace(0.15, 0.9, len(all_data)))

    for (lam, (time_grid, means, stds)), color in zip(
        sorted(all_data.items()), colors
    ):
        mask = time_grid <= xmax_ps if xmax_ps is not None else np.ones_like(time_grid, dtype=bool)
        t_plot = time_grid[mask]
        y = means[column][mask]
        y_std = stds[column][mask]
        ax.plot(t_plot, y, label=f"$\\lambda$={lam:g}", color=color, lw=1.5)
        ax.fill_between(
            t_plot, y - y_std, y + y_std, color=color, alpha=0.15, linewidth=0
        )
        if turnon_ref and ref_key and lam in turnon_ref:
            ax.axhline(
                turnon_ref[lam][ref_key],
                color=color,
                ls=":",
                lw=0.8,
                alpha=0.6,
            )

    if xmax_ps is None or coupling_start_ps <= xmax_ps:
        ax.axvline(
            coupling_start_ps,
            color="k",
            ls="--",
            lw=1.0,
            alpha=0.7,
            label="coupling on",
        )
    if xmax_ps is not None:
        ax.set_xlim(0.0, xmax_ps)
    ax.set_xlabel("time (ps)")
    ax.set_ylabel(ylabel)
    ax.legend(fontsize=8, ncol=2)
    zoom_note = f", first {xmax_ps:g} ps" if xmax_ps is not None else ""
    ax.set_title(
        f"Ensemble mean {ylabel} during step turn-on aging from $\\lambda=0$ IC"
        f"{zoom_note}"
    )
    fig.tight_layout()
    fig.savefig(out_path, dpi=150)
    plt.close(fig)


def _plot_bond_and_dse_all_lambdas(
    all_data: dict[float, tuple[np.ndarray, dict[str, np.ndarray], dict[str, np.ndarray]]],
    out_path: Path,
    coupling_start_ps: float,
    xmax_ps: float,
) -> None:
    """Two-panel plot: bond energy (top) and DSE (bottom) for all lambdas, zoomed."""
    fig, axes = plt.subplots(2, 1, figsize=(9, 8), sharex=True)
    colors = plt.cm.viridis(np.linspace(0.15, 0.9, len(all_data)))

    pre_bond_vals: list[float] = []
    panels = [
        ("E_bond_kjmol", "molecular bond $E_\\mathrm{bond}$ (kJ/mol)"),
        ("E_cav_dse_kjmol", "dipole self-energy $E_\\mathrm{dse}$ (kJ/mol)"),
    ]

    for ax, (column, ylab) in zip(axes, panels):
        for (lam, (time_grid, means, stds)), color in zip(
            sorted(all_data.items()), colors
        ):
            mask = time_grid <= xmax_ps
            t_plot = time_grid[mask]
            y = means[column][mask]
            y_std = stds[column][mask]
            ax.plot(t_plot, y, label=f"$\\lambda$={lam:g}", color=color, lw=1.5)
            ax.fill_between(
                t_plot, y - y_std, y + y_std, color=color, alpha=0.15, linewidth=0
            )
            if column == "E_bond_kjmol":
                pre_mask = t_plot < coupling_start_ps
                if pre_mask.any():
                    pre_bond_vals.append(float(np.nanmean(y[pre_mask])))

        ax.axvline(
            coupling_start_ps,
            color="k",
            ls="--",
            lw=1.0,
            alpha=0.7,
        )
        ax.set_ylabel(ylab)
        ax.set_xlim(0.0, xmax_ps)

    if pre_bond_vals:
        e_bond_pre = float(np.nanmean(pre_bond_vals))
        axes[0].axhline(
            e_bond_pre,
            color="gray",
            ls=":",
            lw=1.0,
            label=f"pre-turn-on $\\langle E_\\mathrm{{bond}}\\rangle$={e_bond_pre:.1f}",
        )
        axes[1].axhline(
            e_bond_pre,
            color="gray",
            ls=":",
            lw=1.0,
            label=f"$\\lambda=0$ baseline $E_\\mathrm{{bond}}$={e_bond_pre:.1f}",
        )

    axes[0].legend(fontsize=7, ncol=3, loc="upper right")
    axes[1].legend(fontsize=7, ncol=2, loc="upper right")
    axes[0].set_title(
        f"Bond energy and dipole self-energy after step turn-on "
        f"(first {xmax_ps:g} ps, ensemble mean)"
    )
    axes[1].set_xlabel("time (ps)")
    fig.tight_layout()
    fig.savefig(out_path, dpi=150)
    plt.close(fig)


def _plot_net_cavity_vs_time(
    all_data: dict[float, tuple[np.ndarray, dict[str, np.ndarray], dict[str, np.ndarray]]],
    out_path: Path,
    coupling_start_ps: float,
) -> None:
    fig, ax = plt.subplots(figsize=(9, 5))
    colors = plt.cm.viridis(np.linspace(0.15, 0.9, len(all_data)))

    for (lam, (time_grid, means, stds)), color in zip(
        sorted(all_data.items()), colors
    ):
        net = (
            means["E_cav_coupling_kjmol"]
            + means["E_cav_dse_kjmol"]
            + means["E_cav_harmonic_kjmol"]
        )
        net_std = np.sqrt(
            stds["E_cav_coupling_kjmol"] ** 2
            + stds["E_cav_dse_kjmol"] ** 2
            + stds["E_cav_harmonic_kjmol"] ** 2
        )
        ax.plot(time_grid, net, label=f"$\\lambda$={lam:g}", color=color, lw=1.5)
        ax.fill_between(
            time_grid, net - net_std, net + net_std, color=color, alpha=0.15, linewidth=0
        )

    ax.axhline(1.3, color="gray", ls=":", lw=1.0, label="equil. net cavity ~1.3 kJ/mol")
    ax.axvline(coupling_start_ps, color="k", ls="--", lw=1.0, alpha=0.7)
    ax.set_xlabel("time (ps)")
    ax.set_ylabel(
        "$E_\\mathrm{coup}+E_\\mathrm{dse}+E_\\mathrm{cav,harm}$ (kJ/mol)"
    )
    ax.legend(fontsize=8, ncol=2)
    ax.set_title("Net cavity energy during step turn-on aging")
    fig.tight_layout()
    fig.savefig(out_path, dpi=150)
    plt.close(fig)


def _plot_four_energies_panel(
    time_grid: np.ndarray,
    means: dict[str, np.ndarray],
    stds: dict[str, np.ndarray],
    lam: float,
    out_path: Path,
    coupling_start_ps: float,
    xmax_ps: float | None = None,
) -> None:
    """Single panel: bilinear coupling, DSE, cavity harmonic, and bond energy vs time."""
    series = [
        ("E_cav_coupling_kjmol", "$E_\\mathrm{coup}$ (bilinear coupling)", "#1f77b4"),
        ("E_cav_dse_kjmol", "$E_\\mathrm{dse}$ (dark-hole cell)", "#ff7f0e"),
        ("E_cav_harmonic_kjmol", "$E_\\mathrm{cav,harm}$ (cavity harmonic)", "#2ca02c"),
        ("E_bond_kjmol", "$E_\\mathrm{bond}$ (molecular bond)", "#d62728"),
    ]

    mask = time_grid <= xmax_ps if xmax_ps is not None else np.ones_like(time_grid, dtype=bool)
    t_plot = time_grid[mask]

    fig, ax = plt.subplots(figsize=(9, 5))
    for col, label, color in series:
        y = means[col][mask]
        y_std = stds[col][mask]
        ax.plot(t_plot, y, label=label, color=color, lw=1.5)
        ax.fill_between(
            t_plot, y - y_std, y + y_std, color=color, alpha=0.12, linewidth=0
        )

    if xmax_ps is None or coupling_start_ps <= xmax_ps:
        ax.axvline(
            coupling_start_ps,
            color="k",
            ls="--",
            lw=1.0,
            alpha=0.7,
            label="coupling on",
        )
    ax.set_xlabel("time (ps)")
    ax.set_ylabel("energy (kJ/mol)")
    if xmax_ps is not None:
        ax.set_xlim(0.0, xmax_ps)
    zoom_note = f", first {xmax_ps:g} ps" if xmax_ps is not None else ""
    ax.set_title(
        f"Coupling, DSE, cavity harmonic, and bond energies "
        f"($\\lambda$={lam:g}, ensemble mean{zoom_note})"
    )
    ax.legend(fontsize=8, loc="best")
    fig.tight_layout()
    fig.savefig(out_path, dpi=150)
    plt.close(fig)


def _plot_ratio_vs_time(
    all_data: dict[float, tuple[np.ndarray, dict[str, np.ndarray], dict[str, np.ndarray]]],
    out_path: Path,
    coupling_start_ps: float,
) -> None:
    fig, ax = plt.subplots(figsize=(9, 5))
    colors = plt.cm.viridis(np.linspace(0.15, 0.9, len(all_data)))

    for (lam, (time_grid, means, _stds)), color in zip(
        sorted(all_data.items()), colors
    ):
        abs_coup = np.abs(means["E_cav_coupling_kjmol"])
        abs_dse = np.abs(means["E_cav_dse_kjmol"])
        ratio = np.where(abs_dse > 1e-6, abs_coup / abs_dse, np.nan)
        ax.plot(time_grid, ratio, label=f"$\\lambda$={lam:g}", color=color, lw=1.5)

    ax.axhline(2.0, color="gray", ls=":", lw=1.0, label="$|$E$_\\mathrm{coup}$$|/|$E$_\\mathrm{dse}$$|=2")
    ax.axvline(coupling_start_ps, color="k", ls="--", lw=1.0, alpha=0.7)
    ax.set_xlabel("time (ps)")
    ax.set_ylabel("$|$E$_\\mathrm{coup}$$| / $|E_\\mathrm{dse}|$")
    ax.legend(fontsize=8, ncol=2)
    ax.set_title(
        "Dynamic $|E_\\mathrm{coup}|/|E_\\mathrm{dse}|$ during step turn-on aging"
    )
    fig.tight_layout()
    fig.savefig(out_path, dpi=150)
    plt.close(fig)


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Ensemble-average aging cavity energies vs time"
    )
    parser.add_argument("--input-dir", type=Path, default=DEFAULT_INPUT_DIR)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUT_DIR)
    parser.add_argument("--lambdas", type=float, nargs="+", default=LAMBDAS)
    parser.add_argument("--csv-interval-ps", type=float, default=0.1)
    parser.add_argument("--runtime-ps", type=float, default=None)
    parser.add_argument("--coupling-start-ps", type=float, default=None)
    args = parser.parse_args()

    input_dir = args.input_dir.resolve()
    output_dir = args.output_dir.resolve()
    output_dir.mkdir(parents=True, exist_ok=True)

    meta = _read_run_meta(input_dir)
    runtime_ps = args.runtime_ps or float(meta.get("runtime_ps", 160.0))
    coupling_start_ps = args.coupling_start_ps or float(
        meta.get("coupling_start_ps", 10.0)
    )

    turnon_ref = _load_turnon_reference(TURNON_SUMMARY)
    all_data: dict[
        float, tuple[np.ndarray, dict[str, np.ndarray], dict[str, np.ndarray]]
    ] = {}
    summary: dict[str, object] = {
        "input_dir": str(input_dir),
        "runtime_ps": runtime_ps,
        "coupling_start_ps": coupling_start_ps,
        "lambdas": {},
    }

    for lam in args.lambdas:
        lam_tag = _lam_tag(lam)
        try:
            time_grid, means, stds, n_traj = _ensemble_average_lambda(
                input_dir, lam, args.csv_interval_ps, runtime_ps
            )
        except FileNotFoundError as exc:
            print(f"Skipping lambda={lam}: {exc}")
            continue

        all_data[lam] = (time_grid, means, stds)
        avg_path = output_dir / f"turnon_aging_lam{lam_tag}_averaged.csv"
        _write_averaged_csv(avg_path, time_grid, means, stds)
        print(f"lambda={lam}: averaged {n_traj} trajectories -> {avg_path.name}")

        lam_summary = _summarize_lambda(time_grid, means, coupling_start_ps)
        lam_summary["n_traj"] = n_traj
        if lam in turnon_ref:
            lam_summary["static_rms_turnon"] = turnon_ref[lam]["rms_turnon"]
            lam_summary["static_mean_relaxed"] = turnon_ref[lam]["mean_relaxed"]
            lam_summary["static_mean_dse"] = turnon_ref[lam]["mean_dse"]
        summary["lambdas"][str(lam)] = lam_summary

    if not all_data:
        sys.exit(f"No aging data found in {input_dir}")

    _plot_energy_vs_time(
        all_data,
        "E_cav_coupling_kjmol",
        "$E_\\mathrm{coup}$ (kJ/mol)",
        output_dir / "turnon_aging_E_coup_vs_time.png",
        coupling_start_ps,
        turnon_ref if turnon_ref else None,
        "rms_turnon",
    )
    _plot_energy_vs_time(
        all_data,
        "E_cav_dse_kjmol",
        "$E_\\mathrm{dse}$ (kJ/mol)",
        output_dir / "turnon_aging_E_dse_vs_time.png",
        coupling_start_ps,
        turnon_ref if turnon_ref else None,
        "mean_dse",
    )
    _plot_energy_vs_time(
        all_data,
        "E_bond_kjmol",
        "molecular bond $E_\\mathrm{bond}$ (kJ/mol)",
        output_dir / "turnon_aging_E_bond_vs_time.png",
        coupling_start_ps,
    )
    _plot_energy_vs_time(
        all_data,
        "E_bond_kjmol",
        "molecular bond $E_\\mathrm{bond}$ (kJ/mol)",
        output_dir / "turnon_aging_E_bond_all_lambdas_first20ps.png",
        coupling_start_ps,
        xmax_ps=20.0,
    )
    _plot_bond_and_dse_all_lambdas(
        all_data,
        output_dir / "turnon_aging_E_bond_E_dse_all_lambdas_first20ps.png",
        coupling_start_ps,
        xmax_ps=20.0,
    )
    _plot_energy_vs_time(
        all_data,
        "E_cav_harmonic_kjmol",
        "cavity harmonic $E_\\mathrm{cav,harm}$ (kJ/mol)",
        output_dir / "turnon_aging_E_cav_harmonic_vs_time.png",
        coupling_start_ps,
    )
    _plot_energy_vs_time(
        all_data,
        "T_v_fictive_K",
        "vibrational fictive $T_v$ (K)",
        output_dir / "turnon_aging_Tv_vs_time.png",
        coupling_start_ps,
    )
    _plot_net_cavity_vs_time(
        all_data,
        output_dir / "turnon_aging_net_cavity_vs_time.png",
        coupling_start_ps,
    )
    _plot_ratio_vs_time(
        all_data,
        output_dir / "turnon_aging_E_coup_ratio_vs_time.png",
        coupling_start_ps,
    )
    for lam, (time_grid, means, stds) in sorted(all_data.items()):
        lam_tag = _lam_tag(lam)
        _plot_four_energies_panel(
            time_grid,
            means,
            stds,
            lam,
            output_dir / f"turnon_aging_four_energies_lam{lam_tag}.png",
            coupling_start_ps,
        )
        _plot_four_energies_panel(
            time_grid,
            means,
            stds,
            lam,
            output_dir / f"turnon_aging_four_energies_lam{lam_tag}_first20ps.png",
            coupling_start_ps,
            xmax_ps=20.0,
        )

    summary_path = output_dir / "turnon_aging_energy_summary.json"
    with open(summary_path, "w") as out:
        json.dump(summary, out, indent=2)
    print(f"Summary -> {summary_path}")
    print(
        "Plots -> turnon_aging_E_coup_vs_time.png, turnon_aging_E_dse_vs_time.png, "
        "turnon_aging_E_bond_vs_time.png, turnon_aging_E_bond_all_lambdas_first20ps.png, "
        "turnon_aging_E_bond_E_dse_all_lambdas_first20ps.png, "
        "turnon_aging_E_cav_harmonic_vs_time.png, "
        "turnon_aging_net_cavity_vs_time.png, turnon_aging_Tv_vs_time.png, "
        "turnon_aging_E_coup_ratio_vs_time.png, "
        "turnon_aging_four_energies_lam*.png"
    )


if __name__ == "__main__":
    main()
