#!/usr/bin/env python
"""Plot energies and fictive temperatures from cavity equilibrium CSV logs."""

from __future__ import annotations

import argparse
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

_SCRIPT_DIR = Path(__file__).resolve().parent
DEFAULT_OUT_DIR = _SCRIPT_DIR / "equilibrium_output"

# mKA system: same as run_c2f.py / TemperatureTracker fallback
NUM_MOL = 250
KJMOL_TO_HARTREE = 4.184 / (6.02214076e23 * 1e3) * 2625.49962  # matches cavitymd Units

TEMP_COLS = {
    "T_kinetic_K": (r"$T_\mathrm{kin}$", "#1f77b4", "-"),
    "T_v_fictive_K": (r"$T_\mathrm{v}$ (empirical)", "#2ca02c", "-"),
    "T_s_fictive_K": (r"$T_\mathrm{s}$", "#d62728", "-"),
}
TV_EQ_LABEL = r"$T_\mathrm{v}^\mathrm{eq}$ ($2E_\mathrm{bond}/N_\mathrm{mol}k_\mathrm{B}$)"
TV_EQ_COLOR = "#17becf"
ENERGY_COLS = {
    "E_bond_kjmol": ("Bond", "#9467bd"),
    "E_nonbonded_kjmol": ("Nonbonded", "#8c564b"),
    "E_cav_harmonic_kjmol": ("Cav. harmonic", "#e377c2"),
    "E_cav_coupling_kjmol": ("Cav. coupling", "#7f7f7f"),
    "E_cav_dse_kjmol": ("Cav. DSE", "#bcbd22"),
}


def load_csv(path: Path) -> dict[str, np.ndarray]:
    data = np.genfromtxt(path, delimiter=",", names=True)
    return {name: np.asarray(data[name], dtype=float) for name in data.dtype.names}


def trim_data(data: dict[str, np.ndarray], t_min: float) -> dict[str, np.ndarray]:
    if t_min <= 0:
        return data
    mask = data["time_ps"] >= t_min
    return {key: values[mask] for key, values in data.items()}


def tv_equipartition_k(e_bond_kjmol: np.ndarray, num_mol: int = NUM_MOL) -> np.ndarray:
    """T_v equipartition: 2 * V_bond / (N_mol * k_B) for one bond DOF per molecule."""
    try:
        from openmm.cavitymd.constants import Units

        e_hartree = np.asarray(e_bond_kjmol, dtype=float) * Units.KJMOL_TO_HARTREE
        kb = Units.KB_HARTREE_PER_K
    except ImportError:
        e_hartree = np.asarray(e_bond_kjmol, dtype=float) * KJMOL_TO_HARTREE
        kb = 3.16681153e-6
    with np.errstate(divide="ignore", invalid="ignore"):
        tv = 2.0 * e_hartree / (num_mol * kb)
    return np.where(e_hartree > 0, tv, 0.0)


def add_tv_equipartition(data: dict[str, np.ndarray], num_mol: int = NUM_MOL) -> dict[str, np.ndarray]:
    out = dict(data)
    out["T_v_equipartition_K"] = tv_equipartition_k(data["E_bond_kjmol"], num_mol)
    return out


def _auto_yscale(ax, arrays: list[np.ndarray], semilog_thresh: float = 500.0) -> None:
    """Use log y for all-positive data, symlog when signed and large."""
    parts = [np.asarray(a)[np.isfinite(a)] for a in arrays]
    flat = np.concatenate([p for p in parts if len(p)])
    if flat.size == 0:
        return

    ymax = float(np.max(np.abs(flat)))
    pos = flat[flat > 0]
    ymin_pos = float(np.min(pos)) if pos.size else ymax
    span_ratio = ymax / max(ymin_pos, 1e-12)

    if ymax <= semilog_thresh and span_ratio <= 50:
        return

    if np.all(flat > 0):
        ax.set_yscale("log")
    else:
        ax.set_yscale("symlog", linthresh=max(1.0, ymax * 1e-3))


def plot_temperatures(
    runs: list[tuple[str, dict[str, np.ndarray]]],
    output_path: Path,
    bath_K: float | None,
    t_min: float,
) -> None:
    n = len(runs)
    fig, axes = plt.subplots(1, n, figsize=(6 * n, 4.5), squeeze=False)
    axes = axes[0]

    for ax, (label, data) in zip(axes, runs):
        t = data["time_ps"]
        series = [data[col] for col in TEMP_COLS if col in data]
        for col, (tex, color, ls) in TEMP_COLS.items():
            ax.plot(t, data[col], color=color, linewidth=1.2, label=tex, ls=ls)

        t_v_eq = data["T_v_equipartition_K"]
        ax.plot(t, t_v_eq, color=TV_EQ_COLOR, linewidth=1.2, ls="--", label=TV_EQ_LABEL)
        series.append(t_v_eq)

        if bath_K is not None and bath_K > 0:
            ax.axhline(bath_K, color="black", ls=":", lw=0.8, alpha=0.5, label="Bath")

        _auto_yscale(ax, series)

        ax.set_title(label)
        ax.set_xlabel("Time (ps)")
        ax.set_ylabel("Temperature (K)")
        ax.legend(loc="best", fontsize=9)
        ax.grid(True, alpha=0.25, which="both")

    title = "Fictive temperatures during NVT equilibrium"
    if t_min > 0:
        title += f" (t ≥ {t_min:g} ps)"
    fig.suptitle(title, y=1.02)
    fig.tight_layout()
    fig.savefig(output_path, dpi=150, bbox_inches="tight")
    plt.close(fig)


def plot_energies(
    runs: list[tuple[str, dict[str, np.ndarray]]],
    output_path: Path,
    t_min: float,
) -> None:
    n = len(runs)
    fig, axes = plt.subplots(len(ENERGY_COLS), n, figsize=(6 * n, 2.2 * len(ENERGY_COLS)), sharex=True)
    if n == 1:
        axes = axes.reshape(-1, 1)

    for col_idx, (col, (name, color)) in enumerate(ENERGY_COLS.items()):
        for run_idx, (label, data) in enumerate(runs):
            ax = axes[col_idx, run_idx]
            y = data[col]
            if col == "E_cav_dse_kjmol" and np.allclose(y, 0.0):
                ax.text(
                    0.5, 0.5, "DSE off", transform=ax.transAxes,
                    ha="center", va="center", fontsize=11, color="0.45",
                )
                ax.set_ylim(-1, 1)
            else:
                ax.plot(data["time_ps"], y, color=color, linewidth=1.0)
                _auto_yscale(ax, [y])
            if run_idx == 0:
                ax.set_ylabel(f"{name}\n(kJ/mol)")
            if col_idx == 0:
                ax.set_title(label)
            if col_idx == len(ENERGY_COLS) - 1:
                ax.set_xlabel("Time (ps)")
            ax.grid(True, alpha=0.25, which="both")

    title = "Energy components during NVT equilibrium"
    if t_min > 0:
        title += f" (t ≥ {t_min:g} ps)"
    fig.suptitle(title, y=1.01)
    fig.tight_layout()
    fig.savefig(output_path, dpi=150, bbox_inches="tight")
    plt.close(fig)


def plot_combined(
    runs: list[tuple[str, dict[str, np.ndarray]]],
    output_path: Path,
    bath_K: float | None,
    t_min: float,
) -> None:
    """Single overview: temperatures (top) + molecular + cavity energy (bottom)."""
    n = len(runs)
    fig, axes = plt.subplots(3, n, figsize=(6 * n, 8), sharex=True)
    if n == 1:
        axes = axes.reshape(-1, 1)

    for idx, (label, data) in enumerate(runs):
        t = data["time_ps"]

        ax_t = axes[0, idx]
        temp_series = [data[col] for col in TEMP_COLS if col in data]
        for col, (tex, color, ls) in TEMP_COLS.items():
            ax_t.plot(t, data[col], color=color, lw=1.2, label=tex, ls=ls)
        t_v_eq = data["T_v_equipartition_K"]
        ax_t.plot(t, t_v_eq, color=TV_EQ_COLOR, lw=1.2, ls="--", label=TV_EQ_LABEL)
        temp_series.append(t_v_eq)
        if bath_K is not None and bath_K > 0:
            ax_t.axhline(bath_K, color="k", ls="--", lw=0.8, alpha=0.4)
        _auto_yscale(ax_t, temp_series)
        ax_t.set_title(label)
        ax_t.set_ylabel("T (K)")
        ax_t.legend(fontsize=8, loc="best")
        ax_t.grid(True, alpha=0.25, which="both")

        ax_mol = axes[1, idx]
        e_bond = data["E_bond_kjmol"]
        e_nb = data["E_nonbonded_kjmol"]
        e_mol = e_bond + e_nb
        ax_mol.plot(t, e_bond, label="Bond", lw=1.0)
        ax_mol.plot(t, e_nb, label="Nonbonded", lw=1.0)
        ax_mol.plot(t, e_mol, color="k", ls=":", lw=1.0, alpha=0.7, label="Sum")
        _auto_yscale(ax_mol, [e_bond, e_nb, e_mol])
        ax_mol.set_ylabel("kJ/mol")
        ax_mol.legend(fontsize=8)
        ax_mol.grid(True, alpha=0.25, which="both")

        ax_cav = axes[2, idx]
        e_h = data["E_cav_harmonic_kjmol"]
        e_c = data["E_cav_coupling_kjmol"]
        e_d = data["E_cav_dse_kjmol"]
        ax_cav.plot(t, e_h, label="Harmonic", lw=1.0)
        ax_cav.plot(t, e_c, label="Coupling", lw=1.0)
        cav_series = [e_h, e_c]
        if not np.allclose(e_d, 0.0):
            ax_cav.plot(t, e_d, label="DSE", lw=1.0)
            cav_series.append(e_d)
        _auto_yscale(ax_cav, cav_series)
        ax_cav.set_xlabel("Time (ps)")
        ax_cav.set_ylabel("kJ/mol")
        ax_cav.legend(fontsize=8)
        ax_cav.grid(True, alpha=0.25, which="both")

    title = "100 K cavity equilibrium overview"
    if t_min > 0:
        title += f" (t ≥ {t_min:g} ps)"
    fig.suptitle(title, y=1.01)
    fig.tight_layout()
    fig.savefig(output_path, dpi=150, bbox_inches="tight")
    plt.close(fig)


def plot_tv_comparison(
    runs: list[tuple[str, dict[str, np.ndarray]]],
    output_path: Path,
    t_min: float,
) -> None:
    """Compare empirical T_v (logged) vs equipartition fallback for both runs."""
    n = len(runs)
    fig, axes = plt.subplots(1, n, figsize=(6 * n, 4.5), squeeze=False)
    axes = axes[0]

    for ax, (label, data) in zip(axes, runs):
        t = data["time_ps"]
        t_v_emp = data["T_v_fictive_K"]
        t_v_eq = data["T_v_equipartition_K"]
        ax.plot(t, t_v_emp, color="#2ca02c", lw=1.3, label=r"$T_\mathrm{v}$ (empirical)")
        ax.plot(t, t_v_eq, color=TV_EQ_COLOR, lw=1.3, ls="--", label=TV_EQ_LABEL)
        _auto_yscale(ax, [t_v_emp, t_v_eq])
        ax.set_title(label)
        ax.set_xlabel("Time (ps)")
        ax.set_ylabel(r"$T_\mathrm{v}$ (K)")
        ax.legend(loc="best", fontsize=9)
        ax.grid(True, alpha=0.25, which="both")

    title = r"$T_\mathrm{v}$: empirical inversion vs equipartition fallback"
    if t_min > 0:
        title += f" (t ≥ {t_min:g} ps)"
    fig.suptitle(title, y=1.02)
    fig.tight_layout()
    fig.savefig(output_path, dpi=150, bbox_inches="tight")
    plt.close(fig)


def _suffix(t_min: float) -> str:
    if t_min <= 0:
        return ""
    return f"_t{t_min:g}ps"


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--dse-on",
        type=Path,
        default=DEFAULT_OUT_DIR / "eq100K_dse_on_energies.csv",
    )
    parser.add_argument(
        "--dse-off",
        type=Path,
        default=DEFAULT_OUT_DIR / "eq100K_dse_off_energies.csv",
        help="Optional; omit panel if file missing",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=DEFAULT_OUT_DIR,
    )
    parser.add_argument(
        "--t-min",
        type=float,
        default=200.0,
        help="Omit data before this time (ps). Default: 200.",
    )
    parser.add_argument(
        "--tag",
        default="",
        help="Output filename tag, e.g. lam0.042 -> eq100K_lam0.042_temperatures_t200ps.png",
    )
    parser.add_argument(
        "--label-prefix",
        default="100 K",
        help="Panel title prefix (e.g. '100 K, λ=0.042')",
    )
    args = parser.parse_args()

    runs: list[tuple[str, dict[str, np.ndarray]]] = []
    for path, stage in (
        (args.dse_on, "DSE on"),
        (args.dse_off, "DSE off"),
    ):
        if path is None or not path.exists():
            if stage == "DSE off":
                print(f"Skipping missing file: {args.dse_off}")
                continue
            raise SystemExit(f"Missing required input: {args.dse_on}")
        label = f"{args.label_prefix}, {stage}"
        runs.append((label, add_tv_equipartition(trim_data(load_csv(path), args.t_min))))

    bath_K = float(runs[0][1]["T_bath_K"][0])
    out = args.output_dir
    out.mkdir(parents=True, exist_ok=True)
    sfx = _suffix(args.t_min)
    tag = f"_{args.tag}" if args.tag else ""

    paths = {
        "temperatures": out / f"eq100K{tag}_temperatures{sfx}.png",
        "tv_comparison": out / f"eq100K{tag}_Tv_empirical_vs_equipartition{sfx}.png",
        "energies": out / f"eq100K{tag}_energies{sfx}.png",
        "overview": out / f"eq100K{tag}_overview{sfx}.png",
    }
    plot_temperatures(runs, paths["temperatures"], bath_K, args.t_min)
    plot_tv_comparison(runs, paths["tv_comparison"], args.t_min)
    plot_energies(runs, paths["energies"], args.t_min)
    plot_combined(runs, paths["overview"], bath_K, args.t_min)

    for path in paths.values():
        print(f"Wrote {path}")


if __name__ == "__main__":
    main()
