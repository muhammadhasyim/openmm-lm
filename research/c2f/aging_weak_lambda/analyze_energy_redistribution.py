#!/usr/bin/env python3
"""Fig 3b: molecular potential-energy redistribution from CSV logs."""

from __future__ import annotations

import argparse
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

from config import CSV_INTERVAL_PS, FIG3_SHOWCASE_LAMBDA, FIGURES_DIR, SWITCH_TIME_PS, job_dir_path
from paper_style import (
    COLOR_HARMONIC,
    COLOR_LJ_COULOMB,
    COLOR_TOTAL,
    apply_paper_style,
    paper_legend,
    save_figure,
    style_axes,
)


def smooth_uniform(arr: np.ndarray, window: int) -> np.ndarray:
    """Centered rolling mean with edge padding (avoids boundary undershoot)."""
    if window <= 1:
        return arr
    pad = window // 2
    padded = np.pad(arr, pad, mode="edge")
    return np.convolve(padded, np.ones(window) / window, mode="valid")


def load_ensemble_csv(
    job_dir: Path,
    lam: float,
    replicas: list[int],
    t_grid: np.ndarray,
) -> dict[str, np.ndarray]:
    """Load replicas, subtract per-replica pre-switch baseline, ensemble-average."""
    from fkt_utils import build_energy_csv_index, resolve_energy_csv

    index = build_energy_csv_index(job_dir, lam)
    bond_stack: list[np.ndarray] = []
    nb_stack: list[np.ndarray] = []
    mol_stack: list[np.ndarray] = []

    for replica in replicas:
        csv_path = resolve_energy_csv(job_dir, lam, replica, index)
        if csv_path is None:
            continue
        data = np.genfromtxt(
            csv_path, delimiter=",", names=True, missing_values="", usemask=False
        )
        t = np.asarray(data["time_ps"], dtype=float)
        bond = np.asarray(data["E_bond_kjmol"], dtype=float)
        nonbonded = np.asarray(data["E_nonbonded_kjmol"], dtype=float)

        pre = t < SWITCH_TIME_PS
        if not pre.any():
            continue
        bond0 = float(np.mean(bond[pre]))
        nb0 = float(np.mean(nonbonded[pre]))

        d_bond = bond - bond0
        d_nb = nonbonded - nb0
        bond_stack.append(np.interp(t_grid, t, d_bond, left=np.nan, right=np.nan))
        nb_stack.append(np.interp(t_grid, t, d_nb, left=np.nan, right=np.nan))
        mol_stack.append(np.interp(t_grid, t, d_bond + d_nb, left=np.nan, right=np.nan))

    if not bond_stack:
        return {}

    return {
        "time": t_grid,
        "bond": np.nanmean(np.vstack(bond_stack), axis=0),
        "nonbonded": np.nanmean(np.vstack(nb_stack), axis=0),
        "mol_total": np.nanmean(np.vstack(mol_stack), axis=0),
        "n_replicas": len(bond_stack),
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--lambda", dest="lam", type=float, default=FIG3_SHOWCASE_LAMBDA)
    parser.add_argument(
        "--campaign-dir",
        type=Path,
        default=None,
        help="Campaign root (default: aging_weak_lambda/).",
    )
    parser.add_argument("--output-dir", type=Path, default=FIGURES_DIR)
    parser.add_argument(
        "--replicas",
        type=int,
        nargs="+",
        default=None,
        help="Replica indices (default: all valid energy CSVs reaching --tmax-ps).",
    )
    parser.add_argument("--tmin-ps", type=float, default=1.0, help="Plot window start (ps).")
    parser.add_argument("--tmax-ps", type=float, default=2000.0, help="Plot window end (ps).")
    parser.add_argument(
        "--smooth-window",
        type=int,
        default=15,
        metavar="N",
        help="Uniform rolling-average window (points); 1 = no smoothing.",
    )
    args = parser.parse_args()

    from fkt_utils import list_available_energy_replicas

    job_dir = job_dir_path(args.lam, campaign_root=args.campaign_dir)
    if args.replicas is None:
        replicas = list_available_energy_replicas(job_dir, args.lam, min_tmax_ps=args.tmax_ps)
    else:
        replicas = args.replicas

    if not replicas:
        raise SystemExit(f"No energy CSV replicas in {job_dir} (tmax >= {args.tmax_ps} ps)")

    t_grid = np.arange(args.tmin_ps, args.tmax_ps + CSV_INTERVAL_PS, CSV_INTERVAL_PS)
    data = load_ensemble_csv(job_dir, args.lam, replicas, t_grid)
    if not data:
        raise SystemExit(f"No CSV data loaded from {job_dir}")

    t = data["time"]
    w = max(1, args.smooth_window)
    d_bond = smooth_uniform(data["bond"], w)
    d_nb = smooth_uniform(data["nonbonded"], w)
    d_total = smooth_uniform(data["mol_total"], w)

    apply_paper_style(grid=True)

    n_rep = data.get("n_replicas", len(replicas))
    fig, ax = plt.subplots(figsize=(8, 5), facecolor="white")
    ax.set_facecolor("white")
    ax.plot(t, d_bond, label="Harmonic", color=COLOR_HARMONIC, lw=2.0)
    ax.plot(t, d_nb, label="LJ+Coulomb", color=COLOR_LJ_COULOMB, lw=2.0)
    ax.plot(t, d_total, label="Total", color=COLOR_TOTAL, lw=2.0)
    ax.axhline(0.0, color="gray", ls="--", lw=1.0, alpha=0.8, label="Equilibrium")
    ax.axvline(SWITCH_TIME_PS, color="k", ls=":", lw=1.0, alpha=0.7)
    ax.set_xlim(args.tmin_ps, args.tmax_ps)
    ax.set_xlabel(r"$t$ (ps)")
    ax.set_ylabel(r"$\Delta V$ (kJ/mol)")
    style_axes(ax, grid=True)
    paper_legend(ax, loc="best", fontsize=10)
    fig.tight_layout()
    args.output_dir.mkdir(parents=True, exist_ok=True)
    out_stem = args.output_dir / f"fig3b_energy_redistribution_lam{args.lam:g}"
    save_figure(fig, out_stem)
    plt.close(fig)
    print(f"Ensemble over {n_rep} replicas, t=[{args.tmin_ps:.0f}, {args.tmax_ps:.0f}] ps")


if __name__ == "__main__":
    main()
