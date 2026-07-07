#!/usr/bin/env python3
"""Plot harmonic and nonbonded energies from an energy-recovery NPZ file."""

from __future__ import annotations

import argparse
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np


def plot_energy_recovery(
    npz_path: Path,
    output_png: Path,
    switch_time_ps: float = 200.0,
    baseline_lo_ps: float = 150.0,
    baseline_hi_ps: float = 200.0,
) -> Path:
    data = np.load(npz_path, allow_pickle=True)
    t = data["time_ps"]
    bond = data["bond_energy_kj_mol"]
    nb = data["nonbonded_energy_kj_mol"]

    pre = (t >= baseline_lo_ps) & (t < baseline_hi_ps)
    pre_bond_mean = bond[pre].mean()
    pre_bond_std = bond[pre].std()
    pre_nb_mean = nb[pre].mean()
    pre_nb_std = nb[pre].std()

    fig, (ax_bond, ax_nb) = plt.subplots(2, 1, figsize=(10, 7), sharex=True)
    fig.suptitle(
        "mKA energy tracking: instant coupling switch at 200 ps (λ = 0.005)",
        fontsize=13,
        fontweight="bold",
    )

    ax_bond.plot(t, bond, color="#d62728", linewidth=1.2, label="Harmonic bonds")
    ax_bond.axvline(switch_time_ps, color="black", linestyle="--", linewidth=1.0, alpha=0.7)
    ax_bond.axhline(pre_bond_mean, color="#d62728", linestyle=":", alpha=0.8)
    ax_bond.fill_between(
        t,
        pre_bond_mean - 2 * pre_bond_std,
        pre_bond_mean + 2 * pre_bond_std,
        color="#d62728",
        alpha=0.12,
        label=f"Pre-switch baseline ±2σ ({baseline_lo_ps:.0f}–{baseline_hi_ps:.0f} ps)",
    )
    ax_bond.set_ylabel("Harmonic bond energy (kJ/mol)")
    ax_bond.legend(loc="upper left", fontsize=9)
    ax_bond.grid(True, alpha=0.25)

    ax_nb.plot(t, nb, color="#1f77b4", linewidth=1.2, label="Nonbonded (LJ + Coulomb)")
    ax_nb.axvline(switch_time_ps, color="black", linestyle="--", linewidth=1.0, alpha=0.7,
                  label="Coupling switch")
    ax_nb.axhline(pre_nb_mean, color="#1f77b4", linestyle=":", alpha=0.8)
    ax_nb.fill_between(
        t,
        pre_nb_mean - 2 * pre_nb_std,
        pre_nb_mean + 2 * pre_nb_std,
        color="#1f77b4",
        alpha=0.12,
        label=f"Pre-switch baseline ±2σ ({baseline_lo_ps:.0f}–{baseline_hi_ps:.0f} ps)",
    )
    ax_nb.set_xlabel("Simulation time (ps)")
    ax_nb.set_ylabel("Nonbonded energy (kJ/mol)")
    ax_nb.legend(loc="lower left", fontsize=9)
    ax_nb.grid(True, alpha=0.25)

    fig.text(
        0.5,
        0.01,
        "250 dimers, T = 100 K, λ = 0 before switch → 0.005 at 200 ps, adaptive VariableVerlet, total 2500 ps",
        ha="center",
        fontsize=9,
        color="#444444",
    )
    fig.tight_layout(rect=[0, 0.03, 1, 0.96])

    output_png.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output_png, dpi=180, bbox_inches="tight")
    plt.close(fig)
    return output_png


def _late_slope(t: np.ndarray, y: np.ndarray, window_ps: float = 1500.0) -> float:
    late = t >= (t.max() - window_ps)
    if np.count_nonzero(late) < 3:
        return float("nan")
    return float(np.polyfit(t[late], y[late], 1)[0])


def plot_comparison(
    nobath_path: Path,
    bath_path: Path,
    output_png: Path,
    switch_time_ps: float = 200.0,
    baseline_lo_ps: float = 150.0,
    baseline_hi_ps: float = 200.0,
) -> Path:
    """Overlay no-bath (buggy) vs cavity-bath (fixed) runs for both energies."""
    d0 = np.load(nobath_path, allow_pickle=True)
    d1 = np.load(bath_path, allow_pickle=True)
    t0, bond0, nb0 = d0["time_ps"], d0["bond_energy_kj_mol"], d0["nonbonded_energy_kj_mol"]
    t1, bond1, nb1 = d1["time_ps"], d1["bond_energy_kj_mol"], d1["nonbonded_energy_kj_mol"]

    pre = (t1 >= baseline_lo_ps) & (t1 < baseline_hi_ps)
    pre_bond = float(bond1[pre].mean())
    pre_nb = float(nb1[pre].mean())

    fig, (ax_bond, ax_nb) = plt.subplots(2, 1, figsize=(10, 7.5), sharex=True)
    fig.suptitle(
        "mKA cavity switch (λ = 0.005 at 200 ps): no cavity bath vs 1 ps Langevin bath",
        fontsize=13,
        fontweight="bold",
    )

    s0 = _late_slope(t0, bond0) * 100.0
    s1 = _late_slope(t1, bond1) * 100.0
    ax_bond.plot(t0, bond0, color="#9467bd", lw=1.1,
                 label=f"No cavity bath (NVE photon): late slope {s0:+.2f} kJ/mol per 100 ps")
    ax_bond.plot(t1, bond1, color="#d62728", lw=1.1,
                 label=f"Cavity Langevin bath (γ=1 ps⁻¹): late slope {s1:+.2f} kJ/mol per 100 ps")
    ax_bond.axvline(switch_time_ps, color="black", ls="--", lw=1.0, alpha=0.7)
    ax_bond.axhline(pre_bond, color="gray", ls=":", alpha=0.8, label="Pre-switch baseline")
    ax_bond.set_ylabel("Harmonic bond energy (kJ/mol)\n(fast / vibrational modes)")
    ax_bond.legend(loc="upper left", fontsize=8.5)
    ax_bond.grid(True, alpha=0.25)

    ax_nb.plot(t0, nb0, color="#9467bd", lw=1.1, label="No cavity bath (NVE photon)")
    ax_nb.plot(t1, nb1, color="#1f77b4", lw=1.1, label="Cavity Langevin bath (γ=1 ps⁻¹)")
    ax_nb.axvline(switch_time_ps, color="black", ls="--", lw=1.0, alpha=0.7, label="Coupling switch")
    ax_nb.axhline(pre_nb, color="gray", ls=":", alpha=0.8, label="Pre-switch baseline")
    ax_nb.set_xlabel("Simulation time (ps)")
    ax_nb.set_ylabel("Nonbonded energy (kJ/mol)\n(structural modes)")
    ax_nb.legend(loc="lower left", fontsize=8.5)
    ax_nb.grid(True, alpha=0.25)
    # Zoom the structural panel to the equilibrated window (exclude the t~0
    # pre-equilibration minimization transient that otherwise dominates).
    eq0 = nb0[t0 >= baseline_lo_ps]
    eq1 = nb1[t1 >= baseline_lo_ps]
    if eq0.size and eq1.size:
        lo = min(eq0.min(), eq1.min()) - 30.0
        hi = max(eq0.max(), eq1.max()) + 30.0
        ax_nb.set_ylim(lo, hi)

    fig.text(
        0.5, 0.01,
        "250 dimers, T = 100 K, molecules on Bussi (τ=1 ps). With the cavity bath the fast-mode energy "
        "saturates (bounded steady state) while the structure ages into deeper basins.",
        ha="center", fontsize=8.5, color="#444444",
    )
    fig.tight_layout(rect=[0, 0.03, 1, 0.96])
    output_png.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output_png, dpi=180, bbox_inches="tight")
    plt.close(fig)
    return output_png


def main() -> None:
    here = Path(__file__).resolve().parent
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--input",
        type=Path,
        default=here / "energy_recovery_lambda0.005_t200.npz",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=here / "energy_recovery_lambda0.005_t200.png",
    )
    parser.add_argument("--switch-time", type=float, default=200.0)
    parser.add_argument("--compare", nargs=2, metavar=("NOBATH_NPZ", "BATH_NPZ"),
                        type=Path, default=None,
                        help="Overlay two runs (no-bath vs cavity-bath) into --output")
    args = parser.parse_args()

    if args.compare is not None:
        out = plot_comparison(args.compare[0], args.compare[1], args.output,
                              switch_time_ps=args.switch_time)
    else:
        out = plot_energy_recovery(args.input, args.output, switch_time_ps=args.switch_time)
    print(f"Wrote {out}")


if __name__ == "__main__":
    main()
