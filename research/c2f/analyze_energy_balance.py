#!/usr/bin/env python
"""Energy bookkeeping for step turn-on aging trajectories.

Under fixed-T NVT (Bussi + cavity Langevin), the *mechanical* energy
E_mech = U_pot + E_kin is **not** conserved: the baths exchange heat with the
system to hold T near 100 K.  This script reconstructs the energy ledger from
logged CSV columns and checks:

  1. Component sum: U_pot = E_bond + E_nb + E_cav,harm + E_coup + E_dse
  2. E_kin from T_kinetic (equipartition, 500 molecular atoms)
  3. Sector split: U_molecular vs U_cavity vs E_kin vs time
  4. Turn-on transfer: mean changes across t = 10 ps step (pre vs post windows)
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

try:
    from scipy.interpolate import interp1d
except ImportError:
    sys.exit("scipy required")

from openmm.cavitymd.constants import Units

_SCRIPT_DIR = Path(__file__).resolve().parent
DEFAULT_INPUT_DIR = _SCRIPT_DIR / "turnon_aging"
DEFAULT_OUT_DIR = _SCRIPT_DIR / "reviewer_response"
N_ATOMS = 500
KB = Units.KB_KJMOL_PER_K
LAMBDAS = [0.01, 0.03, 0.042, 0.07, 0.09, 0.141]


def _lam_tag(lam: float) -> str:
    return f"{lam:g}".replace(".", "p")


def _kin_from_T(T_kin_K: np.ndarray) -> np.ndarray:
    """Molecular kinetic energy (kJ/mol) from kinetic temperature."""
    return 1.5 * N_ATOMS * KB * T_kin_K


def _load_trajectory(path: Path) -> dict[str, np.ndarray]:
    data = np.genfromtxt(path, delimiter=",", names=True)
    t = np.asarray(data["time_ps"], dtype=float)
    bond = np.asarray(data["E_bond_kjmol"], dtype=float)
    nb = np.asarray(data["E_nonbonded_kjmol"], dtype=float)
    harm = np.asarray(data["E_cav_harmonic_kjmol"], dtype=float)
    coup = np.asarray(data["E_cav_coupling_kjmol"], dtype=float)
    dse = np.asarray(data["E_cav_dse_kjmol"], dtype=float)
    T_kin = np.asarray(data["T_kinetic_K"], dtype=float)

    u_mol = bond + nb
    u_cav = harm + coup + dse
    u_pot = u_mol + u_cav
    e_kin = _kin_from_T(T_kin)
    e_mech = u_pot + e_kin

    u_pot_check = bond + nb + harm + coup + dse
    return {
        "time_ps": t,
        "E_bond": bond,
        "E_nonbonded": nb,
        "E_cav_harm": harm,
        "E_coup": coup,
        "E_dse": dse,
        "U_molecular": u_mol,
        "U_cavity": u_cav,
        "U_pot": u_pot,
        "E_kin": e_kin,
        "E_mech": e_mech,
        "U_pot_check_residual": u_pot - u_pot_check,
    }


def _ensemble_mean(
    input_dir: Path, lam: float, dt_ps: float, runtime_ps: float
) -> dict[str, np.ndarray]:
    lam_tag = _lam_tag(lam)
    files = sorted(input_dir.glob(f"lam{lam_tag}_seed*_energies.csv"))
    if not files:
        raise FileNotFoundError(f"No trajectories for lambda={lam}")

    grid = np.arange(0.0, runtime_ps + 0.5 * dt_ps, dt_ps)
    keys = [
        "E_bond", "E_nonbonded", "E_cav_harm", "E_coup", "E_dse",
        "U_molecular", "U_cavity", "U_pot", "E_kin", "E_mech",
    ]
    stack = {k: [] for k in keys}

    for path in files:
        traj = _load_trajectory(path)
        for k in keys:
            f = interp1d(
                traj["time_ps"], traj[k],
                kind="linear", bounds_error=False, fill_value=np.nan,
            )
            stack[k].append(f(grid))

    means = {k: np.nanmean(stack[k], axis=0) for k in keys}
    stds = {k: np.nanstd(stack[k], axis=0) for k in keys}
    means["time_ps"] = grid
    means["n_traj"] = len(files)
    return means, stds


def _turnon_deltas(
    time_ps: np.ndarray,
    series: dict[str, np.ndarray],
    coupling_start_ps: float,
    window_ps: float = 1.0,
) -> dict[str, float]:
    pre = (time_ps >= coupling_start_ps - window_ps) & (time_ps < coupling_start_ps)
    post = (time_ps >= coupling_start_ps) & (time_ps < coupling_start_ps + window_ps)
    out: dict[str, float] = {}
    for key in ("U_pot", "U_molecular", "U_cavity", "E_kin", "E_mech", "E_bond", "E_cav_harm", "E_coup", "E_dse"):
        pre_mean = float(np.nanmean(series[key][pre])) if pre.any() else float("nan")
        post_mean = float(np.nanmean(series[key][post])) if post.any() else float("nan")
        out[f"delta_{key}"] = post_mean - pre_mean
        out[f"pre_{key}"] = pre_mean
        out[f"post_{key}"] = post_mean
    return out


def _plot_balance(
    means: dict[str, np.ndarray],
    stds: dict[str, np.ndarray],
    coupling_start_ps: float,
    lam: float,
    out_path: Path,
) -> None:
    t = means["time_ps"]
    fig, axes = plt.subplots(3, 1, figsize=(9, 10), sharex=True)

    panels = [
        ("U_pot", "Total potential $U_\\mathrm{pot}$ (kJ/mol)", "#1f77b4"),
        ("E_kin", "Kinetic $E_\\mathrm{kin}$ (kJ/mol)", "#ff7f0e"),
        ("E_mech", "Mechanical $E_\\mathrm{mech}=U_\\mathrm{pot}+E_\\mathrm{kin}$ (kJ/mol)", "#2ca02c"),
    ]
    for ax, (key, ylab, color) in zip(axes, panels):
        y, ys = means[key], stds[key]
        ax.plot(t, y, color=color, lw=1.5)
        ax.fill_between(t, y - ys, y + ys, color=color, alpha=0.15, linewidth=0)
        ax.axvline(coupling_start_ps, color="k", ls="--", lw=1.0, alpha=0.7)
        ax.set_ylabel(ylab)
        if key == "E_mech":
            ax.set_title(
                f"Energy ledger $\\lambda$={lam:g}: "
                f"$E_\\mathrm{{mech}}$ not conserved under NVT baths"
            )

    axes[-1].set_xlabel("time (ps)")
    fig.tight_layout()
    fig.savefig(out_path, dpi=150)
    plt.close(fig)


def _plot_sector_split(
    means: dict[str, np.ndarray],
    stds: dict[str, np.ndarray],
    coupling_start_ps: float,
    lam: float,
    out_path: Path,
) -> None:
    t = means["time_ps"]
    fig, axes = plt.subplots(2, 1, figsize=(9, 7), sharex=True)

    for ax, key, ylab in zip(
        axes,
        ("U_molecular", "U_cavity"),
        ("Molecular potential $E_\\mathrm{bond}+E_\\mathrm{nb}$ (kJ/mol)",
         "Cavity potential $E_\\mathrm{harm}+E_\\mathrm{coup}+E_\\mathrm{dse}$ (kJ/mol)"),
    ):
        y, ys = means[key], stds[key]
        ax.plot(t, y, lw=1.5)
        ax.fill_between(t, y - ys, y + ys, alpha=0.15, linewidth=0)
        ax.axvline(coupling_start_ps, color="k", ls="--", lw=1.0, alpha=0.7)
        ax.set_ylabel( ylab)

    axes[0].plot(t, means["E_bond"], ls=":", color="gray", lw=1.0, label="$E_\\mathrm{bond}$ only")
    axes[0].legend(fontsize=8)
    axes[1].axhline(1.3, color="gray", ls=":", lw=0.8, label="equil. net cavity ~1.3")
    axes[1].legend(fontsize=8)
    axes[1].set_xlabel("time (ps)")
    fig.suptitle(f"Sector energies at $\\lambda$={lam:g}", y=1.01)
    fig.tight_layout()
    fig.savefig(out_path, dpi=150, bbox_inches="tight")
    plt.close(fig)


def _plot_turnon_bar(deltas: dict[str, float], lam: float, out_path: Path) -> None:
    labels = [
        ("ΔU_cavity", deltas["delta_U_cavity"]),
        ("ΔU_molecular", deltas["delta_U_molecular"]),
        ("ΔE_kin", deltas["delta_E_kin"]),
        ("ΔE_mech", deltas["delta_E_mech"]),
        ("ΔE_bond", deltas["delta_E_bond"]),
    ]
    names = [x[0] for x in labels]
    vals = [x[1] for x in labels]
    colors = ["#9467bd", "#1f77b4", "#ff7f0e", "#2ca02c", "#8c564b"]

    fig, ax = plt.subplots(figsize=(7, 4))
    ax.bar(names, vals, color=colors)
    ax.axhline(0, color="k", lw=0.8)
    ax.set_ylabel("kJ/mol (post − pre, 1 ps windows at turn-on)")
    ax.set_title(f"Turn-on energy transfer at $t$={deltas.get('coupling_start_ps', 10)} ps, $\\lambda$={lam:g}")
    fig.tight_layout()
    fig.savefig(out_path, dpi=150)
    plt.close(fig)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input-dir", type=Path, default=DEFAULT_INPUT_DIR)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUT_DIR)
    parser.add_argument("--lambdas", type=float, nargs="+", default=[0.01])
    parser.add_argument("--coupling-start-ps", type=float, default=10.0)
    parser.add_argument("--runtime-ps", type=float, default=160.0)
    parser.add_argument("--csv-interval-ps", type=float, default=0.1)
    args = parser.parse_args()

    output_dir = args.output_dir.resolve()
    output_dir.mkdir(parents=True, exist_ok=True)
    summary_all: dict[str, object] = {
        "note": (
            "NVT with Bussi/Langevin baths: E_mech is not conserved. "
            "Turn-on bookkeeping tracks where potential energy is redistributed."
        ),
        "N_atoms_kin_estimate": N_ATOMS,
        "coupling_start_ps": args.coupling_start_ps,
        "lambdas": {},
    }

    for lam in args.lambdas:
        means, stds = _ensemble_mean(
            args.input_dir.resolve(), lam, args.csv_interval_ps, args.runtime_ps
        )
        deltas = _turnon_deltas(means["time_ps"], means, args.coupling_start_ps)
        deltas["coupling_start_ps"] = args.coupling_start_ps

        tag = _lam_tag(lam)
        _plot_balance(
            means, stds, args.coupling_start_ps, lam,
            output_dir / f"turnon_balance_E_mech_lam{tag}.png",
        )
        _plot_sector_split(
            means, stds, args.coupling_start_ps, lam,
            output_dir / f"turnon_balance_sectors_lam{tag}.png",
        )
        _plot_turnon_bar(
            deltas, lam,
            output_dir / f"turnon_balance_turnon_delta_lam{tag}.png",
        )

        steady = means["time_ps"] >= args.coupling_start_ps
        tail = max(1, int(0.2 * steady.sum()))
        summary_all["lambdas"][str(lam)] = {
            "n_traj": int(means["n_traj"]),
            "turnon_window_1ps": deltas,
            "steady_mean_E_mech": float(np.nanmean(means["E_mech"][steady])),
            "steady_std_E_mech": float(np.nanstd(means["E_mech"][steady])),
            "steady_mean_U_pot": float(np.nanmean(means["U_pot"][steady])),
            "steady_mean_E_kin": float(np.nanmean(means["E_kin"][steady])),
            "E_mech_drift_post_turnon": float(
                np.nanmean(means["E_mech"][steady][-tail:])
                - np.nanmean(means["E_mech"][steady][:tail])
            ),
        }
        print(f"lambda={lam}: turn-on ΔU_mol={deltas['delta_U_molecular']:.2f}, "
              f"ΔU_cav={deltas['delta_U_cavity']:.2f}, "
              f"ΔE_kin={deltas['delta_E_kin']:.2f}, "
              f"ΔE_mech={deltas['delta_E_mech']:.2f} kJ/mol")

    out_json = output_dir / "turnon_energy_balance_summary.json"
    with open(out_json, "w", encoding="utf-8") as f:
        json.dump(summary_all, f, indent=2)
    print(f"Summary -> {out_json}")


if __name__ == "__main__":
    main()
