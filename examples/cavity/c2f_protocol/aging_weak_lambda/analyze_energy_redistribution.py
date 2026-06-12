#!/usr/bin/env python3
"""Fig 3b: molecular potential-energy redistribution from CSV logs."""

from __future__ import annotations

import argparse
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

from config import FIGURES_DIR, N_REPLICAS, SWITCH_TIME_PS, job_dir_path, run_prefix


def load_ensemble_csv(job_dir: Path, lam: float, replicas: list[int]) -> dict[str, np.ndarray]:
    series: dict[str, list[np.ndarray]] = {
        "time": [],
        "bond": [],
        "nonbonded": [],
        "mech": [],
    }
    for replica in replicas:
        csv_path = job_dir / f"{run_prefix(lam, replica)}_energies.csv"
        if not csv_path.exists():
            continue
        data = np.genfromtxt(
            csv_path, delimiter=",", names=True, missing_values="", usemask=False
        )
        t = np.asarray(data["time_ps"], dtype=float)
        series["time"].append(t)
        series["bond"].append(np.asarray(data["E_bond_kjmol"], dtype=float))
        series["nonbonded"].append(np.asarray(data["E_nonbonded_kjmol"], dtype=float))
        series["mech"].append(np.asarray(data["E_mech_kjmol"], dtype=float))

    if not series["time"]:
        return {}

    t_common = series["time"][0]
    out: dict[str, np.ndarray] = {"time": t_common}
    for key in ("bond", "nonbonded", "mech"):
        stack = np.vstack(
            [np.interp(t_common, series["time"][i], series[key][i]) for i in range(len(series["time"]))]
        )
        out[key] = np.nanmean(stack, axis=0)
    return out


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--lambda", dest="lam", type=float, default=0.03)
    parser.add_argument("--output-dir", type=Path, default=FIGURES_DIR)
    parser.add_argument("--tmax-ps", type=float, default=2000.0)
    parser.add_argument("--replicas", type=int, nargs="+", default=list(range(N_REPLICAS)))
    args = parser.parse_args()

    from fkt_utils import list_available_replicas

    job_dir = job_dir_path(args.lam)
    replicas = [r for r in args.replicas if r in list_available_replicas(job_dir, args.lam)]
    data = load_ensemble_csv(job_dir, args.lam, replicas)
    if not data:
        raise SystemExit(f"No CSV data in {job_dir}")

    t = data["time"]
    mask = (t >= 0.0) & (t <= args.tmax_ps)
    t = t[mask]
    bond = data["bond"][mask]
    nonbonded = data["nonbonded"][mask]
    mech = data["mech"][mask]

    pre = t < SWITCH_TIME_PS
    bond0 = float(np.mean(bond[pre])) if pre.any() else bond[0]
    nb0 = float(np.mean(nonbonded[pre])) if pre.any() else nonbonded[0]
    mech0 = float(np.mean(mech[pre])) if pre.any() else mech[0]

    fig, ax = plt.subplots(figsize=(8, 5))
    ax.plot(t, bond - bond0, label="harmonic (bond)", color="#1f77b4")
    ax.plot(t, nonbonded - nb0, label="LJ + Coulomb", color="#d62728")
    ax.plot(t, mech - mech0, label="total mechanical", color="#2ca02c")
    ax.axvline(SWITCH_TIME_PS, color="k", ls="--", lw=1.0, alpha=0.7, label="coupling on")
    ax.set_xlabel("time (ps)")
    ax.set_ylabel("$\\Delta U$ (kJ/mol)")
    ax.set_title(f"Energy redistribution after turn-on ($\\lambda$={args.lam:g})")
    ax.legend(fontsize=8)
    fig.tight_layout()
    out_path = args.output_dir / f"fig3b_energy_redistribution_lam{args.lam:g}.png"
    fig.savefig(out_path, dpi=150)
    plt.close(fig)
    print(f"Wrote {out_path}")


if __name__ == "__main__":
    main()
