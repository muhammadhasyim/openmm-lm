#!/usr/bin/env python3
"""DSE / bilinear cavity energies vs time and lambda from CSV logs."""

from __future__ import annotations

import argparse
import csv
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

from config import FIGURES_DIR, LAMBDAS, N_REPLICAS, RESULTS_DIR, SWITCH_TIME_PS, job_dir_path, run_prefix
from paper_style import apply_paper_style, save_figure, style_axes


def _load_csv(job_dir: Path, lam: float, replica: int) -> dict[str, np.ndarray]:
    path = job_dir / f"{run_prefix(lam, replica)}_energies.csv"
    data = np.genfromtxt(path, delimiter=",", names=True, missing_values="", usemask=False)
    return {name: np.asarray(data[name], dtype=float) for name in data.dtype.names}


def main() -> None:
    apply_paper_style()
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-dir", type=Path, default=FIGURES_DIR)
    parser.add_argument("--results-dir", type=Path, default=RESULTS_DIR)
    parser.add_argument("--lambdas", type=float, nargs="+", default=[l for l in LAMBDAS if l > 0])
    parser.add_argument("--replicas", type=int, nargs="+", default=list(range(N_REPLICAS)))
    parser.add_argument("--xmax-ps", type=float, default=800.0)
    args = parser.parse_args()

    args.output_dir.mkdir(parents=True, exist_ok=True)
    fig, axes = plt.subplots(2, 1, figsize=(9, 8), sharex=True)
    colors = plt.cm.viridis(np.linspace(0.15, 0.9, len(args.lambdas)))
    rows_out: list[dict[str, float | str]] = []

    for lam, color in zip(args.lambdas, colors):
        job_dir = job_dir_path(lam)
        t_list, coup_list, dse_list, net_list = [], [], [], []
        for rep in args.replicas:
            csv_path = job_dir / f"{run_prefix(lam, rep)}_energies.csv"
            if not csv_path.exists():
                continue
            d = _load_csv(job_dir, lam, rep)
            mask = d["time_ps"] <= args.xmax_ps
            t_list.append(d["time_ps"][mask])
            coup = d["E_cav_coupling_kjmol"][mask]
            dse = d["E_cav_dse_kjmol"][mask]
            harm = d["E_cav_harmonic_kjmol"][mask]
            coup_list.append(coup)
            dse_list.append(dse)
            net_list.append(coup + dse + harm)

        if not t_list:
            continue
        t = t_list[0]
        coup_mean = np.mean(np.vstack([np.interp(t, t_list[i], coup_list[i]) for i in range(len(t_list))]), axis=0)
        dse_mean = np.mean(np.vstack([np.interp(t, t_list[i], dse_list[i]) for i in range(len(t_list))]), axis=0)
        net_mean = np.mean(np.vstack([np.interp(t, t_list[i], net_list[i]) for i in range(len(t_list))]), axis=0)

        axes[0].plot(t, coup_mean, color=color, lw=1.2, label=f"$\\lambda$={lam:g}")
        axes[1].plot(t, dse_mean, color=color, lw=1.2, label=f"$\\lambda$={lam:g}")

        post = t > SWITCH_TIME_PS
        if post.any():
            rows_out.append(
                {
                    "lambda": lam,
                    "mean_E_coup_kjmol": float(np.mean(coup_mean[post])),
                    "mean_E_dse_kjmol": float(np.mean(dse_mean[post])),
                    "mean_net_cavity_kjmol": float(np.mean(net_mean[post])),
                    "abs_ratio_coup_over_dse": float(
                        abs(np.mean(coup_mean[post])) / max(abs(np.mean(dse_mean[post])), 1e-12)
                    ),
                }
            )

    for ax, ylab in zip(axes, ["$E_\\mathrm{coup}$ (kJ/mol)", "$E_\\mathrm{dse}$ (kJ/mol)"]):
        ax.axvline(SWITCH_TIME_PS, color="k", ls="--", lw=1.0, alpha=0.7)
        ax.set_ylabel(ylab)
        ax.legend(fontsize=7, ncol=2)
        style_axes(ax)
    axes[0].set_title("Bilinear coupling and DSE after step turn-on")
    axes[1].set_xlabel("time (ps)")
    fig.tight_layout()
    save_figure(fig, args.output_dir / "cavity_energies_vs_time")
    plt.close(fig)

    args.results_dir.mkdir(parents=True, exist_ok=True)
    out_csv = args.results_dir / "cavity_energy_decomposition.csv"
    if rows_out:
        with open(out_csv, "w", encoding="utf-8", newline="") as fh:
            writer = csv.DictWriter(fh, fieldnames=list(rows_out[0].keys()))
            writer.writeheader()
            writer.writerows(rows_out)
        print(f"Wrote {out_csv}")


if __name__ == "__main__":
    main()
