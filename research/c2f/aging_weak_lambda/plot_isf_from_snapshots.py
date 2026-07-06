#!/usr/bin/env python3
"""Plot ISF curves recomputed from snapshots with COM-corrected F(k,t) replay."""

from __future__ import annotations

import argparse
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

from config import (
    FKT_MAX_REFS,
    FKT_REF_INTERVAL_PS,
    FIGURES_DIR,
    LAMBDAS,
    N_REPLICAS,
    SWITCH_TIME_PS,
    job_dir_path,
)
from fkt_utils import (
    average_phi_from_snapshots_over_replicas,
    list_available_snapshot_replicas,
    waiting_time_ps,
)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-dir", type=Path, default=FIGURES_DIR / "pilot_n5_rescued")
    parser.add_argument("--lambdas", type=float, nargs="+", default=LAMBDAS)
    parser.add_argument("--replicas", type=int, nargs="+", default=list(range(N_REPLICAS)))
    parser.add_argument("--max-lag-ps", type=float, default=1600.0)
    parser.add_argument("--block-window-ps", type=float, default=10.0)
    parser.add_argument("--fkt-start-ps", type=float, default=SWITCH_TIME_PS)
    parser.add_argument("--ref-interval-ps", type=float, default=FKT_REF_INTERVAL_PS)
    parser.add_argument("--max-refs", type=int, default=FKT_MAX_REFS)
    args = parser.parse_args()

    args.output_dir.mkdir(parents=True, exist_ok=True)
    plot_lams = sorted(set(args.lambdas))
    ncols = max(len(plot_lams), 1)
    fig, axes = plt.subplots(1, ncols, figsize=(3.2 * ncols, 4), sharey=True)
    if ncols == 1:
        axes = [axes]

    for ax, lam in zip(axes, plot_lams):
        job_dir = job_dir_path(lam)
        available = [
            replica
            for replica in args.replicas
            if replica in list_available_snapshot_replicas(job_dir, lam)
        ]
        ref_indices = list(range(args.max_refs))
        colors = plt.cm.plasma(np.linspace(0.1, 0.9, max(len(ref_indices), 1)))
        plotted = 0
        for color, ref_idx in zip(colors, ref_indices):
            ref_time, lags, mean_phi, std_phi, n_used = (
                average_phi_from_snapshots_over_replicas(
                    job_dir,
                    lam,
                    available,
                    ref_idx,
                    fkt_start_ps=args.fkt_start_ps,
                    ref_interval_ps=args.ref_interval_ps,
                    max_refs=args.max_refs,
                    block_window_ps=args.block_window_ps,
                )
            )
            if lags.size == 0 or n_used == 0:
                continue
            plotted += 1
            mask = lags <= args.max_lag_ps
            t_w = waiting_time_ps(ref_time, ref_idx)
            ax.plot(
                lags[mask],
                mean_phi[mask],
                color=color,
                lw=1.2,
                label=f"$t_w$={t_w:.0f} (N={n_used})",
            )
            if std_phi.size:
                ax.fill_between(
                    lags[mask],
                    (mean_phi - std_phi)[mask],
                    (mean_phi + std_phi)[mask],
                    color=color,
                    alpha=0.15,
                    linewidth=0,
                )
        ax.set_title(f"$\\lambda$={lam:g}")
        ax.set_xlabel("lag time (ps)")
        ax.set_ylim(0.0, 1.05)
        if lam == plot_lams[0]:
            ax.set_ylabel("$\\phi_k(t; t_w)$")
        if plotted:
            ax.legend(fontsize=6)

    fig.suptitle(
        "COM-corrected ISF from snapshot replay (weak-coupling aging)",
        y=1.02,
    )
    fig.tight_layout()
    out_path = args.output_dir / "fig2a_isf_vs_time.png"
    fig.savefig(out_path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"Wrote {out_path}")


if __name__ == "__main__":
    main()
