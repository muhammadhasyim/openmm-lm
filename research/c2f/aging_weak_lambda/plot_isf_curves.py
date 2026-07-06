#!/usr/bin/env python3
"""Plot normalized ISF curves (Fig 2a bottom)."""

from __future__ import annotations

import argparse
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

from config import FIGURES_DIR, LAMBDAS, N_REPLICAS, job_dir_path
from fkt_utils import (
    average_phi_over_replicas,
    build_phi_lag_dicts_for_replicas,
    collect_replica_fkt_files,
    count_replicas_for_ref,
    list_available_replicas,
    waiting_time_ps,
)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-dir", type=Path, default=FIGURES_DIR)
    parser.add_argument("--lambdas", type=float, nargs="+", default=LAMBDAS)
    parser.add_argument("--replicas", type=int, nargs="+", default=list(range(N_REPLICAS)))
    parser.add_argument("--max-lag-ps", type=float, default=1600.0)
    parser.add_argument("--block-window-ps", type=float, default=10.0)
    parser.add_argument(
        "--min-replicas",
        type=int,
        default=50,
        help="Skip reference times with fewer than this many contributing replicas",
    )
    args = parser.parse_args()

    args.output_dir.mkdir(parents=True, exist_ok=True)
    plot_lams = sorted(set(args.lambdas))
    ncols = max(len(plot_lams), 1)
    fig, axes = plt.subplots(1, ncols, figsize=(3.2 * ncols, 4), sharey=True)
    if ncols == 1:
        axes = [axes]
    for ax, lam in zip(axes, plot_lams):
        job_dir = job_dir_path(lam)
        available = [r for r in args.replicas if r in list_available_replicas(job_dir, lam)]
        ref_indices = sorted(
            {idx for r in available for idx in collect_replica_fkt_files(job_dir, lam, r)}
        )
        ref_indices = [
            ref_idx
            for ref_idx in ref_indices
            if count_replicas_for_ref(job_dir, lam, available, ref_idx)
            >= args.min_replicas
        ]
        lag_dicts, ref_times, n_used_map = build_phi_lag_dicts_for_replicas(
            job_dir, lam, available, set(ref_indices)
        )
        colors = plt.cm.plasma(np.linspace(0.1, 0.9, max(len(ref_indices), 1)))
        for color, ref_idx in zip(colors, ref_indices):
            ref_time, lags, mean_phi, sem_phi, n_used = average_phi_over_replicas(
                job_dir,
                lam,
                available,
                ref_idx,
                block_window_ps=args.block_window_ps,
                envelope=True,
                error="sem",
                lag_dict=lag_dicts.get(ref_idx),
                ref_time=ref_times.get(ref_idx),
                n_used=n_used_map.get(ref_idx, 0),
            )
            if lags.size == 0:
                continue
            mask = lags <= args.max_lag_ps
            t_w = waiting_time_ps(ref_time, ref_idx)
            ax.plot(
                lags[mask],
                mean_phi[mask],
                color=color,
                lw=1.2,
                label=f"$t_w$={t_w:.0f} ($N$={n_used})",
            )
            if sem_phi.size:
                lower = np.clip((mean_phi - sem_phi)[mask], 0.0, None)
                upper = np.clip((mean_phi + sem_phi)[mask], 0.0, 1.05)
                ax.fill_between(
                    lags[mask],
                    lower,
                    upper,
                    color=color,
                    alpha=0.15,
                    linewidth=0,
                )
        ax.set_title(f"$\\lambda$={lam:g}")
        ax.set_xlabel("lag time (ps)")
        ax.set_ylim(0.0, 1.05)
        if lam == plot_lams[0]:
            ax.set_ylabel("$\\phi_k(t; t_w)$")
        if ref_indices:
            ax.legend(fontsize=6)

    fig.suptitle("Normalized ISF during weak-coupling step turn-on aging", y=1.02)
    fig.tight_layout()
    out_path = args.output_dir / "fig2a_isf_vs_time.png"
    fig.savefig(out_path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"Wrote {out_path}")


if __name__ == "__main__":
    main()
