#!/usr/bin/env python3
"""Plot individual F(k,t) panels per lambda (Fig 2a bottom, cav-hoomd style)."""

from __future__ import annotations

import argparse
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.ticker as ticker
import numpy as np
from matplotlib.colors import Normalize

from config import ANALYSIS_LAMBDAS, FIGURES_DIR, N_REPLICAS, lambda_tag
from fkt_utils import (
    load_lambda_fkt_data,
    process_fkt_data,
    waiting_time_ps,
)
from paper_style import apply_paper_style, save_figure, style_axes

# Tectonic usetex drops axis tick labels in PNG export; mathtext cm is reliable here.
_FKT_RC: dict[str, object] = {
    "text.usetex": False,
    "mathtext.fontset": "cm",
    "font.family": "serif",
    "axes.labelsize": 14,
    "axes.titlesize": 14,
    "xtick.labelsize": 12,
    "ytick.labelsize": 12,
}


def _lambda_filename(lam: float) -> str:
    if lam == 0.0:
        return "fig2a_fkt_lambda0"
    tag = lambda_tag(lam)
    return f"fig2a_fkt_lambda{tag}"


def _set_colorbar_ticks(cbar, *, vmin: float, vmax: float, step: float = 400.0) -> None:
    tick_val = 0.0
    tick_positions: list[float] = []
    while tick_val <= vmax + 1e-9:
        if tick_val >= vmin - 1e-9:
            tick_positions.append(tick_val)
        tick_val += step
    if tick_positions:
        cbar.set_ticks(tick_positions)


def plot_individual_fkt_coupling(
    lam: float,
    replicas: list[int],
    output_dir: Path,
    *,
    max_lag_ps: float = 1000.0,
) -> Path | None:
    data, norm_value, _ = load_lambda_fkt_data(lam, replicas)
    if not data:
        return None

    ref_indices = sorted(data)
    ref_times = [waiting_time_ps(data[r][0], r) for r in ref_indices]
    norm = Normalize(vmin=min(ref_times), vmax=max(ref_times))
    cmap = plt.colormaps.get_cmap("viridis")

    with matplotlib.rc_context(_FKT_RC):
        fig, ax = plt.subplots(figsize=(4, 3))
        has_data = False
        for ref_idx in ref_indices:
            ref_time, lags, fkt = data[ref_idx]
            t_w = waiting_time_ps(ref_time, ref_idx)
            time_processed, fkt_processed, _ = process_fkt_data(lags, fkt, norm_value)
            if time_processed is None or fkt_processed is None:
                continue
            mask = time_processed <= max_lag_ps
            if not np.any(mask):
                continue
            color = cmap(norm(t_w))
            ax.plot(time_processed[mask], fkt_processed[mask], color=color, lw=1.5, alpha=0.85)
            has_data = True

        if not has_data:
            plt.close(fig)
            return None

        ax.set_xlabel(r"$t - t_{\mathrm{w}}$ (ps)")
        ax.set_ylabel(r"$\phi_k(t; t_{\mathrm{w}})$")
        ax.set_title(rf"$\lambda = {lam:g}$")
        style_axes(ax, grid=True)
        ax.set_ylim(bottom=0.0, top=1.05)
        ax.set_xlim(0, max_lag_ps)
        ax.set_xticks([0, 400, 800, max_lag_ps])
        ax.xaxis.set_minor_locator(ticker.MultipleLocator(200))
        ax.set_yticks([0.0, 0.2, 0.4, 0.6, 0.8, 1.0])
        ax.yaxis.set_minor_locator(ticker.MultipleLocator(0.1))
        ax.tick_params(axis="both", which="major", labelsize=12)

        sm = plt.cm.ScalarMappable(cmap=cmap, norm=norm)
        sm.set_array([])
        cbar = fig.colorbar(sm, ax=ax, fraction=0.046, pad=0.04)
        cbar.set_label(r"$t_{\mathrm{w}}$ (ps)")
        _set_colorbar_ticks(cbar, vmin=norm.vmin, vmax=norm.vmax)
        cbar.ax.tick_params(labelsize=10)

        fig.tight_layout()
        output_dir.mkdir(parents=True, exist_ok=True)
        stem = output_dir / _lambda_filename(lam)
        save_figure(fig, stem)
        plt.close(fig)
    return stem.with_suffix(".png")


def main() -> None:
    apply_paper_style(grid=False)
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-dir", type=Path, default=FIGURES_DIR)
    parser.add_argument("--lambdas", type=float, nargs="+", default=ANALYSIS_LAMBDAS)
    parser.add_argument("--replicas", type=int, nargs="+", default=list(range(N_REPLICAS)))
    parser.add_argument("--max-lag-ps", type=float, default=1000.0)
    args = parser.parse_args()

    written: list[Path] = []
    for lam in sorted(set(args.lambdas)):
        out = plot_individual_fkt_coupling(
            lam,
            args.replicas,
            args.output_dir,
            max_lag_ps=args.max_lag_ps,
        )
        if out is not None:
            written.append(out)
            print(f"Wrote {out}")
    if not written:
        raise SystemExit("No F(k,t) panels generated (missing master or replica data)")


if __name__ == "__main__":
    main()
