#!/usr/bin/env python3
"""Assemble Figure 4 (material time / TN) from pre-rendered panel PNGs."""

from __future__ import annotations

import argparse
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec

from config import FIGURES_DIR
from paper_style import apply_paper_style


def make_figure4(*, figures_dir: Path) -> plt.Figure:
    panels = [
        figures_dir / "fig4a_material_time.png",
        figures_dir / "fig4b_isf_collapse.png",
        figures_dir / "fig4c_tn_tau_tilde_vs_lambda.png",
        figures_dir / "fig4d_tn_tau_tilde_vs_tw.png",
    ]
    missing = [p for p in panels if not p.is_file()]
    if missing:
        raise FileNotFoundError(
            "Missing Fig 4 panels; run analyze_material_time_aging.py first:\n"
            + "\n".join(f"  - {p}" for p in missing)
        )

    apply_paper_style(grid=False)
    fig = plt.figure(figsize=(18, 4.5))
    gs = gridspec.GridSpec(1, 4, figure=fig, wspace=0.25)
    labels = ["(a)", "(b)", "(c)", "(d)"]

    for col, (panel_path, label) in enumerate(zip(panels, labels)):
        ax = fig.add_subplot(gs[0, col])
        ax.imshow(plt.imread(panel_path))
        ax.axis("off")
        ax.text(
            0.02,
            0.98,
            label,
            transform=ax.transAxes,
            fontsize=14,
            fontweight="bold",
            va="top",
            ha="left",
        )

    fig.suptitle(
        "Figure 4 — material time and TN model (weak coupling, OpenMM N=500)",
        y=1.02,
        fontsize=12,
    )
    fig.subplots_adjust(top=0.88)
    return fig


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--figures-dir", type=Path, default=FIGURES_DIR)
    parser.add_argument("--output-dir", type=Path, default=FIGURES_DIR)
    args = parser.parse_args()

    args.output_dir.mkdir(parents=True, exist_ok=True)
    fig = make_figure4(figures_dir=args.figures_dir)
    for ext in ("png", "pdf"):
        out = args.output_dir / f"figure4_weak_coupling.{ext}"
        fig.savefig(out, dpi=200, bbox_inches="tight", facecolor="white")
        print(f"Wrote {out}")
    plt.close(fig)


if __name__ == "__main__":
    main()
