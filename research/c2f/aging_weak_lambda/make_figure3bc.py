#!/usr/bin/env python3
"""Assemble Figure 3 panels (b)/(c) from pre-rendered panel PNGs."""

from __future__ import annotations

import argparse
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec

from config import FIGURES_DIR, FIG3_SHOWCASE_LAMBDA
from paper_style import apply_paper_style


def make_figure3bc(*, figures_dir: Path, lam: float | None = None) -> plt.Figure:
    if lam is None:
        lam = FIG3_SHOWCASE_LAMBDA
    panel_b = figures_dir / f"fig3b_energy_redistribution_lam{lam:g}.png"
    panel_c = figures_dir / f"fig3c_fictive_temperatures_lam{lam:g}.png"
    missing = [p for p in (panel_b, panel_c) if not p.is_file()]
    if missing:
        raise FileNotFoundError(
            "Missing Fig 3b/3c panels; run analyze_energy_redistribution.py and "
            "analyze_fictive_temperatures.py first:\n"
            + "\n".join(f"  - {p}" for p in missing)
        )

    apply_paper_style(grid=False)
    fig = plt.figure(figsize=(14, 5))
    gs = gridspec.GridSpec(1, 2, figure=fig, wspace=0.15)

    for col, (panel_path, label) in enumerate(zip((panel_b, panel_c), ("(b)", "(c)"))):
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
        rf"Figure 3 — energy redistribution and fictive temperatures ($\lambda={lam:g}$ a.u.)",
        y=1.02,
        fontsize=12,
    )
    fig.subplots_adjust(top=0.88)
    return fig


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--figures-dir", type=Path, default=FIGURES_DIR)
    parser.add_argument("--output-dir", type=Path, default=FIGURES_DIR)
    parser.add_argument(
        "--lambda",
        dest="lam",
        type=float,
        default=FIG3_SHOWCASE_LAMBDA,
        help="Showcase coupling for Fig 3(b,c) panels (default: highest complete λ).",
    )
    args = parser.parse_args()

    args.output_dir.mkdir(parents=True, exist_ok=True)
    fig = make_figure3bc(figures_dir=args.figures_dir, lam=args.lam)
    for ext in ("png", "pdf"):
        out = args.output_dir / f"figure3bc_weak_coupling.{ext}"
        fig.savefig(out, dpi=200, bbox_inches="tight", facecolor="white")
        print(f"Wrote {out}")
    plt.close(fig)


if __name__ == "__main__":
    main()
