#!/usr/bin/env python3
"""Minimal one-spectrum-per-file IR figures (plot_minimal_spectra.py style).

By default reads ``average_spectrum_lambda_{NN}.txt`` and writes PDFs only.
Use ``--recompute`` to rebuild spectra from dipole ``*.npz`` trajectories.
"""

from __future__ import annotations

import argparse
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.ticker import MultipleLocator

from analyze_ir_from_dipole import ensemble_spectrum
from config import (
    FIGURES_DIR,
    FREQUENCY_CM1,
    IR_SUBSET_REPLICAS,
    LAMBDAS,
    lambda_tag,
)

MIN_FREQ_CM1 = 1300.0
MAX_FREQ_CM1 = 1800.0


def _style_axes(ax: plt.Axes, *, ymax: float) -> None:
    ax.set_xlim(MIN_FREQ_CM1, MAX_FREQ_CM1)
    ax.set_ylim(-0.0005, ymax)
    ax.set_xlabel(r"$\omega$ $(\mathrm{cm^{-1}})$", fontsize=14)
    ax.set_ylabel(r"$n(\omega)\alpha(\omega)$", fontsize=14)
    ax.tick_params(axis="both", which="major", labelsize=10)
    ax.xaxis.set_major_locator(MultipleLocator(200))
    ax.locator_params(axis="y", nbins=5)
    ax.axvline(
        FREQUENCY_CM1,
        color="gray",
        linestyle="--",
        alpha=0.6,
        linewidth=1.5,
        label=f"{FREQUENCY_CM1:g} cm$^{{-1}}$",
    )
    ax.legend(loc="upper right", fontsize=10, framealpha=0.8)
    ax.set_facecolor("white")
    for spine in ax.spines.values():
        spine.set_visible(True)
        spine.set_color("black")
        spine.set_linewidth(1.0)


def export_spectrum(path: Path, freqs: np.ndarray, spec: np.ndarray) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    np.savetxt(path, np.column_stack([freqs, spec]), fmt="%.8e")


def plot_minimal_spectrum(
    freqs: np.ndarray,
    spec: np.ndarray,
    *,
    color: tuple[float, float, float, float],
    out_path: Path,
) -> None:
    mask = (freqs >= MIN_FREQ_CM1) & (freqs <= MAX_FREQ_CM1)
    freq = freqs[mask]
    mean_spec = spec[mask]
    mean_spec = mean_spec / np.trapezoid(mean_spec, freq)
    ymax = max(float(mean_spec.max()) * 1.05, 0.01)

    fig = plt.figure(figsize=(2.5, 1.5))
    ax = fig.add_subplot(111)
    ax.plot(freq, mean_spec, color=color, alpha=0.8, linewidth=2)
    _style_axes(ax, ymax=ymax)
    fig.patch.set_facecolor("white")
    fig.tight_layout(pad=0.2)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, dpi=300, bbox_inches="tight")
    plt.close(fig)


def load_spectrum(path: Path) -> tuple[np.ndarray, np.ndarray]:
    data = np.loadtxt(path)
    return data[:, 0], data[:, 1]


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-dir", type=Path, default=FIGURES_DIR)
    parser.add_argument(
        "--spectrum-dir",
        type=Path,
        default=None,
        help="Directory for average_spectrum_lambda_*.txt (default: output-dir/ir_spectra)",
    )
    parser.add_argument("--lambdas", type=float, nargs="+", default=LAMBDAS)
    parser.add_argument("--replica-end", type=int, default=IR_SUBSET_REPLICAS)
    parser.add_argument(
        "--window-idx",
        type=int,
        default=1,
        help="Dipole window index (default 1 = late-aged window)",
    )
    parser.add_argument(
        "--recompute",
        action="store_true",
        help="Recompute spectra from dipole npz files (slow; default: read cached txt)",
    )
    parser.add_argument("--use-tex",
        action="store_true",
        help="Enable LaTeX text rendering (requires a TeX install)",
    )
    args = parser.parse_args()

    spectrum_dir = args.spectrum_dir or (args.output_dir / "ir_spectra")
    plot_lams = sorted(set(args.lambdas))
    colors = plt.cm.coolwarm(np.linspace(0, 1, len(plot_lams)))

    plt.style.use("classic")
    rc: dict[str, object] = {
        "font.family": "serif",
        "font.serif": ["Computer Modern Roman", "DejaVu Serif"],
    }
    if args.use_tex:
        rc["text.usetex"] = True
    plt.rcParams.update(rc)

    for idx, lam in enumerate(plot_lams):
        txt_path = spectrum_dir / f"average_spectrum_lambda_{idx:02d}.txt"
        pdf_path = args.output_dir / f"minimal_spectrum_{idx:02d}.pdf"

        if args.recompute or not txt_path.is_file():
            result = ensemble_spectrum(lam, args.window_idx, args.replica_end)
            if result is None:
                print(f"Skipping lambda={lam:g}: no dipole spectra found")
                continue
            freqs, spec = result
            export_spectrum(txt_path, freqs, spec)
            source = "dipole npz"
        else:
            freqs, spec = load_spectrum(txt_path)
            source = txt_path.name

        plot_minimal_spectrum(freqs, spec, color=colors[idx], out_path=pdf_path)
        print(
            f"lambda={lam:g} -> {pdf_path.name} (from {source}; tag=lambda{lambda_tag(lam)})"
        )


if __name__ == "__main__":
    main()
