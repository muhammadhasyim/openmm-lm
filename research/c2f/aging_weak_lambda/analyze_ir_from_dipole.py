#!/usr/bin/env python3
"""Fig 2a top: IR from fine-sampled dipole windows (up to 5000 cm^-1).

Uses in-plane dipole components (x, y) only; the cavity-axis (z) component is excluded.
"""

from __future__ import annotations

import argparse
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from scipy import fftpack

from config import (
    DIPOLE_INTERVAL_PS,
    FIGURES_DIR,
    FREQUENCY_CM1,
    IR_SUBSET_REPLICAS,
    IR_WINDOWS,
    ANALYSIS_LAMBDAS,
    TEMPERATURE_K,
    job_dir_path,
)
from fkt_utils import dipole_path
from paper_style import apply_paper_style, save_figure, style_axes

BOLTZ = 1.38064852e-23
LIGHTSPEED = 299792458.0
REDUCED_PLANCK = 1.05457180013e-34


def _vector_dipole_acf(dipole: np.ndarray) -> np.ndarray:
    """Autocorrelation of in-plane dipole components: sum_d <mu_d(t) mu_d(0)>."""
    n_frames = dipole.shape[0]
    n_fft = int(2 ** np.ceil(np.log2(2 * n_frames - 1)))
    acf = np.zeros(n_frames, dtype=float)
    for dim in range(min(dipole.shape[1], 2)):
        comp = dipole[:, dim] - np.mean(dipole[:, dim])
        fx = np.fft.rfft(comp, n=n_fft)
        acf += np.fft.irfft(fx * fx.conj()).real[:n_frames]
    return acf


def ir_spectrum_from_dipole(
    dipole: np.ndarray,
    times_ps: np.ndarray,
    *,
    temperature_k: float = TEMPERATURE_K,
) -> tuple[np.ndarray, np.ndarray]:
    """DCT + quantum correction IR spectrum (cav-hoomd process_dipole_autocorr)."""
    if dipole.shape[0] < 4:
        return np.array([]), np.array([])
    times_fs = np.asarray(times_ps, dtype=float) * 1000.0
    dt_fs = float(np.mean(np.diff(times_fs)))
    if dt_fs <= 0:
        return np.array([]), np.array([])
    dipole = dipole[:, :2] - np.mean(dipole[:, :2], axis=0, keepdims=True)
    acf = _vector_dipole_acf(dipole)
    if acf[0] != 0:
        acf = acf / acf[0]

    timestep_s = dt_fs * 1.0e-15
    lineshape = fftpack.dct(acf, type=1)[1:]
    omega = np.linspace(0, 0.5 / timestep_s, acf.size)[1:]
    freqs_cm1 = omega / (100.0 * LIGHTSPEED)
    field_description = omega * (
        1.0 - np.exp(-REDUCED_PLANCK * omega / (BOLTZ * temperature_k))
    )
    quantum_correction = omega / (
        1.0 - np.exp(-REDUCED_PLANCK * omega / (BOLTZ * temperature_k))
    )
    spectrum = lineshape * field_description * quantum_correction
    return freqs_cm1, spectrum


def _load_dipole_window(path: Path, window_idx: int) -> tuple[np.ndarray, np.ndarray] | None:
    data = np.load(path)
    times_key = f"window_{window_idx}_times_ps"
    dipole_key = f"window_{window_idx}_dipole_nm"
    if times_key not in data or dipole_key not in data:
        return None
    return (
        np.asarray(data[times_key], dtype=float),
        np.asarray(data[dipole_key], dtype=float),
    )


def ensemble_spectrum(
    lam: float,
    window_idx: int,
    replica_end: int,
) -> tuple[np.ndarray, np.ndarray] | None:
    """Ensemble-average IR spectrum for one lambda and dipole window."""
    job_dir = job_dir_path(lam)
    specs: list[np.ndarray] = []
    freqs_ref: np.ndarray | None = None
    for replica in range(replica_end):
        path = dipole_path(job_dir, lam, replica)
        if path is None:
            continue
        loaded = _load_dipole_window(path, window_idx)
        if loaded is None:
            continue
        times, dipole = loaded
        freqs, spec = ir_spectrum_from_dipole(dipole, times)
        if freqs.size == 0:
            continue
        if freqs_ref is None:
            freqs_ref = freqs
        elif freqs_ref.shape != freqs.shape:
            continue
        specs.append(spec)
    if freqs_ref is None or not specs:
        return None
    return freqs_ref, np.mean(np.stack(specs, axis=0), axis=0)


def plot_ir_on_axes(
    axes: list[plt.Axes],
    lambdas: list[float],
    *,
    window_idx: int = 1,
    replica_end: int = IR_SUBSET_REPLICAS,
    min_freq_cm1: float = 1200.0,
    max_freq_cm1: float = 2200.0,
) -> None:
    plot_lams = sorted(set(lambdas))
    for ax, lam in zip(axes, plot_lams):
        result = ensemble_spectrum(lam, window_idx, replica_end)
        if result is None:
            ax.set_title(f"$\\lambda$={lam:g}")
            continue
        freqs, spec = result
        mask = (freqs >= min_freq_cm1) & (freqs <= max_freq_cm1)
        if not np.any(mask):
            ax.set_title(f"$\\lambda$={lam:g}")
            continue
        peak = max(spec[mask].max(), 1e-30)
        ax.plot(freqs[mask], spec[mask] / peak, color="C0", lw=1.5)
        ax.axvline(FREQUENCY_CM1, color="gray", ls="--", lw=1.0)
        ax.set_title(f"$\\lambda$={lam:g}")
        ax.set_xlim(min_freq_cm1, max_freq_cm1)
        ax.set_ylim(0.0, 1.05)
        ax.set_xlabel(r"frequency (cm$^{-1}$)")
        if lam == plot_lams[0]:
            ax.set_ylabel(r"$n(\omega)\alpha(\omega)$ (norm.)")
        style_axes(ax)


def plot_ir_spectra_fig(
    lambdas: list[float] | None = None,
    *,
    window_idx: int = 1,
    replica_end: int = IR_SUBSET_REPLICAS,
    min_freq_cm1: float = 1200.0,
    max_freq_cm1: float = 2200.0,
) -> plt.Figure:
    """Paper-style Fig 2a top: one panel per lambda with polariton splitting."""
    plot_lams = sorted(set(lambdas or ANALYSIS_LAMBDAS))
    ncols = max(len(plot_lams), 1)
    fig, axes = plt.subplots(1, ncols, figsize=(3.2 * ncols, 3.2), sharey=True)
    if ncols == 1:
        axes = [axes]
    plot_ir_on_axes(
        list(axes),
        plot_lams,
        window_idx=window_idx,
        replica_end=replica_end,
        min_freq_cm1=min_freq_cm1,
        max_freq_cm1=max_freq_cm1,
    )
    win_start, win_len = IR_WINDOWS[window_idx]
    fig.suptitle(
        f"IR spectra ($x,y$; late window t={win_start:g}-{win_start + win_len:g} ps, N={replica_end})",
        y=1.02,
        fontsize=10,
    )
    fig.tight_layout()
    return fig


def main() -> None:
    apply_paper_style()
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-dir", type=Path, default=FIGURES_DIR)
    parser.add_argument("--lambdas", type=float, nargs="+", default=ANALYSIS_LAMBDAS)
    parser.add_argument("--replica-end", type=int, default=IR_SUBSET_REPLICAS)
    parser.add_argument(
        "--window-idx",
        type=int,
        default=1,
        help="Dipole window index (default 1 = late-aged window)",
    )
    parser.add_argument("--min-freq-cm1", type=float, default=1200.0)
    parser.add_argument("--max-freq-cm1", type=float, default=2200.0)
    args = parser.parse_args()

    args.output_dir.mkdir(parents=True, exist_ok=True)
    fig = plot_ir_spectra_fig(
        args.lambdas,
        window_idx=args.window_idx,
        replica_end=args.replica_end,
        min_freq_cm1=args.min_freq_cm1,
        max_freq_cm1=args.max_freq_cm1,
    )
    out_path = args.output_dir / "fig2a_ir_spectra"
    save_figure(fig, out_path)
    plt.close(fig)


if __name__ == "__main__":
    main()
