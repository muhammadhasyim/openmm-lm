#!/usr/bin/env python3
"""Fig 2a top: IR from fine-sampled dipole windows (up to 5000 cm^-1)."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

from config import (
    DIPOLE_INTERVAL_PS,
    FIGURES_DIR,
    FREQUENCY_CM1,
    IR_SUBSET_REPLICAS,
    IR_WINDOWS,
    LAMBDAS,
    job_dir_path,
    run_prefix,
)

_C_LIGHT_CM_PS = 2.99792458e10


def ir_spectrum_from_dipole(
    dipole: np.ndarray,
    times_ps: np.ndarray,
) -> tuple[np.ndarray, np.ndarray]:
    """Dipole autocorrelation FFT; times_ps in ps, output freqs in cm^-1."""
    dt_ps = float(np.mean(np.diff(times_ps)))
    if dt_ps <= 0 or dipole.shape[0] < 4:
        return np.array([]), np.array([])
    dipole = dipole - np.mean(dipole, axis=0, keepdims=True)
    n_frames = dipole.shape[0]
    acf = np.zeros(n_frames, dtype=float)
    for lag in range(n_frames):
        acf[lag] = np.mean(np.sum(dipole[: n_frames - lag] * dipole[lag:], axis=1))
    acf = acf / max(acf[0], 1e-30)
    spec = np.fft.rfft(acf).real
    freqs_cm1 = np.fft.rfftfreq(acf.size, d=dt_ps * 1e-12) / _C_LIGHT_CM_PS
    return freqs_cm1, spec


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


def _ensemble_spectrum(
    lam: float,
    window_idx: int,
    replica_end: int,
) -> tuple[np.ndarray, np.ndarray] | None:
    specs: list[np.ndarray] = []
    freqs_ref: np.ndarray | None = None
    for replica in range(replica_end):
        dipole_path = job_dir_path(lam) / f"{run_prefix(lam, replica)}_dipole.npz"
        if not dipole_path.exists():
            continue
        loaded = _load_dipole_window(dipole_path, window_idx)
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


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-dir", type=Path, default=FIGURES_DIR)
    parser.add_argument("--lambdas", type=float, nargs="+", default=LAMBDAS)
    parser.add_argument("--replica-end", type=int, default=IR_SUBSET_REPLICAS)
    parser.add_argument("--max-freq-cm1", type=float, default=5000.0)
    args = parser.parse_args()

    args.output_dir.mkdir(parents=True, exist_ok=True)
    window_labels = [f"t={start:g}-{start + length:g} ps" for start, length in IR_WINDOWS]
    colors = plt.cm.viridis(np.linspace(0.15, 0.9, len(args.lambdas)))

    fig, axes = plt.subplots(
        len(IR_WINDOWS),
        1,
        figsize=(8, 4 * len(IR_WINDOWS)),
        sharex=True,
        squeeze=False,
    )

    for win_idx, (ax, win_label) in enumerate(zip(axes[:, 0], window_labels)):
        for lam, color in zip(args.lambdas, colors):
            result = _ensemble_spectrum(lam, win_idx, args.replica_end)
            if result is None:
                continue
            freqs, spec = result
            mask = (freqs > 0.0) & (freqs <= args.max_freq_cm1)
            if not np.any(mask):
                continue
            peak = max(spec[mask].max(), 1e-30)
            ax.plot(
                freqs[mask],
                spec[mask] / peak,
                color=color,
                label=f"$\\lambda$={lam:g}",
            )
        ax.axvline(
            FREQUENCY_CM1,
            color="gray",
            ls=":",
            lw=1.0,
            label=f"$\\omega_c$={FREQUENCY_CM1:.0f} cm$^{{-1}}$",
        )
        ax.set_ylabel("normalized IR intensity")
        ax.set_title(f"IR dipole ACF ({win_label}, replicas 0–{args.replica_end - 1})")
        ax.legend(fontsize=8)

    axes[-1, 0].set_xlabel("frequency (cm$^{-1}$)")
    fig.suptitle(
        f"IR from dipole windows (dt={DIPOLE_INTERVAL_PS:g} ps, max {args.max_freq_cm1:g} cm$^{{-1}}$)",
        y=1.01,
    )
    fig.tight_layout()
    out_path = args.output_dir / "fig2a_ir_spectra.png"
    fig.savefig(out_path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"Wrote {out_path}")


if __name__ == "__main__":
    main()
