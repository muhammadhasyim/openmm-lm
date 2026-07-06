#!/usr/bin/env python3
"""Fig 2a top: IR from position snapshots and mKA partial charges."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

from config import FIGURES_DIR, FREQUENCY_CM1, LAMBDAS, N_REPLICAS, job_dir_path, run_prefix

_C2F_ROOT = Path(__file__).resolve().parent.parent
if str(_C2F_ROOT) not in sys.path:
    sys.path.insert(0, str(_C2F_ROOT))

from run_c2f import CHARGE_MAG, FRAC_AA, NUM_MOL, build_mka_system  # noqa: E402


def _molecular_charges() -> np.ndarray:
    _, _, n_atoms = build_mka_system(seed=42)
    n_aa = int(round(NUM_MOL * FRAC_AA))
    charges = np.zeros(n_atoms, dtype=float)
    for mol in range(NUM_MOL):
        sign = 1.0 if mol < n_aa else -1.0
        i0 = 2 * mol
        charges[i0] = sign * CHARGE_MAG
        charges[i0 + 1] = -sign * CHARGE_MAG
    return charges


def _dipole_trajectory(positions_nm: np.ndarray, charges: np.ndarray) -> np.ndarray:
    """positions_nm: (n_frames, n_atoms, 3). Returns (n_frames, 3)."""
    return np.sum(positions_nm * charges[None, :, None], axis=1)


def _ir_spectrum(dipole: np.ndarray, times_ps: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    dt_ps = float(np.mean(np.diff(times_ps)))
    if dt_ps <= 0 or dipole.shape[0] < 4:
        return np.array([]), np.array([])
    dipole = dipole - np.mean(dipole, axis=0, keepdims=True)
    acf = np.zeros(dipole.shape[0], dtype=float)
    for lag in range(dipole.shape[0]):
        acf[lag] = np.mean(np.sum(dipole[: dipole.shape[0] - lag] * dipole[lag:], axis=1))
    acf = acf / max(acf[0], 1e-30)
    spec = np.fft.rfft(acf).real
    freqs_cm1 = np.fft.rfftfreq(acf.size, d=dt_ps * 1e-12) / (2.99792458e10)  # ps -> s, Hz -> cm-1
    return freqs_cm1, spec


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-dir", type=Path, default=FIGURES_DIR)
    parser.add_argument("--lambdas", type=float, nargs="+", default=LAMBDAS)
    parser.add_argument("--replica", type=int, default=0)
    args = parser.parse_args()

    charges = _molecular_charges()
    args.output_dir.mkdir(parents=True, exist_ok=True)

    fig, ax = plt.subplots(figsize=(8, 5))
    colors = plt.cm.viridis(np.linspace(0.15, 0.9, len(args.lambdas)))

    for lam, color in zip(args.lambdas, colors):
        snap_path = job_dir_path(lam) / f"{run_prefix(lam, args.replica)}_snapshots.npz"
        if not snap_path.exists():
            continue
        data = np.load(snap_path)
        pos = np.asarray(data["positions_nm"], dtype=float)[..., : charges.size, :]
        times = np.asarray(data["times_ps"], dtype=float)
        dip = _dipole_trajectory(pos, charges)
        freqs, spec = _ir_spectrum(dip, times)
        if freqs.size == 0:
            continue
        # 10 ps frames limit Nyquist to ~1.7 cm^-1; use full resolved band for now.
        mask = (freqs > 0.0) & (freqs < freqs.max() * 0.99)
        if not np.any(mask):
            print(f"Skip lambda={lam:g}: no resolvable IR band (dt={np.mean(np.diff(times)):.1f} ps)")
            continue
        ax.plot(
            freqs[mask],
            spec[mask] / max(spec[mask].max(), 1e-30),
            color=color,
            label=f"$\\lambda$={lam:g}",
        )

    ax.axvline(FREQUENCY_CM1, color="gray", ls=":", lw=1.0, label=f"$\\omega_c$={FREQUENCY_CM1:.0f} cm$^{{-1}}$")
    ax.set_xlabel("frequency (cm$^{-1}$)")
    ax.set_ylabel("normalized IR intensity")
    ax.set_title("IR from snapshot dipole autocorrelation (weak coupling)")
    ax.legend(fontsize=8)
    fig.tight_layout()
    out_path = args.output_dir / "fig2a_ir_spectra.png"
    fig.savefig(out_path, dpi=150)
    plt.close(fig)
    print(f"Wrote {out_path}")


if __name__ == "__main__":
    main()
