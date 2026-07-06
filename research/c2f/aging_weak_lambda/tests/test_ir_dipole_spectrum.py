"""Unit tests for IR spectrum from fine-sampled dipole trajectories."""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np

_CAMPAIGN_DIR = Path(__file__).resolve().parents[1]
if str(_CAMPAIGN_DIR) not in sys.path:
    sys.path.insert(0, str(_CAMPAIGN_DIR))

from analyze_ir_from_dipole import ir_spectrum_from_dipole  # noqa: E402

C_LIGHT_CM_PS = 2.99792458e10  # cm/s; lag in ps -> frequency in cm^-1


def _synthetic_dipole_cosine(
    freq_cm1: float,
    duration_ps: float,
    dt_ps: float,
    amplitude: float = 1.0,
) -> tuple[np.ndarray, np.ndarray]:
    """Dipole along x: mu(t) = A cos(2 pi nu t) with t in ps, nu in cm^-1."""
    times = np.arange(0.0, duration_ps, dt_ps, dtype=float)
    omega_rad_ps = 2.0 * np.pi * freq_cm1 * C_LIGHT_CM_PS * 1e-12
    dipole = np.zeros((times.size, 3), dtype=float)
    dipole[:, 0] = amplitude * np.cos(omega_rad_ps * times)
    return times, dipole


def test_ir_spectrum_resolves_1560_cm1_peak() -> None:
    dt_ps = 0.001
    times, dipole = _synthetic_dipole_cosine(1560.0, duration_ps=50.0, dt_ps=dt_ps)
    freqs, spec = ir_spectrum_from_dipole(dipole, times)
    assert freqs.size > 0
    nyquist = 0.5 / (dt_ps * 1e-12) / C_LIGHT_CM_PS
    assert nyquist > 5000.0
    idx_peak = int(np.argmax(spec[(freqs > 1000.0) & (freqs < 2000.0)]))
    freqs_band = freqs[(freqs > 1000.0) & (freqs < 2000.0)]
    assert abs(freqs_band[idx_peak] - 1560.0) < 15.0


def test_ir_spectrum_resolves_4800_cm1_peak() -> None:
    dt_ps = 0.001
    times, dipole = _synthetic_dipole_cosine(4800.0, duration_ps=50.0, dt_ps=dt_ps)
    freqs, spec = ir_spectrum_from_dipole(dipole, times)
    mask = (freqs > 4500.0) & (freqs < 5100.0)
    assert np.any(mask)
    idx_peak = int(np.argmax(spec[mask]))
    freqs_band = freqs[mask]
    assert abs(freqs_band[idx_peak] - 4800.0) < 30.0


def test_ir_spectrum_frequency_resolution_under_2_cm1() -> None:
    """50 ps window at 1 fs gives ~0.7 cm^-1 spacing; peaks within 2 cm^-1."""
    dt_ps = 0.001
    times, dipole = _synthetic_dipole_cosine(2000.0, duration_ps=50.0, dt_ps=dt_ps)
    freqs, spec = ir_spectrum_from_dipole(dipole, times)
    positive = freqs > 0.0
    spacing = float(np.min(np.diff(freqs[positive])))
    assert spacing < 2.0
    mask = (freqs > 1900.0) & (freqs < 2100.0)
    peak_freq = float(freqs[mask][int(np.argmax(spec[mask]))])
    assert abs(peak_freq - 2000.0) < 2.0
