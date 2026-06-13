"""Unit tests for COM removal in F(k,t) helpers."""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np

_C2F_ROOT = Path(__file__).resolve().parents[2]
if str(_C2F_ROOT) not in sys.path:
    sys.path.insert(0, str(_C2F_ROOT))

from fkt_tracker import compute_rhok, fibonacci_sphere, subtract_com_positions_nm  # noqa: E402


def _wavevectors_nm(kmag_nm_inv: float, num_wavevectors: int = 8) -> np.ndarray:
    return fibonacci_sphere(num_wavevectors) * kmag_nm_inv


def test_subtract_com_positions_nm_zeros_mean() -> None:
    positions = np.array(
        [
            [0.0, 0.0, 0.0],
            [2.0, 0.0, 0.0],
            [0.0, 2.0, 0.0],
            [2.0, 2.0, 0.0],
        ],
        dtype=np.float64,
    )
    centered = subtract_com_positions_nm(positions)
    assert np.allclose(centered.mean(axis=0), 0.0)


def test_compute_rhok_invariant_to_uniform_translation() -> None:
    rng = np.random.default_rng(0)
    positions = rng.normal(size=(20, 3))
    shift = np.array([1.7, -0.4, 2.1])
    wavevectors = _wavevectors_nm(19.0, num_wavevectors=8)
    rho0_r, rho0_i = compute_rhok(positions, wavevectors)
    rho1_r, rho1_i = compute_rhok(positions + shift, wavevectors)
    assert np.allclose(rho0_r, rho1_r)
    assert np.allclose(rho0_i, rho1_i)


def test_compute_rhok_without_com_removal_changes_under_translation() -> None:
    rng = np.random.default_rng(1)
    positions = rng.normal(size=(20, 3))
    shift = np.array([1.7, -0.4, 2.1])
    wavevectors = _wavevectors_nm(19.0, num_wavevectors=8)
    rho0_r, rho0_i = compute_rhok(positions, wavevectors, subtract_com=False)
    rho1_r, rho1_i = compute_rhok(positions + shift, wavevectors, subtract_com=False)
    assert not np.allclose(rho0_r, rho1_r) or not np.allclose(rho0_i, rho1_i)
