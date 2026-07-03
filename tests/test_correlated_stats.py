"""Tests for correlated time-series error estimation (block reblocking)."""

from __future__ import annotations

import importlib.util
import sys
from pathlib import Path

import numpy as np
import pytest

_MODULE = (
    Path(__file__).resolve().parents[1]
    / "wrappers/python/openmm/cavitymd/correlated_stats.py"
)
_spec = importlib.util.spec_from_file_location("correlated_stats", _MODULE)
assert _spec and _spec.loader
_mod = importlib.util.module_from_spec(_spec)
sys.modules[_spec.name] = _mod
_spec.loader.exec_module(_mod)
summarize_correlated = _mod.summarize_correlated


def _ar1_series(n: int, phi: float, rng: np.random.Generator) -> np.ndarray:
    """AR(1) process with innovation variance chosen for unit marginal variance."""
    noise_scale = np.sqrt(1.0 - phi**2)
    x = np.zeros(n)
    for i in range(1, n):
        x[i] = phi * x[i - 1] + noise_scale * rng.standard_normal()
    return x


def test_uncorrelated_gaussian_sem_near_naive():
    rng = np.random.default_rng(42)
    samples = rng.normal(loc=3.0, scale=2.0, size=2000)
    summary = summarize_correlated(samples)
    naive_sem = float(np.std(samples, ddof=1) / np.sqrt(len(samples)))
    assert summary.n_eff > 0.5 * len(samples)
    assert summary.sem == pytest.approx(naive_sem, rel=0.35)
    assert summary.mean == pytest.approx(3.0, abs=0.15)


def test_ar1_reblocked_sem_larger_than_naive():
    rng = np.random.default_rng(7)
    samples = _ar1_series(2000, phi=0.95, rng=rng) + 1.0
    summary = summarize_correlated(samples)
    assert summary.sem > summary.naive_sem * 2.0
    assert summary.n_eff < len(samples) / 5.0


def test_constant_series_zero_sem():
    samples = np.full(500, -0.42)
    summary = summarize_correlated(samples)
    assert summary.mean == pytest.approx(-0.42)
    assert summary.sem == pytest.approx(0.0, abs=1e-12)


def test_too_few_samples_raises():
    with pytest.raises(ValueError, match="at least 2"):
        summarize_correlated(np.array([1.0]))
