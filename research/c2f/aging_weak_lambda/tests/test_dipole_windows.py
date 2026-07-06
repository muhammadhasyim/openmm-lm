"""Tests for dipole-window helpers used in IR production runs."""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pytest

_C2F_ROOT = Path(__file__).resolve().parents[2]
_CAMPAIGN_DIR = Path(__file__).resolve().parents[1]
for path in (_C2F_ROOT, _CAMPAIGN_DIR):
    if str(path) not in sys.path:
        sys.path.insert(0, str(path))

from run_cavity_equilibrium import (  # noqa: E402
    _dipole_nm,
    _molecular_charges,
    _time_in_dipole_window,
)
from config import IR_SUBSET_REPLICAS, IR_WINDOWS  # noqa: E402


def test_molecular_charges_sum_zero() -> None:
    charges = _molecular_charges(500)
    assert charges.shape == (500,)
    assert np.isclose(charges.sum(), 0.0)


def test_dipole_nm_known_geometry() -> None:
    charges = np.array([1.0, -1.0], dtype=float)
    positions = np.array([[0.0, 0.0, 0.0], [1.0, 0.0, 0.0]], dtype=float)
    dipole = _dipole_nm(positions, charges, n_atoms=2)
    assert np.allclose(dipole, [-1.0, 0.0, 0.0])


def test_time_in_dipole_window_boundaries() -> None:
    windows = [(150.0, 50.0), (2450.0, 50.0)]
    assert _time_in_dipole_window(150.0, windows) == 0
    assert _time_in_dipole_window(199.999, windows) == 0
    assert _time_in_dipole_window(200.0, windows) is None
    assert _time_in_dipole_window(2450.0, windows) == 1
    assert _time_in_dipole_window(100.0, windows) is None


def test_ir_subset_replicas_matches_window_plan() -> None:
    assert IR_SUBSET_REPLICAS == 10
    assert len(IR_WINDOWS) == 2


@pytest.mark.skipif(
    not Path(_CAMPAIGN_DIR.parent / "equilibrium_output" / "eq10ns100K_lam0_final_state.npz").exists(),
    reason="IC npz required for OpenMM smoke dipole test",
)
def test_smoke_run_writes_dipole_for_subset_replica() -> None:
    """Integration: replica 0 with smoke still records dipole when windows overlap runtime."""
    import subprocess

    result = subprocess.run(
        [
            sys.executable,
            str(_CAMPAIGN_DIR / "run_single.py"),
            "--lambda",
            "0.01",
            "--replica",
            "0",
            "--smoke",
        ],
        cwd=_CAMPAIGN_DIR,
        check=False,
        capture_output=True,
        text=True,
    )
    if result.returncode != 0 and "OpenMM" in (result.stdout + result.stderr):
        pytest.skip("OpenMM/cavity-md not available")
    assert result.returncode == 0, result.stderr
    dipole_path = _CAMPAIGN_DIR / "lambda0p01" / "lam0p01_seed0042_dipole.npz"
    assert dipole_path.exists(), "Smoke subset replica must write at least baseline dipole window"
    data = np.load(dipole_path)
    assert "window_0_times_ps" in data
    assert data["window_0_times_ps"].size > 0
