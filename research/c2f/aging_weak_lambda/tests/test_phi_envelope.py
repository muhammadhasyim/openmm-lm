"""Tests for ISF envelope averaging (|phi| and SEM)."""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np

_AGING_DIR = Path(__file__).resolve().parents[1]
if str(_AGING_DIR) not in sys.path:
    sys.path.insert(0, str(_AGING_DIR))

from fkt_utils import aggregate_replica_phi_stats  # noqa: E402


def test_aggregate_signed_vs_envelope_mean() -> None:
    lag_dict = {1.0: [0.5, -0.5, 0.2], 2.0: [0.4, -0.4]}
    _, signed_mean, _ = aggregate_replica_phi_stats(
        lag_dict, envelope=False, error="std"
    )
    _, envelope_mean, _ = aggregate_replica_phi_stats(
        lag_dict, envelope=True, error="std"
    )
    assert signed_mean[0] == np.mean([0.5, -0.5, 0.2])
    assert envelope_mean[0] == np.mean([0.5, 0.5, 0.2])
    assert envelope_mean[0] > signed_mean[0]


def test_aggregate_sem_scales_with_replica_count() -> None:
    lag_dict = {1.0: [0.0, 1.0, 2.0, 3.0]}
    _, _, std_err = aggregate_replica_phi_stats(lag_dict, envelope=True, error="std")
    _, _, sem_err = aggregate_replica_phi_stats(lag_dict, envelope=True, error="sem")
    assert sem_err[0] == std_err[0] / 2.0
