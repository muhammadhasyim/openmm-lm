"""Tests for N=10k campaign scaling constants."""

from __future__ import annotations

import sys
from pathlib import Path

import pytest

CAMPAIGN_DIR = Path(__file__).resolve().parents[1]
if str(CAMPAIGN_DIR) not in sys.path:
    sys.path.insert(0, str(CAMPAIGN_DIR))

from config import (  # noqa: E402
    N10K_IR_WINDOWS,
    N10K_LAMBDA_EFF,
    N10K_LAMBDA_REF,
    N10K_NUM_MOL,
    N10K_RUNTIME_PS,
    N10K_SWITCH_PS,
    REFERENCE_NUM_MOL,
    collective_coupling_g,
    job_dir_path,
    scale_lambda,
)


def test_n10k_lambda_scales_constant_g() -> None:
    g_ref = collective_coupling_g(N10K_LAMBDA_REF, REFERENCE_NUM_MOL)
    g_n10k = collective_coupling_g(N10K_LAMBDA_EFF, N10K_NUM_MOL)
    assert g_ref == pytest.approx(g_n10k)
    assert N10K_LAMBDA_EFF == pytest.approx(scale_lambda(N10K_LAMBDA_REF, N10K_NUM_MOL))
    assert N10K_LAMBDA_EFF == pytest.approx(0.00474341649025444)


def test_n10k_protocol_timing() -> None:
    assert N10K_SWITCH_PS == pytest.approx(200.0)
    assert N10K_RUNTIME_PS == pytest.approx(2200.0)
    assert N10K_IR_WINDOWS[1][0] == pytest.approx(N10K_RUNTIME_PS - 50.0)


def test_n10k_job_dir_isolated() -> None:
    from config import N10K_CAMPAIGN_DIR

    path = job_dir_path(0.03, campaign_root=N10K_CAMPAIGN_DIR)
    assert path.parts[-2:] == ("N10k", "lambda0p03")
