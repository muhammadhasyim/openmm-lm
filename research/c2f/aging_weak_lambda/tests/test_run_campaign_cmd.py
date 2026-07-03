"""Regression tests for run_campaign command construction."""

from __future__ import annotations

import sys
from pathlib import Path

import pytest

_CAMPAIGN_DIR = Path(__file__).resolve().parents[1]
_C2F_ROOT = _CAMPAIGN_DIR.parent
for path in (_C2F_ROOT, _CAMPAIGN_DIR):
    if str(path) not in sys.path:
        sys.path.insert(0, str(path))

from run_campaign import _run_one  # noqa: E402


def _dry_run_cmd(*, adaptive: bool, dt_ps: float = 0.0001, capsys: pytest.CaptureFixture[str]) -> list[str]:
    _run_one(
        lam=0.0,
        replica=0,
        runtime_ps=10.0,
        switch_time_ps=200.0,
        smoke=False,
        dry_run=True,
        adaptive=adaptive,
        dt_ps=dt_ps,
    )
    out = capsys.readouterr().out
    assert "DRY-RUN would execute:" in out
    line = out.strip().splitlines()[-1]
    return line.split("DRY-RUN would execute: ", 1)[1].split()


def test_run_one_fixed_dt_forwards_no_adaptive(capsys: pytest.CaptureFixture[str]) -> None:
    cmd = _dry_run_cmd(adaptive=False, dt_ps=0.0001, capsys=capsys)
    assert "--no-adaptive" in cmd
    assert "--adaptive" not in cmd
    dt_idx = cmd.index("--dt-ps")
    assert cmd[dt_idx + 1] == "0.0001"


def test_run_one_adaptive_forwards_adaptive_not_no_adaptive(
    capsys: pytest.CaptureFixture[str],
) -> None:
    cmd = _dry_run_cmd(adaptive=True, capsys=capsys)
    assert "--adaptive" in cmd
    assert "--no-adaptive" not in cmd
    assert "--dt-max-ps" in cmd
    assert "--dt-ps" not in cmd
