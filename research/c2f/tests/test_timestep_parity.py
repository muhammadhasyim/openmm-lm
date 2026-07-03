"""Fast pure-Python parity tests: cav-hoomd AdaptiveTimestepUpdater vs OpenMM adaptive.py."""

from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np
import pytest

_C2F_ROOT = Path(__file__).resolve().parent.parent
if str(_C2F_ROOT) not in sys.path:
    sys.path.insert(0, str(_C2F_ROOT))

DIAG_DIR = _C2F_ROOT / "aging_weak_lambda" / "diagnose_fkt"

from timestep_parity_harness import (
    HoomdPolicyDriver,
    OpenmmPolicyDriver,
    build_parity_matrix,
    compare_dt_from_tolerance_shape,
    compare_tolerance_curves,
    step_turn_on_epsilon,
)

TARGET_ET = 1.0
INITIAL_FRACTION = 1e-5
SWITCH_PS = 200.0
LAMBDA = 0.03
OMEGAC_AU = 0.0071
EPSILON_ON = LAMBDA * OMEGAC_AU
STEADY_FORCE_MAX_NORM = 1.061221e7


def _run_both(
    *,
    time_ps_grid: np.ndarray,
    force_fn,
    epsilon_fn,
    error_tolerance: float = TARGET_ET,
    initial_fraction: float = INITIAL_FRACTION,
    omm_dt_max_ps: float = 0.001,
    hoomd_dt_max_fs: float = 10.0,
) -> tuple[HoomdPolicyDriver, OpenmmPolicyDriver]:
    hoomd_drv = HoomdPolicyDriver(
        error_tolerance=error_tolerance,
        initial_fraction=initial_fraction,
        switch_time_ps=SWITCH_PS,
        min_update_interval=1,
    )
    omm_drv = OpenmmPolicyDriver(
        error_tolerance=error_tolerance,
        initial_fraction=initial_fraction,
        coupling_start_ps=SWITCH_PS,
        lambda_coupling=LAMBDA,
        omegac_au=OMEGAC_AU,
        dt_max_ps=omm_dt_max_ps,
        min_update_interval=1,
    )
    hoomd_drv.set_force_fn(force_fn)
    hoomd_drv.set_epsilon_fn(epsilon_fn)
    omm_drv.set_force_fn(force_fn)
    omm_drv.set_epsilon_fn(epsilon_fn)
    hoomd_drv.run(time_ps_grid=time_ps_grid)
    omm_drv.run(time_ps_grid=time_ps_grid)
    return hoomd_drv, omm_drv


def test_pre_switch_steady_tolerance_constant() -> None:
    """Pre-switch: epsilon=0, constant target tolerance, no shock."""
    times = np.linspace(0.0, 199.0, 40)

    def epsilon_fn(t: float) -> float:
        return 0.0

    def force_fn(_t: float) -> float:
        return STEADY_FORCE_MAX_NORM

    hoomd_drv, omm_drv = _run_both(
        time_ps_grid=times,
        force_fn=force_fn,
        epsilon_fn=epsilon_fn,
    )
    compare_tolerance_curves(
        hoomd_drv.samples,
        omm_drv.samples,
        target_tolerance=TARGET_ET,
    )
    h_et = [s.error_tolerance for s in hoomd_drv.samples]
    assert np.allclose(h_et, TARGET_ET, rtol=1e-6)


def test_step_turn_on_shock_and_recovery() -> None:
    """Step turn-on at 200 ps: shock then exponential recovery."""
    times = np.array(
        [195.0, 199.0, 200.0, 201.0, 210.0, 250.0, 300.0],
        dtype=float,
    )

    def epsilon_fn(t: float) -> float:
        return step_turn_on_epsilon(
            t, switch_ps=SWITCH_PS, epsilon_on=EPSILON_ON
        )

    def force_fn(_t: float) -> float:
        return STEADY_FORCE_MAX_NORM

    hoomd_drv, omm_drv = _run_both(
        time_ps_grid=times,
        force_fn=force_fn,
        epsilon_fn=epsilon_fn,
    )
    compare_tolerance_curves(
        hoomd_drv.samples,
        omm_drv.samples,
        target_tolerance=TARGET_ET,
    )

    shock_h = hoomd_drv.samples[2]
    shock_o = omm_drv.samples[2]
    assert shock_h.error_tolerance == pytest.approx(TARGET_ET * INITIAL_FRACTION)
    assert shock_o.error_tolerance == pytest.approx(TARGET_ET * INITIAL_FRACTION)

    final_h = hoomd_drv.samples[-1]
    final_o = omm_drv.samples[-1]
    # tau=50 ps: full recovery needs t_shock + ~150 ps; t=300 is only 100 ps post-shock.
    assert final_h.error_tolerance == pytest.approx(TARGET_ET, rel=0.15)
    assert final_o.error_tolerance == pytest.approx(TARGET_ET, rel=0.15)


def test_dt_policy_shape_matches_after_turn_on() -> None:
    """dt ratios relative to each side's cap match across shock recovery."""
    times = np.linspace(195.0, 300.0, 22)

    def epsilon_fn(t: float) -> float:
        return step_turn_on_epsilon(
            t, switch_ps=SWITCH_PS, epsilon_on=EPSILON_ON
        )

    def force_fn(_t: float) -> float:
        return STEADY_FORCE_MAX_NORM

    hoomd_drv, omm_drv = _run_both(
        time_ps_grid=times,
        force_fn=force_fn,
        epsilon_fn=epsilon_fn,
        omm_dt_max_ps=0.001,
    )
    compare_dt_from_tolerance_shape(
        hoomd_drv.samples,
        omm_drv.samples,
    )


def test_force_spike_clamp_behavior() -> None:
    """Large force spike drives dt down on both sides (shape parity)."""
    times = np.array([100.0, 101.0, 102.0], dtype=float)
    spike_fmn = STEADY_FORCE_MAX_NORM * 1.0e6

    def epsilon_fn(_t: float) -> float:
        return 0.0

    def force_fn(t: float) -> float:
        return spike_fmn if t >= 101.0 else STEADY_FORCE_MAX_NORM

    hoomd_drv, omm_drv = _run_both(
        time_ps_grid=times,
        force_fn=force_fn,
        epsilon_fn=epsilon_fn,
    )
    assert hoomd_drv.samples[-1].dt_fs <= hoomd_drv.samples[0].dt_fs + 1e-9
    assert omm_drv.samples[-1].dt_fs <= omm_drv.samples[0].dt_fs + 1e-9
    compare_dt_from_tolerance_shape(
        hoomd_drv.samples,
        omm_drv.samples,
        rtol=0.1,
    )


def test_aged_regime_representative_force() -> None:
    """Aged-regime forces (lambda=0.01 @752 ps) yield stable sub-cap dt."""
    aged_fmn = STEADY_FORCE_MAX_NORM * 1.05
    times = np.linspace(740.0, 760.0, 11)

    def epsilon_fn(t: float) -> float:
        return 0.01 * OMEGAC_AU if t >= SWITCH_PS else 0.0

    def force_fn(_t: float) -> float:
        return aged_fmn

    hoomd_drv, omm_drv = _run_both(
        time_ps_grid=times,
        force_fn=force_fn,
        epsilon_fn=epsilon_fn,
        omm_dt_max_ps=0.001,
    )
    compare_tolerance_curves(
        hoomd_drv.samples,
        omm_drv.samples,
        target_tolerance=TARGET_ET,
    )
    for sample in hoomd_drv.samples + omm_drv.samples:
        assert sample.dt_fs <= 10.0 + 1e-6
        assert sample.dt_fs > 0.0


def test_cav_hoomd_dt_slew_params_unused() -> None:
    """Document: cav-hoomd stores slew params but act() applies dt directly."""
    import importlib.util
    import sys
    import types
    from pathlib import Path

    repo = Path(__file__).resolve().parents[3]
    timestep_path = (
        repo / "third_party" / "cav-hoomd" / "src" / "cavitymd" / "simulation" / "timestep.py"
    )
    source = timestep_path.read_text(encoding="utf-8")
    act_start = source.index("    def act(self, timestep):")
    act_end = source.index("\n    @property", act_start)
    act_body = source[act_start:act_end]
    assert "timestep_change_threshold" not in act_body
    assert "max_timestep_change_factor" not in act_body


def test_parity_matrix_written() -> None:
    """Emit parity_matrix.json for documented gap audit."""
    DIAG_DIR.mkdir(parents=True, exist_ok=True)
    out = DIAG_DIR / "parity_matrix.json"
    matrix = build_parity_matrix()
    out.write_text(json.dumps(matrix, indent=2), encoding="utf-8")
    assert out.is_file()
    gap_ids = {g["id"] for g in matrix["gaps"]}
    assert gap_ids == set(range(1, 15))
