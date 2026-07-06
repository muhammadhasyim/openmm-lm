"""Tests for max-metric adaptive integration in run_cavity_equilibrium.

Adaptive timestepping uses cav-hoomd's max-force metric via
``openmm.cavitymd.adaptive``: plain Verlet with externally set dt from
N·max(|F|/m), epsilon calibrated to dt_max, and shock ramp at lambda turn-on.

Pure-function tests run without a GPU. Integration smoke tests use CUDA when
available, else Reference; skipped if the cavity-md plugin is unavailable.
"""

from __future__ import annotations

import math
import os
import sys
from pathlib import Path
from unittest.mock import MagicMock

import numpy as np
import pytest

C2F_ROOT = Path(__file__).resolve().parents[1]
if str(C2F_ROOT) not in sys.path:
    sys.path.insert(0, str(C2F_ROOT))

import run_cavity_equilibrium as rce  # noqa: E402


# --------------------------------------------------------------------------
# Pure-function tests: ramp anchor and max-metric helpers
# --------------------------------------------------------------------------


def test_ramp_start_none_when_lambda_zero() -> None:
    """A pure lambda=0 equilibration has no turn-on edge: tolerance stays relaxed."""
    assert rce._adaptive_ramp_start_ps(coupling_start_ps=0.0, lambda_coupling=0.0) is None
    assert rce._adaptive_ramp_start_ps(coupling_start_ps=100.0, lambda_coupling=0.0) is None


def test_ramp_start_anchors_at_delayed_turn_on() -> None:
    """With a delayed turn-on, the ramp anchors exactly at coupling_start_ps."""
    assert rce._adaptive_ramp_start_ps(
        coupling_start_ps=100.0, lambda_coupling=0.03
    ) == pytest.approx(100.0)


def test_ramp_start_anchors_at_zero_for_immediate_coupling() -> None:
    """lambda>0 from t=0 anchors the ramp at t=0 (strict tolerance at start)."""
    assert rce._adaptive_ramp_start_ps(
        coupling_start_ps=0.0, lambda_coupling=0.03
    ) == pytest.approx(0.0)
    assert rce._adaptive_ramp_start_ps(
        coupling_start_ps=-5.0, lambda_coupling=0.03
    ) == pytest.approx(0.0)


def test_epsilon_ramp_fraction_at_edge_and_relaxes() -> None:
    from openmm.cavitymd.adaptive import F0, TAU_RAMP_PS, epsilon_ramp_fraction

    ramp = 100.0
    assert epsilon_ramp_fraction(50.0, ramp) == pytest.approx(1.0)
    assert epsilon_ramp_fraction(100.0, ramp) == pytest.approx(F0)
    far = epsilon_ramp_fraction(100.0 + 10.0 * TAU_RAMP_PS, ramp)
    assert far > F0
    assert far == pytest.approx(1.0, rel=1e-3)


def test_step_coupling_on_delayed_turn_on() -> None:
    from openmm.cavitymd.adaptive import step_coupling_on, step_coupling_value

    assert not step_coupling_on(199.9, 200.0, 0.03)
    assert step_coupling_on(200.0, 200.0, 0.03)
    assert step_coupling_value(199.0, 200.0, 0.03) == pytest.approx(0.0)
    assert step_coupling_value(201.0, 200.0, 0.03) == pytest.approx(0.03)


def test_create_adaptive_state_before_turn_on() -> None:
    from openmm.cavitymd.adaptive import create_adaptive_state

    state = create_adaptive_state(0.03, 200.0, initial_time_ps=0.0, eps_relaxed=1.0)
    assert state["ramp_t0"] is None
    assert state["prev_lambda_on"] is False
    assert state["last_lambda"] == pytest.approx(0.0)
    assert state["eps_relaxed"] == pytest.approx(1.0)


def test_compute_force_max_norm_known_values() -> None:
    from openmm.cavitymd.adaptive import compute_force_max_norm

    masses = [1.0, 2.0, 4.0]
    forces = np.array([[3.0, 0.0, 0.0], [0.0, 4.0, 0.0], [0.0, 0.0, 8.0]])
    # per-mass: 3, 2, 2 -> max=3, N=3 -> 9
    assert compute_force_max_norm(masses, forces) == pytest.approx(9.0)


def test_cavhoomd_optimal_dt_clamps() -> None:
    from openmm.cavitymd.adaptive import (
        DT_MAX_PS,
        DT_MIN_PS,
        cavhoomd_optimal_dt,
    )

    eps = 1e-6
    fmn = 1.0
    dt = cavhoomd_optimal_dt(fmn, eps)
    assert dt == pytest.approx(math.sqrt(eps))

    assert cavhoomd_optimal_dt(fmn, eps, dt_max_ps=1e-4) == pytest.approx(1e-4)
    assert cavhoomd_optimal_dt(fmn, 1e-20, dt_min_ps=DT_MIN_PS) == pytest.approx(DT_MIN_PS)
    assert cavhoomd_optimal_dt(0.0, eps) == pytest.approx(DT_MAX_PS)


def test_calibrate_epsilon_gives_target_dt() -> None:
    from openmm.cavitymd.adaptive import (
        TARGET_DT_PS,
        calibrate_epsilon,
        cavhoomd_optimal_dt,
        compute_force_max_norm,
    )

    masses = [12.0, 12.0]
    forces = np.array([[10.0, 0.0, 0.0], [0.0, 5.0, 0.0]])
    fmn = compute_force_max_norm(masses, forces)
    eps = TARGET_DT_PS**2 * fmn
    dt = cavhoomd_optimal_dt(fmn, eps, dt_max_ps=TARGET_DT_PS)
    assert dt == pytest.approx(TARGET_DT_PS)

    context = MagicMock()
    system = MagicMock()
    system.getNumParticles.return_value = 2
    system.getParticleMass.side_effect = lambda i: MagicMock(
        value_in_unit=lambda u: masses[i]
    )
    state_mock = MagicMock()
    state_mock.getForces.return_value.value_in_unit.return_value = forces
    context.getState.return_value = state_mock

    eps_cal, fmn_cal = calibrate_epsilon(context, system, target_dt_ps=TARGET_DT_PS)
    assert fmn_cal == pytest.approx(fmn)
    assert eps_cal == pytest.approx(eps)


def test_effective_epsilon_scaled_shock_and_ramp() -> None:
    from openmm.cavitymd.adaptive import (
        F0,
        create_adaptive_state,
        effective_epsilon_scaled,
    )

    eps_relaxed = 2.0
    state = create_adaptive_state(0.03, 200.0, eps_relaxed=eps_relaxed)
    assert effective_epsilon_scaled(
        100.0, state, 200.0, 0.03, current_dt_ps=0.001
    ) == pytest.approx(eps_relaxed)
    eps_near = effective_epsilon_scaled(
        199.999, state, 200.0, 0.03, current_dt_ps=0.001
    )
    assert eps_near == pytest.approx(eps_relaxed * F0)

    state_shock = {"ramp_t0": 200.0, "eps_relaxed": eps_relaxed}
    assert effective_epsilon_scaled(
        200.0, state_shock, 200.0, 0.03, current_dt_ps=1e-6
    ) == pytest.approx(eps_relaxed * F0)
    later = effective_epsilon_scaled(
        210.0, state_shock, 200.0, 0.03, current_dt_ps=0.001
    )
    assert later > eps_relaxed * F0
    assert later < eps_relaxed


def test_effective_epsilon_tolerance_legacy() -> None:
    from openmm.cavitymd.adaptive import (
        EPS_STAR_NM,
        F0,
        create_adaptive_state,
        effective_epsilon_tolerance,
        shock_epsilon_tolerance,
    )

    state = create_adaptive_state(0.03, 200.0)
    assert effective_epsilon_tolerance(
        100.0, state, 200.0, 0.03, current_dt_ps=0.001
    ) == pytest.approx(EPS_STAR_NM)
    eps_near = effective_epsilon_tolerance(
        199.999, state, 200.0, 0.03, current_dt_ps=0.001
    )
    assert eps_near == pytest.approx(shock_epsilon_tolerance())
    assert eps_near == pytest.approx(EPS_STAR_NM * F0)


# --------------------------------------------------------------------------
# Integration smoke tests
# --------------------------------------------------------------------------


def _reference_cavity_available() -> bool:
    plugin_dir = os.environ.get("OPENMM_PLUGIN_DIR")
    if plugin_dir is None:
        candidate = C2F_ROOT.parents[2] / ".pixi" / "envs" / "test" / "lib" / "plugins"
        if candidate.is_dir():
            os.environ["OPENMM_PLUGIN_DIR"] = str(candidate)
    try:
        import openmm  # noqa: F401

        openmm.Platform.getPlatformByName("Reference")
        return hasattr(openmm, "CavityForce")
    except Exception:
        return False


pytestmark_integration = pytest.mark.skipif(
    not _reference_cavity_available(),
    reason="cavity-md OpenMM plugin / Reference platform unavailable",
)


@pytestmark_integration
def test_adaptive_equilibrium_runs_and_stays_finite(tmp_path: Path) -> None:
    """A short adaptive lambda=0 run completes with finite, physical energies."""
    prefix = tmp_path / "adapt_lam0"
    rce.run_cavity_equilibrium(
        temperature_K=100.0,
        runtime_ps=2.0,
        lambda_coupling=0.0,
        include_dipole_self_energy=True,
        output_prefix=str(prefix),
        seed=7,
        sample_interval_ps=1.0,
        finite_q=False,
        platform_name="Reference",
        adaptive=True,
        num_molecules=250,
    )
    final_state = Path(f"{prefix}_final_state.npz")
    assert final_state.exists()

    csv = np.genfromtxt(f"{prefix}_energies.csv", delimiter=",", names=True)
    T_kin = np.atleast_1d(csv["T_kinetic_K"])
    E_pot = np.atleast_1d(csv["E_potential_kjmol"])
    assert np.all(np.isfinite(T_kin))
    assert np.all(np.isfinite(E_pot))
    assert np.max(T_kin) < 1.0e4


@pytestmark_integration
def test_adaptive_turn_on_is_stable(tmp_path: Path) -> None:
    """An instantaneous lambda turn-on under adaptive dt does not explode."""
    prefix = tmp_path / "adapt_turnon"
    rce.run_cavity_equilibrium(
        temperature_K=100.0,
        runtime_ps=3.0,
        lambda_coupling=0.03,
        include_dipole_self_energy=True,
        output_prefix=str(prefix),
        seed=11,
        sample_interval_ps=1.0,
        finite_q=False,
        platform_name="Reference",
        coupling_start_ps=1.0,
        adaptive=True,
        num_molecules=250,
    )
    csv = np.genfromtxt(f"{prefix}_energies.csv", delimiter=",", names=True)
    T_kin = np.atleast_1d(csv["T_kinetic_K"])
    assert np.all(np.isfinite(T_kin))
    assert np.max(T_kin) < 1.0e4


def _cuda_cavity_available() -> bool:
    plugin_dir = os.environ.get("OPENMM_PLUGIN_DIR")
    if plugin_dir is None:
        candidate = C2F_ROOT.parents[2] / ".pixi" / "envs" / "test" / "lib" / "plugins"
        if candidate.is_dir():
            os.environ["OPENMM_PLUGIN_DIR"] = str(candidate)
    try:
        import openmm

        openmm.Platform.getPlatformByName("CUDA")
        return hasattr(openmm, "CavityForce")
    except Exception:
        return False


def _integration_platform() -> str:
    return "CUDA" if _cuda_cavity_available() else "Reference"


@pytestmark_integration
def test_adaptive_lam003_turn_on_200ps_stable(tmp_path: Path) -> None:
    """Reproduce aging campaign turn-on; must stay stable through 250 ps."""
    prefix = tmp_path / "adapt_lam003_200"
    ic = C2F_ROOT / "equilibrium_output" / "eq10ns100K_lam0_final_state.npz"
    if not ic.is_file():
        pytest.skip(f"Missing IC: {ic}")

    rce.run_cavity_equilibrium(
        temperature_K=100.0,
        runtime_ps=250.0,
        lambda_coupling=0.03,
        include_dipole_self_energy=True,
        output_prefix=str(prefix),
        seed=89,
        sample_interval_ps=25.0,
        finite_q=False,
        platform_name=_integration_platform(),
        coupling_start_ps=200.0,
        adaptive=True,
        initial_state=ic,
        resample_velocities=True,
        num_molecules=250,
        enable_fkt=False,
    )
    csv = np.genfromtxt(f"{prefix}_energies.csv", delimiter=",", names=True)
    T_kin = np.atleast_1d(csv["T_kinetic_K"])
    assert np.all(np.isfinite(T_kin))
    assert np.max(T_kin) < 500.0


@pytestmark_integration
def test_adaptive_seed89_regression_past_turnon(tmp_path: Path) -> None:
    """Known-bad seed 89 must not blow up before 250 ps after the fix."""
    prefix = tmp_path / "adapt_seed89"
    ic = C2F_ROOT / "equilibrium_output" / "eq10ns100K_lam0_final_state.npz"
    if not ic.is_file():
        pytest.skip(f"Missing IC: {ic}")

    rce.run_cavity_equilibrium(
        temperature_K=100.0,
        runtime_ps=250.0,
        lambda_coupling=0.03,
        include_dipole_self_energy=True,
        output_prefix=str(prefix),
        seed=89,
        sample_interval_ps=25.0,
        finite_q=False,
        platform_name=_integration_platform(),
        coupling_start_ps=200.0,
        adaptive=True,
        initial_state=ic,
        resample_velocities=True,
        num_molecules=250,
        enable_fkt=False,
    )
    csv = np.genfromtxt(f"{prefix}_energies.csv", delimiter=",", names=True)
    T_kin = np.atleast_1d(csv["T_kinetic_K"])
    assert np.all(np.isfinite(T_kin))
    assert np.max(T_kin) < 500.0


@pytestmark_integration
def test_adaptive_seed363_regression_past_turnon(tmp_path: Path) -> None:
    """Seed 363 blew up at t=199 ps with lambda=0.03 under old adaptive integrator."""
    prefix = tmp_path / "adapt_seed363"
    ic = C2F_ROOT / "equilibrium_output" / "eq10ns100K_lam0_final_state.npz"
    if not ic.is_file():
        pytest.skip(f"Missing IC: {ic}")

    rce.run_cavity_equilibrium(
        temperature_K=100.0,
        runtime_ps=250.0,
        lambda_coupling=0.03,
        include_dipole_self_energy=True,
        output_prefix=str(prefix),
        seed=363,
        sample_interval_ps=25.0,
        finite_q=False,
        platform_name=_integration_platform(),
        coupling_start_ps=200.0,
        adaptive=True,
        initial_state=ic,
        resample_velocities=True,
        num_molecules=250,
        enable_fkt=False,
    )
    csv = np.genfromtxt(f"{prefix}_energies.csv", delimiter=",", names=True)
    T_kin = np.atleast_1d(csv["T_kinetic_K"])
    assert np.all(np.isfinite(T_kin))
    assert np.max(T_kin) < 500.0


@pytestmark_integration
def test_dipole_sampling_does_not_mutate_velocities(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Dipole IR sampling must not strip COM velocity during adaptive integration."""
    com_calls: list[float] = []

    def _track_com(*_args, **_kwargs) -> None:
        com_calls.append(1.0)

    monkeypatch.setattr(rce, "remove_molecular_com_velocity", _track_com)

    prefix = tmp_path / "adapt_dipole_no_com"
    ic = C2F_ROOT / "equilibrium_output" / "eq10ns100K_lam0_final_state.npz"
    if not ic.is_file():
        pytest.skip(f"Missing IC: {ic}")

    rce.run_cavity_equilibrium(
        temperature_K=100.0,
        runtime_ps=155.0,
        lambda_coupling=0.0,
        include_dipole_self_energy=True,
        output_prefix=str(prefix),
        seed=42,
        sample_interval_ps=1.0,
        finite_q=False,
        platform_name=_integration_platform(),
        coupling_start_ps=200.0,
        adaptive=True,
        initial_state=ic,
        resample_velocities=True,
        num_molecules=250,
        enable_fkt=False,
        dipole_windows=[(150.0, 50.0)],
        dipole_interval_ps=0.001,
    )
    # One call at velocity initialization; none during dipole/FKT observers.
    assert len(com_calls) <= 1


@pytestmark_integration
def test_adaptive_seed42_lambda0_past_dipole_window(tmp_path: Path) -> None:
    """Pilot replica 0: seed 42, lambda=0, IR dipole window must stay stable past 200 ps."""
    prefix = tmp_path / "adapt_seed42_lam0"
    ic = C2F_ROOT / "equilibrium_output" / "eq10ns100K_lam0_final_state.npz"
    if not ic.is_file():
        pytest.skip(f"Missing IC: {ic}")

    rce.run_cavity_equilibrium(
        temperature_K=100.0,
        runtime_ps=250.0,
        lambda_coupling=0.0,
        include_dipole_self_energy=True,
        output_prefix=str(prefix),
        seed=42,
        sample_interval_ps=1.0,
        finite_q=False,
        platform_name=_integration_platform(),
        coupling_start_ps=200.0,
        adaptive=True,
        initial_state=ic,
        resample_velocities=True,
        num_molecules=250,
        enable_fkt=True,
        dipole_windows=[(150.0, 50.0), (2450.0, 50.0)],
        dipole_interval_ps=0.001,
    )
    csv = np.genfromtxt(f"{prefix}_energies.csv", delimiter=",", names=True)
    T_kin = np.atleast_1d(csv["T_kinetic_K"])
    assert np.all(np.isfinite(T_kin))
    assert np.max(T_kin) < 500.0
    times = np.atleast_1d(csv["time_ps"])
    assert float(times[-1]) >= 200.0


@pytestmark_integration
@pytest.mark.skipif(not _cuda_cavity_available(), reason="CUDA required for lam01 stress")
def test_adaptive_lam01_stress(tmp_path: Path) -> None:
    """lambda=0.1 stress test (cav-hoomd parity)."""
    prefix = tmp_path / "adapt_lam01"
    ic = C2F_ROOT / "equilibrium_output" / "eq10ns100K_lam0_final_state.npz"
    if not ic.is_file():
        pytest.skip(f"Missing IC: {ic}")

    rce.run_cavity_equilibrium(
        temperature_K=100.0,
        runtime_ps=500.0,
        lambda_coupling=0.1,
        include_dipole_self_energy=True,
        output_prefix=str(prefix),
        seed=42,
        sample_interval_ps=50.0,
        finite_q=False,
        platform_name="CUDA",
        coupling_start_ps=200.0,
        adaptive=True,
        initial_state=ic,
        resample_velocities=True,
        num_molecules=250,
        enable_fkt=False,
    )
    csv = np.genfromtxt(f"{prefix}_energies.csv", delimiter=",", names=True)
    T_kin = np.atleast_1d(csv["T_kinetic_K"])
    assert np.all(np.isfinite(T_kin))
    assert np.max(T_kin) < 500.0
