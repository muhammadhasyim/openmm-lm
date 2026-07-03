"""Pure-Python cav-hoomd vs OpenMM adaptive timestep policy parity harness.

Drives the real cav-hoomd ``AdaptiveTimestepUpdater.act()`` (via a minimal
``hoomd`` stub) against OpenMM ``adaptive.py`` policy functions on identical
synthetic inputs.  No MD integration or GPU required.
"""

from __future__ import annotations

import importlib.util
import sys
import types
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Callable

import numpy as np

_REPO_ROOT = Path(__file__).resolve().parents[2]
_CAV_HOOMD_SRC = _REPO_ROOT / "third_party" / "cav-hoomd" / "src"

# ---------------------------------------------------------------------------
# hoomd stub (must be installed before importing cavitymd.simulation.timestep)
# ---------------------------------------------------------------------------


def _install_hoomd_stub() -> None:
    if "hoomd" in sys.modules and hasattr(sys.modules["hoomd"], "version"):
        return

    hoomd = types.ModuleType("hoomd")
    custom = types.ModuleType("hoomd.custom")
    logging_mod = types.ModuleType("hoomd.logging")
    version_mod = types.ModuleType("hoomd.version")

    class Action:
        pass

    def log(func=None, **_kwargs):
        if func is None:
            return lambda f: f
        return func

    custom.Action = Action
    logging_mod.log = log
    version_mod.version = "5.2.0"
    hoomd.custom = custom
    hoomd.logging = logging_mod
    hoomd.version = version_mod
    sys.modules["hoomd"] = hoomd
    sys.modules["hoomd.custom"] = custom
    sys.modules["hoomd.logging"] = logging_mod
    sys.modules["hoomd.version"] = version_mod


def _load_module_from_path(name: str, path: Path):
    spec = importlib.util.spec_from_file_location(name, path)
    if spec is None or spec.loader is None:
        raise ImportError(f"Cannot load module {name} from {path}")
    mod = importlib.util.module_from_spec(spec)
    sys.modules[name] = mod
    spec.loader.exec_module(mod)
    return mod


def _load_physical_constants():
    _install_hoomd_stub()
    utils_path = _CAV_HOOMD_SRC / "cavitymd" / "utils.py"
    cavitymd = types.ModuleType("cavitymd")
    sys.modules["cavitymd"] = cavitymd
    utils_mod = _load_module_from_path("cavitymd.utils", utils_path)
    cavitymd.utils = utils_mod
    return utils_mod.PhysicalConstants


def _load_adaptive_timestep_updater():
    _install_hoomd_stub()
    utils_path = _CAV_HOOMD_SRC / "cavitymd" / "utils.py"
    timestep_path = _CAV_HOOMD_SRC / "cavitymd" / "simulation" / "timestep.py"

    cavitymd = sys.modules.get("cavitymd")
    if cavitymd is None:
        cavitymd = types.ModuleType("cavitymd")
        sys.modules["cavitymd"] = cavitymd

    if "cavitymd.utils" not in sys.modules:
        utils_mod = _load_module_from_path("cavitymd.utils", utils_path)
        cavitymd.utils = utils_mod

    sim_pkg = types.ModuleType("cavitymd.simulation")
    sys.modules["cavitymd.simulation"] = sim_pkg
    cavitymd.simulation = sim_pkg

    timestep_mod = _load_module_from_path("cavitymd.simulation.timestep", timestep_path)
    sim_pkg.timestep = timestep_mod
    return timestep_mod.AdaptiveTimestepUpdater


PhysicalConstants = _load_physical_constants()
AdaptiveTimestepUpdater = _load_adaptive_timestep_updater()

TIME_PS_CONVERSION = PhysicalConstants.TIME_PS_CONVERSION

from openmm.cavitymd.adaptive import (  # noqa: E402
    COUPLING_CHANGE_THRESHOLD,
    DT_MAX_PS,
    DT_MIN_PS,
    FORCE_UPDATE_INTERVAL,
    TAU_RAMP_PS,
    AdaptiveParityConfig,
    cavhoomd_optimal_dt,
    cavhoomd_runtime_parity_config,
    coupling_epsilon_au,
    create_adaptive_state,
    effective_dt_max_ps,
    effective_epsilon_scaled,
    effective_force_update_interval,
    lambda_transition,
    step_coupling_on,
    step_coupling_value,
)

# ---------------------------------------------------------------------------
# Fake HOOMD interfaces for AdaptiveTimestepUpdater.act()
# ---------------------------------------------------------------------------


class _FakeForce:
    def __init__(self, n_particles: int, mass: float) -> None:
        self._n = n_particles
        self._mass = mass
        self.forces: np.ndarray | None = None

    def set_force_max_norm(self, force_max_norm: float) -> None:
        """Set forces so N * max(|F|/m) equals *force_max_norm*."""
        forces = np.zeros((self._n, 3), dtype=np.float64)
        if force_max_norm > 0.0:
            force_mag = force_max_norm * self._mass / self._n
            forces[0, 0] = force_mag
        self.forces = forces


class _FakeCavityForce:
    def __init__(self, epsilon_fn: Callable[[float], float], n_particles: int) -> None:
        self._epsilon_fn = epsilon_fn
        self.lambda_coupling_variant = self._variant
        self._elapsed_time_ps = 0.0
        self.forces = np.zeros((n_particles, 3), dtype=np.float64)

    def _variant(self, _timestep: int) -> float:
        return float(self._epsilon_fn(self._elapsed_time_ps))

    def set_elapsed_time_ps(self, time_ps: float) -> None:
        self._elapsed_time_ps = time_ps


class _FakeIntegrator:
    def __init__(
        self,
        n_particles: int,
        mass: float,
        initial_dt_au: float,
        epsilon_fn: Callable[[float], float],
    ) -> None:
        self.dt = initial_dt_au
        self._particle_force = _FakeForce(n_particles, mass)
        self._cavity_force = _FakeCavityForce(epsilon_fn, n_particles)
        self.forces = [self._particle_force, self._cavity_force]

    def set_elapsed_time_ps(self, time_ps: float) -> None:
        self._cavity_force.set_elapsed_time_ps(time_ps)

    def set_force_max_norm(self, force_max_norm: float) -> None:
        self._particle_force.set_force_max_norm(force_max_norm)


class _FakeSnapshot:
    def __init__(self, mass: float, n_particles: int) -> None:
        self.particles = types.SimpleNamespace(mass=np.full(n_particles, mass))


class _FakeState:
    def __init__(self, mass: float, n_particles: int) -> None:
        self._snap = _FakeSnapshot(mass, n_particles)

    def get_snapshot(self) -> _FakeSnapshot:
        return self._snap


class _FakeTimeTracker:
    def __init__(self) -> None:
        self.elapsed_time = 0.0


@dataclass
class PolicySample:
    """One recorded policy evaluation."""

    step: int
    time_ps: float
    error_tolerance: float
    dt_fs: float
    force_max_norm: float
    updated: bool


class HoomdPolicyDriver:
    """Drive real cav-hoomd ``AdaptiveTimestepUpdater.act()`` on synthetic inputs."""

    def __init__(
        self,
        *,
        error_tolerance: float = 1.0,
        initial_fraction: float = 1e-5,
        time_constant_ps: float = TAU_RAMP_PS,
        switch_time_ps: float | None = 200.0,
        n_particles: int = 251,
        mass: float = 1.0,
        min_update_interval: int = 1,
    ) -> None:
        self.n_particles = n_particles
        self._epsilon_fn: Callable[[float], float] = lambda _t: 0.0
        self._force_fn: Callable[[float], float] = lambda _t: 1.0e7
        self._time_tracker = _FakeTimeTracker()
        self._integrator = _FakeIntegrator(
            n_particles,
            mass,
            PhysicalConstants.fs_to_atomic_units(1.0),
            self._epsilon_at_time,
        )
        self._state = _FakeState(mass, n_particles)
        self.updater = AdaptiveTimestepUpdater(
            self._state,
            self._integrator,
            error_tolerance=error_tolerance,
            time_constant_ps=time_constant_ps,
            initial_fraction=initial_fraction,
            adaptiveerror=True,
            time_tracker=self._time_tracker,
            switch_time_ps=switch_time_ps,
            shock_dampening_factor=initial_fraction,
            shock_dampening_enabled=switch_time_ps is not None,
            shock_dampening_time_constant_ps=time_constant_ps,
            dynamic_coupling_detection=True,
            coupling_change_threshold=COUPLING_CHANGE_THRESHOLD,
        )
        self.updater.min_update_interval = min_update_interval
        self.samples: list[PolicySample] = []
        self._prev_dt_au = self._integrator.dt

    def _epsilon_at_time(self, time_ps: float) -> float:
        return float(self._epsilon_fn(time_ps))

    def set_epsilon_fn(self, fn: Callable[[float], float]) -> None:
        self._epsilon_fn = fn

    def set_force_fn(self, fn: Callable[[float], float]) -> None:
        self._force_fn = fn

    def run(
        self,
        *,
        time_ps_grid: np.ndarray,
        step_stride: int = 1,
    ) -> list[PolicySample]:
        """Evaluate policy at each time in *time_ps_grid* (monotone increasing)."""
        self.samples = []
        if len(time_ps_grid) == 0:
            return self.samples

        # Warm-start: avoid false switch crossing when grid starts after switch_time.
        t0 = float(time_ps_grid[0])
        self._time_tracker.elapsed_time = t0
        self._integrator.set_elapsed_time_ps(t0)
        self.updater.last_elapsed_time_ps = t0
        if self.updater.switch_time_ps is not None and t0 >= self.updater.switch_time_ps:
            self.updater.switch_detected = True

        step = 0
        for time_ps in time_ps_grid:
            step += step_stride
            self._time_tracker.elapsed_time = float(time_ps)
            self._integrator.set_elapsed_time_ps(float(time_ps))
            force_max_norm = float(self._force_fn(float(time_ps)))
            self._integrator.set_force_max_norm(force_max_norm)
            prev_dt = self._integrator.dt
            self.updater.act(step)
            updated = self._integrator.dt != prev_dt
            dt_fs = PhysicalConstants.atomic_units_to_ps(self._integrator.dt) * 1000.0
            self.samples.append(
                PolicySample(
                    step=step,
                    time_ps=float(time_ps),
                    error_tolerance=float(self.updater.error_tolerance),
                    dt_fs=dt_fs,
                    force_max_norm=force_max_norm,
                    updated=updated,
                )
            )
            self.updater.last_elapsed_time_ps = float(time_ps)
        return self.samples


class OpenmmPolicyDriver:
    """OpenMM adaptive policy driver mirroring ``advance_to_time_step_on`` updates."""

    def __init__(
        self,
        *,
        error_tolerance: float = 1.0,
        initial_fraction: float = 1e-5,
        coupling_start_ps: float = 200.0,
        lambda_coupling: float = 0.03,
        omegac_au: float = 0.0071,
        dt_max_ps: float = DT_MAX_PS,
        dt_min_ps: float = DT_MIN_PS,
        parity_config: AdaptiveParityConfig | None = None,
        min_update_interval: int = 1,
    ) -> None:
        cfg = parity_config or cavhoomd_runtime_parity_config(
            error_tolerance=error_tolerance,
            initial_fraction=initial_fraction,
        )
        self.coupling_start_ps = coupling_start_ps
        self.lambda_coupling = lambda_coupling
        self.omegac_au = omegac_au
        self.dt_max_ps = dt_max_ps
        self.dt_min_ps = dt_min_ps
        self.min_update_interval = min_update_interval
        self.state = create_adaptive_state(
            lambda_coupling,
            coupling_start_ps,
            initial_time_ps=0.0,
            eps_relaxed=error_tolerance,
            omegac_au=omegac_au,
            parity_config=cfg,
        )
        self._epsilon_fn: Callable[[float], float] = lambda _t: 0.0
        self._force_fn: Callable[[float], float] = lambda _t: 1.0e7
        self.current_dt_ps = dt_max_ps
        self.samples: list[PolicySample] = []
        self._step = 0

    def set_epsilon_fn(self, fn: Callable[[float], float]) -> None:
        self._epsilon_fn = fn

    def set_force_fn(self, fn: Callable[[float], float]) -> None:
        self._force_fn = fn

    def _register_shock(self, time_ps: float) -> None:
        self.state["ramp_t0"] = time_ps
        self.state["steps_since_dt_update"] = FORCE_UPDATE_INTERVAL
        self.current_dt_ps = min(self.current_dt_ps, DT_MIN_PS)

    def evaluate_at(self, time_ps: float) -> PolicySample:
        """Single policy evaluation at *time_ps*."""
        self._step += 1
        force_max_norm = float(self._force_fn(float(time_ps)))
        self.state["last_force_max_norm"] = force_max_norm

        curr_on = step_coupling_on(
            time_ps, self.coupling_start_ps, self.lambda_coupling
        )
        curr_lambda = step_coupling_value(
            time_ps, self.coupling_start_ps, self.lambda_coupling
        )
        curr_epsilon = float(self._epsilon_fn(float(time_ps)))

        force_update = False
        if lambda_transition(self.state["prev_lambda_on"], curr_on):
            self._register_shock(time_ps)
            force_update = True

        last_lambda = float(self.state.get("last_lambda", 0.0))
        if abs(curr_lambda - last_lambda) >= COUPLING_CHANGE_THRESHOLD:
            self._register_shock(time_ps)
            force_update = True

        last_epsilon = float(self.state.get("last_epsilon", 0.0))
        if abs(curr_epsilon - last_epsilon) >= COUPLING_CHANGE_THRESHOLD:
            self._register_shock(time_ps)
            force_update = True

        self.state["prev_lambda_on"] = curr_on
        self.state["last_lambda"] = curr_lambda
        self.state["last_epsilon"] = curr_epsilon

        steps_since = int(self.state.get("steps_since_dt_update", 0))
        update_interval = effective_force_update_interval(self.state, time_ps)
        if self.min_update_interval > 1:
            update_interval = max(update_interval, self.min_update_interval)

        prev_dt = self.current_dt_ps
        updated = False
        if force_update or steps_since >= update_interval:
            eps = effective_epsilon_scaled(
                time_ps,
                self.state,
                self.coupling_start_ps,
                self.lambda_coupling,
                self.current_dt_ps,
            )
            dt_cap = effective_dt_max_ps(time_ps, self.state, self.dt_max_ps)
            self.current_dt_ps = cavhoomd_optimal_dt(
                force_max_norm,
                eps,
                dt_min_ps=self.dt_min_ps,
                dt_max_ps=dt_cap,
            )
            self.state["steps_since_dt_update"] = 0
            updated = self.current_dt_ps != prev_dt
        else:
            self.state["steps_since_dt_update"] = steps_since + 1
            eps = effective_epsilon_scaled(
                time_ps,
                self.state,
                self.coupling_start_ps,
                self.lambda_coupling,
                self.current_dt_ps,
            )

        sample = PolicySample(
            step=self._step,
            time_ps=float(time_ps),
            error_tolerance=float(eps),
            dt_fs=self.current_dt_ps * 1000.0,
            force_max_norm=force_max_norm,
            updated=updated,
        )
        self.samples.append(sample)
        return sample

    def warm_start(self, time_ps: float) -> None:
        """Initialize state when the grid starts mid-trajectory."""
        if time_ps >= self.coupling_start_ps - 1e-15:
            eps = float(self._epsilon_fn(float(time_ps)))
            self.state["prev_lambda_on"] = True
            self.state["last_lambda"] = self.lambda_coupling
            self.state["last_epsilon"] = eps
            self.state["ramp_t0"] = None
        else:
            self.state["prev_lambda_on"] = False
            self.state["last_lambda"] = 0.0
            self.state["last_epsilon"] = 0.0
            self.state["ramp_t0"] = None

    def run(self, *, time_ps_grid: np.ndarray) -> list[PolicySample]:
        self.samples = []
        if len(time_ps_grid) == 0:
            return self.samples
        self.warm_start(float(time_ps_grid[0]))
        for time_ps in time_ps_grid:
            self.evaluate_at(float(time_ps))
        return self.samples


def hoomd_error_tolerance_at_time(
    driver: HoomdPolicyDriver, time_ps: float
) -> float:
    """Return cav-hoomd effective error tolerance at *time_ps* (after run)."""
    for sample in driver.samples:
        if abs(sample.time_ps - time_ps) < 1e-9:
            return sample.error_tolerance
    raise KeyError(f"No sample at t={time_ps} ps")


def compare_tolerance_curves(
    hoomd_samples: list[PolicySample],
    omm_samples: list[PolicySample],
    *,
    target_tolerance: float,
    rtol: float = 1e-4,
) -> None:
    """Assert normalized error-tolerance ramps match."""
    assert len(hoomd_samples) == len(omm_samples)
    h_norm = np.array([s.error_tolerance / target_tolerance for s in hoomd_samples])
    o_norm = np.array([s.error_tolerance / target_tolerance for s in omm_samples])
    np.testing.assert_allclose(h_norm, o_norm, rtol=rtol, atol=1e-8)


def compare_dt_from_tolerance_shape(
    hoomd_samples: list[PolicySample],
    omm_samples: list[PolicySample],
    *,
    rtol: float = 0.05,
) -> None:
    """dt ∝ sqrt(eps) at fixed force; compare relative sqrt(tolerance) curves."""
    assert len(hoomd_samples) == len(omm_samples)
    h_root = np.sqrt(np.array([max(s.error_tolerance, 0.0) for s in hoomd_samples]))
    o_root = np.sqrt(np.array([max(s.error_tolerance, 0.0) for s in omm_samples]))
    h0 = max(h_root[0], 1e-15)
    o0 = max(o_root[0], 1e-15)
    np.testing.assert_allclose(h_root / h0, o_root / o0, rtol=rtol, atol=1e-6)


def step_turn_on_epsilon(
    time_ps: float,
    *,
    switch_ps: float,
    epsilon_on: float,
) -> float:
    return epsilon_on if time_ps >= switch_ps - 1e-15 else 0.0


def build_parity_matrix() -> dict[str, Any]:
    """Document PASS/DEVIATION for each documented parity gap."""
    return {
        "gaps": [
            {
                "id": 1,
                "topic": "absolute_error_tolerance",
                "cav_hoomd": "target error_tolerance (configurable, e.g. 1.0)",
                "openmm": "eps_relaxed via absolute_error_tolerance in AdaptiveParityConfig",
                "status": "PASS",
                "note": (
                    "Policy-shape parity verified on identical numeric tolerance; "
                    "physical unit systems differ (a.u. vs kJ/mol/nm/amu) but "
                    "sqrt(tolerance/force_max_norm) ramp shape matches."
                ),
            },
            {
                "id": 2,
                "topic": "shock_factor",
                "cav_hoomd": "shock_dampening_factor = initial_fraction (1e-5)",
                "openmm": "f0 in AdaptiveParityConfig",
                "status": "PASS",
                "note": "Shock tolerance = target * f0 on both sides.",
            },
            {
                "id": 3,
                "topic": "pre_switch_guard",
                "cav_hoomd": "none",
                "openmm": "PRE_SWITCH_GUARD_PS (legacy); disabled in runtime parity",
                "status": "DEVIATION",
                "intentional": False,
                "fix": "pre_switch_guard_ps=0.0 in default_parity_config",
            },
            {
                "id": 4,
                "topic": "dt_max",
                "cav_hoomd": "10 fs",
                "openmm": "1 fs campaign cap",
                "status": "DEVIATION",
                "intentional": True,
                "note": "Campaign stability cap; compare dt ratios not absolute fs.",
            },
            {
                "id": 5,
                "topic": "dt_slew",
                "cav_hoomd": "params stored but unused in act()",
                "openmm": "dt_slew_threshold in _update_timestep_max_metric",
                "status": "DEVIATION",
                "intentional": False,
                "fix": "dt_slew_threshold=0.0 in default parity config",
            },
            {
                "id": 6,
                "topic": "shock_dt_cap",
                "cav_hoomd": "min dt 0.001 fs via clamp",
                "openmm": "DT_SHOCK_CAP_PS=1e-6 at shock registration",
                "status": "DEVIATION",
                "intentional": True,
                "note": "OpenMM uses stricter shock cap for campaign stability.",
            },
            {
                "id": 7,
                "topic": "thermostat_tau",
                "cav_hoomd": "5 ps default",
                "openmm": "BUSSI_TAU_PS=1.0",
                "status": "DEVIATION",
                "intentional": True,
                "note": "Outside timestep policy; not tested by harness.",
            },
            {
                "id": 8,
                "topic": "dipole_subchunking",
                "cav_hoomd": "none",
                "openmm": "1 fs sub-chunks in dipole windows",
                "status": "DEVIATION",
                "intentional": True,
                "note": "Observer scheduling; outside timestep policy harness.",
            },
            {
                "id": 9,
                "topic": "stability_check_cadence",
                "cav_hoomd": "every HOOMD step via updater",
                "openmm": "T_kin at CSV sample interval",
                "status": "DEVIATION",
                "intentional": True,
                "note": "Observer only; not timestep policy.",
            },
            {
                "id": 10,
                "topic": "dt_max_ramp_scaling",
                "cav_hoomd": "dt_max fixed at 10 fs during shock recovery",
                "openmm": "effective_dt_max_ps scaled by ramp fraction during shock",
                "status": "DEVIATION",
                "intentional": True,
                "note": (
                    "Production restores dt_max ramp for post-switch stability; "
                    "harness parity tests use cavhoomd_runtime_parity_config explicitly."
                ),
            },
            {
                "id": 11,
                "topic": "production_epsilon_calibration",
                "cav_hoomd": "configurable error_tolerance (often 0.01 a.u.)",
                "openmm": "default_parity_config uses force-calibrated eps_relaxed",
                "status": "DEVIATION",
                "intentional": True,
                "note": (
                    "OpenMM kJ/mol/nm units require sqrt(eps/F) calibration at t=0; "
                    "absolute_error_tolerance override kept for harness-only parity."
                ),
            },
            {
                "id": 12,
                "topic": "coupling_switch_substep",
                "cav_hoomd": "HOOMD integrator adapts within step",
                "openmm": "would_cross_coupling_switch splits Verlet steps at t0",
                "status": "DEVIATION",
                "intentional": True,
                "note": "Prevents first coupled step from spanning λ step discontinuity.",
            },
            {
                "id": 13,
                "topic": "hybrid_variable_verlet_safety",
                "cav_hoomd": "HOOMD integrator adapts per internal step",
                "openmm": "Optional VariableVerlet diagnostic mode only; production uses velocity-Verlet CustomIntegrator + max-force setStepSize",
                "status": "DEVIATION",
                "intentional": True,
                "note": (
                    "hybrid_safety_verlet default False (2026-07-02): VariableVerletIntegrator.step() "
                    "overwrites setStepSize, so max-force policy must not use VariableVerlet."
                ),
            },
            {
                "id": 14,
                "topic": "leapfrog_dt_churn_energy_injection",
                "cav_hoomd": "HOOMD velocity-Verlet (full-step state); setStepSize safe",
                "openmm": "create_velocity_verlet_integrator CustomIntegrator (2026-07-02); not leapfrog VerletIntegrator",
                "status": "DEVIATION",
                "intentional": True,
                "note": (
                    "Plain VerletIntegrator stores half-step velocities; external setStepSize() "
                    "during adaptive dt churn injects spurious KE (verify_leapfrog_dt_churn_injection.py)."
                ),
            },
        ],
        "failure_root_causes": {
            "lambda_0.03_at_105ps": {
                "verdict": "parallel_gpu_contamination",
                "evidence": (
                    "Serial t105_probe stable T_kin=97.3 K; SLURM jobs=5 on one GPU "
                    "is primary suspect, not lambda-dependent pre-switch physics."
                ),
            },
            "lambda_0.01_at_752ps": {
                "verdict": "policy_testable_without_md",
                "evidence": (
                    "Aged-force scenario in test_timestep_parity uses representative "
                    "force_max_norm; full 2500 ps gate deferred."
                ),
            },
            "post_switch_blowup_seed42": {
                "verdict": "adaptive_policy_regression",
                "evidence": (
                    "Serial λ=0.01/0.03 fail 19-26 ps after t=200 ps with "
                    "absolute_error_tolerance=1.0; June 12 pilot λ=0.01 seed 42 "
                    "completed 2500 ps; fixed via force calibration + switch substep."
                ),
            },
        },
    }
