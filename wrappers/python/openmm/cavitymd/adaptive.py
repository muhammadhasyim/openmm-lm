"""SI-accurate adaptive timestepping helpers for C2F cavity-MD.

Metric story (three layers):

- **Paper SI** (integration theory): RMS displacement error with ``ε* = 5.0``
  (nm-scale tolerance) and shock ramp ``f₀ = 10⁻³``, ``τ* = 50`` ps.
- **cav-hoomd runtime** (``AdaptiveTimestepUpdater``): max-force metric
  ``dt = sqrt(error_tolerance / (N·max|F|/m))`` with dynamic ``|Δε|`` shock
  detection on cavity coupling strength ε.
- **OpenMM aging** (this module): max-force metric with self-calibrated
  ``eps_relaxed`` (targets ``dt_max = 1`` fs at t=0) and the same ramp shape as
  the paper.  Intentionally matches cav-hoomd *code*, not the RST displacement
  doc.

Uses velocity-Verlet ``CustomIntegrator`` + external ``setStepSize`` (not leapfrog
``VerletIntegrator``, which injects energy when dt changes; not VariableVerlet RMS):

  force_max_norm = N * max_i |F_i| / m_i
  dt = sqrt(eps_eff / force_max_norm)

Epsilon is self-calibrated from initial forces to target dt = 1.0 fs.
At coupling shocks, eps_eff scales by F0 and recovers over TAU_RAMP_PS.

Parity notes vs cav-hoomd ``AdaptiveTimestepUpdater``:
- Shock triggers on ``|Δε| ≥ COUPLING_CHANGE_THRESHOLD`` (ε = λ·ω_c) plus λ
  step edges.
- dt is recomputed every ``FORCE_UPDATE_INTERVAL`` steps, or every step during
  the post-shock recovery window ``[t_shock, t_shock + τ*]``.
- COM drift is handled by Bussi ``setSubtractCMMotion``; cav-hoomd optional
  ``ZeroMomentum`` is not applied during integration observers.
"""

from __future__ import annotations

import hashlib
import math
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Callable, Dict, List, Optional, Sequence, Tuple

import numpy as np

# Shock ramp (SI Eq. 3.16 shape, applied to calibrated epsilon)
F0 = 1e-3
TAU_RAMP_PS = 50.0

# Proven-stable fixed-dt cap (1.0 fs); adaptive must not exceed this.
TARGET_DT_PS = 0.001
DT_MAX_PS = 0.001
DT_MIN_PS = 1e-4  # 0.1 fs floor (safe for mixed/double integration)
DT_SHOCK_CAP_PS = 1e-4
COUPLING_CHANGE_THRESHOLD = 1e-5
PRE_SWITCH_GUARD_PS = 5.0
SWITCH_TIME_EPS_PS = 1e-9

# Legacy paper constant (VariableVerlet RMS path only)
EPS_STAR_NM = 5.0

FORCE_UPDATE_INTERVAL = 1000


@dataclass(frozen=True)
class AdaptiveParityConfig:
    """Runtime knobs for cav-hoomd ``AdaptiveTimestepUpdater`` parity experiments."""

    f0: float = F0
    pre_switch_guard_ps: float = PRE_SWITCH_GUARD_PS
    dt_slew_threshold: float = 0.0
    max_timestep_change_factor: float = 1.5
    absolute_error_tolerance: float | None = None


def default_parity_config() -> AdaptiveParityConfig:
    """Production defaults: force-calibrated epsilon + cav-hoomd shock ``f0``."""
    return AdaptiveParityConfig(
        f0=1e-5,
        pre_switch_guard_ps=0.0,
        dt_slew_threshold=0.0,
        max_timestep_change_factor=1.5,
        absolute_error_tolerance=None,
    )


def cavhoomd_runtime_parity_config(
    *,
    error_tolerance: float = 0.01,
    initial_fraction: float = 1e-5,
) -> AdaptiveParityConfig:
    """Settings aligned with cav-hoomd ``core.py`` adaptive wiring."""
    return AdaptiveParityConfig(
        f0=initial_fraction,
        pre_switch_guard_ps=0.0,
        dt_slew_threshold=0.0,
        max_timestep_change_factor=1.5,
        absolute_error_tolerance=error_tolerance,
    )


def parity_config_from_state(state: Dict[str, Any]) -> AdaptiveParityConfig:
    """Read ``parity_config`` from adaptive state, else defaults."""
    raw = state.get("parity_config")
    if raw is None:
        return default_parity_config()
    if isinstance(raw, AdaptiveParityConfig):
        return raw
    if isinstance(raw, dict):
        return AdaptiveParityConfig(**raw)
    raise TypeError(f"Unexpected parity_config type: {type(raw)!r}")


def epsilon_ramp_fraction_for_config(
    time_ps: float,
    ramp_start_ps: Optional[float],
    *,
    f0: float,
) -> float:
    """Ramp multiplier with configurable shock factor *f0*."""
    if ramp_start_ps is None or time_ps < ramp_start_ps:
        return 1.0
    t_since = time_ps - ramp_start_ps
    return 1.0 - (1.0 - f0) * math.exp(-t_since / TAU_RAMP_PS)


def adaptive_module_sha256() -> str:
    """SHA-256 of this module file (deployment provenance)."""
    return hashlib.sha256(Path(__file__).read_bytes()).hexdigest()


def coupling_epsilon_au(
    time_ps: float,
    coupling_start_ps: float,
    lambda_coupling: float,
    omegac_au: float,
) -> float:
    """Cavity coupling strength ε = λ(t)·ω_c in atomic units."""
    return step_coupling_value(time_ps, coupling_start_ps, lambda_coupling) * omegac_au


def effective_dt_max_ps(
    time_ps: float,
    state: Dict[str, Any],
    dt_max_ps: float = DT_MAX_PS,
) -> float:
    """Cap maximum dt; ramp down during post-shock recovery for stability."""
    cfg = parity_config_from_state(state)
    ramp_t0 = state.get("ramp_t0")
    if ramp_t0 is not None:
        return dt_max_ps * epsilon_ramp_fraction_for_config(
            time_ps, float(ramp_t0), f0=cfg.f0
        )
    if cfg.pre_switch_guard_ps > 0.0 or cfg.dt_slew_threshold > 0.0:
        return dt_max_ps
    return dt_max_ps


def effective_force_update_interval(
    state: Dict[str, Any],
    time_ps: float,
    *,
    current_dt_ps: float | None = None,
    dt_max_ps: float = DT_MAX_PS,
) -> int:
    """Return 1 during post-shock recovery or near-dt-floor, else interval."""
    ramp_t0 = state.get("ramp_t0")
    if ramp_t0 is not None and time_ps - float(ramp_t0) < TAU_RAMP_PS - 1e-15:
        return 1
    if current_dt_ps is not None and current_dt_ps > 0.0:
        if current_dt_ps <= 4.0 * DT_MIN_PS or current_dt_ps < 0.05 * dt_max_ps:
            return 1
    return FORCE_UPDATE_INTERVAL


def epsilon_ramp_fraction(time_ps: float, ramp_start_ps: Optional[float]) -> float:
    """Return multiplicative factor on relaxed epsilon (1.0 = full, F0 = strict)."""
    return epsilon_ramp_fraction_for_config(time_ps, ramp_start_ps, f0=F0)


def epsilon_tolerance(time_ps: float, ramp_start_ps: Optional[float]) -> float:
    """Legacy SI Eq. 3.16 for VariableVerlet error tolerance (nm)."""
    return EPS_STAR_NM * epsilon_ramp_fraction(time_ps, ramp_start_ps)


def shock_epsilon_tolerance() -> float:
    """Strict error tolerance immediately after a coupling shock."""
    return EPS_STAR_NM * F0


def step_coupling_on(
    time_ps: float,
    coupling_start_ps: float,
    lambda_coupling: float,
) -> bool:
    """Return True when step-profile λ(t) = λ Θ(t - t₀) is ON."""
    if lambda_coupling <= 0.0:
        return False
    return time_ps >= coupling_start_ps - 1e-15


def step_coupling_value(
    time_ps: float,
    coupling_start_ps: float,
    lambda_coupling: float,
) -> float:
    """Current λ for the analytic step turn-on profile."""
    return lambda_coupling if step_coupling_on(time_ps, coupling_start_ps, lambda_coupling) else 0.0


def create_adaptive_state(
    lambda_coupling: float,
    coupling_start_ps: float,
    initial_time_ps: float = 0.0,
    *,
    eps_relaxed: float | None = None,
    omegac_au: float = 0.0,
    parity_config: AdaptiveParityConfig | None = None,
) -> Dict[str, Any]:
    """Initialize mutable state for :func:`advance_to_time_step_on`."""
    initial_on = step_coupling_on(initial_time_ps, coupling_start_ps, lambda_coupling)
    initial_lambda = step_coupling_value(
        initial_time_ps, coupling_start_ps, lambda_coupling
    )
    return {
        "ramp_t0": initial_time_ps if initial_on else None,
        "prev_lambda_on": initial_on,
        "last_lambda": initial_lambda,
        "last_epsilon": coupling_epsilon_au(
            initial_time_ps, coupling_start_ps, lambda_coupling, omegac_au
        ),
        "omegac_au": float(omegac_au),
        "eps_relaxed": eps_relaxed,
        "steps_since_dt_update": 0,
        "last_force_max_norm": None,
        "parity_config": parity_config or default_parity_config(),
    }


def effective_epsilon_scaled(
    time_ps: float,
    state: Dict[str, Any],
    coupling_start_ps: float,
    lambda_coupling: float,
    current_dt_ps: float,
) -> float:
    """Scaled epsilon for max-metric dt from calibrated *eps_relaxed*."""
    cfg = parity_config_from_state(state)
    eps_relaxed = float(state["eps_relaxed"])
    ramp_t0 = state.get("ramp_t0")
    if ramp_t0 is not None:
        return eps_relaxed * epsilon_ramp_fraction_for_config(
            time_ps, ramp_t0, f0=cfg.f0
        )

    if (
        cfg.pre_switch_guard_ps > 0.0
        and lambda_coupling > 0.0
        and time_ps < coupling_start_ps - 1e-15
    ):
        time_to_switch = coupling_start_ps - time_ps
        if time_to_switch <= cfg.pre_switch_guard_ps + 1e-12:
            return eps_relaxed * cfg.f0
    return eps_relaxed


def effective_epsilon_tolerance(
    time_ps: float,
    state: Dict[str, Any],
    coupling_start_ps: float,
    lambda_coupling: float,
    current_dt_ps: float,
) -> float:
    """Legacy VariableVerlet tolerance (nm); kept for square-wave path."""
    ramp_t0 = state.get("ramp_t0")
    if ramp_t0 is not None:
        return epsilon_tolerance(time_ps, ramp_t0)

    if lambda_coupling > 0.0 and time_ps < coupling_start_ps - 1e-15:
        time_to_switch = coupling_start_ps - time_ps
        if time_to_switch <= PRE_SWITCH_GUARD_PS + 1e-12:
            return shock_epsilon_tolerance()
    return EPS_STAR_NM


def compute_force_max_norm(
    masses_amu: Sequence[float],
    forces_kj_mol_nm: np.ndarray,
) -> float:
    """Return N * max_i |F_i|/m_i (OpenMM kJ/(mol·nm), amu)."""
    n = len(masses_amu)
    force_mag = np.linalg.norm(forces_kj_mol_nm, axis=1)
    masses = np.asarray(masses_amu, dtype=np.float64)
    positive = masses > 0.0
    if not np.any(positive):
        return 0.0
    per_mass = force_mag[positive] / masses[positive]
    return float(n * np.max(per_mass))


def cavhoomd_optimal_dt(
    force_max_norm: float,
    eps: float,
    *,
    dt_min_ps: float = DT_MIN_PS,
    dt_max_ps: float = DT_MAX_PS,
) -> float:
    """dt (ps) from cav-hoomd max-metric formula with clamps."""
    if force_max_norm <= 0.0 or eps <= 0.0:
        return dt_max_ps
    dt = math.sqrt(eps / force_max_norm)
    return max(dt_min_ps, min(dt_max_ps, dt))


def particle_masses_amu(system) -> List[float]:
    """Per-particle masses in amu."""
    from openmm import unit

    return [
        system.getParticleMass(i).value_in_unit(unit.dalton)
        for i in range(system.getNumParticles())
    ]


def read_forces_kj_mol_nm(context) -> np.ndarray:
    """Forces on all particles in kJ/(mol·nm)."""
    from openmm import unit

    state = context.getState(getForces=True)
    return np.asarray(
        state.getForces(asNumpy=True).value_in_unit(
            unit.kilojoule_per_mole / unit.nanometer
        ),
        dtype=np.float64,
    )


def calibrate_epsilon(
    context,
    system,
    *,
    target_dt_ps: float = TARGET_DT_PS,
    absolute_error_tolerance: float | None = None,
) -> Tuple[float, float]:
    """Calibrate relaxed epsilon so initial dt equals *target_dt_ps*.

    When *absolute_error_tolerance* is set, return it directly (cav-hoomd style).
    """
    masses = particle_masses_amu(system)
    forces = read_forces_kj_mol_nm(context)
    force_max_norm = compute_force_max_norm(masses, forces)
    if force_max_norm <= 0.0:
        raise ValueError("Cannot calibrate epsilon: zero force_max_norm")
    if absolute_error_tolerance is not None:
        return float(absolute_error_tolerance), force_max_norm
    eps = target_dt_ps**2 * force_max_norm
    return eps, force_max_norm


def _clamp_integrator_step(integrator, dt_cap_ps: float) -> None:
    """Reduce integrator step size to at most *dt_cap_ps* (ps)."""
    from openmm import unit

    current_dt_ps = integrator.getStepSize().value_in_unit(unit.picosecond)
    if current_dt_ps > dt_cap_ps:
        integrator.setStepSize(dt_cap_ps * unit.picosecond)


def _register_coupling_shock(
    state: Dict[str, Any],
    time_ps: float,
    integrator,
    dt_shock_cap_ps: float,
) -> None:
    """Anchor the error ramp and clamp dt at a coupling discontinuity."""
    state["ramp_t0"] = time_ps
    state["steps_since_dt_update"] = FORCE_UPDATE_INTERVAL
    _clamp_integrator_step(integrator, dt_shock_cap_ps)


def square_wave_on(
    time_ps: float,
    start_ps: float,
    period_ps: float,
    duty: float,
) -> bool:
    """Return True when analytic square-wave λ(t) is ON."""
    if time_ps < start_ps or period_ps <= 0.0:
        return False
    phase = ((time_ps - start_ps) / period_ps) % 1.0
    return phase < duty


def lambda_transition(prev_on: bool, curr_on: bool) -> bool:
    """True on every rising or falling square-wave edge."""
    return prev_on != curr_on


def would_cross_coupling_switch(
    time_ps: float,
    dt_ps: float,
    coupling_start_ps: float,
    lambda_coupling: float,
) -> bool:
    """True when a Verlet step would span the delayed λ step turn-on time."""
    if lambda_coupling <= 0.0 or coupling_start_ps <= 0.0:
        return False
    if time_ps >= coupling_start_ps - SWITCH_TIME_EPS_PS:
        return False
    return time_ps + dt_ps > coupling_start_ps + SWITCH_TIME_EPS_PS


def create_velocity_verlet_integrator(
    dt_ps: float,
    *,
    random_number_seed: int | None = None,
):
    """Create a velocity-Verlet CustomIntegrator safe under external setStepSize().

    OpenMM's plain ``VerletIntegrator`` is leapfrog (half-step velocity storage).
    Changing ``dt`` between steps reinterprets that half-step state and injects
    spurious kinetic energy.  This integrator stores full-step position/velocity
    state, matching HOOMD's velocity-Verlet scheme and cav-hoomd adaptive parity.

    On CUDA, *random_number_seed* must match ``BussiThermostat.setRandomNumberSeed``
    (non-zero); otherwise Context construction fails with conflicting RNG seeds.
    """
    import openmm
    from openmm import unit

    integrator = openmm.CustomIntegrator(dt_ps * unit.picosecond)
    integrator.addPerDofVariable("x1", 0)
    integrator.addUpdateContextState()
    integrator.addComputePerDof("v", "v + 0.5*dt*f/m")
    integrator.addComputePerDof("x", "x + dt*v")
    integrator.addComputePerDof("x1", "x")
    integrator.addConstrainPositions()
    integrator.addComputePerDof("v", "v + 0.5*dt*f/m + (x-x1)/dt")
    integrator.addConstrainVelocities()
    if random_number_seed is not None and random_number_seed != 0:
        integrator.setRandomNumberSeed(random_number_seed)
    return integrator


def create_adaptive_integrator(
    dt_max_ps: float = DT_MAX_PS,
    *,
    use_variable_verlet: bool = False,
    hybrid_safety_verlet: bool = False,
    use_leapfrog_verlet: bool = False,
    random_number_seed: int | None = None,
):
    """Create integrator for adaptive stepping.

    Default: velocity-Verlet ``CustomIntegrator`` + external max-force
    ``setStepSize`` (safe when dt changes every step during shock recovery).

    Set *use_variable_verlet* or *hybrid_safety_verlet* for VariableVerlet
    diagnostic modes.  Set *use_leapfrog_verlet* True to reproduce the old
    leapfrog ``VerletIntegrator`` path (unsafe under dt churn; tests only).
    """
    import openmm
    from openmm import unit

    if use_variable_verlet or hybrid_safety_verlet:
        integrator = openmm.VariableVerletIntegrator(EPS_STAR_NM)
        integrator.setMaximumStepSize(dt_max_ps * unit.picosecond)
        return integrator
    if use_leapfrog_verlet:
        return openmm.VerletIntegrator(dt_max_ps * unit.picosecond)
    return create_velocity_verlet_integrator(
        dt_max_ps, random_number_seed=random_number_seed
    )


def _uses_variable_verlet_integrator(integrator) -> bool:
    return integrator.__class__.__name__ == "VariableVerletIntegrator"


def _update_timestep_max_metric(
    context,
    system,
    integrator,
    state: Dict[str, Any],
    *,
    time_ps: float,
    coupling_start_ps: float,
    lambda_coupling: float,
    dt_min_ps: float,
    dt_max_ps: float,
    masses_amu: Sequence[float],
) -> float:
    """Recompute and apply dt from forces; return dt (ps)."""
    from openmm import unit

    forces = read_forces_kj_mol_nm(context)
    force_max_norm = compute_force_max_norm(masses_amu, forces)
    state["last_force_max_norm"] = force_max_norm

    current_dt_ps = integrator.getStepSize().value_in_unit(unit.picosecond)
    eps = effective_epsilon_scaled(
        time_ps,
        state,
        coupling_start_ps,
        lambda_coupling,
        current_dt_ps,
    )
    dt_ps = cavhoomd_optimal_dt(
        force_max_norm,
        eps,
        dt_min_ps=dt_min_ps,
        dt_max_ps=effective_dt_max_ps(time_ps, state, dt_max_ps),
    )
    cfg = parity_config_from_state(state)
    if cfg.dt_slew_threshold > 0.0 and current_dt_ps > 0.0:
        rel_change = abs(dt_ps - current_dt_ps) / current_dt_ps
        if rel_change < cfg.dt_slew_threshold:
            dt_ps = current_dt_ps
        else:
            max_dt = current_dt_ps * cfg.max_timestep_change_factor
            min_dt = current_dt_ps / cfg.max_timestep_change_factor
            dt_ps = max(min_dt, min(max_dt, dt_ps))
    integrator.setStepSize(dt_ps * unit.picosecond)
    state["steps_since_dt_update"] = 0
    return dt_ps


def advance_to_time_step_on(
    context,
    integrator,
    thermostat,
    *,
    system,
    target_time_ps: float,
    lambda_coupling: float,
    coupling_start_ps: float,
    state: Dict[str, Any],
    dt_shock_cap_ps: float = DT_SHOCK_CAP_PS,
    coupling_change_threshold: float = COUPLING_CHANGE_THRESHOLD,
    dt_min_ps: float = DT_MIN_PS,
    dt_max_ps: float = DT_MAX_PS,
    force_update_interval: int = FORCE_UPDATE_INTERVAL,
    masses_amu: Sequence[float] | None = None,
    on_step: Optional[Callable[[float, float], None]] = None,
    log_dt: bool = False,
) -> List[float]:
    """Integrate to *target_time_ps* with cav-hoomd max-metric adaptive dt.

    Uses plain Verlet + external ``setStepSize``.  Requires *state* to contain
    calibrated ``eps_relaxed`` from :func:`calibrate_epsilon`.
    """
    from openmm import unit

    if state.get("eps_relaxed") is None:
        raise ValueError("advance_to_time_step_on requires state['eps_relaxed']")

    if masses_amu is None:
        masses_amu = particle_masses_amu(system)

    logged_dts: List[float] = []
    target_time_ps = float(target_time_ps)

    while True:
        time_ps = context.getState().getTime().value_in_unit(unit.picosecond)
        if time_ps >= target_time_ps - 1e-15:
            break

        current_dt_ps = integrator.getStepSize().value_in_unit(unit.picosecond)
        if would_cross_coupling_switch(
            time_ps,
            current_dt_ps,
            coupling_start_ps,
            lambda_coupling,
        ):
            dt_to_switch_ps = max(
                dt_min_ps,
                coupling_start_ps - time_ps,
            )
            integrator.setStepSize(dt_to_switch_ps * unit.picosecond)
            integrator.step(1)
            dt_ps = integrator.getStepSize().value_in_unit(unit.picosecond)
            thermostat.apply_cavity_thermostat_step(dt_ps)
            step_time_ps = context.getState().getTime().value_in_unit(unit.picosecond)
            if on_step is not None:
                on_step(step_time_ps, dt_ps)
            if log_dt:
                logged_dts.append(dt_ps)
            continue

        curr_on = step_coupling_on(time_ps, coupling_start_ps, lambda_coupling)
        curr_lambda = step_coupling_value(time_ps, coupling_start_ps, lambda_coupling)

        force_update = False
        if lambda_transition(state["prev_lambda_on"], curr_on):
            _register_coupling_shock(state, time_ps, integrator, dt_shock_cap_ps)
            force_update = True

        last_lambda = float(state.get("last_lambda", 0.0))
        if abs(curr_lambda - last_lambda) >= coupling_change_threshold:
            _register_coupling_shock(state, time_ps, integrator, dt_shock_cap_ps)
            force_update = True

        omegac_au = float(state.get("omegac_au", 0.0))
        if omegac_au > 0.0:
            curr_epsilon = curr_lambda * omegac_au
            last_epsilon = float(state.get("last_epsilon", 0.0))
            if abs(curr_epsilon - last_epsilon) >= coupling_change_threshold:
                _register_coupling_shock(state, time_ps, integrator, dt_shock_cap_ps)
                force_update = True
            state["last_epsilon"] = curr_epsilon

        state["prev_lambda_on"] = curr_on
        state["last_lambda"] = curr_lambda

        update_interval = effective_force_update_interval(
            state,
            time_ps,
            current_dt_ps=current_dt_ps,
            dt_max_ps=dt_max_ps,
        )
        steps_since = int(state.get("steps_since_dt_update", 0))
        if force_update or steps_since >= update_interval:
            current_dt_ps = _update_timestep_max_metric(
                context,
                system,
                integrator,
                state,
                time_ps=time_ps,
                coupling_start_ps=coupling_start_ps,
                lambda_coupling=lambda_coupling,
                dt_min_ps=dt_min_ps,
                dt_max_ps=dt_max_ps,
                masses_amu=masses_amu,
            )
        else:
            state["steps_since_dt_update"] = steps_since + 1

        if _uses_variable_verlet_integrator(integrator):
            cfg = parity_config_from_state(state)
            ramp_t0 = state.get("ramp_t0")
            integrator.setErrorTolerance(
                EPS_STAR_NM
                * epsilon_ramp_fraction_for_config(
                    time_ps, ramp_t0, f0=cfg.f0
                )
            )

        integrator.step(1)
        dt_ps = integrator.getStepSize().value_in_unit(unit.picosecond)
        if dt_ps < dt_min_ps:
            integrator.setStepSize(dt_min_ps * unit.picosecond)
            dt_ps = dt_min_ps

        thermostat.apply_cavity_thermostat_step(dt_ps)
        step_time_ps = context.getState().getTime().value_in_unit(unit.picosecond)
        if on_step is not None:
            on_step(step_time_ps, dt_ps)
        if log_dt:
            logged_dts.append(dt_ps)

    return logged_dts


def advance_to_time_step_on_rms(
    context,
    integrator,
    thermostat,
    *,
    target_time_ps: float,
    coupling_start_ps: float,
    lambda_coupling: float,
    state: Dict[str, Any],
    dt_shock_cap_ps: float = DT_SHOCK_CAP_PS,
    dt_min_ps: float = DT_MIN_PS,
    on_step: Optional[Callable[[float, float], None]] = None,
    log_dt: bool = False,
) -> List[float]:
    """Integrate with VariableVerlet RMS tolerance ramp (legacy shock path).

    Used for diagnostic comparison against max-metric ``advance_to_time_step_on``.
    """
    from openmm import unit

    logged_dts: List[float] = []
    target_time_ps = float(target_time_ps)
    cfg = parity_config_from_state(state)

    while True:
        time_ps = context.getState().getTime().value_in_unit(unit.picosecond)
        if time_ps >= target_time_ps - 1e-15:
            break

        current_dt_ps = integrator.getStepSize().value_in_unit(unit.picosecond)
        if would_cross_coupling_switch(
            time_ps,
            current_dt_ps,
            coupling_start_ps,
            lambda_coupling,
        ):
            dt_to_switch_ps = max(
                dt_min_ps,
                coupling_start_ps - time_ps,
            )
            integrator.setStepSize(dt_to_switch_ps * unit.picosecond)
            integrator.step(1)
            dt_ps = integrator.getStepSize().value_in_unit(unit.picosecond)
            thermostat.apply_cavity_thermostat_step(dt_ps)
            step_time_ps = context.getState().getTime().value_in_unit(unit.picosecond)
            if on_step is not None:
                on_step(step_time_ps, dt_ps)
            if log_dt:
                logged_dts.append(dt_ps)
            continue

        curr_on = step_coupling_on(time_ps, coupling_start_ps, lambda_coupling)
        if lambda_transition(state["prev_lambda_on"], curr_on):
            state["ramp_t0"] = time_ps
            _clamp_integrator_step(integrator, dt_shock_cap_ps)

        ramp_t0 = state.get("ramp_t0")
        tol_nm = EPS_STAR_NM * epsilon_ramp_fraction_for_config(
            time_ps, ramp_t0, f0=cfg.f0
        )
        integrator.setErrorTolerance(tol_nm)
        state["prev_lambda_on"] = curr_on

        integrator.step(1)
        dt_ps = integrator.getStepSize().value_in_unit(unit.picosecond)
        thermostat.apply_cavity_thermostat_step(dt_ps)
        step_time_ps = context.getState().getTime().value_in_unit(unit.picosecond)
        if on_step is not None:
            on_step(step_time_ps, dt_ps)
        if log_dt:
            logged_dts.append(dt_ps)

    return logged_dts


def advance_to_time(
    context,
    integrator,
    thermostat,
    displacer,
    lambda_coupling: float,
    target_time_ps: float,
    coupling_start_ps: float,
    period_ps: float,
    duty: float,
    state: Dict[str, Any],
    log_dt: bool = False,
) -> List[float]:
    """Integrate to *target_time_ps* with VariableVerlet (square-wave legacy)."""
    from openmm import unit

    logged_dts: List[float] = []
    target_time_ps = float(target_time_ps)

    while True:
        time_ps = context.getState().getTime().value_in_unit(unit.picosecond)
        if time_ps >= target_time_ps - 1e-15:
            break

        curr_on = square_wave_on(time_ps, coupling_start_ps, period_ps, duty)
        if lambda_transition(state["prev_lambda_on"], curr_on):
            state["ramp_t0"] = time_ps
            _clamp_integrator_step(integrator, DT_SHOCK_CAP_PS)
            if curr_on and displacer is not None and lambda_coupling > 0.0:
                displacer.displaceToEquilibrium(context, lambda_coupling)

        integrator.setErrorTolerance(
            epsilon_tolerance(time_ps, state["ramp_t0"])
        )
        state["prev_lambda_on"] = curr_on

        integrator.step(1)
        dt_ps = integrator.getStepSize().value_in_unit(unit.picosecond)
        thermostat.apply_cavity_thermostat_step(dt_ps)
        if log_dt:
            logged_dts.append(dt_ps)

    return logged_dts
