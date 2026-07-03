"""Unified coupling configuration for CavityForce GPU/CPU paths."""

from __future__ import annotations

from typing import Any, Tuple

import openmm

from .variants import (
    AdaptiveSquareWaveVariant,
    CouplingVariant,
    DecayingSquareWaveVariant,
    ExponentialWaveVariant,
    SinusoidVariant,
    SquareWaveVariant,
    StepVariant,
)


def _stop_time(stop_time_ps: float | None) -> float:
    return -1.0 if stop_time_ps is None else float(stop_time_ps)


def configure_coupling(
    cavity_force,
    variant: CouplingVariant,
    *,
    use_gpu: bool = True,
    time_ps: float = 0.0,
) -> None:
    """Apply a coupling variant to CavityForce.

    GPU path (default): one-time kernel modulation setup.
    CPU path: set initial lambda from ``variant.evaluate(time_ps)``.
    """
    if not use_gpu:
        cavity_force.setLambdaCoupling(variant.evaluate(time_ps))
        return

    if isinstance(variant, AdaptiveSquareWaveVariant):
        cavity_force.setAdaptiveSquareWaveModulation(
            variant._target_coupling,
            variant.target_temperature_K,
            variant.period_ps,
            variant.duty_cycle,
            variant.start_time_ps,
            _stop_time(variant.stop_time_ps),
            variant.min_amplitude,
            variant.max_amplitude,
        )
        return

    if isinstance(variant, DecayingSquareWaveVariant):
        cavity_force.setDecayingSquareWaveModulation(
            variant._initial_amplitude,
            variant.period_ps,
            variant.duty_cycle,
            variant.decay_rate_per_period,
            variant.start_time_ps,
            _stop_time(variant.stop_time_ps),
            variant.minimum_amplitude,
        )
        return

    if isinstance(variant, SinusoidVariant):
        cavity_force.setSinusoidModulation(
            variant._amplitude,
            variant.period_ps,
            variant.phase_offset,
            variant.start_time_ps,
            _stop_time(variant.stop_time_ps),
        )
        return

    if isinstance(variant, ExponentialWaveVariant):
        cavity_force.setExponentialWaveModulation(
            variant._amplitude,
            variant.period_ps,
            variant.decay_tau_ps,
            variant.start_time_ps,
            _stop_time(variant.stop_time_ps),
        )
        return

    if hasattr(variant, "to_modulation_params"):
        mod_type, params = variant.to_modulation_params()
        cavity_force.setCouplingModulation(mod_type, *params)
        return

    raise TypeError(
        f"Variant {type(variant).__name__} has no GPU modulation mapping; "
        "use use_gpu=False for host-side evaluation"
    )


def configure_multimode_adaptive_square_wave(
    force,
    period_ps: float,
    duty_cycle: float,
    mode_params,
    start_time_ps: float = 0.0,
    stop_time_ps: float = -1.0,
) -> None:
    """Configure per-mode adaptive square-wave on MultiModeCavityForce."""
    force.setAdaptiveSquareWaveModulation(
        period_ps, duty_cycle, start_time_ps, stop_time_ps
    )
    for i, (g, t, mn, mx) in enumerate(mode_params):
        force.setModeModulationParams(i, g, t, mn, mx)
