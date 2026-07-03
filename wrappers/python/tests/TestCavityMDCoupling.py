"""Tests for configure_coupling and variant modulation mapping."""

import unittest
from unittest.mock import MagicMock

import openmm

from openmm.cavitymd.coupling import configure_coupling
from openmm.cavitymd.variants import (
    AdaptiveSquareWaveVariant,
    ConstantVariant,
    DecayingSquareWaveVariant,
    ExponentialWaveVariant,
    SinusoidVariant,
    SquareWaveVariant,
    StepVariant,
)


class TestConfigureCoupling(unittest.TestCase):
    def test_square_wave_gpu(self):
        force = MagicMock()
        variant = SquareWaveVariant(0.03, 10.0, duty_cycle=0.5, start_time_ps=200.0)
        configure_coupling(force, variant, use_gpu=True)
        force.setCouplingModulation.assert_called_once_with(
            openmm.CavityForce.ModulationSquareWave,
            0.03,
            10.0,
            0.5,
            200.0,
            -1.0,
            1.0,
        )

    def test_step_gpu(self):
        force = MagicMock()
        variant = StepVariant(0.03, switch_time_ps=200.0)
        configure_coupling(force, variant, use_gpu=True)
        force.setCouplingModulation.assert_called_once_with(
            openmm.CavityForce.ModulationStep,
            0.03,
            0.0,
            0.5,
            200.0,
            -1.0,
            1.0,
        )

    def test_decaying_step_gpu(self):
        force = MagicMock()
        variant = StepVariant(
            0.03, switch_time_ps=200.0, decay_time_constant_ps=50.0
        )
        configure_coupling(force, variant, use_gpu=True)
        force.setCouplingModulation.assert_called_once_with(
            openmm.CavityForce.ModulationDecayingStep,
            0.03,
            0.0,
            0.5,
            200.0,
            -1.0,
            50.0,
        )

    def test_adaptive_square_wave_gpu(self):
        force = MagicMock()
        variant = AdaptiveSquareWaveVariant(
            0.03,
            100.0,
            10.0,
            lambda: 100.0,
            start_time_ps=200.0,
        )
        configure_coupling(force, variant, use_gpu=True)
        force.setAdaptiveSquareWaveModulation.assert_called_once()

    def test_decaying_square_wave_gpu(self):
        force = MagicMock()
        variant = DecayingSquareWaveVariant(0.03, 10.0, decay_rate_per_period=0.1)
        configure_coupling(force, variant, use_gpu=True)
        force.setDecayingSquareWaveModulation.assert_called_once()

    def test_sinusoid_gpu(self):
        force = MagicMock()
        variant = SinusoidVariant(0.03, 10.0)
        configure_coupling(force, variant, use_gpu=True)
        force.setSinusoidModulation.assert_called_once()

    def test_exponential_wave_gpu(self):
        force = MagicMock()
        variant = ExponentialWaveVariant(0.03, 10.0, decay_tau_ps=2.0)
        configure_coupling(force, variant, use_gpu=True)
        force.setExponentialWaveModulation.assert_called_once()

    def test_cpu_path_sets_lambda(self):
        force = MagicMock()
        variant = ConstantVariant(0.05)
        configure_coupling(force, variant, use_gpu=False, time_ps=0.0)
        force.setLambdaCoupling.assert_called_once_with(0.05)
        force.setCouplingModulation.assert_not_called()


if __name__ == "__main__":
    unittest.main()
