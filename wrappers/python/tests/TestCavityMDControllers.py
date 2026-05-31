"""Unit tests for C2F controllers and Tool-Narayanaswamy analysis."""

import unittest
import numpy as np

from openmm.cavitymd.controllers import (
    DiffEqController,
    SimpleSetpointController,
    PIDControl,
)
from openmm.cavitymd.analysis import RelaxationTimeModel, ToolNarayanaswamy


class _MockTracker:
    """Minimal TemperatureTracker stand-in for controller tests."""

    def __init__(self, kinetic=300.0, harmonic=280.0, structural=250.0):
        self.kinetic_temperature = kinetic
        self.harmonic_equipartition_temperature = harmonic
        self.structural_fictive_temperature = structural


class TestDiffEqController(unittest.TestCase):
    def test_tracks_structural_temperature(self):
        tracker = _MockTracker(structural=200.0)
        ctrl = DiffEqController(
            tracker,
            time_constant_ps=10.0,
            update_interval_ps=5.0,
            turn_on_time_ps=0.0,
        )
        new_T = ctrl.step(5.0, current_bath_T=300.0)
        self.assertIsNotNone(new_T)
        self.assertLess(new_T, 300.0)
        self.assertGreater(new_T, 200.0)

    def test_inactive_before_turn_on(self):
        tracker = _MockTracker(structural=200.0)
        ctrl = DiffEqController(tracker, turn_on_time_ps=100.0)
        self.assertIsNone(ctrl.step(50.0, current_bath_T=300.0))


class TestSimpleSetpointController(unittest.TestCase):
    def test_captures_setpoint_and_relaxes(self):
        tracker = _MockTracker(kinetic=280.0, structural=220.0)
        ctrl = SimpleSetpointController(
            tracker,
            time_constant_ps=10.0,
            update_interval_ps=5.0,
            turn_on_time_ps=0.0,
        )
        ctrl.step(0.0, current_bath_T=300.0)
        self.assertEqual(ctrl.setpoint_temperature, 220.0)
        new_T = ctrl.step(5.0, current_bath_T=300.0)
        self.assertIsNotNone(new_T)
        self.assertLess(new_T, 300.0)


class TestPIDControl(unittest.TestCase):
    def test_reduces_bath_when_signal_above_target(self):
        tracker = _MockTracker(structural=400.0)
        ctrl = PIDControl(
            tracker,
            target_temperature=100.0,
            Kp=0.5,
            Ti=20.0,
            Td=0.0,
            update_interval_ps=1.0,
            turn_on_time_ps=0.0,
        )
        new_T = ctrl.step(1.0, current_bath_T=300.0)
        self.assertIsNotNone(new_T)
        self.assertLess(new_T, 300.0)


class TestToolNarayanaswamy(unittest.TestCase):
    def test_stretched_exponential(self):
        h = np.array([0.0, 1.0, 2.0])
        phi = ToolNarayanaswamy.stretched_exponential(h, beta=1.0)
        np.testing.assert_allclose(phi, np.exp(-h))

    def test_integrate_tn_monotonic(self):
        model = RelaxationTimeModel()
        tn = ToolNarayanaswamy(relaxation_model=model, beta=0.55)
        times = np.linspace(0.0, 100.0, 50)
        T_s = np.linspace(300.0, 100.0, 50)
        h = tn.integrate_tn(times, T_s)
        self.assertTrue(np.all(np.diff(h) >= 0.0))

    def test_reconstruct_material_time_shape(self):
        tn = ToolNarayanaswamy(smoothness_alpha=0.1)
        tw = np.array([0.0, 10.0, 20.0])
        tau = np.array([5.0, 5.0, 5.0])
        t_grid, h = tn.reconstruct_material_time(tw, tau)
        self.assertEqual(len(t_grid), len(h))
        self.assertTrue(h[-1] >= h[0])


if __name__ == "__main__":
    unittest.main()
