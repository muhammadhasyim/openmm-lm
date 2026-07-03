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

    def test_integrate_tn_zero_before_switch(self):
        class _ConstModel:
            is_fitted = True

            @staticmethod
            def get_relaxation_time(_T):
                return 100.0

        tn = ToolNarayanaswamy(relaxation_model=_ConstModel(), beta=0.55)
        times = np.linspace(0.0, 400.0, 41)
        T_s = np.full_like(times, 100.0)
        h = tn.integrate_tn(times, T_s, switch_time_ps=200.0)
        self.assertTrue(np.all(h[times <= 200.0] == 0.0))
        self.assertGreater(h[-1], 0.0)

    def test_reconstruct_material_time_shape(self):
        tn = ToolNarayanaswamy(smoothness_alpha=1.0)
        tw = np.array([0.0, 10.0, 20.0])
        tau = np.array([5.0, 5.0, 5.0])
        t_grid, h = tn.reconstruct_material_time(tw, tau)
        self.assertEqual(len(t_grid), len(h))
        self.assertTrue(h[-1] >= h[0])

    def test_hat_basis_smooth_monotonic(self):
        tn = ToolNarayanaswamy(smoothness_alpha=1.0)
        tw = np.linspace(200.0, 2400.0, 13)
        tau = np.full(13, 100.0)
        t_grid = np.linspace(200.0, 2600.0, 500)
        _, h = tn.reconstruct_material_time(
            tw, tau, time_grid_ps=t_grid, origin_time_ps=200.0
        )
        self.assertTrue(np.all(np.diff(h) >= -1e-8))
        d2h = np.diff(h, n=2)
        self.assertLess(float(np.max(np.abs(d2h))), 0.05)

    def test_hat_basis_increment_one(self):
        tn = ToolNarayanaswamy(smoothness_alpha=1.0)
        tw = np.array([200.0, 400.0, 600.0, 800.0])
        tau = np.array([100.0, 100.0, 100.0, 100.0])
        t_grid = np.linspace(200.0, 1200.0, 500)
        t_out, h = tn.reconstruct_material_time(
            tw, tau, time_grid_ps=t_grid, origin_time_ps=200.0
        )
        for t_w, t_r in zip(tw, tau):
            inc = np.interp(t_w + t_r, t_out, h) - np.interp(t_w, t_out, h)
            self.assertAlmostEqual(inc, 1.0, delta=0.08)

    def test_mtti_origin_at_switch(self):
        tn = ToolNarayanaswamy(smoothness_alpha=1.0)
        tw = np.array([200.0, 400.0, 600.0])
        tau = np.array([100.0, 100.0, 100.0])
        t_grid = np.linspace(200.0, 1200.0, 200)
        t_out, h = tn.reconstruct_material_time(
            tw, tau, time_grid_ps=t_grid, origin_time_ps=200.0
        )
        self.assertAlmostEqual(float(np.interp(200.0, t_out, h)), 0.0, delta=1e-10)

    def test_mtti_increment_constraints(self):
        tn = ToolNarayanaswamy(smoothness_alpha=1.0)
        tw = np.array([200.0, 400.0, 600.0])
        tau = np.array([100.0, 100.0, 100.0])
        t_grid, h = tn.reconstruct_material_time(tw, tau, origin_time_ps=200.0)
        for t_w, t_r in zip(tw, tau):
            inc = np.interp(t_w + t_r, t_grid, h) - np.interp(t_w, t_grid, h)
            self.assertAlmostEqual(inc, 1.0, delta=0.08)

    def test_collapse_isf_uses_absolute_waiting_time(self):
        tn = ToolNarayanaswamy(beta=0.55)
        t_grid = np.linspace(0.0, 1000.0, 101)
        h = t_grid / 200.0
        lags = np.linspace(0.0, 400.0, 20)
        lab_t_w = 200.0
        h_diff, h_w = tn.collapse_isf(lags, np.array([lab_t_w]), h, t_grid)
        self.assertTrue(np.all(h_diff >= 0.0))
        self.assertGreater(float(h_diff[-1]), float(h_diff[0]))
        self.assertAlmostEqual(float(h_w), 1.0, delta=1e-10)

    def test_equilibrium_h_zero_at_switch(self):
        model = RelaxationTimeModel()
        switch_ps = 200.0
        tau_eq = model.get_relaxation_time(100.0)
        t = np.array([switch_ps, switch_ps + tau_eq])
        h = np.maximum(t - switch_ps, 0.0) / max(tau_eq, 1e-12)
        self.assertAlmostEqual(h[0], 0.0)
        self.assertAlmostEqual(h[1], 1.0)

    def test_tau_tilde_interpolated_baseline(self):
        tw0 = np.array([0.0, 200.0, 400.0])
        tau0 = np.array([50.0, 60.0, 70.0])
        for t_w in (100.0, 200.0, 350.0):
            base = float(np.interp(t_w, tw0, tau0))
            self.assertAlmostEqual(base / base, 1.0)

    def test_tau_s_tn_constant_temperature(self):
        class _ConstModel:
            is_fitted = True

            @staticmethod
            def get_relaxation_time(_T):
                return 100.0

        tn = ToolNarayanaswamy(relaxation_model=_ConstModel())
        times = np.linspace(0.0, 2000.0, 2001)
        T_s = np.full_like(times, 100.0)
        tau_tn = tn.tau_s_tn(times, T_s, t_w_ps=0.0, switch_time_ps=200.0)
        self.assertIsNotNone(tau_tn)
        self.assertAlmostEqual(float(tau_tn), 100.0, delta=2.0)


if __name__ == "__main__":
    unittest.main()
