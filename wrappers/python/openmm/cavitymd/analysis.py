"""Relaxation-time and Tool-Narayanaswamy analysis for C2F cavity-MD."""

from pathlib import Path
from typing import Dict, List, Optional, Tuple
import numpy as np

k_B = 8.617333e-5  # Boltzmann constant in eV/K


class RelaxationTimeModel:
    """Temperature-dependent tau_s(T) from dual-regime (Arrhenius + parabolic) fit."""

    def __init__(
        self,
        data_file_path: Optional[str] = None,
        *,
        bootstrap_samples: int = 200,
        bootstrap_seed: int = 42,
    ) -> None:
        self.data_file_path = data_file_path
        self.is_fitted = False
        self.T_onset: Optional[float] = None
        self.fit_results: Dict = {}
        self.bootstrap_samples = int(bootstrap_samples)
        self.bootstrap_seed = int(bootstrap_seed)
        self._temperatures: Optional[np.ndarray] = None
        self._relaxation_times: Optional[np.ndarray] = None
        self._bootstrap_tau_at_100K: Optional[np.ndarray] = None

        if data_file_path and Path(data_file_path).exists():
            self._load_and_fit_data()

    def _load_and_fit_data(self) -> None:
        try:
            data = np.loadtxt(self.data_file_path, skiprows=3, usecols=(0, 2))
            temperatures = data[:, 0]
            relaxation_times = data[:, 1]
            self._temperatures = temperatures
            self._relaxation_times = relaxation_times
            self.T_onset, self.fit_results = self._find_onset_temperature(
                temperatures, relaxation_times
            )
            self.is_fitted = not np.isnan(self.T_onset)
            if self.is_fitted and self.bootstrap_samples > 0:
                self._bootstrap_tau_at_100K = self._bootstrap_relaxation_times(
                    temperatures, relaxation_times
                )
        except Exception as exc:
            print(f"RelaxationTimeModel: failed to load {self.data_file_path}: {exc}")

    def _bootstrap_relaxation_times(
        self,
        temperatures: np.ndarray,
        relaxation_times: np.ndarray,
        *,
        reference_temperature_K: float = 100.0,
    ) -> np.ndarray:
        """Resample calibration points and collect tau(reference T) samples."""
        rng = np.random.default_rng(self.bootstrap_seed)
        valid = ~np.isnan(relaxation_times)
        T = temperatures[valid]
        tau = relaxation_times[valid]
        if len(T) < 6:
            return np.array([], dtype=float)

        samples: list[float] = []
        for _ in range(self.bootstrap_samples):
            idx = rng.integers(0, len(T), size=len(T))
            T_boot = T[idx]
            tau_boot = tau[idx]
            try:
                T_onset, fit_results = self._find_onset_temperature(T_boot, tau_boot)
                if not fit_results or np.isnan(T_onset):
                    continue
                model = RelaxationTimeModel(bootstrap_samples=0)
                model.is_fitted = True
                model.T_onset = float(T_onset)
                model.fit_results = fit_results
                samples.append(model.get_relaxation_time(reference_temperature_K))
            except Exception:
                continue
        return np.asarray(samples, dtype=float)

    @staticmethod
    def _fit_arrhenius(beta: np.ndarray, log_tau: np.ndarray) -> Tuple[float, float, float]:
        from scipy.optimize import curve_fit

        def model(b, Ea, beta_0, ln_tau_0):
            return ln_tau_0 + Ea * (b - beta_0)

        coeffs = np.polyfit(beta, log_tau, 1)
        beta_0_guess = float(np.mean(beta))
        ln_tau_0_guess = float(coeffs[1] + coeffs[0] * beta_0_guess)
        popt, _ = curve_fit(
            model, beta, log_tau,
            p0=[coeffs[0], beta_0_guess, ln_tau_0_guess],
        )
        return float(popt[0]), float(popt[1]), float(popt[2])

    @staticmethod
    def _fit_parabolic(
        beta: np.ndarray, log_tau: np.ndarray, beta_0: float, arr: Tuple[float, float]
    ) -> Tuple[float, float, float]:
        from scipy.optimize import curve_fit

        Ea_arr, ln_tau_0_arr = arr

        def model(b, Ea, J):
            d = b - beta_0
            return ln_tau_0_arr + Ea * d + J ** 2 * d ** 2

        delta = beta - beta_0
        residuals = log_tau - (ln_tau_0_arr + Ea_arr * delta)
        J_guess = float(np.sqrt(max(np.mean(np.abs(residuals / (delta ** 2 + 1e-10))), 1e-6)))
        popt, _ = curve_fit(
            model,
            beta,
            log_tau,
            p0=[Ea_arr, J_guess],
            bounds=([-np.inf, 1e-8], [np.inf, np.inf]),
            maxfev=10000,
        )
        return float(popt[0]), float(popt[1]), ln_tau_0_arr

    def _find_onset_temperature(
        self, temperatures: np.ndarray, relaxation_times: np.ndarray
    ) -> Tuple[float, Dict]:
        valid = ~np.isnan(relaxation_times)
        T = temperatures[valid]
        tau = relaxation_times[valid]
        if len(T) < 6:
            return np.nan, {}

        order = np.argsort(T)
        T = T[order]
        tau = tau[order]
        beta = 1.0 / (k_B * T)
        log_tau = np.log(tau)

        min_pts = 5
        best_r2 = -np.inf
        best_idx = None
        best_fits: Dict = {}

        for split in range(min_pts, len(T) - min_pts + 1):
            low_b, low_lt = beta[:split], log_tau[:split]
            high_b, high_lt = beta[split:], log_tau[split:]
            try:
                Ea, beta_0, ln_tau_0 = self._fit_arrhenius(high_b, high_lt)
                pred_h = ln_tau_0 + Ea * (high_b - beta_0)
                r2_h = 1 - np.sum((high_lt - pred_h) ** 2) / (
                    np.sum((high_lt - np.mean(high_lt)) ** 2) + 1e-10
                )
                Ea_p, J_p, ln_tau_0_p = self._fit_parabolic(
                    low_b, low_lt, beta_0, (Ea, ln_tau_0)
                )
                d = low_b - beta_0
                pred_l = ln_tau_0_p + Ea_p * d + J_p ** 2 * d ** 2
                r2_l = 1 - np.sum((low_lt - pred_l) ** 2) / (
                    np.sum((low_lt - np.mean(low_lt)) ** 2) + 1e-10
                )
                n_h, n_l = len(high_b), len(low_b)
                total_r2 = (n_h * r2_h + n_l * r2_l) / (n_h + n_l)
                if J_p > 0 and total_r2 > best_r2:
                    best_r2 = total_r2
                    best_idx = split
                    best_fits = {
                        "beta_0": beta_0,
                        "arrhenius": {"Ea": Ea, "ln_tau_0": ln_tau_0, "r2": r2_h},
                        "parabolic": {"Ea": Ea_p, "J": J_p, "ln_tau_0": ln_tau_0_p, "r2": r2_l},
                    }
            except Exception:
                continue

        if best_idx is None:
            return np.nan, {}
        return float(T[best_idx]), best_fits

    def get_relaxation_time(self, temperature_K: float) -> float:
        if temperature_K <= 0:
            return 1.0
        if not self.is_fitted:
            return 100.0 * np.exp(0.1 / (k_B * temperature_K))

        beta = 1.0 / (k_B * temperature_K)
        beta_0 = self.fit_results["beta_0"]
        if temperature_K > self.T_onset:
            p = self.fit_results["arrhenius"]
            ln_tau = p["ln_tau_0"] + p["Ea"] * (beta - beta_0)
        else:
            p = self.fit_results["parabolic"]
            d = beta - beta_0
            ln_tau = p["ln_tau_0"] + p["Ea"] * d + p["J"] ** 2 * d ** 2
        return float(np.exp(ln_tau))

    def get_relaxation_time_std(self, temperature_K: float) -> float:
        """Bootstrap standard deviation of tau at *temperature_K* (0 if unavailable)."""
        if self._bootstrap_tau_at_100K is None or self._bootstrap_tau_at_100K.size == 0:
            return 0.0
        if abs(temperature_K - 100.0) > 1e-6:
            return 0.0
        return float(np.std(self._bootstrap_tau_at_100K))

    def tau_uncertainty_band_at_temperature(
        self, temperature_K: float, n_sigma: float = 1.0
    ) -> Tuple[float, float]:
        """Return (tau_lo, tau_hi) from bootstrap spread at *temperature_K*."""
        tau = self.get_relaxation_time(temperature_K)
        std = self.get_relaxation_time_std(temperature_K) * float(n_sigma)
        if std <= 0.0:
            return tau, tau
        return max(tau - std, 1e-12), tau + std


class ToolNarayanaswamy:
    """Material-time reconstruction and TN-model integration."""

    def __init__(
        self,
        relaxation_model: Optional[RelaxationTimeModel] = None,
        beta: float = 0.55,
        smoothness_alpha: float = 1.0,
    ) -> None:
        self.relaxation_model = relaxation_model
        self.beta = float(beta)
        self.smoothness_alpha = float(smoothness_alpha)

    @staticmethod
    def stretched_exponential(h: np.ndarray, beta: float = 0.55) -> np.ndarray:
        return np.exp(-np.power(np.maximum(h, 0.0), beta))

    @staticmethod
    def _second_difference_matrix(n: int) -> np.ndarray:
        D = np.zeros((max(n - 2, 0), n))
        for i in range(n - 2):
            D[i, i] = 1.0
            D[i, i + 1] = -2.0
            D[i, i + 2] = 1.0
        return D

    @staticmethod
    def _hat_basis_at_time(t_val: float, t_grid: np.ndarray) -> np.ndarray:
        """Piecewise-linear hat function values θ_m(t) on uniform grid knots."""
        n = len(t_grid)
        theta = np.zeros(n, dtype=float)
        if n == 0:
            return theta
        t_val = float(t_val)
        if t_val <= float(t_grid[0]):
            theta[0] = 1.0
            return theta
        if t_val >= float(t_grid[-1]):
            theta[-1] = 1.0
            return theta
        idx = int(np.searchsorted(t_grid, t_val, side="right")) - 1
        idx = max(0, min(idx, n - 2))
        t_lo, t_hi = float(t_grid[idx]), float(t_grid[idx + 1])
        dt = t_hi - t_lo
        if dt <= 0.0:
            return theta
        theta[idx] = (t_hi - t_val) / dt
        theta[idx + 1] = (t_val - t_lo) / dt
        return theta

    @staticmethod
    def _evaluate_hat_expansion(t_val: float, t_grid: np.ndarray, h_coeffs: np.ndarray) -> float:
        return float(np.dot(h_coeffs, ToolNarayanaswamy._hat_basis_at_time(t_val, t_grid)))

    @staticmethod
    def _monotonic_hat_coefficients(
        constraint: np.ndarray,
        target: np.ndarray,
        smoothness_alpha: float,
        n: int,
    ) -> np.ndarray:
        """Solve for monotonic nodal hat coefficients via cumulative increments d >= 0."""
        from scipy.optimize import lsq_linear

        if n <= 1:
            return np.zeros(n, dtype=float)

        D = ToolNarayanaswamy._second_difference_matrix(n)
        n_free = n - 1
        L = np.zeros((n, n_free), dtype=float)
        for i in range(1, n):
            L[i, :i] = 1.0

        A_mono = constraint @ L
        if D.size:
            D_mono = D @ L
            stack = np.vstack([A_mono, np.sqrt(max(smoothness_alpha, 0.0)) * D_mono])
            rhs = np.concatenate([target, np.zeros(D_mono.shape[0], dtype=float)])
        else:
            stack = A_mono
            rhs = target

        result = lsq_linear(stack, rhs, bounds=(0.0, np.inf), lsmr_tol="auto")
        return L @ result.x

    @staticmethod
    def _default_solve_grid(n_constraints: int, n_grid: int | None = None) -> int:
        """Choose solve-grid size commensurate with sparse MTTI constraints."""
        if n_grid is not None and n_grid > 0:
            return max(int(n_grid), 50)
        return int(np.clip(8 * max(n_constraints, 3), 50, 200))

    def reconstruct_material_time(
        self,
        waiting_times_ps: np.ndarray,
        relaxation_times_ps: np.ndarray,
        time_grid_ps: Optional[np.ndarray] = None,
        origin_time_ps: float = 0.0,
        n_grid: Optional[int] = None,
    ) -> Tuple[np.ndarray, np.ndarray]:
        """Regularized LS reconstruction of h(t) from MTTI constraints (paper hat basis)."""
        tw = np.asarray(waiting_times_ps, dtype=float)
        tau = np.asarray(relaxation_times_ps, dtype=float)
        if tw.size == 0:
            return np.array([]), np.array([])

        t_origin = float(origin_time_ps)
        t_end = float(np.max(tw + tau))
        t_out = np.asarray(time_grid_ps, dtype=float) if time_grid_ps is not None else None
        t_max = max(t_end * 1.05, float(t_out[-1]) if t_out is not None and t_out.size else t_end * 1.05)

        n = self._default_solve_grid(len(tw), n_grid)
        t_solve = np.linspace(t_origin, t_max, n)

        rows: List[np.ndarray] = []
        rhs: List[float] = []
        for t_w, t_r in zip(tw, tau):
            row = self._hat_basis_at_time(float(t_w + t_r), t_solve) - self._hat_basis_at_time(
                float(t_w), t_solve
            )
            rows.append(row)
            rhs.append(1.0)

        anchor = self._hat_basis_at_time(t_origin, t_solve)
        rows.append(anchor)
        rhs.append(0.0)

        constraint = np.vstack(rows)
        target = np.asarray(rhs, dtype=float)
        h_coeffs = self._monotonic_hat_coefficients(
            constraint, target, self.smoothness_alpha, n
        )

        if t_out is None:
            t_max_out = t_end * 1.05
            t_out = np.linspace(t_origin, t_max_out, max(len(tw) * 20, 100))

        h_out = np.array(
            [self._evaluate_hat_expansion(float(t), t_solve, h_coeffs) for t in t_out],
            dtype=float,
        )
        h_out = np.maximum.accumulate(h_out)
        return t_out, h_out

    def integrate_tn(
        self,
        times_ps: np.ndarray,
        structural_temperatures_K: np.ndarray,
        switch_time_ps: float = 0.0,
        rate_scale: float = 1.0,
    ) -> np.ndarray:
        """Integrate dh/dt = rate_scale/tau_s,eq[T_s(t)] by trapezoidal rule."""
        t = np.asarray(times_ps, dtype=float)
        T_s = np.asarray(structural_temperatures_K, dtype=float)
        if len(t) < 2:
            return np.zeros_like(t)

        scale = max(float(rate_scale), 1e-12)
        h = np.zeros(len(t))
        for i in range(1, len(t)):
            if t[i] <= switch_time_ps:
                h[i] = 0.0
                continue
            t_start = max(t[i - 1], switch_time_ps)
            dt = t[i] - t_start
            if dt <= 0.0:
                h[i] = h[i - 1]
                continue
            T_start = float(np.interp(t_start, t, T_s))
            if self.relaxation_model is not None:
                tau_start = self.relaxation_model.get_relaxation_time(T_start)
                tau_end = self.relaxation_model.get_relaxation_time(T_s[i])
            else:
                tau_start = tau_end = 100.0
            rate = scale * 0.5 * (
                1.0 / max(tau_start, 1e-12) + 1.0 / max(tau_end, 1e-12)
            )
            h[i] = h[i - 1] + dt * rate
        return h

    def tn_material_increment(
        self,
        times_ps: np.ndarray,
        structural_temperatures_K: np.ndarray,
        t_start_ps: float,
        t_end_ps: float,
    ) -> float:
        """Integrate dh/dt = 1/tau_s,eq[T_s(t)] from t_start to t_end."""
        if self.relaxation_model is None or t_end_ps <= t_start_ps:
            return 0.0
        t = np.asarray(times_ps, dtype=float)
        T_s = np.asarray(structural_temperatures_K, dtype=float)
        n_steps = max(int((t_end_ps - t_start_ps) / 0.5), 4)
        t_fine = np.linspace(t_start_ps, t_end_ps, n_steps)
        T_fine = np.interp(t_fine, t, T_s)
        h_inc = 0.0
        for i in range(1, len(t_fine)):
            dt = t_fine[i] - t_fine[i - 1]
            tau_prev = self.relaxation_model.get_relaxation_time(T_fine[i - 1])
            tau_curr = self.relaxation_model.get_relaxation_time(T_fine[i])
            rate = 0.5 * (
                1.0 / max(tau_prev, 1e-12) + 1.0 / max(tau_curr, 1e-12)
            )
            h_inc += dt * rate
        return float(h_inc)

    def tau_s_tn_from_h(
        self,
        times_ps: np.ndarray,
        h_ps: np.ndarray,
        t_w_ps: float,
        switch_time_ps: float,
    ) -> Optional[float]:
        """Lab-time tau where pre-integrated TN material time increases by 1."""
        t0 = switch_time_ps + float(t_w_ps)
        t = np.asarray(times_ps, dtype=float)
        h = np.asarray(h_ps, dtype=float)
        if t.size < 2 or t0 >= float(t[-1]):
            return None
        h0 = float(np.interp(t0, t, h))
        h_target = h0 + 1.0
        mask = t >= t0
        t_seg = t[mask]
        h_seg = h[mask]
        if h_seg.size == 0 or float(h_seg[-1]) < h_target:
            return None
        idx = int(np.searchsorted(h_seg, h_target))
        if idx == 0:
            return float(t_seg[0] - t0)
        t_lo, t_hi = t_seg[idx - 1], t_seg[idx]
        h_lo, h_hi = h_seg[idx - 1], h_seg[idx]
        if h_hi == h_lo:
            return float(t_hi - t0)
        frac = (h_target - h_lo) / (h_hi - h_lo)
        return float(t_lo + frac * (t_hi - t_lo) - t0)

    def tau_s_tn(
        self,
        times_ps: np.ndarray,
        structural_temperatures_K: np.ndarray,
        t_w_ps: float,
        switch_time_ps: float,
        tau_max_ps: float = 5000.0,
    ) -> Optional[float]:
        """Lab-time tau where TN material-time increment equals 1 after switch + t_w."""
        h = self.integrate_tn(
            times_ps, structural_temperatures_K, switch_time_ps=switch_time_ps
        )
        return self.tau_s_tn_from_h(times_ps, h, t_w_ps, switch_time_ps)

    def collapse_isf(
        self,
        correlation_times_ps: np.ndarray,
        waiting_times_ps: np.ndarray,
        h_at_times: np.ndarray,
        time_grid_ps: np.ndarray,
    ) -> Tuple[np.ndarray, np.ndarray]:
        """Reparameterize ISF onto material time using interpolated h(t).

        ``waiting_times_ps`` must be absolute laboratory times (same frame as
        ``time_grid_ps``), e.g. switch_time + t_w.

        Returns ``(h_diff, h_w)`` where ``h_diff = h(t_w + tau) - h(t_w)``.
        Callers should pair ``h_diff`` with measured ISF values, not the second
        return value, for collapse plots.
        """
        t_corr = np.asarray(correlation_times_ps, dtype=float)
        t_w = np.asarray(waiting_times_ps, dtype=float)
        if t_w.size == 1:
            t_w_val = float(t_w[0])
            h_grid = np.interp(t_w_val + t_corr, time_grid_ps, h_at_times)
            h_w = float(np.interp(t_w_val, time_grid_ps, h_at_times))
        else:
            h_grid = np.array(
                [
                    np.interp(t_w[i] + t_corr[i], time_grid_ps, h_at_times)
                    for i in range(len(t_corr))
                ]
            )
            h_w = np.interp(t_w, time_grid_ps, h_at_times)
        h_diff = np.maximum(h_grid - h_w, 0.0)
        if np.ndim(h_w) == 0:
            h_w_out = float(h_w)
        else:
            h_w_out = h_w
        return h_diff, h_w_out
