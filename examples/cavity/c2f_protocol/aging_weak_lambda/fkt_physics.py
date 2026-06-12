"""Shared F(k,t) physics helpers for unit audits and k diagnostics."""

from __future__ import annotations

from typing import Iterable, Optional

import numpy as np

from config import BOHR_TO_NM, FKT_KMAG_AU, FKT_KMAG_NM_INV, FKT_NUM_WAVEVECTORS

_C2F_ROOT = __import__("pathlib").Path(__file__).resolve().parent.parent
import sys

if str(_C2F_ROOT) not in sys.path:
    sys.path.insert(0, str(_C2F_ROOT))

from fkt_tracker import (  # noqa: E402
    compute_fkt,
    compute_rhok,
    fibonacci_sphere,
    fkt_positions_nm,
)
from run_c2f import (  # noqa: E402
    BOX_AU,
    NUM_MOL,
    R0_AA_AU,
    R0_BB_AU,
    SIG_AA_AU,
)


def kmag_nm_from_au(kmag_au: float) -> float:
    return kmag_au / BOHR_TO_NM


def kmag_au_from_nm(kmag_nm_inv: float) -> float:
    return kmag_nm_inv * BOHR_TO_NM


def wavevectors_nm(kmag_nm_inv: float, num_wavevectors: int = FKT_NUM_WAVEVECTORS) -> np.ndarray:
    return fibonacci_sphere(num_wavevectors) * kmag_nm_inv


def wavevectors_au(kmag_au: float, num_wavevectors: int = FKT_NUM_WAVEVECTORS) -> np.ndarray:
    return fibonacci_sphere(num_wavevectors) * kmag_au


def compute_f0_from_positions_nm(
    positions_nm: np.ndarray,
    kmag_nm_inv: float,
    num_molecules: int = NUM_MOL,
    site_mode: str = "atomic",
) -> float:
    """Self-correlation F(0) = mean_k Re(rho_k rho_k*)."""
    pos = fkt_positions_nm(positions_nm, num_molecules, site_mode=site_mode)
    wv = wavevectors_nm(kmag_nm_inv)
    rhok_r, rhok_i = compute_rhok(pos, wv)
    return compute_fkt(rhok_r, rhok_i, rhok_r, rhok_i)


def compute_f0_hoomd_style(positions_bohr: np.ndarray, kmag_au: float) -> float:
    """HOOMD FieldAutocorrelationTracker F(0) with positions in Bohr."""
    wv = wavevectors_au(kmag_au)
    k_dot_r = np.dot(wv, positions_bohr.T)
    rhok = np.sum(np.exp(1j * k_dot_r), axis=1)
    return float(np.mean(np.real(rhok * np.conj(rhok))))


def compute_sk_shell(
    positions_nm: np.ndarray,
    kmag_au: float,
    num_molecules: int = NUM_MOL,
    site_mode: str = "atomic",
) -> float:
    """
    Shell-averaged static structure factor S(k) for one frame.

    S(k) = (1/N) mean_{|k_s|=k} |rho_{k_s}|^2
    """
    pos = fkt_positions_nm(positions_nm, num_molecules, site_mode=site_mode)
    n_sites = pos.shape[0]
    wv = wavevectors_au(kmag_au)
    k_dot_r = np.dot(wv, (pos / BOHR_TO_NM).T)
    rhok = np.sum(np.exp(1j * k_dot_r), axis=1)
    return float(np.mean(np.abs(rhok) ** 2) / n_sites)


def sk_curve(
    positions_nm: np.ndarray,
    kmag_au_grid: Iterable[float],
    site_mode: str = "atomic",
) -> tuple[np.ndarray, np.ndarray]:
    grid = np.asarray(list(kmag_au_grid), dtype=float)
    values = np.array(
        [compute_sk_shell(positions_nm, k, site_mode=site_mode) for k in grid]
    )
    return grid, values


def bond_lengths_nm(positions_nm: np.ndarray, num_molecules: int = NUM_MOL) -> np.ndarray:
    pairs = positions_nm.reshape(num_molecules, 2, 3)
    return np.linalg.norm(pairs[:, 0, :] - pairs[:, 1, :], axis=1)


def replay_fkt_from_trajectory_nm(
    trajectory_nm: np.ndarray,
    kmag_nm_inv: float,
    lag_ps: float = 1.0,
    num_molecules: int = NUM_MOL,
    site_mode: str = "atomic",
) -> tuple[np.ndarray, np.ndarray]:
    """
    Recompute F(k,t) from a position time series (T, N_atoms, 3) in nm.

    Returns lag_times_ps, fkt_values for reference at frame 0.
    """
    n_frames = trajectory_nm.shape[0]
    wv = wavevectors_nm(kmag_nm_inv)
    pos0 = fkt_positions_nm(trajectory_nm[0], num_molecules, site_mode=site_mode)
    r0_r, r0_i = compute_rhok(pos0, wv)
    lags: list[float] = []
    values: list[float] = []
    for frame in range(n_frames):
        pos = fkt_positions_nm(trajectory_nm[frame], num_molecules, site_mode=site_mode)
        rr, ri = compute_rhok(pos, wv)
        lags.append(frame * lag_ps)
        values.append(compute_fkt(r0_r, r0_i, rr, ri))
    return np.asarray(lags, dtype=float), np.asarray(values, dtype=float)


def estimate_sk_time_average(
    trajectory_nm: np.ndarray,
    kmag_au: float,
    num_molecules: int = NUM_MOL,
    site_mode: str = "atomic",
) -> float:
    """Time-averaged S(k) = <|rho_k|^2>/N over trajectory frames."""
    values = [
        compute_sk_shell(trajectory_nm[t], kmag_au, num_molecules, site_mode)
        for t in range(trajectory_nm.shape[0])
    ]
    return float(np.mean(values))


def normalize_fkt_by_sk(
    lag_times: np.ndarray,
    fkt_values: np.ndarray,
    sk: float,
) -> tuple[np.ndarray, np.ndarray]:
    if sk == 0.0:
        return np.array([]), np.array([])
    order = np.argsort(lag_times)
    return lag_times[order], fkt_values[order] / sk


def dimensionless_products(kmag_au: float = FKT_KMAG_AU) -> dict[str, float]:
    return {
        "k_sigma_AA": kmag_au * SIG_AA_AU,
        "k_r0_AA": kmag_au * R0_AA_AU,
        "k_r0_BB": kmag_au * R0_BB_AU,
        "k_L_box": kmag_au * BOX_AU,
        "k_2pi_over_sigma_AA": 2.0 * np.pi / SIG_AA_AU,
        "k_2pi_over_L": 2.0 * np.pi / BOX_AU,
    }
