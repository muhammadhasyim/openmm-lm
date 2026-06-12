"""Parse F(k,t) files from OpenMM aging runs."""

from __future__ import annotations

import re
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np

from config import FKT_KMAG_AU, SWITCH_TIME_PS, lambda_tag, run_prefix
from fkt_physics import estimate_sk_time_average


def parse_fkt_file(path: Path) -> Tuple[Optional[float], np.ndarray, np.ndarray]:
    ref_time: Optional[float] = None
    lags: List[float] = []
    values: List[float] = []
    with open(path, encoding="utf-8") as fh:
        for line in fh:
            line = line.strip()
            if not line:
                continue
            if line.startswith("#"):
                match = re.search(r"Reference time:\s+([\d.]+)", line)
                if match:
                    ref_time = float(match.group(1))
                continue
            parts = line.split()
            if len(parts) >= 2:
                try:
                    lags.append(float(parts[0]))
                    values.append(float(parts[1]))
                except ValueError:
                    continue
    return ref_time, np.asarray(lags, dtype=float), np.asarray(values, dtype=float)


def collect_replica_fkt_files(job_dir: Path, lam: float, replica: int) -> Dict[int, Path]:
    prefix = run_prefix(lam, replica)
    files: Dict[int, Path] = {}
    for path in sorted(job_dir.glob(f"{prefix}_fkt_ref_*.txt")):
        match = re.search(r"_fkt_ref_(\d+)\.txt$", path.name)
        if match:
            files[int(match.group(1))] = path
    return files


def waiting_time_ps(ref_time_ps: Optional[float], ref_idx: int) -> float:
    """FKT reference times are relative to coupling turn-on in OpenMM output."""
    if ref_time_ps is None:
        return float(ref_idx) * 200.0
    return max(0.0, ref_time_ps)


def normalize_fkt_to_phi(
    lag_times: np.ndarray,
    fkt_values: np.ndarray,
    sk: float | None = None,
) -> tuple[np.ndarray, np.ndarray] | tuple[None, None]:
    """Return sorted (lags, phi) using F(0) or static S_k normalization."""
    if lag_times.size == 0:
        return None, None
    order = np.argsort(lag_times)
    lags = lag_times[order]
    values = fkt_values[order]
    idx0 = int(np.argmin(np.abs(lags)))
    f0 = float(sk if sk is not None else values[idx0])
    if f0 == 0.0:
        return None, None
    return lags, values / f0


def estimate_sk_from_fkt_file(
    path: Path,
    trajectory_nm: np.ndarray | None = None,
    kmag_au: float | None = None,
) -> float | None:
    """
    Estimate S_k for normalization: time-averaged shell S(k) from trajectory if
    available, else |F(0)| from the FKT reference file.
    """
    k_au = FKT_KMAG_AU if kmag_au is None else kmag_au
    if trajectory_nm is not None:
        return estimate_sk_time_average(trajectory_nm, k_au, site_mode="atomic")
    _, lags, vals = parse_fkt_file(path)
    if vals.size == 0:
        return None
    idx0 = int(np.argmin(np.abs(lags)))
    return abs(float(vals[idx0]))


def block_average_abs_phi(
    lag_times: np.ndarray,
    phi: np.ndarray,
    window_ps: float = 10.0,
    min_lag_ps: float = 0.0,
) -> tuple[np.ndarray, np.ndarray]:
    """Block average of |phi| for smoother ISF envelopes."""
    if lag_times.size == 0:
        return np.array([]), np.array([])
    order = np.argsort(lag_times)
    lags = lag_times[order]
    values = np.abs(phi[order])
    max_lag = float(lags[-1])
    if max_lag <= min_lag_ps:
        return np.array([]), np.array([])
    block_centers: list[float] = []
    block_values: list[float] = []
    start = min_lag_ps
    while start + window_ps <= max_lag + 1e-9:
        mask = (lags >= start) & (lags < start + window_ps)
        if np.any(mask):
            block_centers.append(start + 0.5 * window_ps)
            block_values.append(float(np.mean(values[mask])))
        start += window_ps
    return np.asarray(block_centers, dtype=float), np.asarray(block_values, dtype=float)


def extract_tau_s(
    lag_times: np.ndarray,
    fkt_values: np.ndarray,
    threshold: float = 0.1,
    min_lag_ps: float = 10.0,
    use_block_average: bool = True,
    block_window_ps: float = 10.0,
) -> Optional[float]:
    """
    Extract tau_s where phi(tau_s) = threshold.

    Ignores lags below ``min_lag_ps`` to avoid spurious early crossings from
    beta/vibrational dephasing in the real part of F(k,t). When
    ``use_block_average`` is True (default), uses block-averaged |phi| which
    matches the envelope used for cav-hoomd calibration on oscillatory F(k,t).
    """
    normalized = normalize_fkt_to_phi(lag_times, fkt_values)
    if normalized[0] is None:
        return None
    lags, phi = normalized
    if use_block_average:
        lags, phi = block_average_abs_phi(
            lags, phi, window_ps=block_window_ps, min_lag_ps=min_lag_ps
        )
    else:
        mask = lags >= min_lag_ps
        lags = lags[mask]
        phi = np.abs(phi[mask])
    if lags.size < 2:
        return None
    below = np.where(phi <= threshold)[0]
    if below.size == 0:
        return None
    idx = int(below[0])
    if idx == 0:
        return float(lags[0])
    t0, t1 = lags[idx - 1], lags[idx]
    p0, p1 = phi[idx - 1], phi[idx]
    if p1 == p0:
        return float(t1)
    return float(t0 + (threshold - p0) * (t1 - t0) / (p1 - p0))


def fit_kww_tau(
    lag_times: np.ndarray,
    fkt_values: np.ndarray,
    min_lag_ps: float = 10.0,
    beta: float = 0.55,
) -> Optional[float]:
    """
    Fit phi(t) ~ exp(-(t/tau)^beta) on lags >= min_lag_ps; return tau in ps.
    """
    normalized = normalize_fkt_to_phi(lag_times, fkt_values)
    if normalized[0] is None:
        return None
    lags, phi = normalized
    mask = (lags >= min_lag_ps) & (phi > 1e-6)
    lags = lags[mask]
    phi = phi[mask]
    if lags.size < 4:
        return None
    log_phi = np.log(phi)
    x = np.power(lags, beta)
    slope, _ = np.polyfit(x, log_phi, 1)
    if slope >= 0.0:
        return None
    tau = np.power(-1.0 / slope, 1.0 / beta)
    return float(tau)


def average_fkt_over_replicas(
    job_dir: Path,
    lam: float,
    replicas: List[int],
    ref_idx: int,
) -> Tuple[Optional[float], np.ndarray, np.ndarray, int]:
    ref_time: Optional[float] = None
    lag_dict: Dict[float, List[float]] = {}
    n_used = 0
    for replica in replicas:
        files = collect_replica_fkt_files(job_dir, lam, replica)
        if ref_idx not in files:
            continue
        rt, lags, vals = parse_fkt_file(files[ref_idx])
        if rt is not None:
            ref_time = rt
        for lag, val in zip(lags, vals):
            lag_dict.setdefault(round(float(lag), 4), []).append(float(val))
        n_used += 1
    if not lag_dict:
        return ref_time, np.array([]), np.array([]), 0
    lags_sorted = sorted(lag_dict)
    means = np.array([np.mean(lag_dict[k]) for k in lags_sorted])
    return ref_time, np.asarray(lags_sorted, dtype=float), means, n_used


def average_phi_over_replicas(
    job_dir: Path,
    lam: float,
    replicas: List[int],
    ref_idx: int,
    block_window_ps: float = 0.0,
) -> Tuple[Optional[float], np.ndarray, np.ndarray, np.ndarray, int]:
    """
    Normalize each replica to phi, then average phi and std at common lags.

    If ``block_window_ps`` > 0, return block-averaged |phi| on a uniform grid.
    """
    ref_time: Optional[float] = None
    lag_dict: Dict[float, List[float]] = {}
    n_used = 0
    for replica in replicas:
        files = collect_replica_fkt_files(job_dir, lam, replica)
        if ref_idx not in files:
            continue
        rt, lags, vals = parse_fkt_file(files[ref_idx])
        if rt is not None:
            ref_time = rt
        norm = normalize_fkt_to_phi(lags, vals)
        if norm[0] is None:
            continue
        lags_n, phi = norm
        for lag, value in zip(lags_n, phi):
            lag_dict.setdefault(round(float(lag), 4), []).append(float(value))
        n_used += 1
    if not lag_dict or n_used == 0:
        return ref_time, np.array([]), np.array([]), np.array([]), 0
    lags_sorted = sorted(lag_dict)
    means = np.array([np.mean(lag_dict[k]) for k in lags_sorted])
    stds = np.array([np.std(lag_dict[k]) for k in lags_sorted])
    lags_arr = np.asarray(lags_sorted, dtype=float)
    if block_window_ps > 0.0:
        block_lags, block_means = block_average_abs_phi(
            lags_arr, means, window_ps=block_window_ps, min_lag_ps=0.0
        )
        _, block_stds = block_average_abs_phi(
            lags_arr, stds, window_ps=block_window_ps, min_lag_ps=0.0
        )
        return ref_time, block_lags, block_means, block_stds, n_used
    return ref_time, lags_arr, means, stds, n_used


def list_available_replicas(job_dir: Path, lam: float) -> List[int]:
    from config import BASE_SEED

    replicas: set[int] = set()
    for path in job_dir.glob(f"lam{lambda_tag(lam)}_seed*_fkt_ref_000.txt"):
        match = re.search(r"_seed(\d+)_fkt_ref_", path.name)
        if match:
            replicas.add(int(match.group(1)) - BASE_SEED)
    return sorted(replicas)


def replica_complete(job_dir: Path, lam: float, replica: int, runtime_ps: float) -> bool:
    csv_path = job_dir / f"{run_prefix(lam, replica)}_energies.csv"
    fkt_path = job_dir / f"{run_prefix(lam, replica)}_fkt_ref_000.txt"
    if not csv_path.exists() or not fkt_path.exists():
        return False
    try:
        lines = csv_path.read_text().strip().splitlines()
        if len(lines) < 2:
            return False
        last_time = float(lines[-1].split(",")[0])
    except (ValueError, IndexError):
        return False
    return last_time >= 0.98 * runtime_ps
