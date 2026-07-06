"""Parse F(k,t) files from OpenMM aging runs."""

from __future__ import annotations

import re
from pathlib import Path
from typing import Dict, List, Literal, Optional, Tuple

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


def aggregate_replica_phi_stats(
    lag_dict: Dict[float, List[float]],
    *,
    envelope: bool,
    error: Literal["std", "sem"],
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    Aggregate normalized phi samples at common lags across replicas.

    When ``envelope`` is True, average |phi| (ISF magnitude) rather than signed phi.
    """
    lags_sorted = sorted(lag_dict)
    means: list[float] = []
    errors: list[float] = []
    for lag_key in lags_sorted:
        values = np.asarray(lag_dict[lag_key], dtype=np.float64)
        if envelope:
            values = np.abs(values)
        means.append(float(np.mean(values)))
        spread = float(np.std(values))
        if error == "sem":
            spread /= np.sqrt(values.size)
        errors.append(spread)
    return (
        np.asarray(lags_sorted, dtype=np.float64),
        np.asarray(means, dtype=np.float64),
        np.asarray(errors, dtype=np.float64),
    )


def build_phi_lag_dicts_for_replicas(
    job_dir: Path,
    lam: float,
    replicas: List[int],
    ref_indices: set[int] | None = None,
) -> tuple[dict[int, Dict[float, List[float]]], dict[int, Optional[float]], dict[int, int]]:
    """
    Load normalized phi samples for multiple reference indices in one pass.

    Returns lag_dicts[ref_idx], ref_times[ref_idx], n_used[ref_idx].
    """
    lag_dicts: dict[int, Dict[float, List[float]]] = {}
    ref_times: dict[int, Optional[float]] = {}
    n_used: dict[int, int] = {}
    allowed = ref_indices
    for replica in replicas:
        files = collect_replica_fkt_files(job_dir, lam, replica)
        for ref_idx, path in files.items():
            if allowed is not None and ref_idx not in allowed:
                continue
            rt, lags, vals = parse_fkt_file(path)
            norm = normalize_fkt_to_phi(lags, vals)
            if norm[0] is None:
                continue
            lags_n, phi = norm
            lag_dict = lag_dicts.setdefault(ref_idx, {})
            for lag, value in zip(lags_n, phi):
                lag_dict.setdefault(round(float(lag), 4), []).append(float(value))
            if rt is not None:
                ref_times[ref_idx] = rt
            n_used[ref_idx] = n_used.get(ref_idx, 0) + 1
    return lag_dicts, ref_times, n_used


def average_phi_over_replicas(
    job_dir: Path,
    lam: float,
    replicas: List[int],
    ref_idx: int,
    block_window_ps: float = 0.0,
    *,
    envelope: bool = False,
    error: Literal["std", "sem"] = "std",
    lag_dict: Dict[float, List[float]] | None = None,
    ref_time: Optional[float] = None,
    n_used: int | None = None,
) -> Tuple[Optional[float], np.ndarray, np.ndarray, np.ndarray, int]:
    """
    Normalize each replica to phi, then average at common lags.

    With ``envelope=True``, average |phi| for the ISF magnitude (recommended for
    oscillatory F(k,t)). ``error`` selects across-replica std or SEM for bands.
    """
    if lag_dict is None:
        lag_dict = {}
        ref_time = None
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
    elif n_used is None:
        raise ValueError("n_used is required when lag_dict is provided")
    if not lag_dict or n_used == 0:
        return ref_time, np.array([]), np.array([]), np.array([]), 0
    lags_arr, means, errors = aggregate_replica_phi_stats(
        lag_dict, envelope=envelope, error=error
    )
    if block_window_ps > 0.0:
        block_lags, block_means = block_average_abs_phi(
            lags_arr, means, window_ps=block_window_ps, min_lag_ps=0.0
        )
        _, block_errors = block_average_abs_phi(
            lags_arr, errors, window_ps=block_window_ps, min_lag_ps=0.0
        )
        return ref_time, block_lags, block_means, block_errors, n_used
    return ref_time, lags_arr, means, errors, n_used


def count_replicas_for_ref(
    job_dir: Path,
    lam: float,
    replicas: List[int],
    ref_idx: int,
) -> int:
    """Return how many replicas have an FKT file for a reference index."""
    from config import run_prefix

    return sum(
        1
        for replica in replicas
        if (job_dir / f"{run_prefix(lam, replica)}_fkt_ref_{ref_idx:03d}.txt").exists()
    )


def list_available_replicas(job_dir: Path, lam: float) -> List[int]:
    from config import BASE_SEED

    replicas: set[int] = set()
    for path in job_dir.glob(f"lam{lambda_tag(lam)}_seed*_fkt_ref_000.txt"):
        match = re.search(r"_seed(\d+)_fkt_ref_", path.name)
        if match:
            replicas.add(int(match.group(1)) - BASE_SEED)
    return sorted(replicas)


def list_available_snapshot_replicas(job_dir: Path, lam: float) -> List[int]:
    from config import BASE_SEED

    replicas: set[int] = set()
    for path in job_dir.glob(f"lam{lambda_tag(lam)}_seed*_snapshots.npz"):
        match = re.search(r"_seed(\d+)_snapshots\.npz$", path.name)
        if match:
            replicas.add(int(match.group(1)) - BASE_SEED)
    return sorted(replicas)


def snapshot_path(job_dir: Path, lam: float, replica: int) -> Path:
    return job_dir / f"{run_prefix(lam, replica)}_snapshots.npz"


def replay_fkt_from_snapshots_npz(
    snapshot_file: Path,
    kmag_nm_inv: float,
    fkt_start_ps: float,
    ref_interval_ps: float,
    max_refs: int,
) -> Dict[int, Tuple[float, np.ndarray, np.ndarray]]:
    """
    Recompute F(k,t) from saved snapshots with COM-removed replay physics.

    Returns mapping ref_idx -> (ref_time_ps, lag_times_ps, fkt_values).
    Reference times are spaced by ``ref_interval_ps`` from ``fkt_start_ps``.
    """
    from config import FKT_KMAG_NM_INV
    from fkt_physics import replay_fkt_from_trajectory_nm

    kmag = kmag_nm_inv if kmag_nm_inv is not None else FKT_KMAG_NM_INV
    data = np.load(snapshot_file)
    trajectory = np.asarray(data["positions_nm"], dtype=np.float64)
    times_ps = np.asarray(data["times_ps"], dtype=np.float64)
    if trajectory.ndim != 3 or times_ps.size < 2:
        return {}
    lag_ps = float(np.median(np.diff(times_ps)))
    start_idx = int(np.argmin(np.abs(times_ps - fkt_start_ps)))
    ref_indices: list[int] = []
    next_ref_time = float(times_ps[start_idx])
    max_time = float(times_ps[-1])
    while next_ref_time <= max_time + 1e-9 and len(ref_indices) < max_refs:
        ref_idx = int(np.argmin(np.abs(times_ps - next_ref_time)))
        if not ref_indices or ref_idx > ref_indices[-1]:
            ref_indices.append(ref_idx)
        next_ref_time += ref_interval_ps

    results: Dict[int, Tuple[float, np.ndarray, np.ndarray]] = {}
    for file_idx, frame_idx in enumerate(ref_indices):
        lags, values = replay_fkt_from_trajectory_nm(
            trajectory,
            kmag,
            lag_ps=lag_ps,
            reference_frame=frame_idx,
        )
        results[file_idx] = (float(times_ps[frame_idx] - fkt_start_ps), lags, values)
    return results


def average_phi_from_snapshots_over_replicas(
    job_dir: Path,
    lam: float,
    replicas: List[int],
    ref_idx: int,
    fkt_start_ps: float,
    ref_interval_ps: float = 200.0,
    max_refs: int = 13,
    block_window_ps: float = 10.0,
    kmag_nm_inv: float | None = None,
) -> Tuple[Optional[float], np.ndarray, np.ndarray, np.ndarray, int]:
    """Ensemble-average phi from COM-corrected snapshot replay."""
    ref_time: Optional[float] = None
    lag_dict: Dict[float, List[float]] = {}
    n_used = 0
    for replica in replicas:
        snap_path = snapshot_path(job_dir, lam, replica)
        if not snap_path.exists():
            continue
        replay = replay_fkt_from_snapshots_npz(
            snap_path,
            kmag_nm_inv=kmag_nm_inv,
            fkt_start_ps=fkt_start_ps,
            ref_interval_ps=ref_interval_ps,
            max_refs=max_refs,
        )
        if ref_idx not in replay:
            continue
        rt, lags, vals = replay[ref_idx]
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


def measure_com_drift_from_snapshots(
    snapshot_file: Path,
    n_atoms: int = 500,
    drift_window_ps: float = 100.0,
) -> Tuple[float, float]:
    """
    Return (max_com_displacement_nm, raw_msd_at_window_nm2) from snapshots.

    Uses molecular atoms only (first ``n_atoms`` sites per frame).
    """
    data = np.load(snapshot_file)
    atoms = np.asarray(data["positions_nm"], dtype=np.float64)[:, :n_atoms, :]
    times_ps = np.asarray(data["times_ps"], dtype=np.float64)
    com = atoms.mean(axis=1)
    com_disp = np.linalg.norm(com - com[0], axis=1)
    max_disp = float(np.max(com_disp)) if com_disp.size else 0.0
    msd_window = float("nan")
    if atoms.shape[0] > 1:
        target_time = float(times_ps[0] + drift_window_ps)
        frame_lag = int(np.argmin(np.abs(times_ps - target_time)))
        dr = atoms[frame_lag] - atoms[0]
        msd_window = float(np.mean(np.sum(dr**2, axis=1)))
    return max_disp, msd_window


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
