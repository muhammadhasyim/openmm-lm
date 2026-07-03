"""Parse F(k,t) files from OpenMM aging runs."""

from __future__ import annotations

import re
from pathlib import Path
from typing import Dict, List, Literal, Optional, Tuple

import numpy as np
from scipy.interpolate import interp1d

from config import FKT_KMAG_AU, MASTER_FKT_DIR, SWITCH_TIME_PS, lambda_tag, run_prefix
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


_BAD_ENERGY_ARCHIVE_MARKERS = (
    "poisoned",
    "timeout",
    "no_checkpoint",
    "no_resume",
    "blown_up",
    "lambda003_rerun",
)


def _is_bad_archive_path(path: Path) -> bool:
    path_str = str(path)
    return any(marker in path_str for marker in _BAD_ENERGY_ARCHIVE_MARKERS)


def _artifact_archive_preference(path: Path) -> tuple[int, float]:
    """Sort key for archived artifacts: prefer top-level, then full_rerun, then mtime."""
    path_str = str(path)
    if "archive" not in path_str:
        tier = 2
    elif "full_rerun" in path_str:
        tier = 1
    else:
        tier = 0
    try:
        mtime = path.stat().st_mtime
    except OSError:
        mtime = 0.0
    return tier, mtime


def _replica_artifact_roots(
    job_dir: Path,
    lam: float,
    replica: int,
    artifact: str,
) -> list[Path]:
    """Return parent directories containing ``{prefix}_{artifact}`` for *replica*."""
    prefix = run_prefix(lam, replica)
    pattern = f"**/{prefix}_{artifact}"
    roots: list[Path] = []
    for path in job_dir.glob(pattern):
        if _is_bad_archive_path(path):
            continue
        roots.append(path.parent)
    return roots


def _score_replica_root(root: Path, lam: float, replica: int) -> tuple[int, int, float]:
    """Prefer roots with more FKT references, then archive tier, then mtime."""
    prefix = run_prefix(lam, replica)
    ref_files = list(root.glob(f"{prefix}_fkt_ref_*.txt"))
    anchor = root / f"{prefix}_fkt_ref_000.txt"
    tier, mtime = _artifact_archive_preference(anchor if anchor.is_file() else root)
    return len(ref_files), tier, mtime


_REPLICA_ROOT_INDEX: dict[tuple[str, float], dict[int, Path]] = {}


def build_replica_root_index(job_dir: Path, lam: float) -> Dict[int, Path]:
    """Build replica -> best data root mapping with one recursive glob."""
    from config import BASE_SEED

    key = (str(job_dir.resolve()), lam)
    cached = _REPLICA_ROOT_INDEX.get(key)
    if cached is not None:
        return cached

    candidates: Dict[int, list[Path]] = {}
    pattern = f"**/lam{lambda_tag(lam)}_seed*_fkt_ref_000.txt"
    for path in job_dir.glob(pattern):
        if _is_bad_archive_path(path):
            continue
        match = re.search(r"_seed(\d+)_fkt_ref_", path.name)
        if not match:
            continue
        replica = int(match.group(1)) - BASE_SEED
        candidates.setdefault(replica, []).append(path.parent)

    index = {
        replica: max(roots, key=lambda root: _score_replica_root(root, lam, replica))
        for replica, roots in candidates.items()
    }
    _REPLICA_ROOT_INDEX[key] = index
    return index


def resolve_replica_root(job_dir: Path, lam: float, replica: int) -> Optional[Path]:
    """Return the best data root (top-level or archive dir) for *replica*."""
    return build_replica_root_index(job_dir, lam).get(replica)


def resolve_replica_artifact(
    job_dir: Path,
    lam: float,
    replica: int,
    artifact_suffix: str,
) -> Optional[Path]:
    """Resolve ``{prefix}_{artifact_suffix}`` from the best replica root."""
    root = resolve_replica_root(job_dir, lam, replica)
    if root is None:
        return None
    path = root / f"{run_prefix(lam, replica)}_{artifact_suffix}"
    return path if path.is_file() else None


def collect_replica_fkt_files(job_dir: Path, lam: float, replica: int) -> Dict[int, Path]:
    root = resolve_replica_root(job_dir, lam, replica)
    if root is None:
        return {}
    prefix = run_prefix(lam, replica)
    files: Dict[int, Path] = {}
    for path in sorted(root.glob(f"{prefix}_fkt_ref_*.txt")):
        match = re.search(r"_fkt_ref_(\d+)\.txt$", path.name)
        if match:
            files[int(match.group(1))] = path
    return files


def waiting_time_ps(ref_time_ps: Optional[float], ref_idx: int) -> float:
    """FKT reference times are relative to coupling turn-on in OpenMM output."""
    if ref_time_ps is None:
        return float(ref_idx) * 200.0
    return max(0.0, ref_time_ps)


def master_fkt_dir(lam: float) -> Path:
    return MASTER_FKT_DIR / f"lambda{lambda_tag(lam)}"


def master_fkt_path(lam: float, ref_idx: int) -> Path:
    return master_fkt_dir(lam) / f"master_fkt_ref{ref_idx:03d}.txt"


def _parse_master_header(path: Path) -> tuple[Optional[float], int]:
    ref_time: Optional[float] = None
    n_replicas = 0
    with open(path, encoding="utf-8") as fh:
        for line in fh:
            if not line.startswith("#"):
                break
            match = re.search(r"Reference time:\s+([\d.]+)", line)
            if match:
                ref_time = float(match.group(1))
            match = re.search(r"n_replicas:\s+(\d+)", line)
            if match:
                n_replicas = int(match.group(1))
    return ref_time, n_replicas


def read_master_fkt(
    lam: float,
    ref_idx: int,
) -> Tuple[Optional[float], np.ndarray, np.ndarray, int]:
    """Read ensemble-averaged raw F(k,t) from a master file."""
    path = master_fkt_path(lam, ref_idx)
    if not path.is_file():
        return None, np.array([]), np.array([]), 0
    ref_time, n_replicas = _parse_master_header(path)
    lags: list[float] = []
    values: list[float] = []
    with open(path, encoding="utf-8") as fh:
        header_seen = False
        for line in fh:
            line = line.strip()
            if not line or line.startswith("#"):
                continue
            if not header_seen:
                header_seen = True
                if line.lower().startswith("lag_time"):
                    continue
            parts = line.split()
            if len(parts) >= 2:
                try:
                    lags.append(float(parts[0]))
                    values.append(float(parts[1]))
                except ValueError:
                    continue
    if not lags and n_replicas == 0:
        ref_time, n_replicas = _parse_master_header(path)
    return ref_time, np.asarray(lags, dtype=float), np.asarray(values, dtype=float), n_replicas


def list_master_refs(lam: float) -> List[int]:
    job = master_fkt_dir(lam)
    if not job.is_dir():
        return []
    refs: list[int] = []
    for path in sorted(job.glob("master_fkt_ref*.txt")):
        match = re.search(r"master_fkt_ref(\d+)\.txt$", path.name)
        if match:
            refs.append(int(match.group(1)))
    return refs


def first_nonzero_fkt(fkt_values: np.ndarray) -> Optional[float]:
    nonzero_mask = fkt_values != 0
    if not np.any(nonzero_mask):
        return None
    return float(fkt_values[int(np.where(nonzero_mask)[0][0])])


def ref0_normalization_value(
    lam: float,
    *,
    job_dir: Path | None = None,
    replicas: List[int] | None = None,
) -> Optional[float]:
    """Return ref0 F(k,0) used to normalize all waiting times at *lam*."""
    _, _, fkt, _ = read_master_fkt(lam, 0)
    if fkt.size:
        return first_nonzero_fkt(fkt)
    if job_dir is None:
        from config import job_dir_path

        job_dir = job_dir_path(lam)
    if replicas is None:
        replicas = list_analysis_replicas(job_dir, lam)
    else:
        replicas = list_analysis_replicas(job_dir, lam, replicas)
    ref_time, lags, mean_fkt, n_used = average_fkt_over_replicas(
        job_dir, lam, replicas, 0
    )
    if n_used == 0 or mean_fkt.size == 0:
        return None
    return first_nonzero_fkt(mean_fkt)


def normalize_fkt_by_ref0(
    lag_times: np.ndarray,
    fkt_values: np.ndarray,
    normalization_value: float | None,
) -> tuple[np.ndarray, np.ndarray] | tuple[None, None]:
    """Normalize raw F(k,t) by a single ref0 amplitude."""
    if lag_times.size == 0 or normalization_value is None or normalization_value == 0.0:
        return None, None
    order = np.argsort(lag_times)
    lags = lag_times[order]
    values = fkt_values[order]
    return lags, values / normalization_value


def process_fkt_data(
    time: np.ndarray,
    fkt: np.ndarray,
    normalization_value: float | None = None,
    *,
    min_normalized: float = 0.001,
) -> tuple[np.ndarray, np.ndarray, float | None] | tuple[None, None, None]:
    """Normalize F(k,t) and trim below *min_normalized* (cav-hoomd convention)."""
    time = np.asarray(time, dtype=np.float64)
    fkt = np.asarray(fkt, dtype=np.float64)
    valid_mask = ~(np.isnan(time) | np.isnan(fkt)) & (fkt != 0)
    time_clean = time[valid_mask]
    fkt_clean = fkt[valid_mask]
    if time_clean.size < 2:
        return None, None, None
    order = np.argsort(time_clean)
    time_sorted = time_clean[order]
    fkt_sorted = fkt_clean[order]
    if normalization_value is not None and normalization_value != 0:
        fkt_normalized = fkt_sorted / normalization_value
    elif fkt_sorted.size:
        fkt_normalized = fkt_sorted / fkt_sorted[0]
    else:
        return None, None, None
    above = fkt_normalized >= min_normalized
    if not np.any(above):
        return None, None, None
    time_filtered = time_sorted[above]
    fkt_filtered = fkt_normalized[above]
    max_time = float(time_filtered[-1]) if time_filtered.size else None
    return time_filtered, fkt_filtered, max_time


def find_relaxation_time(
    time: np.ndarray,
    fkt: np.ndarray,
    target_value: float = 0.1,
    normalization_value: float | None = None,
) -> Optional[float]:
    """Find τ where F(k,t)=target_value after ref0-single normalization."""
    try:
        valid_mask = ~(np.isnan(time) | np.isnan(fkt)) & (fkt != 0)
        time_clean = time[valid_mask]
        fkt_clean = fkt[valid_mask]
        if time_clean.size < 2:
            return None
        order = np.argsort(time_clean)
        time_sorted = time_clean[order]
        fkt_sorted = fkt_clean[order]
        if fkt_sorted.size > 10:
            n_keep = int(0.995 * fkt_sorted.size)
            fkt_sorted = fkt_sorted[:n_keep]
            time_sorted = time_sorted[:n_keep]
        if normalization_value is not None and normalization_value != 0:
            fkt_sorted = fkt_sorted / normalization_value
        elif fkt_sorted.size:
            fkt_sorted = fkt_sorted / fkt_sorted[0]
        else:
            return None
        if target_value < fkt_sorted.min() or target_value > fkt_sorted.max():
            return None
        _, unique_indices = np.unique(fkt_sorted, return_index=True)
        unique_indices = np.sort(unique_indices)
        fkt_unique = fkt_sorted[unique_indices]
        time_unique = time_sorted[unique_indices]
        if fkt_unique.size < 2:
            return None
        if target_value < fkt_unique.min() or target_value > fkt_unique.max():
            return None
        if not np.all(np.diff(fkt_unique) <= 0):
            crossing_indices = np.where(
                (fkt_sorted[:-1] >= target_value) & (fkt_sorted[1:] <= target_value)
            )[0]
            if crossing_indices.size == 0:
                return None
            idx = int(crossing_indices[0])
            t1, t2 = time_sorted[idx], time_sorted[idx + 1]
            f1, f2 = fkt_sorted[idx], fkt_sorted[idx + 1]
            if f1 != f2:
                tau = t1 + (target_value - f1) * (t2 - t1) / (f2 - f1)
                return tau if tau >= 0 else None
            return t1 if t1 >= 0 else None
        if fkt_unique.size > 3:
            interp_func = interp1d(
                fkt_unique, time_unique, kind="cubic", bounds_error=False, fill_value=np.nan
            )
        else:
            interp_func = interp1d(
                fkt_unique, time_unique, kind="linear", bounds_error=False, fill_value=np.nan
            )
        relaxation_time = float(interp_func(target_value))
        if np.isnan(relaxation_time) or relaxation_time < 0:
            return None
        return relaxation_time
    except Exception:
        return None


def load_lambda_fkt_data(
    lam: float,
    replicas: List[int],
    *,
    job_dir: Path | None = None,
) -> tuple[dict[int, tuple[Optional[float], np.ndarray, np.ndarray]], Optional[float], int]:
    """Load all reference-index F(k,t) curves for *lam* from master files or replicas."""
    if job_dir is None:
        from config import job_dir_path

        job_dir = job_dir_path(lam)
    ref_indices = list_master_refs(lam)
    data: dict[int, tuple[Optional[float], np.ndarray, np.ndarray]] = {}
    n_replicas = 0
    if ref_indices:
        for ref_idx in ref_indices:
            ref_time, lags, fkt, n_used = read_master_fkt(lam, ref_idx)
            if lags.size == 0:
                continue
            data[ref_idx] = (ref_time, lags, fkt)
            n_replicas = max(n_replicas, n_used)
    else:
        available = list_analysis_replicas(job_dir, lam, replicas)
        ref_indices = sorted(
            {
                idx
                for replica in available
                for idx in collect_replica_fkt_files(job_dir, lam, replica)
            }
        )
        for ref_idx in ref_indices:
            ref_time, lags, mean_fkt, n_used = average_fkt_over_replicas(
                job_dir, lam, available, ref_idx
            )
            if lags.size == 0:
                continue
            data[ref_idx] = (ref_time, lags, mean_fkt)
            n_replicas = max(n_replicas, n_used)
    norm_value = ref0_normalization_value(lam, job_dir=job_dir, replicas=replicas)
    return data, norm_value, n_replicas


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
    use_block_average: bool = False,
    block_window_ps: float = 10.0,
    normalization_value: float | None = None,
) -> Optional[float]:
    """
    Extract τ_s where F(k,t)/F_ref0 = threshold using cav-hoomd interpolation.

    When *normalization_value* is set, all curves for a λ share ref0-single
    normalization. Legacy block-averaging remains available for diagnostics.
    """
    if use_block_average:
        normalized = normalize_fkt_to_phi(lag_times, fkt_values)
        if normalized[0] is None:
            return None
        lags, phi = normalized
        lags, phi = block_average_abs_phi(
            lags, phi, window_ps=block_window_ps, min_lag_ps=min_lag_ps
        )
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
    return find_relaxation_time(
        lag_times,
        fkt_values,
        target_value=threshold,
        normalization_value=normalization_value,
    )


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
    return sum(
        1
        for replica in replicas
        if ref_idx in collect_replica_fkt_files(job_dir, lam, replica)
    )


def list_available_replicas(job_dir: Path, lam: float) -> List[int]:
    return sorted(build_replica_root_index(job_dir, lam))


def list_analysis_replicas(
    job_dir: Path,
    lam: float,
    replicas: List[int] | None = None,
    *,
    strict_qc: bool = True,
) -> List[int]:
    """Replicas with FKT data, optionally filtered by stability-critical QC."""
    if replicas is None:
        candidates = list_available_replicas(job_dir, lam)
    else:
        available = set(list_available_replicas(job_dir, lam))
        candidates = [r for r in replicas if r in available]
    if not strict_qc:
        return sorted(candidates)
    from replica_qc import list_qc_passing_replicas

    passing = set(list_qc_passing_replicas(lam, candidates))
    return sorted(r for r in candidates if r in passing)


def _energy_csv_is_valid(path: Path) -> bool:
    """Reject trajectories with non-physical energies or kinetic temperatures."""
    try:
        data = np.genfromtxt(
            path, delimiter=",", names=True, missing_values="", usemask=False
        )
        tk = np.asarray(data["T_kinetic_K"], dtype=float)
        eb = np.asarray(data["E_bond_kjmol"], dtype=float)
        enb = np.asarray(data["E_nonbonded_kjmol"], dtype=float)
        if not (np.all(np.isfinite(tk)) and np.all(np.isfinite(eb)) and np.all(np.isfinite(enb))):
            return False
        if np.max(tk) > 300.0 or np.min(tk) < 0.0:
            return False
        if np.max(eb) > 500.0 or np.min(eb) < 0.0:
            return False
        if np.max(enb) > 0.0 or np.min(enb) < -5000.0:
            return False
        return True
    except (OSError, ValueError, IndexError):
        return False


def _archive_preference(path: Path) -> tuple[int, float, float]:
    """Sort key: prefer top-level, then full_rerun, then longest runtime."""
    path_str = str(path)
    if "archive" not in path_str:
        tier = 2
    elif "full_rerun" in path_str:
        tier = 1
    else:
        tier = 0
    t_min, t_max = _energy_csv_time_bounds(path)
    return (tier, t_max, -t_min)


def _energy_csv_time_bounds(path: Path) -> Tuple[float, float]:
    """Return (t_min, t_max) in ps from the first and last rows of an energy CSV."""
    with path.open(encoding="utf-8") as fh:
        header = fh.readline()
        first = fh.readline().split(",")[0]
        last_line = first
        for line in fh:
            if line.strip():
                last_line = line
    return float(first), float(last_line.split(",")[0])


def build_energy_csv_index(job_dir: Path, lam: float) -> Dict[int, List[Path]]:
    """Map replica index -> all discovered ``*_energies.csv`` paths (top-level + archives)."""
    from config import BASE_SEED

    index: Dict[int, List[Path]] = {}
    pattern = f"**/lam{lambda_tag(lam)}_seed*_energies.csv"
    for path in job_dir.glob(pattern):
        match = re.search(r"_seed(\d+)_energies\.csv$", path.name)
        if not match:
            continue
        replica = int(match.group(1)) - BASE_SEED
        index.setdefault(replica, []).append(path)
    return index


def resolve_energy_csv(
    job_dir: Path,
    lam: float,
    replica: int,
    index: Optional[Dict[int, List[Path]]] = None,
) -> Optional[Path]:
    """Return the best valid energy CSV for *replica*."""
    if index is None:
        index = build_energy_csv_index(job_dir, lam)

    candidates: List[Path] = []
    top = job_dir / f"{run_prefix(lam, replica)}_energies.csv"
    if top.is_file():
        candidates.append(top)
    for path in index.get(replica, []):
        if path not in candidates:
            candidates.append(path)

    good = [
        p
        for p in candidates
        if not any(m in str(p) for m in _BAD_ENERGY_ARCHIVE_MARKERS)
        and _energy_csv_is_valid(p)
    ]
    if not good:
        return None
    return max(good, key=_archive_preference)


def list_available_energy_replicas(
    job_dir: Path,
    lam: float,
    *,
    min_tmax_ps: float = 0.0,
) -> List[int]:
    """Replica indices with a resolvable energy CSV reaching at least *min_tmax_ps*."""
    index = build_energy_csv_index(job_dir, lam)
    replicas: List[int] = []
    for replica in sorted(index):
        path = resolve_energy_csv(job_dir, lam, replica, index)
        if path is None:
            continue
        _t_min, t_max = _energy_csv_time_bounds(path)
        if t_max >= min_tmax_ps:
            replicas.append(replica)
    return replicas


def list_available_snapshot_replicas(job_dir: Path, lam: float) -> List[int]:
    replicas: list[int] = []
    for replica, root in build_replica_root_index(job_dir, lam).items():
        path = root / f"{run_prefix(lam, replica)}_snapshots.npz"
        if path.is_file():
            replicas.append(replica)
    return sorted(replicas)


def snapshot_path(job_dir: Path, lam: float, replica: int) -> Path:
    resolved = resolve_replica_artifact(job_dir, lam, replica, "snapshots.npz")
    if resolved is not None:
        return resolved
    return job_dir / f"{run_prefix(lam, replica)}_snapshots.npz"


def dipole_path(job_dir: Path, lam: float, replica: int) -> Optional[Path]:
    """Return archive-resolved dipole trajectory for IR post-processing."""
    return resolve_replica_artifact(job_dir, lam, replica, "dipole.npz")


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
