"""Error estimates for serially correlated MD/Monte Carlo time series."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np


@dataclass(frozen=True)
class CorrelatedSummary:
    """Mean and standard error for a correlated scalar time series."""

    mean: float
    sem: float
    sem_crosscheck: float
    n_samples: int
    n_eff: float
    tau_int_samples: float
    block_size: int
    naive_sem: float

    @property
    def sem_pymbar(self) -> float:
        """Backward-compatible alias for ``sem_crosscheck``."""
        return self.sem_crosscheck


def _naive_sem(samples: np.ndarray) -> float:
    n = samples.size
    if n < 2:
        return 0.0
    return float(np.std(samples, ddof=1) / np.sqrt(n))


def _manual_block_sem(samples: np.ndarray) -> tuple[float, int]:
    """Flyvbjerg-style blocking without external deps; returns (sem, block_size)."""
    n = samples.size
    if n < 2:
        return 0.0, 1

    sem_by_block: list[tuple[int, float]] = []
    block_size = 1
    while block_size <= n // 2:
        n_blocks = n // block_size
        trimmed = samples[: n_blocks * block_size]
        block_means = trimmed.reshape(n_blocks, block_size).mean(axis=1)
        if n_blocks >= 2:
            sem = float(block_means.std(ddof=1) / np.sqrt(n_blocks))
            sem_by_block.append((block_size, sem))
        block_size *= 2

    if not sem_by_block:
        return _naive_sem(samples), 1

    # Plateau: use the largest block size with at least 5 blocks.
    eligible = [(b, s) for b, s in sem_by_block if (n // b) >= 5]
    if eligible:
        block_size, sem = eligible[-1]
        return sem, block_size
    block_size, sem = sem_by_block[-1]
    return sem, block_size


def _pyblock_sem(samples: np.ndarray) -> tuple[float | None, int | None, float | None]:
    try:
        import pyblock.blocking as blocking
    except ImportError:
        return None, None, None

    data = np.asarray(samples, dtype=float).ravel()
    stats = blocking.reblock(data)
    if not stats:
        return None, None, None

    opt_indices = blocking.find_optimal_block(data.size, stats)
    opt_idx = opt_indices[0]
    if np.isnan(opt_idx):
        return None, None, None

    opt_idx = int(opt_idx)

    entry = stats[opt_idx]
    sem = float(np.ravel(entry.std_err)[0])
    mean = float(np.ravel(entry.mean)[0])
    block_size = 2**entry.block
    return sem, block_size, mean


def summarize_correlated(samples: np.ndarray) -> CorrelatedSummary:
    """Compute mean and reblocked SEM using pyblock (manual blocking fallback)."""
    arr = np.asarray(samples, dtype=float).ravel()
    if arr.size < 2:
        raise ValueError("summarize_correlated requires at least 2 samples")

    mean = float(np.mean(arr))
    naive = _naive_sem(arr)
    if float(np.var(arr)) == 0.0:
        return CorrelatedSummary(
            mean=mean,
            sem=0.0,
            sem_crosscheck=0.0,
            n_samples=int(arr.size),
            n_eff=float(arr.size),
            tau_int_samples=0.0,
            block_size=1,
            naive_sem=0.0,
        )

    pyblock_sem, block_size, pyblock_mean = _pyblock_sem(arr)
    if pyblock_sem is not None and block_size is not None:
        sem = pyblock_sem
        mean = pyblock_mean if pyblock_mean is not None else mean
    else:
        sem, block_size = _manual_block_sem(arr)

    g = max(1.0, (naive / sem) ** 2) if sem > 0 else 1.0
    n_eff = float(arr.size / g)

    return CorrelatedSummary(
        mean=mean,
        sem=float(sem),
        sem_crosscheck=float(sem),
        n_samples=int(arr.size),
        n_eff=n_eff,
        tau_int_samples=float(0.5 * (g - 1.0)),
        block_size=int(block_size),
        naive_sem=float(naive),
    )
