"""Checkpoint helpers for restartable cavity equilibrium runs."""

from __future__ import annotations

import csv
import math
import shutil
from datetime import datetime, timezone
from pathlib import Path

import numpy as np

COMPLETION_FRACTION = 0.98


def checkpoint_path(output_prefix: str | Path) -> Path:
    return Path(f"{output_prefix}_checkpoint.npz")


def energies_csv_path(output_prefix: str | Path) -> Path:
    return Path(f"{output_prefix}_energies.csv")


def read_csv_last_time_ps(csv_path: Path) -> float | None:
    if not csv_path.exists():
        return None
    try:
        with csv_path.open(encoding="utf-8") as fh:
            rows = list(csv.DictReader(fh))
        if not rows:
            return None
        return float(rows[-1]["time_ps"])
    except (ValueError, KeyError, OSError):
        return None


def trajectory_complete(last_time_ps: float | None, runtime_ps: float) -> bool:
    if last_time_ps is None:
        return False
    return last_time_ps >= COMPLETION_FRACTION * runtime_ps


def read_csv_row_near_time(
    csv_path: Path,
    time_ps: float,
    *,
    time_tolerance_ps: float = 1e-3,
) -> dict[str, str] | None:
    """Return the CSV row whose time_ps is closest to *time_ps* within tolerance."""
    if not csv_path.exists():
        return None
    try:
        with csv_path.open(encoding="utf-8") as fh:
            rows = list(csv.DictReader(fh))
        if not rows:
            return None
        best: dict[str, str] | None = None
        best_delta = float("inf")
        for row in rows:
            row_time = float(row["time_ps"])
            delta = abs(row_time - time_ps)
            if delta < best_delta:
                best_delta = delta
                best = row
        if best is None or best_delta > time_tolerance_ps:
            return None
        return best
    except (ValueError, KeyError, OSError):
        return None


def is_poisoned_checkpoint(
    checkpoint: dict,
    energies_csv: Path | None,
    *,
    t_kin_max_k: float = 5000.0,
    time_tolerance_ps: float = 1e-3,
) -> bool:
    """True if checkpoint or trailing CSV row shows numerical blow-up."""
    if energies_csv is None:
        return False
    row = read_csv_row_near_time(
        energies_csv,
        float(checkpoint["time_ps"]),
        time_tolerance_ps=time_tolerance_ps,
    )
    if row is None:
        return False
    try:
        t_kin = float(row["T_kinetic_K"])
        e_kin = float(row["E_kinetic_kjmol"])
        e_pot = float(row["E_potential_kjmol"])
    except (ValueError, KeyError):
        return False
    if not math.isfinite(t_kin) or not math.isfinite(e_kin) or not math.isfinite(e_pot):
        return True
    return t_kin > t_kin_max_k


def save_checkpoint(
    path: Path,
    *,
    time_ps: float,
    positions_nm: np.ndarray,
    velocities_nm_per_ps: np.ndarray,
    fkt_state: dict | None = None,
    dipole_state: dict | None = None,
    snapshot_times_ps: np.ndarray | None = None,
    snapshot_positions_nm: np.ndarray | None = None,
) -> None:
    payload: dict[str, object] = {
        "time_ps": float(time_ps),
        "positions_nm": np.asarray(positions_nm, dtype=np.float64),
        "velocities_nm_per_ps": np.asarray(velocities_nm_per_ps, dtype=np.float64),
        "saved_utc": datetime.now(timezone.utc).isoformat(),
    }
    if fkt_state is not None:
        payload["fkt_state"] = fkt_state
    if dipole_state is not None:
        payload["dipole_state"] = dipole_state
    if snapshot_times_ps is not None and snapshot_positions_nm is not None:
        payload["snapshot_times_ps"] = np.asarray(snapshot_times_ps, dtype=np.float64)
        payload["snapshot_positions_nm"] = np.asarray(snapshot_positions_nm, dtype=np.float64)
    path.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(path, **payload)


def load_checkpoint(path: Path) -> dict:
    data = np.load(path, allow_pickle=True)
    out: dict[str, object] = {
        "time_ps": float(data["time_ps"]),
        "positions_nm": np.asarray(data["positions_nm"], dtype=np.float64),
        "velocities_nm_per_ps": np.asarray(data["velocities_nm_per_ps"], dtype=np.float64),
    }
    if "fkt_state" in data:
        out["fkt_state"] = data["fkt_state"].item()
    if "dipole_state" in data:
        out["dipole_state"] = data["dipole_state"].item()
    if "snapshot_times_ps" in data and "snapshot_positions_nm" in data:
        out["snapshot_times_ps"] = np.asarray(data["snapshot_times_ps"], dtype=np.float64)
        out["snapshot_positions_nm"] = np.asarray(data["snapshot_positions_nm"], dtype=np.float64)
    return out


def archive_stale_partial_outputs(
    output_prefix: str | Path,
    *,
    runtime_ps: float,
    reason: str = "no_checkpoint",
) -> Path | None:
    """
    Move incomplete outputs (CSV/FKT/dipole) to a timestamped archive directory.

    Called when restarting from IC without a checkpoint so partial observables
    are preserved rather than overwritten.
    """
    prefix = Path(output_prefix)
    csv_path = energies_csv_path(prefix)
    last_time = read_csv_last_time_ps(csv_path)
    if trajectory_complete(last_time, runtime_ps):
        return None
    if last_time is None or last_time <= 0.0:
        return None

    stamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    archive_dir = prefix.parent / f"{prefix.name}_archive_{reason}_{stamp}"
    archive_dir.mkdir(parents=True, exist_ok=True)

    patterns = [
        f"{prefix.name}_energies.csv",
        f"{prefix.name}_fkt_ref_*.txt",
        f"{prefix.name}_dipole.npz",
        f"{prefix.name}_meta.txt",
        f"{prefix.name}_checkpoint.npz",
    ]
    moved = 0
    for pattern in patterns:
        for path in prefix.parent.glob(pattern):
            shutil.move(str(path), str(archive_dir / path.name))
            moved += 1
    if moved == 0:
        archive_dir.rmdir()
        return None
    manifest = archive_dir / "ARCHIVE_MANIFEST.txt"
    manifest.write_text(
        f"reason={reason}\n"
        f"last_time_ps={last_time}\n"
        f"runtime_ps={runtime_ps}\n"
        f"files_moved={moved}\n",
        encoding="utf-8",
    )
    return archive_dir


def _replica_output_patterns(prefix: Path) -> list[str]:
    return [
        f"{prefix.name}_energies.csv",
        f"{prefix.name}_fkt_ref_*.txt",
        f"{prefix.name}_dipole.npz",
        f"{prefix.name}_meta.txt",
        f"{prefix.name}_final_state.npz",
        f"{prefix.name}_snapshots.npz",
        f"{prefix.name}_checkpoint.npz",
    ]


def archive_replica_outputs(
    output_prefix: str | Path,
    *,
    reason: str,
    runtime_ps: float | None = None,
    lambda_coupling: float | None = None,
    replica: int | None = None,
) -> Path | None:
    """
    Move all replica outputs (including complete trajectories) to a timestamped archive.

    Unlike ``archive_stale_partial_outputs``, this archives finished runs too and
    includes final_state, snapshots, and checkpoint files.
    """
    prefix = Path(output_prefix)
    stamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    archive_dir = prefix.parent / f"{prefix.name}_archive_{reason}_{stamp}"
    archive_dir.mkdir(parents=True, exist_ok=True)

    moved = 0
    for pattern in _replica_output_patterns(prefix):
        for path in prefix.parent.glob(pattern):
            shutil.move(str(path), str(archive_dir / path.name))
            moved += 1

    if moved == 0:
        archive_dir.rmdir()
        return None

    csv_path = archive_dir / f"{prefix.name}_energies.csv"
    last_time = read_csv_last_time_ps(csv_path) if csv_path.exists() else None

    manifest_lines = [
        f"reason={reason}",
        f"timestamp_utc={stamp}",
        f"prefix={prefix.name}",
        f"files_moved={moved}",
    ]
    if lambda_coupling is not None:
        manifest_lines.append(f"lambda={lambda_coupling:g}")
    if replica is not None:
        manifest_lines.append(f"replica={replica}")
    if runtime_ps is not None:
        manifest_lines.append(f"runtime_ps={runtime_ps}")
    if last_time is not None:
        manifest_lines.append(f"last_time_ps={last_time}")

    (archive_dir / "ARCHIVE_MANIFEST.txt").write_text(
        "\n".join(manifest_lines) + "\n",
        encoding="utf-8",
    )
    return archive_dir
