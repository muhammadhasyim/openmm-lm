"""Tests for checkpoint/resume helpers and FKT tracker state."""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pytest

C2F_ROOT = Path(__file__).resolve().parents[1]
if str(C2F_ROOT) not in sys.path:
    sys.path.insert(0, str(C2F_ROOT))

from checkpoint_utils import (  # noqa: E402
    archive_replica_outputs,
    archive_stale_partial_outputs,
    checkpoint_path,
    load_checkpoint,
    read_csv_last_time_ps,
    save_checkpoint,
    trajectory_complete,
)
from fkt_tracker import FKTTracker  # noqa: E402


def test_fkt_tracker_state_roundtrip(tmp_path: Path) -> None:
    prefix = str(tmp_path / "run")
    tracker = FKTTracker(
        kmag_nm_inv=19.0,
        num_wavevectors=4,
        reference_interval_ps=10.0,
        max_references=3,
        output_period_ps=1.0,
        output_prefix=prefix,
    )
    pos = np.random.default_rng(0).random((10, 3))
    tracker.update(0.0, pos)
    tracker.update(1.0, pos + 0.01)
    tracker.update(11.0, pos + 0.02)

    state = tracker.to_state_dict()
    restored = FKTTracker.from_state_dict(state, output_prefix=prefix + "_resume")
    assert restored.last_reference_time_ps == tracker.last_reference_time_ps
    assert restored._next_ref_file_idx == tracker._next_ref_file_idx
    assert len(restored.references) == len(tracker.references)
    np.testing.assert_allclose(
        restored.references[0]["rhok_real"],
        tracker.references[0]["rhok_real"],
    )


def test_checkpoint_save_load_roundtrip(tmp_path: Path) -> None:
    prefix = tmp_path / "lam0_seed0042"
    ckpt = checkpoint_path(prefix)
    pos = np.ones((3, 3))
    vel = np.zeros((3, 3))
    fkt_state = {"last_reference_time_ps": 5.0, "references": []}
    save_checkpoint(
        ckpt,
        time_ps=42.0,
        positions_nm=pos,
        velocities_nm_per_ps=vel,
        fkt_state=fkt_state,
    )
    loaded = load_checkpoint(ckpt)
    assert loaded["time_ps"] == pytest.approx(42.0)
    np.testing.assert_array_equal(loaded["positions_nm"], pos)
    assert loaded["fkt_state"]["last_reference_time_ps"] == 5.0


def test_read_csv_last_time_and_completion(tmp_path: Path) -> None:
    csv_path = tmp_path / "energies.csv"
    csv_path.write_text(
        "time_ps,T_bath_K\n"
        "1.0,100\n"
        "2450.0,100\n",
        encoding="utf-8",
    )
    assert read_csv_last_time_ps(csv_path) == pytest.approx(2450.0)
    assert trajectory_complete(2450.0, 2500.0)
    assert not trajectory_complete(1750.0, 2500.0)


def test_archive_stale_partial_outputs(tmp_path: Path) -> None:
    prefix = tmp_path / "lam0_seed0099"
    csv_path = tmp_path / "lam0_seed0099_energies.csv"
    csv_path.write_text("time_ps,T\n1.0,100\n500.0,100\n", encoding="utf-8")
    fkt_path = tmp_path / "lam0_seed0099_fkt_ref_000.txt"
    fkt_path.write_text("# Reference time: 0.0 ps\n0.0\t1.0\n", encoding="utf-8")

    archive_dir = archive_stale_partial_outputs(prefix, runtime_ps=2500.0)
    assert archive_dir is not None
    assert archive_dir.is_dir()
    assert not csv_path.exists()
    assert (archive_dir / csv_path.name).exists()
    assert (archive_dir / fkt_path.name).exists()


def test_archive_replica_outputs_includes_complete_trajectory(tmp_path: Path) -> None:
    prefix = tmp_path / "lam0p03_seed0042"
    name = prefix.name
    (tmp_path / f"{name}_energies.csv").write_text(
        "time_ps,T\n2450.0,100\n", encoding="utf-8"
    )
    (tmp_path / f"{name}_fkt_ref_000.txt").write_text(
        "# Reference time: 0.0 ps\n0.0\t1.0\n", encoding="utf-8"
    )
    (tmp_path / f"{name}_final_state.npz").write_bytes(b"")
    (tmp_path / f"{name}_snapshots.npz").write_bytes(b"")
    save_checkpoint(
        checkpoint_path(prefix),
        time_ps=2450.0,
        positions_nm=np.ones((3, 3)),
        velocities_nm_per_ps=np.zeros((3, 3)),
    )

    archive_dir = archive_replica_outputs(
        prefix,
        reason="lambda003_rerun",
        runtime_ps=2500.0,
        lambda_coupling=0.03,
        replica=42,
    )
    assert archive_dir is not None
    assert archive_dir.is_dir()
    assert not (tmp_path / f"{name}_energies.csv").exists()
    assert not checkpoint_path(prefix).exists()
    assert (archive_dir / f"{name}_final_state.npz").exists()
    manifest = (archive_dir / "ARCHIVE_MANIFEST.txt").read_text(encoding="utf-8")
    assert "reason=lambda003_rerun" in manifest
    assert "lambda=0.03" in manifest
    assert "replica=42" in manifest
    assert trajectory_complete(2450.0, 2500.0)


def test_archive_replica_outputs_no_files(tmp_path: Path) -> None:
    prefix = tmp_path / "lam0p03_seed0999"
    assert archive_replica_outputs(prefix, reason="lambda003_rerun") is None
