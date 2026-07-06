#!/usr/bin/env python3
"""Quality checks for aging campaign production runs."""

from __future__ import annotations

import argparse
import json
import re
import subprocess
import sys
from dataclasses import asdict, dataclass, field
from pathlib import Path

import numpy as np

from config import (
    LAMBDAS,
    N_REPLICAS,
    RUNTIME_PS,
    SWITCH_TIME_PS,
    TEMPERATURE_K,
    job_dir_path,
    run_prefix,
)
from fkt_utils import collect_replica_fkt_files, normalize_fkt_to_phi, parse_fkt_file, replica_complete, snapshot_path


@dataclass
class RunIssue:
    replica: int
    lam: float
    check: str
    detail: str


@dataclass
class QCSummary:
    replica_range: list[int]
    n_replicas: int
    n_runs_expected: int
    n_runs_complete: int
    n_issues: int
    issues: list[dict] = field(default_factory=list)
    checks: dict = field(default_factory=dict)
    slurm_completed: int = 0


def _slurm_completed_replicas(job_id: int = 10756504) -> set[int]:
    try:
        out = subprocess.check_output(
            ["sacct", "-j", str(job_id), "--starttime=2026-06-13", "--format=JobID,State", "-n", "-P"],
            text=True,
            stderr=subprocess.DEVNULL,
        )
    except (subprocess.CalledProcessError, FileNotFoundError):
        return set()
    done: set[int] = set()
    for line in out.splitlines():
        m = re.match(rf"^{job_id}_(\d+)\|COMPLETED", line)
        if m:
            done.add(int(m.group(1)))
    return done


def _read_fkt_phi0(job_dir: Path, lam: float, replica: int) -> float | None:
    prefix = run_prefix(lam, replica)
    path = job_dir / f"{prefix}_fkt_ref_000.txt"
    if not path.exists():
        return None
    _, lags, values = parse_fkt_file(path)
    lags_n, phi = normalize_fkt_to_phi(lags, values)
    if lags_n is None or phi is None or phi.size == 0:
        return None
    return float(phi[int(np.argmin(np.abs(lags_n)))])


def validate_run(
    lam: float,
    replica: int,
    runtime_ps: float = RUNTIME_PS,
) -> list[RunIssue]:
    issues: list[RunIssue] = []
    job_dir = job_dir_path(lam)
    prefix = run_prefix(lam, replica)

    if not replica_complete(job_dir, lam, replica, runtime_ps):
        issues.append(
            RunIssue(replica, lam, "completion", "missing output or trajectory too short")
        )
        return issues

    csv_path = job_dir / f"{prefix}_energies.csv"
    data = np.genfromtxt(
        csv_path, delimiter=",", names=True, missing_values="", usemask=False
    )
    t = np.asarray(data["time_ps"], dtype=float)
    if t.size != int(runtime_ps):
        issues.append(
            RunIssue(
                replica,
                lam,
                "csv_rows",
                f"expected {int(runtime_ps)} rows, got {t.size}",
            )
        )
    if not np.allclose(np.diff(t), 1.0, atol=0.01):
        issues.append(RunIssue(replica, lam, "csv_time", "non-uniform 1 ps spacing"))

    T_bath = np.asarray(data["T_bath_K"], dtype=float)
    T_k = np.asarray(data["T_kinetic_K"], dtype=float)
    T_v = np.asarray(data["T_v_fictive_K"], dtype=float)
    T_s = np.asarray(data["T_s_fictive_K"], dtype=float)
    E_cpl = np.asarray(data["E_cav_coupling_kjmol"], dtype=float)

    if np.nanmax(np.abs(T_bath - TEMPERATURE_K)) > 1.0:
        issues.append(
            RunIssue(
                replica,
                lam,
                "bath_temperature",
                f"T_bath range {np.nanmin(T_bath):.1f}-{np.nanmax(T_bath):.1f} K",
            )
        )

    pre = t < SWITCH_TIME_PS - 0.5
    post = t >= SWITCH_TIME_PS + 0.5
    if pre.any():
        pre_k = T_k[pre]
        if np.nanmedian(pre_k) < 80 or np.nanmedian(pre_k) > 120:
            issues.append(
                RunIssue(
                    replica,
                    lam,
                    "pre_switch_kinetic",
                    f"median T_k={np.nanmedian(pre_k):.1f} K before switch",
                )
            )
        if lam > 0 and np.nanmax(np.abs(E_cpl[pre])) > 1e-3:
            issues.append(
                RunIssue(replica, lam, "pre_switch_coupling", "cavity coupling active before switch")
            )

    if lam == 0 and post.any() and np.nanmax(np.abs(E_cpl[post])) > 1e-3:
        issues.append(
            RunIssue(replica, lam, "lambda0_coupling", "unexpected coupling at λ=0 post-switch")
        )

    if lam > 0 and post.any():
        if np.nanmax(np.abs(E_cpl[post])) < 1e-6:
            issues.append(
                RunIssue(replica, lam, "post_switch_coupling", "cavity coupling never turned on")
            )

    late = t >= runtime_ps - 50
    if late.any():
        zero_ts = np.sum((T_s[late] == 0) | ~np.isfinite(T_s[late]))
        if zero_ts > 0.5 * np.sum(late):
            issues.append(
                RunIssue(
                    replica,
                    lam,
                    "T_s_calibration",
                    f"{100*zero_ts/max(np.sum(late),1):.0f}% late T_s zero "
                    "(likely incomplete OpenMM calibration table, not MD failure)",
                )
            )
        late_v = T_v[late]
        if np.nanmedian(late_v) < 5 or np.nanmedian(late_v) > 500:
            issues.append(
                RunIssue(
                    replica,
                    lam,
                    "T_v_fictive",
                    f"late median T_v={np.nanmedian(late_v):.1f} K",
                )
            )

    for col in ("E_potential_kjmol", "E_kinetic_kjmol", "E_mech_kjmol"):
        vals = np.asarray(data[col], dtype=float)
        if not np.all(np.isfinite(vals)):
            issues.append(RunIssue(replica, lam, "energy_nan", f"non-finite values in {col}"))

    fkt_files = collect_replica_fkt_files(job_dir, lam, replica)
    if not fkt_files:
        issues.append(RunIssue(replica, lam, "fkt_missing", "no FKT reference files"))
    else:
        phi0 = _read_fkt_phi0(job_dir, lam, replica)
        if phi0 is None or not (0.95 <= phi0 <= 1.05):
            issues.append(
                RunIssue(replica, lam, "fkt_normalize", f"phi(0)={phi0} at ref 0 (expected ~1)")
            )
        _, lags, values = parse_fkt_file(job_dir / f"{prefix}_fkt_ref_000.txt")
        lags_n, phi = normalize_fkt_to_phi(lags, values)
        if lags_n is not None and phi is not None and phi.size > 10:
            if not np.any(phi[lags_n >= 10.0] < 0.5):
                issues.append(
                    RunIssue(replica, lam, "fkt_decay", "FKT shows no decay below 0.5 by 10 ps lag")
                )

    snap = snapshot_path(job_dir, lam, replica)
    if not snap.exists():
        issues.append(RunIssue(replica, lam, "snapshots", "missing snapshots.npz"))
    else:
        try:
            snap_data = np.load(snap)
            times = np.asarray(snap_data["times_ps"], dtype=float)
            pos = np.asarray(snap_data["positions_nm"], dtype=float)
            if times.size < 2 or pos.ndim != 3:
                issues.append(RunIssue(replica, lam, "snapshots", "malformed snapshot array"))
            else:
                com = pos.mean(axis=1)
                drift = float(np.linalg.norm(com[-1] - com[0]))
                if drift > 2.0:
                    issues.append(
                        RunIssue(replica, lam, "com_drift", f"COM drift {drift:.2f} nm")
                    )
        except Exception as exc:
            issues.append(RunIssue(replica, lam, "snapshots", f"load error: {exc}"))

    return issues


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--replica-start", type=int, default=0)
    parser.add_argument("--replica-end", type=int, default=N_REPLICAS - 1)
    parser.add_argument("--output", type=Path, default=Path(__file__).resolve().parent / "results" / "qc_report.json")
    args = parser.parse_args()

    replicas = list(range(args.replica_start, args.replica_end + 1))
    slurm_done = _slurm_completed_replicas()
    slurm_in_range = slurm_done & set(replicas)

    all_issues: list[RunIssue] = []
    n_complete = 0
    for rep in replicas:
        for lam in LAMBDAS:
            if replica_complete(job_dir_path(lam), lam, rep, RUNTIME_PS):
                n_complete += 1
            all_issues.extend(validate_run(lam, rep))

    by_check: dict[str, int] = {}
    for issue in all_issues:
        by_check[issue.check] = by_check.get(issue.check, 0) + 1

    summary = QCSummary(
        replica_range=[args.replica_start, args.replica_end],
        n_replicas=len(replicas),
        n_runs_expected=len(replicas) * len(LAMBDAS),
        n_runs_complete=n_complete,
        n_issues=len(all_issues),
        issues=[asdict(i) for i in all_issues[:200]],
        checks={
            "completion_rate": n_complete / max(len(replicas) * len(LAMBDAS), 1),
            "issues_by_check": by_check,
            "slurm_completed_in_range": len(slurm_in_range),
            "slurm_complete_missing_outputs": len(
                [r for r in slurm_in_range if any(
                    not replica_complete(job_dir_path(lam), lam, r, RUNTIME_PS) for lam in LAMBDAS
                )]
            ),
        },
        slurm_completed=len(slurm_done),
    )

    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(asdict(summary), indent=2) + "\n", encoding="utf-8")

    print(f"Replicas {args.replica_start}-{args.replica_end}")
    print(f"  Complete runs: {n_complete}/{summary.n_runs_expected}")
    print(f"  QC issues: {len(all_issues)}")
    if by_check:
        print("  Issues by check:")
        for k, v in sorted(by_check.items(), key=lambda x: -x[1]):
            print(f"    {k}: {v}")
    if len(all_issues) > 200:
        print(f"  (first 200 issues written to {args.output})")
    print(f"Report -> {args.output}")
    return 1 if all_issues else 0


if __name__ == "__main__":
    sys.exit(main())
