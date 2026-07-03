#!/usr/bin/env python3
"""Stability-critical QC for aging campaign replicas (FKT/tau analysis)."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np

from config import ANALYSIS_LAMBDAS, N_REPLICAS, RESULTS_DIR, RUNTIME_PS, job_dir_path, run_prefix
from fkt_utils import (
    collect_replica_fkt_files,
    list_available_replicas,
    normalize_fkt_to_phi,
    parse_fkt_file,
    resolve_replica_root,
)

STABILITY_T_KIN_MAX_K = 5000.0
PHI0_MIN = 0.95
PHI0_MAX = 1.05
FKT_DECAY_LAG_PS = 10.0
FKT_DECAY_THRESHOLD = 0.5


def _replica_has_complete_trajectory(
    job_dir: Path,
    lam: float,
    replica: int,
    runtime_ps: float,
) -> tuple[bool, str | None]:
    root = resolve_replica_root(job_dir, lam, replica)
    if root is None:
        return False, "no FKT data root found"
    prefix = run_prefix(lam, replica)
    fkt_path = root / f"{prefix}_fkt_ref_000.txt"
    if not fkt_path.is_file():
        return False, "missing fkt_ref_000"
    csv_path = root / f"{prefix}_energies.csv"
    if not csv_path.is_file():
        return False, "missing energies CSV"
    try:
        lines = csv_path.read_text(encoding="utf-8").strip().splitlines()
        if len(lines) < 2:
            return False, "empty energies CSV"
        last_time = float(lines[-1].split(",")[0])
    except (ValueError, IndexError, OSError) as exc:
        return False, f"cannot read trajectory length: {exc}"
    if last_time < 0.98 * runtime_ps:
        return False, f"trajectory too short ({last_time:.0f} ps < {0.98 * runtime_ps:.0f} ps)"
    return True, None


def _read_fkt_phi0_from_root(root: Path, lam: float, replica: int) -> float | None:
    prefix = run_prefix(lam, replica)
    path = root / f"{prefix}_fkt_ref_000.txt"
    if not path.exists():
        return None
    _, lags, values = parse_fkt_file(path)
    lags_n, phi = normalize_fkt_to_phi(lags, values)
    if lags_n is None or phi is None or phi.size == 0:
        return None
    return float(phi[int(np.argmin(np.abs(lags_n)))])


def qc_replica(
    lam: float,
    replica: int,
    *,
    runtime_ps: float = RUNTIME_PS,
    t_kin_max_k: float = STABILITY_T_KIN_MAX_K,
) -> list[str]:
    """Return failure reasons; empty list means QC pass."""
    job_dir = job_dir_path(lam)
    failures: list[str] = []

    ok, reason = _replica_has_complete_trajectory(job_dir, lam, replica, runtime_ps)
    if not ok:
        failures.append(reason or "incomplete trajectory")
        return failures

    root = resolve_replica_root(job_dir, lam, replica)
    assert root is not None
    prefix = run_prefix(lam, replica)
    csv_path = root / f"{prefix}_energies.csv"
    try:
        data = np.genfromtxt(
            csv_path, delimiter=",", names=True, missing_values="", usemask=False
        )
    except (OSError, ValueError) as exc:
        failures.append(f"cannot read energies CSV: {exc}")
        return failures

    t_k = np.asarray(data["T_kinetic_K"], dtype=float)
    if not np.all(np.isfinite(t_k)):
        failures.append("non-finite T_kinetic_K")
    else:
        max_t_kin = float(np.max(t_k))
        if max_t_kin > t_kin_max_k:
            failures.append(f"T_kin max={max_t_kin:.4g} K > {t_kin_max_k:g} K")

    for col in ("E_potential_kjmol", "E_kinetic_kjmol", "E_mech_kjmol"):
        vals = np.asarray(data[col], dtype=float)
        if not np.all(np.isfinite(vals)):
            failures.append(f"non-finite values in {col}")

    fkt_files = collect_replica_fkt_files(job_dir, lam, replica)
    if not fkt_files:
        failures.append("no FKT reference files")
    else:
        phi0 = _read_fkt_phi0_from_root(root, lam, replica)
        if phi0 is None or not (PHI0_MIN <= phi0 <= PHI0_MAX):
            failures.append(f"phi(0)={phi0} at ref 0 (expected ~1)")
        ref0_path = root / f"{prefix}_fkt_ref_000.txt"
        if ref0_path.exists():
            _, lags, values = parse_fkt_file(ref0_path)
            lags_n, phi = normalize_fkt_to_phi(lags, values)
            if lags_n is not None and phi is not None and phi.size > 10:
                if not np.any(phi[lags_n >= FKT_DECAY_LAG_PS] < FKT_DECAY_THRESHOLD):
                    failures.append(
                        f"FKT shows no decay below {FKT_DECAY_THRESHOLD} by "
                        f"{FKT_DECAY_LAG_PS:g} ps lag"
                    )

    return failures


def list_qc_passing_replicas(
    lam: float,
    replicas: list[int] | None = None,
    *,
    runtime_ps: float = RUNTIME_PS,
) -> list[int]:
    if replicas is None:
        job_dir = job_dir_path(lam)
        replicas = list_available_replicas(job_dir, lam)
    return [r for r in replicas if not qc_replica(lam, r, runtime_ps=runtime_ps)]


def build_exclusion_report(
    lambdas: list[float],
    replicas: list[int],
    *,
    runtime_ps: float = RUNTIME_PS,
) -> dict:
    report: dict = {
        "runtime_ps": runtime_ps,
        "t_kin_max_k": STABILITY_T_KIN_MAX_K,
        "checks": [
            "replica_complete",
            "T_kin_blowup",
            "energy_finite",
            "fkt_present",
            "fkt_phi0",
            "fkt_decay",
        ],
        "by_lambda": {},
    }
    for lam in lambdas:
        available = [
            r for r in replicas if r in list_available_replicas(job_dir_path(lam), lam)
        ]
        passing: list[int] = []
        excluded: dict[str, list[str]] = {}
        for replica in available:
            reasons = qc_replica(lam, replica, runtime_ps=runtime_ps)
            if reasons:
                excluded[str(replica)] = reasons
            else:
                passing.append(replica)
        report["by_lambda"][str(lam)] = {
            "n_available": len(available),
            "n_qc_passed": len(passing),
            "n_excluded": len(excluded),
            "passing_replicas": passing,
            "excluded": excluded,
        }
    return report


def load_exclusion_report(path: Path) -> dict | None:
    if not path.is_file():
        return None
    return json.loads(path.read_text(encoding="utf-8"))


def qc_passing_replicas_from_report(report: dict, lam: float) -> list[int] | None:
    entry = report.get("by_lambda", {}).get(str(lam))
    if not entry:
        return None
    return [int(r) for r in entry.get("passing_replicas", [])]


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--lambdas", type=float, nargs="+", default=ANALYSIS_LAMBDAS)
    parser.add_argument("--replicas", type=int, nargs="+", default=list(range(N_REPLICAS)))
    parser.add_argument("--runtime-ps", type=float, default=RUNTIME_PS)
    parser.add_argument(
        "--write",
        type=Path,
        default=RESULTS_DIR / "replica_exclusion.json",
        help="Write exclusion report JSON",
    )
    args = parser.parse_args()

    report = build_exclusion_report(args.lambdas, args.replicas, runtime_ps=args.runtime_ps)
    args.write.parent.mkdir(parents=True, exist_ok=True)
    args.write.write_text(json.dumps(report, indent=2), encoding="utf-8")
    print(f"Wrote {args.write}")

    for lam in args.lambdas:
        entry = report["by_lambda"][str(lam)]
        print(
            f"lambda={lam:g}: available={entry['n_available']} "
            f"passed={entry['n_qc_passed']} excluded={entry['n_excluded']}"
        )


if __name__ == "__main__":
    main()
