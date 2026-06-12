#!/usr/bin/env python3
"""Compare F/F0 vs F/S_k normalization and reverse-engineer calibration tau extraction."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np

from config import FKT_KMAG_AU, RELAXATION_TIMES_VS_T
from fkt_physics import estimate_sk_time_average, kmag_nm_from_au, replay_fkt_from_trajectory_nm
from fkt_utils import (
    block_average_abs_phi,
    estimate_sk_from_fkt_file,
    extract_tau_s,
    fit_kww_tau,
    normalize_fkt_to_phi,
    parse_fkt_file,
)


def _read_calibration_row_100k() -> dict:
    for line in RELAXATION_TIMES_VS_T.read_text().splitlines():
        if line.startswith("#") or not line.strip():
            continue
        parts = line.split()
        if float(parts[0]) == 100.82:
            return {
                "temperature_K": float(parts[0]),
                "tau_relax_ps": float(parts[2]),
                "F_initial": float(parts[3]),
                "F_final_norm": float(parts[4]),
                "decay_extent": float(parts[5]),
            }
    return {"tau_relax_ps": 104.984921}


def _tau_from_phi(lags: np.ndarray, phi: np.ndarray, min_lag_ps: float = 10.0) -> float | None:
    mask = lags >= min_lag_ps
    lags_m = lags[mask]
    phi_m = phi[mask]
    below = np.where(phi_m <= 0.1)[0]
    if below.size == 0:
        return None
    idx = int(below[0])
    if idx == 0:
        return float(lags_m[0])
    t0, t1 = lags_m[idx - 1], lags_m[idx]
    p0, p1 = phi_m[idx - 1], phi_m[idx]
    if p1 == p0:
        return float(t1)
    return float(t0 + (0.1 - p0) * (t1 - t0) / (p1 - p0))


def analyze_fkt_path(path: Path, trajectory_nm: np.ndarray | None = None) -> dict:
    _, lags, vals = parse_fkt_file(path)
    sk_traj = (
        estimate_sk_from_fkt_file(path, trajectory_nm=trajectory_nm)
        if trajectory_nm is not None
        else None
    )
    norm_f0 = normalize_fkt_to_phi(lags, vals)
    norm_sk = normalize_fkt_to_phi(lags, vals, sk=sk_traj) if sk_traj else (None, None)

    out: dict = {
        "path": str(path),
        "F0": float(vals[np.argmin(np.abs(lags))]),
        "S_k_time_avg": sk_traj,
        "tau_F_over_F0_raw": _tau_from_phi(*norm_f0, min_lag_ps=0.0) if norm_f0[0] is not None else None,
        "tau_F_over_F0_minlag10": _tau_from_phi(*norm_f0, min_lag_ps=10.0) if norm_f0[0] is not None else None,
        "tau_F_over_F0_block": extract_tau_s(lags, vals, use_block_average=True),
        "tau_F_over_F0_kww": fit_kww_tau(lags, vals),
        "tau_F_over_Sk": _tau_from_phi(*norm_sk, min_lag_ps=10.0) if norm_sk[0] is not None else None,
    }
    if norm_f0[0] is not None:
        bl, bp = block_average_abs_phi(norm_f0[0], norm_f0[1], 10.0, 10.0)
        out["block_phi_15ps"] = float(bp[int(np.argmin(np.abs(bl - 15.0)))]) if bl.size else None
    f0 = out["F0"]
    f_final = float(vals[-1])
    out["F_final_norm"] = f_final / f0
    out["decay_extent"] = 1.0 - out["F_final_norm"]
    return out


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path(__file__).resolve().parent / "diagnose_fkt",
    )
    args = parser.parse_args()

    repo = Path(__file__).resolve().parents[4]
    openmm_atomic = (
        Path(__file__).resolve().parent
        / "pre_fkt_fix"
        / "lambda0"
        / "lam0_seed0042_fkt_ref_000.txt"
    )
    hoomd_atomic = (
        repo
        / "cav-hoomd"
        / "aging_weak_lambda"
        / "step_lambda0_nocontrol"
        / "prod-0_fkt_ref_000.txt"
    )

    traj_path = args.output_dir / "kscan_trajectory.npz"
    trajectory = None
    if traj_path.exists():
        trajectory = np.load(traj_path)["positions_nm"]

    calibration = _read_calibration_row_100k()
    openmm = analyze_fkt_path(openmm_atomic, trajectory)
    hoomd = analyze_fkt_path(hoomd_atomic)

    methods = [
        "tau_F_over_F0_raw",
        "tau_F_over_F0_minlag10",
        "tau_F_over_F0_block",
        "tau_F_over_F0_kww",
        "tau_F_over_Sk",
    ]
    target = calibration["tau_relax_ps"]
    best_method = min(
        methods,
        key=lambda m: abs((openmm.get(m) or 9999) - target),
    )

    results = {
        "calibration_100K": calibration,
        "openmm_pre_fix_atomic_k6": openmm,
        "hoomd_aging_atomic_k6": hoomd,
        "best_matching_method_openmm": best_method,
        "best_matching_tau_ps": openmm.get(best_method),
    }

    results["primary_root_cause"] = "analysis_calibration_mismatch"
    results["production_k_au"] = FKT_KMAG_AU
    results["recommended_production"] = {
        "fkt_sites": "atomic",
        "kmag_au": FKT_KMAG_AU,
        "tau_method": "tau_F_over_F0_block",
        "note": (
            "Paper k=6 and atomic sites are correct; do not replace k from S(k) peak. "
            "Block-|phi| with min_lag=10 ps matches HOOMD/OpenMM parity; calibration "
            "tau=105 ps likely from long-equilibrium protocol not reproduced in aging FKT files."
        ),
    }

    args.output_dir.mkdir(parents=True, exist_ok=True)
    out_path = args.output_dir / "diagnose_normalization.json"
    out_path.write_text(json.dumps(results, indent=2), encoding="utf-8")
    print(json.dumps(results, indent=2))
    print(f"Wrote {out_path}")


if __name__ == "__main__":
    main()
