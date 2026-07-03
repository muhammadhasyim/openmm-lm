#!/usr/bin/env python3
"""
Benchmark F(k,t) on equilibrium lambda=0 at 100 K with COM-drift gate.

Production defaults: atomic 500 sites, |k| = 2π/σ_AA a.u., resampled velocities
with COM removal (matches aging campaign). Block-averaged |phi| tau extraction.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np

_C2F_ROOT = Path(__file__).resolve().parent.parent
if str(_C2F_ROOT) not in sys.path:
    sys.path.insert(0, str(_C2F_ROOT))

from run_cavity_equilibrium import run_cavity_equilibrium  # noqa: E402

from config import FKT_KMAG_AU, FKT_KMAG_NM_INV, INITIAL_STATE  # noqa: E402
from fkt_utils import (  # noqa: E402
    block_average_abs_phi,
    extract_tau_s,
    measure_com_drift_from_snapshots,
    normalize_fkt_to_phi,
    parse_fkt_file,
)

CALIBRATION_TAU_PS = 105.0
TAU_TOLERANCE_PS = 55.0
MAX_COM_DRIFT_NM = 0.05
MAX_RAW_MSD_100PS_NM2 = 0.15


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--runtime-ps", type=float, default=600.0)
    parser.add_argument("--output-dir", type=Path, default=Path(__file__).resolve().parent / "benchmark_fkt")
    parser.add_argument("--initial-state", type=Path, default=INITIAL_STATE)
    parser.add_argument("--fkt-sites", choices=("atomic", "molecular_com"), default="atomic")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--evaluate-only", type=Path, default=None, help="Existing *_fkt_ref_000.txt")
    parser.add_argument(
        "--resample-velocities",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Resample velocities from IC (production parity; requires COM removal)",
    )
    args = parser.parse_args()

    args.output_dir.mkdir(parents=True, exist_ok=True)
    prefix = str(args.output_dir / f"eq100K_{args.fkt_sites}")
    snapshot_path = Path(f"{prefix}_snapshots.npz")

    if args.evaluate_only is None:
        run_cavity_equilibrium(
            temperature_K=100.0,
            runtime_ps=args.runtime_ps,
            lambda_coupling=0.0,
            include_dipole_self_energy=True,
            output_prefix=prefix,
            seed=args.seed,
            initial_state=args.initial_state,
            finite_q=False,
            resample_velocities=args.resample_velocities,
            enable_fkt=True,
            fkt_kmag_nm_inv=FKT_KMAG_NM_INV,
            fkt_ref_interval_ps=200.0,
            fkt_max_refs=3,
            fkt_output_period_ps=1.0,
            fkt_start_ps=0.0,
            fkt_sites=args.fkt_sites,
            snapshot_interval_ps=10.0,
            snapshots_out=snapshot_path,
        )
        fkt_path = Path(f"{prefix}_fkt_ref_000.txt")
    else:
        fkt_path = args.evaluate_only
        snapshot_path = fkt_path.with_name(fkt_path.name.replace("_fkt_ref_000.txt", "_snapshots.npz"))

    _, lags, vals = parse_fkt_file(fkt_path)
    norm = normalize_fkt_to_phi(lags, vals)
    if norm[0] is None:
        raise SystemExit(f"Failed to normalize {fkt_path}")
    lags_n, phi = norm
    f0 = float(vals[np.argmin(np.abs(lags))])
    idx1 = int(np.argmin(np.abs(lags_n - 1.0)))
    block_lags, block_phi = block_average_abs_phi(lags_n, phi, 10.0, 10.0)
    idx_block = int(np.argmin(np.abs(block_lags - 15.0))) if block_lags.size else 0
    tau_s = extract_tau_s(
        lags,
        vals,
        threshold=0.1,
        min_lag_ps=10.0,
        use_block_average=True,
        block_window_ps=10.0,
    )

    com_drift_nm = None
    raw_msd_100ps_nm2 = None
    if snapshot_path.exists():
        com_drift_nm, raw_msd_100ps_nm2 = measure_com_drift_from_snapshots(snapshot_path)

    report = {
        "fkt_sites": args.fkt_sites,
        "fkt_kmag_au": FKT_KMAG_AU,
        "fkt_kmag_nm_inv": FKT_KMAG_NM_INV,
        "runtime_ps": args.runtime_ps,
        "resample_velocities": args.resample_velocities,
        "F0": f0,
        "phi_at_1ps": float(phi[idx1]),
        "phi_block_15ps": float(block_phi[idx_block]) if block_phi.size else None,
        "tau_s_ps": tau_s,
        "calibration_tau_ps": CALIBRATION_TAU_PS,
        "com_drift_nm": com_drift_nm,
        "raw_msd_100ps_nm2": raw_msd_100ps_nm2,
        "pass_units": f0 > 300.0,
        "pass_f0_coherent": f0 > 300.0 and f0 < 1500.0,
        "pass_com_drift": com_drift_nm is not None and com_drift_nm <= MAX_COM_DRIFT_NM,
        "pass_glassy_msd": raw_msd_100ps_nm2 is not None
        and raw_msd_100ps_nm2 <= MAX_RAW_MSD_100PS_NM2,
        "pass_tau_vs_calibration": tau_s is not None
        and abs(tau_s - CALIBRATION_TAU_PS) <= TAU_TOLERANCE_PS,
    }
    report["pass"] = (
        report["pass_units"]
        and report["pass_f0_coherent"]
        and report["pass_com_drift"]
        and report["pass_glassy_msd"]
        and report["pass_tau_vs_calibration"]
    )

    out_json = args.output_dir / f"benchmark_{args.fkt_sites}.json"
    out_json.write_text(json.dumps(report, indent=2), encoding="utf-8")
    print(json.dumps(report, indent=2))
    if not report["pass"]:
        raise SystemExit("FKT equilibrium benchmark FAILED")
    print(f"PASS: benchmark written to {out_json}")


if __name__ == "__main__":
    main()
