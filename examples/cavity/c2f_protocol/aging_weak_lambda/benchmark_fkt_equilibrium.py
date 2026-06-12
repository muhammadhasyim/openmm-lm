#!/usr/bin/env python3
"""
Benchmark F(k,t) on equilibrium lambda=0 at 100 K.

Production defaults: atomic 500 sites, |k| = 2π/σ_AA a.u. Block-averaged |phi|
tau extraction (matches calibration-scale relaxation at this k).
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
    normalize_fkt_to_phi,
    parse_fkt_file,
)

CALIBRATION_TAU_PS = 105.0
TAU_TOLERANCE_PS = 30.0


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--runtime-ps", type=float, default=600.0)
    parser.add_argument("--output-dir", type=Path, default=Path(__file__).resolve().parent / "benchmark_fkt")
    parser.add_argument("--initial-state", type=Path, default=INITIAL_STATE)
    parser.add_argument("--fkt-sites", choices=("atomic", "molecular_com"), default="atomic")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--evaluate-only", type=Path, default=None, help="Existing *_fkt_ref_000.txt")
    args = parser.parse_args()

    args.output_dir.mkdir(parents=True, exist_ok=True)
    prefix = str(args.output_dir / f"eq100K_{args.fkt_sites}")

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
            resample_velocities=False,
            enable_fkt=True,
            fkt_kmag_nm_inv=FKT_KMAG_NM_INV,
            fkt_ref_interval_ps=200.0,
            fkt_max_refs=3,
            fkt_output_period_ps=1.0,
            fkt_start_ps=0.0,
            fkt_sites=args.fkt_sites,
        )
        fkt_path = Path(f"{prefix}_fkt_ref_000.txt")
    else:
        fkt_path = args.evaluate_only

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

    report = {
        "fkt_sites": args.fkt_sites,
        "fkt_kmag_au": FKT_KMAG_AU,
        "fkt_kmag_nm_inv": FKT_KMAG_NM_INV,
        "runtime_ps": args.runtime_ps,
        "F0": f0,
        "phi_at_1ps": float(phi[idx1]),
        "phi_block_15ps": float(block_phi[idx_block]) if block_phi.size else None,
        "tau_s_ps": tau_s,
        "calibration_tau_ps": CALIBRATION_TAU_PS,
        "pass_units": f0 > 300.0,
        "pass_f0_coherent": f0 > 300.0 and f0 < 1500.0,
        "pass_tau_vs_calibration": tau_s is not None
        and abs(tau_s - CALIBRATION_TAU_PS) <= TAU_TOLERANCE_PS,
    }
    report["pass"] = report["pass_units"] and report["pass_f0_coherent"]

    out_json = args.output_dir / f"benchmark_{args.fkt_sites}.json"
    out_json.write_text(json.dumps(report, indent=2), encoding="utf-8")
    print(json.dumps(report, indent=2))
    if not report["pass"]:
        raise SystemExit("FKT equilibrium benchmark FAILED (units/F0 gate)")
    print(f"PASS: benchmark written to {out_json}")
    if not report["pass_tau_vs_calibration"]:
        print(
            f"NOTE: tau_s={tau_s} ps vs calibration {CALIBRATION_TAU_PS} ps "
            f"(|k|={FKT_KMAG_AU:.4f} a.u.)"
        )


if __name__ == "__main__":
    main()
