#!/usr/bin/env python3
"""Compute S(k) from equilibrium IC and k-scan F(k,t) via snapshot replay."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

_C2F_ROOT = Path(__file__).resolve().parent.parent
if str(_C2F_ROOT) not in sys.path:
    sys.path.insert(0, str(_C2F_ROOT))

from run_cavity_equilibrium import run_cavity_equilibrium  # noqa: E402

from config import (  # noqa: E402
    BOHR_TO_NM,
    FKT_KMAG_AU,
    FKT_KMAG_NM_INV,
    FKT_KMAG_PAPER_AU,
    INITIAL_STATE,
)
from fkt_physics import (  # noqa: E402
    kmag_nm_from_au,
    replay_fkt_from_trajectory_nm,
    sk_curve,
)
from fkt_utils import extract_tau_s, fit_kww_tau, normalize_fkt_to_phi  # noqa: E402
from run_c2f import SIG_AA_AU, BOX_AU  # noqa: E402


def _summarize_fkt(lags: np.ndarray, fkt: np.ndarray) -> dict:
    norm = normalize_fkt_to_phi(lags, fkt)
    phi1 = None
    if norm[0] is not None:
        lags_n, phi = norm
        idx1 = int(np.argmin(np.abs(lags_n - 1.0)))
        phi1 = float(phi[idx1])
    return {
        "F0": float(fkt[np.argmin(np.abs(lags))]),
        "phi_at_1ps": phi1,
        "tau_s_raw": extract_tau_s(lags, fkt, use_block_average=False, min_lag_ps=10.0),
        "tau_s_block": extract_tau_s(lags, fkt, use_block_average=True, min_lag_ps=10.0),
        "tau_s_kww": fit_kww_tau(lags, fkt, min_lag_ps=10.0),
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--initial-state", type=Path, default=INITIAL_STATE)
    parser.add_argument("--output-dir", type=Path, default=Path(__file__).resolve().parent / "diagnose_fkt")
    parser.add_argument("--runtime-ps", type=float, default=600.0)
    parser.add_argument("--snapshot-interval-ps", type=float, default=2.0)
    parser.add_argument("--skip-md", action="store_true", help="Reuse existing snapshot npz")
    args = parser.parse_args()

    args.output_dir.mkdir(parents=True, exist_ok=True)
    snap_path = args.output_dir / "kscan_trajectory.npz"
    prefix = str(args.output_dir / "kscan_eq100K_atomic")

    if not args.skip_md or not snap_path.exists():
        run_cavity_equilibrium(
            temperature_K=100.0,
            runtime_ps=args.runtime_ps,
            lambda_coupling=0.0,
            include_dipole_self_energy=True,
            output_prefix=prefix,
            seed=42,
            initial_state=args.initial_state,
            finite_q=False,
            resample_velocities=False,
            enable_fkt=False,
            fkt_sites="atomic",
            snapshot_interval_ps=args.snapshot_interval_ps,
            snapshots_out=snap_path,
        )

    snap = np.load(snap_path)
    positions = np.asarray(snap["positions_nm"], dtype=np.float64)
    times = np.asarray(snap["times_ps"], dtype=np.float64)
    lag_ps = float(np.median(np.diff(times))) if times.size > 1 else args.snapshot_interval_ps

    ic_positions = np.load(args.initial_state)["positions_nm"][:500]
    k_grid = np.linspace(0.5, 10.0, 40)
    k_vals, s_vals = sk_curve(ic_positions, k_grid, site_mode="atomic")
    k_peak = float(k_vals[int(np.argmax(s_vals))])

    reference_k_au = [
        k_peak,
        1.0,
        2.0,
        4.0,
        FKT_KMAG_AU,
        FKT_KMAG_AU / SIG_AA_AU,
        2.0 * np.pi / SIG_AA_AU,
        2.0 * np.pi / BOX_AU,
    ]
    reference_k_au = sorted(set(round(k, 4) for k in reference_k_au))

    kscan_results: dict[str, dict] = {}
    for k_au in reference_k_au:
        k_nm = kmag_nm_from_au(k_au)
        lags, fkt = replay_fkt_from_trajectory_nm(
            positions, k_nm, lag_ps=lag_ps, site_mode="atomic"
        )
        kscan_results[f"k_au_{k_au:g}"] = {
            "kmag_au": k_au,
            "kmag_nm_inv": k_nm,
            **_summarize_fkt(lags, fkt),
        }

    sk_report = {
        "k_peak_au": k_peak,
        "S_at_k_peak": float(np.max(s_vals)),
        "S_at_k6": float(s_vals[int(np.argmin(np.abs(k_vals - FKT_KMAG_PAPER_AU)))]),
        "k_2pi_over_sigma_AA": float(FKT_KMAG_AU),
        "k_production_au": float(FKT_KMAG_AU),
        "k_paper_au": FKT_KMAG_PAPER_AU,
    }

    fig, ax = plt.subplots(figsize=(7, 4))
    ax.plot(k_vals, s_vals, "b-", lw=1.5)
    ax.axvline(FKT_KMAG_PAPER_AU, color="r", ls="--", label=f"paper k={FKT_KMAG_PAPER_AU} a.u.")
    ax.axvline(k_peak, color="g", ls="--", label=f"S(k) peak k={k_peak:.2f} a.u.")
    ax.axvline(FKT_KMAG_AU, color="orange", ls=":", label=f"production k=2π/σ={FKT_KMAG_AU:.3f} a.u.")
    ax.set_xlabel("|k| (Bohr⁻¹)")
    ax.set_ylabel("S(k) (single-frame shell average)")
    ax.legend(fontsize=8)
    ax.set_title("Static S(k) from equilibrium IC")
    fig.tight_layout()
    fig.savefig(args.output_dir / "sk_equilibrium_ic.png", dpi=150)
    plt.close(fig)

    report = {
        "sk_summary": sk_report,
        "kscan": kscan_results,
        "calibration_tau_ps": 105.0,
        "trajectory": {
            "n_frames": int(positions.shape[0]),
            "lag_ps": lag_ps,
            "runtime_ps": args.runtime_ps,
        },
    }
    out_json = args.output_dir / "compute_sk_kscan.json"
    out_json.write_text(json.dumps(report, indent=2), encoding="utf-8")
    print(json.dumps(report, indent=2))
    print(f"Wrote {out_json} and sk_equilibrium_ic.png")


if __name__ == "__main__":
    main()
