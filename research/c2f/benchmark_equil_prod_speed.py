#!/usr/bin/env python3
"""Measure equil vs aging-production throughput; write JSON + ns/day estimates."""

from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

_SCRIPT_DIR = Path(__file__).resolve().parent
if str(_SCRIPT_DIR) not in sys.path:
    sys.path.insert(0, str(_SCRIPT_DIR))

from run_cavity_equilibrium import run_cavity_equilibrium  # noqa: E402

FKT_KMAG_NM_INV = 19.05789556235437


def _bench(label: str, out_dir: Path, runtime_ps: float, **kwargs) -> dict:
    prefix = str(out_dir / label)
    t0 = time.perf_counter()
    run_cavity_equilibrium(**kwargs, output_prefix=prefix, platform_name="CUDA")
    wall_s = time.perf_counter() - t0
    ps_per_s = runtime_ps / wall_s
    ns_per_day = ps_per_s * 86400.0 / 1000.0
    return {
        "label": label,
        "runtime_ps": runtime_ps,
        "runtime_ns": runtime_ps / 1000.0,
        "wall_s": wall_s,
        "wall_h": wall_s / 3600.0,
        "ps_per_s": ps_per_s,
        "ns_per_day": ns_per_day,
    }


def _extrapolate(ps_per_s: float, target_ps: float) -> dict:
    wall_s = target_ps / ps_per_s
    return {
        "target_ps": target_ps,
        "target_ns": target_ps / 1000.0,
        "wall_s": wall_s,
        "wall_h": wall_s / 3600.0,
        "wall_min": wall_s / 60.0,
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--out-dir", type=Path, default=_SCRIPT_DIR / "benchmark_output" / "speed")
    parser.add_argument("--equil-probe-ps", type=float, default=500.0)
    parser.add_argument("--prod-probe-ps", type=float, default=600.0,
                        help="Total ps incl. lam=0 prefix (default 600 = 100+500)")
    parser.add_argument("--switch-time-ps", type=float, default=100.0)
    parser.add_argument("--target-equil-ps", type=float, default=5000.0,
                        help="Planned equil length (default 5 ns)")
    parser.add_argument("--target-prod-ps", type=float, default=2600.0,
                        help="Planned prod length: 100 ps lam=0 + 2500 ps aging")
    args = parser.parse_args()

    args.out_dir.mkdir(parents=True, exist_ok=True)

    equil = _bench(
        "equil_lam0",
        args.out_dir,
        args.equil_probe_ps,
        temperature_K=100.0,
        runtime_ps=args.equil_probe_ps,
        lambda_coupling=0.0,
        include_dipole_self_energy=True,
        seed=42,
        sample_interval_ps=10.0,
        finite_q=False,
        resample_velocities=False,
        enable_fkt=False,
    )

    prod = _bench(
        "aging_lam003_fkt",
        args.out_dir,
        args.prod_probe_ps,
        temperature_K=100.0,
        runtime_ps=args.prod_probe_ps,
        lambda_coupling=0.03,
        include_dipole_self_energy=True,
        seed=42,
        sample_interval_ps=1.0,
        finite_q=False,
        resample_velocities=True,
        enable_fkt=True,
        fkt_kmag_nm_inv=FKT_KMAG_NM_INV,
        fkt_num_wavevectors=50,
        fkt_ref_interval_ps=200.0,
        fkt_output_period_ps=1.0,
        fkt_max_refs=13,
        fkt_start_ps=args.switch_time_ps,
        fkt_sites="atomic",
        coupling_start_ps=args.switch_time_ps,
        snapshot_interval_ps=10.0,
        snapshots_out=args.out_dir / "aging_lam003_fkt_snapshots.npz",
    )

    report = {
        "benchmarks": [equil, prod],
        "extrapolated": {
            "equil_5ns": _extrapolate(equil["ps_per_s"], args.target_equil_ps),
            "prod_lam003": _extrapolate(prod["ps_per_s"], args.target_prod_ps),
        },
        "combined_wall_h": (
            args.target_equil_ps / equil["ps_per_s"] + args.target_prod_ps / prod["ps_per_s"]
        ) / 3600.0,
        "slurm_reference_a100": {
            "equil_10ns_ps_per_s": 5.1,
            "prod_fkt_ps_per_s": 0.4,
            "note": "From aging_weak_lambda/slurm/logs/ic_* and prod_* on a100_chemistry",
        },
    }
    report["extrapolated_slurm"] = {
        "equil_5ns": _extrapolate(5.1, args.target_equil_ps),
        "prod_lam003": _extrapolate(0.4, args.target_prod_ps),
        "combined_wall_h": (
            args.target_equil_ps / 5.1 + args.target_prod_ps / 0.4
        ) / 3600.0,
    }

    out_json = args.out_dir / "speed_report.json"
    out_json.write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")

    print(json.dumps(report, indent=2))
    print(f"\nWrote {out_json}")


if __name__ == "__main__":
    main()
