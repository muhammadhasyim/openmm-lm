#!/usr/bin/env python3
"""Run one OpenMM weak-coupling aging production trajectory."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

_SCRIPT_DIR = Path(__file__).resolve().parent
_C2F_ROOT = _SCRIPT_DIR.parent
if str(_C2F_ROOT) not in sys.path:
    sys.path.insert(0, str(_C2F_ROOT))

from run_cavity_equilibrium import run_cavity_equilibrium  # noqa: E402

from config import (  # noqa: E402
    CSV_INTERVAL_PS,
    FKT_KMAG_NM_INV,
    FKT_MAX_REFS,
    FKT_NUM_WAVEVECTORS,
    FKT_OUTPUT_PERIOD_PS,
    FKT_REF_INTERVAL_PS,
    FREQUENCY_CM1,
    INITIAL_STATE,
    RUNTIME_PS,
    SNAPSHOT_INTERVAL_PS,
    SWITCH_TIME_PS,
    TEMPERATURE_K,
    job_dir_path,
    replica_seed,
    run_prefix,
)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--lambda", dest="lam", type=float, required=True)
    parser.add_argument("--replica", type=int, required=True)
    parser.add_argument("--runtime-ps", type=float, default=RUNTIME_PS)
    parser.add_argument("--switch-time-ps", type=float, default=SWITCH_TIME_PS)
    parser.add_argument("--smoke", action="store_true")
    parser.add_argument("--no-fkt", action="store_true")
    parser.add_argument("--platform", default=None)
    args = parser.parse_args()

    job_dir = job_dir_path(args.lam)
    job_dir.mkdir(parents=True, exist_ok=True)
    seed = replica_seed(args.replica)
    prefix = str(job_dir / run_prefix(args.lam, args.replica))

    runtime_ps = args.runtime_ps
    fkt_max_refs = FKT_MAX_REFS
    snapshot_interval = SNAPSHOT_INTERVAL_PS
    if args.smoke:
        runtime_ps = min(runtime_ps, args.switch_time_ps + 5.0)
        fkt_max_refs = 2
        snapshot_interval = 2.0

    if not INITIAL_STATE.exists():
        raise FileNotFoundError(f"Missing IC: {INITIAL_STATE}")

    run_cavity_equilibrium(
        temperature_K=TEMPERATURE_K,
        runtime_ps=runtime_ps,
        lambda_coupling=args.lam,
        include_dipole_self_energy=True,
        output_prefix=prefix,
        seed=seed,
        sample_interval_ps=CSV_INTERVAL_PS,
        initial_state=INITIAL_STATE,
        platform_name=args.platform,
        finite_q=False,
        omega_c_cm1=FREQUENCY_CM1,
        snapshot_interval_ps=snapshot_interval,
        snapshots_out=Path(f"{prefix}_snapshots.npz"),
        coupling_start_ps=args.switch_time_ps,
        resample_velocities=True,
        enable_fkt=not args.no_fkt,
        fkt_kmag_nm_inv=FKT_KMAG_NM_INV,
        fkt_num_wavevectors=FKT_NUM_WAVEVECTORS,
        fkt_ref_interval_ps=FKT_REF_INTERVAL_PS,
        fkt_output_period_ps=FKT_OUTPUT_PERIOD_PS,
        fkt_max_refs=fkt_max_refs,
        fkt_start_ps=args.switch_time_ps,
        fkt_sites="atomic",
    )


if __name__ == "__main__":
    main()
