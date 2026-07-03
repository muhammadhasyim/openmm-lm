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
    DIPOLE_INTERVAL_PS,
    FKT_KMAG_NM_INV,
    FKT_MAX_REFS,
    FKT_NUM_WAVEVECTORS,
    FKT_OUTPUT_PERIOD_PS,
    FKT_REF_INTERVAL_PS,
    FREQUENCY_CM1,
    INITIAL_STATE,
    IR_SUBSET_REPLICAS,
    IR_WINDOWS,
    REFERENCE_NUM_MOL,
    RUNTIME_PS,
    SNAPSHOT_INTERVAL_PS,
    SWITCH_TIME_PS,
    TEMPERATURE_K,
    collective_coupling_g,
    job_dir_path,
    replica_seed,
    run_prefix,
    scale_lambda,
)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--lambda", dest="lam", type=float, required=True)
    parser.add_argument("--replica", type=int, required=True)
    parser.add_argument("--runtime-ps", type=float, default=RUNTIME_PS)
    parser.add_argument("--switch-time-ps", type=float, default=SWITCH_TIME_PS)
    parser.add_argument(
        "--initial-state",
        type=Path,
        default=INITIAL_STATE,
        help="Equilibrated IC npz (default: config INITIAL_STATE)",
    )
    parser.add_argument(
        "--num-molecules",
        type=int,
        default=REFERENCE_NUM_MOL,
        help=f"System size N (default {REFERENCE_NUM_MOL}). "
        "λ is scaled as λ(N)=λ(N_ref)√(N_ref/N) so g=λ√N is constant.",
    )
    parser.add_argument(
        "--no-scale-lambda",
        action="store_true",
        help="Use --lambda literally (do not scale with √N)",
    )
    parser.add_argument("--smoke", action="store_true")
    parser.add_argument("--no-fkt", action="store_true")
    parser.add_argument("--platform", default=None)
    parser.add_argument(
        "--adaptive",
        action="store_true",
        help="Use cav-hoomd max-metric adaptive Verlet integrator (recommended at λ=0.03)",
    )
    parser.add_argument(
        "--campaign-dir",
        type=Path,
        default=None,
        help="Root directory for lambda job outputs (default: aging_weak_lambda/)",
    )
    parser.add_argument(
        "--ir-windows",
        action="append",
        nargs=2,
        metavar=("START_PS", "LENGTH_PS"),
        type=float,
        default=None,
        help="IR dipole recording windows; repeat for multiple. "
        "Default: config IR_WINDOWS for IR subset replicas.",
    )
    parser.add_argument(
        "--no-resume",
        action="store_true",
        help="Ignore checkpoint and start fresh from IC",
    )
    args = parser.parse_args()

    campaign_root = args.campaign_dir
    job_dir = job_dir_path(args.lam, campaign_root=campaign_root)
    job_dir.mkdir(parents=True, exist_ok=True)
    seed = replica_seed(args.replica)
    prefix = str(job_dir / run_prefix(args.lam, args.replica))

    runtime_ps = args.runtime_ps
    fkt_max_refs = FKT_MAX_REFS
    snapshot_interval = SNAPSHOT_INTERVAL_PS
    dipole_windows: list[tuple[float, float]] | None = None
    if args.replica < IR_SUBSET_REPLICAS:
        if args.ir_windows:
            dipole_windows = [(float(s), float(l)) for s, l in args.ir_windows]
        else:
            dipole_windows = list(IR_WINDOWS)
    if args.smoke:
        runtime_ps = min(runtime_ps, args.switch_time_ps + 5.0)
        fkt_max_refs = 2
        snapshot_interval = 2.0

    if not args.initial_state.exists():
        raise FileNotFoundError(f"Missing IC: {args.initial_state}")

    platform_name = args.platform

    lam_ref = args.lam
    lam_eff = lam_ref if args.no_scale_lambda else scale_lambda(lam_ref, args.num_molecules)
    g_eff = collective_coupling_g(lam_eff, args.num_molecules)
    print(
        f"Coupling: λ={lam_eff:.6g} at N={args.num_molecules} "
        f"(λ_ref={lam_ref:g} at N={REFERENCE_NUM_MOL}, g=λ√N={g_eff:.6g})"
    )

    run_cavity_equilibrium(
        temperature_K=TEMPERATURE_K,
        runtime_ps=runtime_ps,
        lambda_coupling=lam_eff,
        include_dipole_self_energy=True,
        output_prefix=prefix,
        seed=seed,
        sample_interval_ps=CSV_INTERVAL_PS,
        initial_state=args.initial_state,
        platform_name=platform_name,
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
        dipole_windows=dipole_windows,
        dipole_interval_ps=DIPOLE_INTERVAL_PS,
        num_molecules=args.num_molecules,
        adaptive=args.adaptive,
        no_resume=args.no_resume,
    )

    if dipole_windows is not None:
        dipole_path = Path(f"{prefix}_dipole.npz")
        if not dipole_path.exists():
            raise RuntimeError(f"Expected dipole output for IR subset replica: {dipole_path}")


if __name__ == "__main__":
    main()
