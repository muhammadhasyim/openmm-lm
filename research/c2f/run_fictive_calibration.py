#!/usr/bin/env python
"""
NVT calibration sweep for fictive temperature energy inversion.

Measures harmonic (intramolecular), total PE, and LJ+Coulomb vs temperature
for use with EmpiricalTemperatureData (C2F protocol).

Supports τ-scaled manifests from plan_fictive_calibration.py and parallel shards.

Usage:
    python plan_fictive_calibration.py --output calibration_output/plan.json
    python run_fictive_calibration.py --manifest calibration_output/plan.json \\
        --shard-id 0 --n-shards 8 --output calibration_output/shard_0.txt
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np

try:
    import openmm
except ImportError:
    sys.exit("OpenMM required. Use: pixi install -e test")

from openmm.cavitymd.calibration import (
    crosscheck_calibration_against_reference,
    load_calibration_manifest,
    run_nvt_energy_calibration,
    validate_calibration_file,
)

_SCRIPT_DIR = Path(__file__).resolve().parent
if str(_SCRIPT_DIR) not in sys.path:
    sys.path.insert(0, str(_SCRIPT_DIR))

from run_c2f import build_mka_system  # noqa: E402

DEFAULT_OUT_DIR = _SCRIPT_DIR / "calibration_output"
REFERENCE_FILE = _SCRIPT_DIR / "reference_potential_energy_vs_T.txt"


def _parse_temperatures(args) -> np.ndarray:
    if args.temperatures:
        return np.array([float(x) for x in args.temperatures.split(",")], dtype=float)
    return np.linspace(args.temperature_max, args.temperature_min, args.n_temperatures)


def _select_shard_entries(manifest_path: Path, shard_id: int, n_shards: int):
    specs = load_calibration_manifest(manifest_path)
    if n_shards < 1:
        raise ValueError("n_shards must be >= 1")
    if not (0 <= shard_id < n_shards):
        raise ValueError(f"shard_id must be in [0, {n_shards})")
    return [spec for i, spec in enumerate(specs) if i % n_shards == shard_id]


def main() -> int:
    parser = argparse.ArgumentParser(
        description="NVT energy calibration for fictive temperatures (mKA system)"
    )
    parser.add_argument("--temperature-min", type=float, default=80.0)
    parser.add_argument("--temperature-max", type=float, default=400.0)
    parser.add_argument("--n-temperatures", type=int, default=50)
    parser.add_argument(
        "--temperatures",
        type=str,
        default=None,
        help="Comma-separated K values (overrides min/max/n)",
    )
    parser.add_argument(
        "--only-temperatures",
        type=str,
        default=None,
        help="Comma-separated K values to run (filters manifest/shard selection)",
    )
    parser.add_argument(
        "--manifest",
        type=Path,
        default=None,
        help="JSON manifest from plan_fictive_calibration.py",
    )
    parser.add_argument("--shard-id", type=int, default=None)
    parser.add_argument("--n-shards", type=int, default=1)
    parser.add_argument(
        "--prod-ns",
        type=float,
        default=100.0,
        help="Production duration per temperature (ns) without manifest",
    )
    parser.add_argument(
        "--equil-ns",
        type=float,
        default=10.0,
        help="Equilibration before production (ns) without manifest",
    )
    parser.add_argument(
        "--n-samples",
        type=int,
        default=1000,
        help="Energy samples during production",
    )
    parser.add_argument("--dt-ps", type=float, default=0.001)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--platform", default=None)
    parser.add_argument(
        "--output",
        default="potential_energy_components_vs_temperature.txt",
        help="Full cav-hoomd-style calibration output",
    )
    parser.add_argument(
        "--slim-output",
        default="calibration_data.txt",
        help="Slim 4-column file for EmpiricalTemperatureData / run_c2f.py",
    )
    parser.add_argument(
        "--timeseries-dir",
        default=None,
        help="Optional directory for per-T sample CSVs",
    )
    parser.add_argument(
        "--n-eff-report",
        default=None,
        help="JSON path for per-T N_eff / block diagnostics",
    )
    parser.add_argument(
        "--resume",
        action="store_true",
        help="Skip temperatures already in --output",
    )
    parser.add_argument(
        "--quick",
        action="store_true",
        help="Smoke: 3 temps, 1 ns prod, 0.1 ns equil, 50 samples",
    )
    parser.add_argument(
        "--no-validate",
        action="store_true",
        help="Skip post-run EmpiricalTemperatureData validation",
    )
    parser.add_argument(
        "--crosscheck-reference",
        type=Path,
        default=None,
        help="Compare slim output against a reference calibration table",
    )
    args = parser.parse_args()

    if args.quick:
        args.n_temperatures = 3
        args.prod_ns = 1.0
        args.equil_ns = 0.1
        args.n_samples = 50
        print("Quick mode: 3 temperatures, 1 ns prod, 0.1 ns equil, 50 samples")

    run_specs = None
    temperatures = _parse_temperatures(args)

    if args.manifest is not None:
        if not args.manifest.exists():
            print(f"ERROR: manifest not found: {args.manifest}", file=sys.stderr)
            return 1
        all_specs = load_calibration_manifest(args.manifest)
        if args.shard_id is not None:
            run_specs = _select_shard_entries(args.manifest, args.shard_id, args.n_shards)
            print(
                f"Shard {args.shard_id}/{args.n_shards}: "
                f"{len(run_specs)} temperatures from {args.manifest}"
            )
        else:
            run_specs = all_specs
            print(f"Manifest: {len(run_specs)} temperatures from {args.manifest}")
        if args.only_temperatures:
            targets = [float(x) for x in args.only_temperatures.split(",")]
            run_specs = [
                spec
                for spec in run_specs
                if any(abs(spec.temperature_K - target) < 0.05 for target in targets)
            ]
            print(f"Filtered to {len(run_specs)} temperatures: --only-temperatures")
        temperatures = np.array([s.temperature_K for s in run_specs], dtype=float)

    print(
        f"Temperature grid ({len(temperatures)} points): "
        f"{temperatures[0]:.1f} → {temperatures[-1]:.1f} K"
    )

    def _make_system(T):
        return build_mka_system(seed=args.seed, sample_bonds_at_T=T)

    n_eff_report = args.n_eff_report
    if n_eff_report is None and args.output:
        out_path = Path(args.output)
        n_eff_report = str(out_path.parent / f"{out_path.stem}_n_eff.json")

    output_path = run_nvt_energy_calibration(
        _make_system,
        temperatures,
        prod_ns=args.prod_ns,
        equil_ns=args.equil_ns,
        n_samples=args.n_samples,
        run_specs=run_specs,
        dt_ps=args.dt_ps,
        output_file=args.output,
        slim_output_file=args.slim_output,
        platform_name=args.platform,
        timeseries_dir=args.timeseries_dir,
        n_eff_report_file=n_eff_report,
        resume=args.resume,
    )

    if not args.no_validate:
        print("\n=== Validation (EmpiricalTemperatureData) ===")
        ok = validate_calibration_file(args.output, args.slim_output)
        if not ok:
            return 1

    ref = args.crosscheck_reference or REFERENCE_FILE
    if ref.exists():
        print("\n=== Cross-check vs reference ===")
        crosscheck_calibration_against_reference(args.slim_output, ref)

    print(f"\nDone. Full: {output_path}")
    print(f"      Slim: {args.slim_output}")
    if n_eff_report:
        print(f"      N_eff: {n_eff_report}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
