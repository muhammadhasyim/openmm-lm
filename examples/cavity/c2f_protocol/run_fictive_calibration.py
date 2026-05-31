#!/usr/bin/env python
"""
NVT calibration sweep for fictive temperature energy inversion.

Measures harmonic (intramolecular), total PE, and LJ+Coulomb vs temperature
for use with EmpiricalTemperatureData (C2F protocol).

Default: 50 temperatures from 400 K to 80 K, 10 ns equil + 100 ns production,
1000 samples per temperature.

Usage:
    pixi run -e test calibrate-fictive
    pixi run -e test calibrate-fictive-quick
    python examples/cavity/c2f_protocol/run_fictive_calibration.py --quick
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np

try:
    import openmm
except ImportError:
    sys.exit("OpenMM required. Use: pixi install -e test")

from openmm.cavitymd.calibration import (
    run_nvt_energy_calibration,
    validate_calibration_file,
)

_SCRIPT_DIR = Path(__file__).resolve().parent
if str(_SCRIPT_DIR) not in sys.path:
    sys.path.insert(0, str(_SCRIPT_DIR))

from run_c2f import build_mka_system  # noqa: E402


def _parse_temperatures(args) -> np.ndarray:
    if args.temperatures:
        return np.array([float(x) for x in args.temperatures.split(",")], dtype=float)
    return np.linspace(args.temperature_max, args.temperature_min, args.n_temperatures)


def main() -> int:
    parser = argparse.ArgumentParser(
        description="NVT energy calibration for fictive temperatures (mKA system)"
    )
    parser.add_argument("--temperature-min", type=float, default=80.0)
    parser.add_argument("--temperature-max", type=float, default=400.0)
    parser.add_argument("--n-temperatures", type=int, default=50)
    parser.add_argument("--temperatures", type=str, default=None,
                        help="Comma-separated K values (overrides min/max/n)")
    parser.add_argument("--prod-ns", type=float, default=100.0,
                        help="Production duration per temperature (ns)")
    parser.add_argument("--equil-ns", type=float, default=10.0,
                        help="Equilibration before production (ns)")
    parser.add_argument("--n-samples", type=int, default=1000,
                        help="Energy samples during production")
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
    parser.add_argument("--timeseries-dir", default=None,
                        help="Optional directory for per-T sample CSVs")
    parser.add_argument("--resume", action="store_true",
                        help="Skip temperatures already in --output")
    parser.add_argument("--quick", action="store_true",
                        help="Smoke: 3 temps, 1 ns prod, 0.1 ns equil, 50 samples")
    parser.add_argument("--no-validate", action="store_true",
                        help="Skip post-run EmpiricalTemperatureData validation")
    args = parser.parse_args()

    if args.quick:
        args.n_temperatures = 3
        args.prod_ns = 1.0
        args.equil_ns = 0.1
        args.n_samples = 50
        print("Quick mode: 3 temperatures, 1 ns prod, 0.1 ns equil, 50 samples")

    temperatures = _parse_temperatures(args)
    print(f"Temperature grid ({len(temperatures)} points): "
          f"{temperatures[0]:.1f} → {temperatures[-1]:.1f} K")

    def _make_system(T):
        return build_mka_system(seed=args.seed, sample_bonds_at_T=T)

    output_path = run_nvt_energy_calibration(
        _make_system,
        temperatures,
        prod_ns=args.prod_ns,
        equil_ns=args.equil_ns,
        n_samples=args.n_samples,
        dt_ps=args.dt_ps,
        output_file=args.output,
        slim_output_file=args.slim_output,
        platform_name=args.platform,
        timeseries_dir=args.timeseries_dir,
        resume=args.resume,
    )

    if not args.no_validate:
        print("\n=== Validation (EmpiricalTemperatureData) ===")
        ok = validate_calibration_file(args.output, args.slim_output)
        if not ok:
            return 1

    print(f"\nDone. Full: {output_path}")
    print(f"      Slim: {args.slim_output}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
