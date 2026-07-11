#!/usr/bin/env python3
"""Tutorial 02: Bussi NVT single A–A dimer — kinetic temperature and IR spectrum."""

from __future__ import annotations

import argparse
import sys

from tutorial_common import OMEGA_C_CM1, run_nvt_bussi_single_dimer, select_platform


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--lambda-coupling", type=float, default=0.01)
    parser.add_argument("--temperature-K", type=float, default=100.0)
    parser.add_argument("--steps", type=int, default=8000)
    parser.add_argument("--equilibration-steps", type=int, default=500)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--platform", default=None)
    parser.add_argument("--peak-tolerance-cm1", type=float, default=200.0)
    parser.add_argument("--temperature-tolerance-K", type=float, default=40.0)
    args = parser.parse_args()

    platform_name = args.platform or select_platform(prefer_cuda=False).getName()
    print(
        f"Tutorial 02 — NVT single dimer: lambda={args.lambda_coupling}, "
        f"T={args.temperature_K} K, steps={args.steps}, platform={platform_name}"
    )

    result = run_nvt_bussi_single_dimer(
        lambda_coupling=args.lambda_coupling,
        temperature_K=args.temperature_K,
        n_steps=args.steps,
        equilibration_steps=args.equilibration_steps,
        seed=args.seed,
        platform_name=platform_name,
    )

    mean_t = result["mean_temperature_K"]
    std_t = result["std_temperature_K"]
    peak = result["peak_frequency_cm1"]
    print(f"\nMean molecular T_kin = {mean_t:.1f} ± {std_t:.1f} K  (target {args.temperature_K} K)")
    print(f"IR peak: {peak:.0f} cm^-1  (omega_c = {OMEGA_C_CM1:.0f} cm^-1)")

    failures = []
    if abs(mean_t - args.temperature_K) > args.temperature_tolerance_K:
        failures.append(
            f"mean T_kin {mean_t:.1f} K deviates from {args.temperature_K} K by more than "
            f"{args.temperature_tolerance_K:.0f} K"
        )
    if abs(peak - OMEGA_C_CM1) > args.peak_tolerance_cm1:
        failures.append(
            f"peak {peak:.0f} cm^-1 deviates from omega_c by more than "
            f"{args.peak_tolerance_cm1:.0f} cm^-1"
        )

    if failures:
        print("\nFAIL:")
        for msg in failures:
            print(f"  - {msg}")
        return 1

    print("\nPASS: Tutorial 02 checks succeeded.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
