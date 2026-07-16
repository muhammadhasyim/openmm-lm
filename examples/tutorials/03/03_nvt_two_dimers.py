#!/usr/bin/env python3
"""Tutorial 03: NVT two dimers (LJ + Coulomb) — resonant polariton splitting."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from tutorial_common import OMEGA_C_CM1, run_nvt_bussi_two_dimers, select_platform


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--lambda-coupling", type=float, default=0.03)
    parser.add_argument("--temperature-K", type=float, default=100.0)
    parser.add_argument("--steps", type=int, default=8000)
    parser.add_argument("--equilibration-steps", type=int, default=0)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--platform", default=None)
    parser.add_argument("--min-split-cm1", type=float, default=15.0)
    args = parser.parse_args()

    platform_name = args.platform or select_platform(prefer_cuda=False).getName()
    print(
        f"Tutorial 03 — NVT two dimers: lambda={args.lambda_coupling}, "
        f"T={args.temperature_K} K, steps={args.steps}, platform={platform_name}"
    )

    result = run_nvt_bussi_two_dimers(
        lambda_coupling=args.lambda_coupling,
        temperature_K=args.temperature_K,
        n_steps=args.steps,
        equilibration_steps=args.equilibration_steps,
        seed=args.seed,
        platform_name=platform_name,
    )

    lp = result["lp_frequency_cm1"]
    up = result["up_frequency_cm1"]
    split = result["polariton_split_cm1"]
    print(f"\nDominant peak: {result['peak_frequency_cm1']:.0f} cm^-1")
    if lp is not None and up is not None:
        print(
            f"Polaritons: LP = {lp:.0f} cm^-1, UP = {up:.0f} cm^-1, "
            f"split = {split:.0f} cm^-1  (omega_c = {OMEGA_C_CM1:.0f} cm^-1)"
        )
    if result["local_peaks_cm1"]:
        peaks = ", ".join(f"{f:.0f}" for f, _ in result["local_peaks_cm1"][:4])
        print(f"  Peaks near omega_c: {peaks} cm^-1")

    failures = []
    if lp is None or up is None:
        failures.append("could not resolve LP and UP polariton peaks straddling omega_c")
    elif split is not None and split < args.min_split_cm1:
        failures.append(f"polariton split {split:.0f} cm^-1 < {args.min_split_cm1:.0f} cm^-1")
    if lp is not None and lp >= OMEGA_C_CM1:
        failures.append(f"LP peak {lp:.0f} cm^-1 is not below omega_c")
    if up is not None and up <= OMEGA_C_CM1:
        failures.append(f"UP peak {up:.0f} cm^-1 is not above omega_c")

    if failures:
        print("\nFAIL:")
        for msg in failures:
            print(f"  - {msg}")
        return 1

    print("\nPASS: Tutorial 03 checks succeeded.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
