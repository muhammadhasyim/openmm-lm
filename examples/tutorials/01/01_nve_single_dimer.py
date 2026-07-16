#!/usr/bin/env python3
"""Tutorial 01: NVE single A–A dimer with cavity coupling and IR spectrum."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from tutorial_common import (
    OMEGA_C_CM1,
    compare_finite_q_energy_exchange,
    run_nve_single_dimer,
    select_platform,
)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--lambda-coupling", type=float, default=0.01)
    parser.add_argument("--temperature-K", type=float, default=100.0)
    parser.add_argument("--steps", type=int, default=12000)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--platform", default=None)
    parser.add_argument(
        "--peak-tolerance-cm1",
        type=float,
        default=200.0,
        help="Allowed deviation of spectral peak from omega_c (cm^-1)",
    )
    args = parser.parse_args()

    platform_name = args.platform or select_platform(prefer_cuda=False).getName()
    print(
        f"Tutorial 01 — NVE single dimer: lambda={args.lambda_coupling}, "
        f"T={args.temperature_K} K, steps={args.steps}, platform={platform_name}"
    )

    q_demo = compare_finite_q_energy_exchange(
        lambda_coupling=args.lambda_coupling,
        platform_name=platform_name,
    )
    print("\nFinite-q displacement demo (zero initial velocities):")
    print(
        f"  No shift:  max |q-q_eq| = {q_demo['no_shift']['max_q_deviation_nm']:.4f} nm, "
        f"dU_amp = {q_demo['no_shift']['potential_energy_amplitude_kj_mol']:.3f} kJ/mol"
    )
    print(
        f"  With shift: max |q-q_eq| = {q_demo['with_shift']['max_q_deviation_nm']:.4f} nm, "
        f"dU_amp = {q_demo['with_shift']['potential_energy_amplitude_kj_mol']:.3f} kJ/mol"
    )

    result = run_nve_single_dimer(
        lambda_coupling=args.lambda_coupling,
        temperature_K=args.temperature_K,
        n_steps=args.steps,
        seed=args.seed,
        platform_name=platform_name,
    )
    peak = result["peak_frequency_cm1"]
    print(
        f"\nIR peak: {peak:.0f} cm^-1  (omega_c = {OMEGA_C_CM1:.0f} cm^-1)"
    )
    if result["local_peaks_cm1"]:
        top = ", ".join(f"{f:.0f}" for f, _ in result["local_peaks_cm1"][:3])
        print(f"  Local maxima: {top} cm^-1")

    failures = []
    if abs(peak - OMEGA_C_CM1) > args.peak_tolerance_cm1:
        failures.append(
            f"peak {peak:.0f} cm^-1 deviates from omega_c by more than "
            f"{args.peak_tolerance_cm1:.0f} cm^-1"
        )
    if q_demo["no_shift"]["max_q_deviation_nm"] <= q_demo["with_shift"]["max_q_deviation_nm"]:
        failures.append("finite-q shift did not reduce photon displacement from equilibrium")
    if q_demo["exchange_ratio"] < 5.0:
        failures.append(
            f"expected stronger energy exchange without shift (ratio={q_demo['exchange_ratio']:.1f})"
        )

    if failures:
        print("\nFAIL:")
        for msg in failures:
            print(f"  - {msg}")
        return 1

    print("\nPASS: Tutorial 01 checks succeeded.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
