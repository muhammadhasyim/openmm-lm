#!/usr/bin/env python3
"""Regression tests for mKA cavity MD tutorial physics (01–03)."""

from __future__ import annotations

import sys
from pathlib import Path

import pytest

TUTORIAL_DIR = Path(__file__).resolve().parents[2] / "examples" / "tutorial"
sys.path.insert(0, str(TUTORIAL_DIR))

openmm = pytest.importorskip("openmm")

from tutorial_common import (  # noqa: E402
    OMEGA_C_CM1,
    build_single_aa_dimer_charged_system,
    compare_finite_q_energy_exchange,
    create_context,
    run_nve_single_dimer,
    run_nvt_bussi_single_dimer,
    run_nvt_bussi_two_dimers,
    run_nvt_single_dimer,
    select_platform,
)

PLATFORM = select_platform(prefer_cuda=False).getName()


@pytest.mark.timeout(300)
def test_tutorial01_nve_peak_and_finite_q():
    """Tutorial 01: single IR peak near omega_c and finite-q suppresses exchange."""
    result = run_nve_single_dimer(
        n_steps=4000,
        seed=42,
        platform_name=PLATFORM,
    )
    assert abs(result["peak_frequency_cm1"] - OMEGA_C_CM1) < 250.0

    q_demo = compare_finite_q_energy_exchange(
        n_steps=300,
        platform_name=PLATFORM,
    )
    assert q_demo["no_shift"]["max_q_deviation_nm"] > q_demo["with_shift"]["max_q_deviation_nm"]
    assert q_demo["exchange_ratio"] > 5.0


@pytest.mark.timeout(300)
def test_tutorial02_nvt_temperature_and_spectrum_peak():
    """Tutorial 02: molecular T near bath and spectral peak near omega_c."""
    result = run_nvt_bussi_single_dimer(
        lambda_coupling=0.01,
        temperature_K=100.0,
        n_steps=8000,
        equilibration_steps=500,
        seed=42,
        platform_name=PLATFORM,
    )

    assert abs(result["mean_temperature_K"] - 100.0) < 40.0, (
        f"Mean T={result['mean_temperature_K']:.1f} K too far from 100 K"
    )
    assert abs(result["peak_frequency_cm1"] - OMEGA_C_CM1) < 250.0, (
        f"Peak {result['peak_frequency_cm1']:.0f} cm^-1 too far from "
        f"omega_c={OMEGA_C_CM1:.0f} cm^-1"
    )


@pytest.mark.timeout(300)
def test_tutorial03_polariton_splitting():
    """Tutorial 03: LP and UP peaks straddle cavity frequency."""
    result = run_nvt_bussi_two_dimers(
        lambda_coupling=0.03,
        temperature_K=100.0,
        n_steps=8000,
        equilibration_steps=500,
        seed=42,
        platform_name=PLATFORM,
    )

    lp = result["lp_frequency_cm1"]
    up = result["up_frequency_cm1"]
    assert lp is not None and up is not None, "LP/UP polariton peaks not resolved"
    assert lp < OMEGA_C_CM1 < up
    assert result["polariton_split_cm1"] is not None
    assert result["polariton_split_cm1"] >= 15.0


@pytest.mark.timeout(300)
def test_nvt_langevin_temperature_and_spectrum_peak():
    """Legacy Section 2 (Langevin NVT) validation."""
    result = run_nvt_single_dimer(
        lambda_coupling=0.01,
        temperature_K=100.0,
        n_steps=5000,
        seed=42,
        platform_name=PLATFORM,
    )

    assert abs(result["mean_system_temperature_K"] - 100.0) < 50.0
    assert abs(result["peak_frequency_cm1"] - OMEGA_C_CM1) < 600.0


@pytest.mark.timeout(120)
def test_displace_to_equilibrium_is_small():
    """Photon equilibrium displacement should be tiny for a single dimer."""
    from tutorial_common import Units

    omegac_au = Units.cm1_to_au(OMEGA_C_CM1)
    system, displacer, positions = build_single_aa_dimer_charged_system(
        lambda_coupling=0.01, omegac_au=omegac_au
    )
    context = create_context(
        system, dt_fs=1.0, temperature_K=100.0, seed=42, positions=positions
    )
    displacer.displaceToEquilibrium(context, 0.01)

    pos = context.getState(getPositions=True).getPositions(asNumpy=True)
    pos_nm = pos.value_in_unit(openmm.unit.nanometer)
    q_xy = float((pos_nm[2, 0] ** 2 + pos_nm[2, 1] ** 2) ** 0.5)
    assert q_xy < 0.1, f"Photon displacement {q_xy:.4f} nm is too large"
