#!/usr/bin/env python3
"""Tutorial 01: NVE single A–A dimer (external copy; uses repo tutorial_common)."""

from __future__ import annotations

import sys
from pathlib import Path

REPO_TUTORIAL = Path(__file__).resolve().parents[2] / "openmm-lm" / "examples" / "tutorial"
sys.path.insert(0, str(REPO_TUTORIAL))

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
import numpy as np

from tutorial_common import (
    OMEGA_C_CM1,
    compare_finite_q_energy_exchange,
    run_nve_single_dimer,
    select_platform,
)

lambda_coupling = 0.01
T_K = 100.0
dt_fs = 1.0
N_steps = 12000
seed = 42
platform_name = select_platform(prefer_cuda=True).getName()

print(f"λ = {lambda_coupling},  T = {T_K} K,  dt = {dt_fs} fs,  ω_c = {OMEGA_C_CM1} cm⁻¹")
print(f"Platform: {platform_name}")

q_demo = compare_finite_q_energy_exchange(
    lambda_coupling=lambda_coupling,
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
    lambda_coupling=lambda_coupling,
    temperature_K=T_K,
    dt_fs=dt_fs,
    n_steps=N_steps,
    seed=seed,
    platform_name=platform_name,
)
dipoles = result["dipoles"]
t_ps = np.arange(len(dipoles)) * dt_fs * 1e-3

fig, ax = plt.subplots(figsize=(10, 3))
ax.plot(t_ps, dipoles, lw=0.6, label="|d|")
ax.set_xlabel("time (ps)")
ax.set_ylabel("dipole magnitude (e·nm)")
ax.set_title("NVE single A–A dimer — rotation-invariant dipole")
ax.legend()
plt.tight_layout()
plt.savefig("01_dipole_trace.png", dpi=120)

freqs = result["freqs_cm1"]
spectrum = result["spectrum"]
mask = (freqs > 0) & (freqs <= 4000)
fig, ax = plt.subplots(figsize=(9, 4))
if spectrum[mask].max() > 0:
    ax.plot(freqs[mask], spectrum[mask] / spectrum[mask].max(), lw=0.9)
ax.axvline(OMEGA_C_CM1, color="r", ls="--", lw=1.2, label=f"ω_c = {OMEGA_C_CM1} cm⁻¹")
ax.set_xlim(0, 4000)
ax.set_xlabel("Frequency (cm⁻¹)")
ax.set_ylabel("Intensity (arb.)")
ax.set_title("IR spectrum from dipole autocorrelation")
ax.legend()
plt.tight_layout()
plt.savefig("01_ir_spectrum.png", dpi=120)

peak = result["peak_frequency_cm1"]
print(f"\nPeak frequency: {peak:.0f} cm⁻¹   (reference ω_c = {OMEGA_C_CM1:.0f} cm⁻¹)")
