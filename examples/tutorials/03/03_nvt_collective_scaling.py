#!/usr/bin/env python3
"""Tutorial 03: NVT collective coupling at fixed lambda*sqrt(N) for N=1,2,4,8."""

from __future__ import annotations

from pathlib import Path

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
import numpy as np
import openmm
from openmm import unit
from scipy import fftpack, signal

from openmm.cavitymd.constants import Units
from openmm.cavitymd.forcefields.mka import (
    MASS_A,
    PHOTON_MASS_AMU,
    K_AA_AU,
    R0_AA_AU,
    CHARGE_MAG,
    OMEGA_C_CM1,
)
from openmm.cavitymd.thermostats import DualThermostat


def dipole_magnitude(pos, charges, indices):
    d = np.dot(charges, pos[indices])
    return np.linalg.norm(d)


def ir_spectrum_from_dipole(dipoles, dt_fs, T_K, fraction_acf=0.25):
    dipole_signal = np.asarray(dipoles, dtype=float) - np.mean(dipoles)
    N_sig = len(dipole_signal)
    n_acf = max(3, int(N_sig * fraction_acf))
    shifted = np.zeros(2 * N_sig if N_sig % 2 == 0 else 2 * N_sig - 1)
    shifted[N_sig // 2 : N_sig // 2 + N_sig] = dipole_signal
    acf_full = signal.fftconvolve(shifted, dipole_signal[::-1], mode="same")[-N_sig:] / np.arange(
        N_sig, 0, -1
    )
    autocorr = acf_full[:n_acf]
    timestep = dt_fs * 1e-15
    lineshape = fftpack.dct(autocorr, type=1)[1:]
    freqs_hz = np.linspace(0, 0.5 / timestep, len(autocorr))[1:]
    freqs_cm1 = freqs_hz / (100.0 * 299792458.0)
    boltz = 1.38064852e-23
    hbar = 1.05457180013e-34
    field = freqs_hz * (1.0 - np.exp(-hbar * freqs_hz / (boltz * T_K)))
    return freqs_cm1, lineshape * field


def select_platform():
    for name in ("CUDA", "OpenCL", "CPU", "Reference"):
        try:
            return openmm.Platform.getPlatformByName(name)
        except Exception:
            continue
    raise RuntimeError("No OpenMM platform available")


def build_n_aa_dimers_cavity(
    n_mol,
    lambda_coupling,
    omegac_au,
    sep_nm,
    T_K,
    seed,
    dt_ps,
    k_aa_omm,
    r0_aa_omm,
    half_r0,
):
    system = openmm.System()
    bond_force = openmm.HarmonicBondForce()
    nb = openmm.NonbondedForce()
    nb.setNonbondedMethod(openmm.NonbondedForce.NoCutoff)

    positions = []
    mol_indices = []
    charges = []
    side = int(np.ceil(n_mol ** (1.0 / 3.0)))
    mol_id = 0
    for ix in range(side):
        for iy in range(side):
            for iz in range(side):
                if mol_id >= n_mol:
                    break
                cx = (ix - 0.5 * (side - 1)) * sep_nm
                cy = (iy - 0.5 * (side - 1)) * sep_nm
                cz = (iz - 0.5 * (side - 1)) * sep_nm
                i0 = system.getNumParticles()
                system.addParticle(MASS_A)
                system.addParticle(MASS_A)
                bond_force.addBond(i0, i0 + 1, r0_aa_omm, k_aa_omm)
                nb.addParticle(-CHARGE_MAG, 0.1, 0.0)
                nb.addParticle(+CHARGE_MAG, 0.1, 0.0)
                nb.addException(i0, i0 + 1, 0.0, 1.0, 0.0)
                positions.append(openmm.Vec3(cx - half_r0, cy, cz) * unit.nanometer)
                positions.append(openmm.Vec3(cx + half_r0, cy, cz) * unit.nanometer)
                mol_indices.extend([i0, i0 + 1])
                charges.extend([-CHARGE_MAG, +CHARGE_MAG])
                mol_id += 1
            if mol_id >= n_mol:
                break
        if mol_id >= n_mol:
            break

    for m1 in range(n_mol):
        for m2 in range(m1 + 1, n_mol):
            for a in (0, 1):
                for b in (0, 1):
                    nb.addException(2 * m1 + a, 2 * m2 + b, 0.0, 1.0, 0.0)

    photon_index = system.addParticle(PHOTON_MASS_AMU)
    nb.addParticle(0.0, 0.1, 0.0)
    positions.append(openmm.Vec3(0, 0, 0) * unit.nanometer)
    system.addForce(bond_force)
    system.addForce(nb)

    system.addForce(openmm.CavityForce(photon_index, omegac_au, lambda_coupling, PHOTON_MASS_AMU))
    displacer = openmm.CavityParticleDisplacer(photon_index, omegac_au, PHOTON_MASS_AMU)
    displacer.setSwitchOnStep(0)
    displacer.setSwitchOnLambda(lambda_coupling)
    system.addForce(displacer)
    DualThermostat.setup_bussi_for_system(
        system, molecular_indices=mol_indices, temperature_K=T_K, tau_ps=1.0, random_number_seed=seed
    )

    integrator = openmm.VerletIntegrator(dt_ps)
    platform = select_platform()
    if platform.getName() == "CUDA":
        platform.setPropertyDefaultValue("Precision", "mixed")
    context = openmm.Context(system, integrator, platform)
    context.setPositions(positions)
    context.setVelocitiesToTemperature(T_K, seed)
    integrator.step(1)
    return system, context, integrator, mol_indices, np.asarray(charges, dtype=float)


def main():
    g_collective = 0.5
    N_list = [1, 2, 4, 8]
    T_K = 100.0
    dt_fs = 1.0
    N_steps = 12_000
    seed = 42
    sep_nm = 2.0

    dt_ps = dt_fs * 1e-3
    omegac_au = Units.cm1_to_au(OMEGA_C_CM1)
    B2NM = Units.BOHR_TO_NM
    H2K = Units.HARTREE_TO_KJMOL
    k_aa_omm = K_AA_AU * H2K / B2NM**2
    r0_aa_omm = R0_AA_AU * B2NM
    half_r0 = r0_aa_omm / 2.0

    print(f"g = lambda*sqrt(N) = {g_collective}")
    spectra = {}
    for N in N_list:
        lam = g_collective / np.sqrt(N)
        system, context, integrator, mol_indices, charges = build_n_aa_dimers_cavity(
            N, lam, omegac_au, sep_nm, T_K, seed + N, dt_ps, k_aa_omm, r0_aa_omm, half_r0
        )
        dipoles = []
        for _ in range(N_steps):
            integrator.step(1)
            pos = context.getState(getPositions=True).getPositions(asNumpy=True).value_in_unit(
                unit.nanometer
            )
            dipoles.append(dipole_magnitude(pos, charges, mol_indices))
        freqs, spectrum = ir_spectrum_from_dipole(dipoles, dt_fs, T_K)
        spectra[N] = (freqs, spectrum, lam)
        print(f"N={N}: lambda={lam:.6f}, g={lam * np.sqrt(N):.6f}")

    fig, ax = plt.subplots(figsize=(9, 4))
    for N in N_list:
        freqs, spec, lam = spectra[N]
        mask = (freqs > 1000) & (freqs < 2200)
        s = spec[mask]
        if s.max() > 0:
            s = s / s.max()
        ax.plot(freqs[mask], s, lw=1.1, label=f"N={N}, g={lam * np.sqrt(N):.3f}")
    ax.axvline(OMEGA_C_CM1, color="k", ls="--", lw=1.0)
    ax.set_xlabel("Frequency (cm^-1)")
    ax.set_ylabel("Intensity (arb.)")
    ax.set_title(f"IR overlay at fixed g = {g_collective}")
    ax.legend()
    plt.tight_layout()
    out = Path(__file__).resolve().parent / "03_ir_spectrum.png"
    plt.savefig(out, dpi=120)
    print(f"Wrote {out}")


if __name__ == "__main__":
    main()
