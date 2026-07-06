#!/usr/bin/env python
"""
Diagnostic probe to confirm CUDA_ERROR_INVALID_PTX root cause.
Tests N_mol=16000 with the interaction group removed and photon type=2.
"""
import os, subprocess, gc, sys
from pathlib import Path

sys.path.insert(0, "/scratch/mh7373/openmm/research/c2f")
import openmm
from openmm import unit

# Import constants from run_c2f
from run_c2f import (
    build_mka_system as build_mka_system_original,
    add_cavity_particle as add_cavity_particle_original,
    OMEGA_C_CM1,
    PHOTON_MASS_AMU,
    BOHR_TO_NM,
)
from openmm.cavitymd import DualThermostat, assign_force_groups

OMEGAC_AU = OMEGA_C_CM1 / 219474.63


def build_mka_system_fixed(num_molecules, box_au, seed=42, frac_aa=0.8, sample_bonds_at_T=None):
    """
    Build mKA system WITHOUT interaction group, using 3x3 LJ tables
    where type 2 (photon) has zero interactions.
    """
    import numpy as np

    BOHR_TO_NM = 0.0529177
    MASS_A, MASS_B = 16.0, 14.0
    K_AA_AU, R0_AA_AU = 0.73204, 2.281655158
    K_BB_AU, R0_BB_AU = 1.4325, 2.0743522177
    EPS_AA_AU, SIG_AA_AU = 1.6685e-4, 6.2304
    EPS_BB_AU, SIG_BB_AU = 8.3426e-5, 5.4828
    EPS_AB_AU, SIG_AB_AU = 2.5028e-4, 4.9832
    RCUT_AU = 15.0
    CHARGE_MAG = 0.3

    def _au_to_openmm_lj(eps_au, sig_au):
        eps_kjmol = eps_au * 2625.5
        sig_nm = sig_au * BOHR_TO_NM
        return eps_kjmol, sig_nm

    def _au_to_openmm_bond(k_au, r0_au):
        k_kjmol_nm2 = k_au * 2625.5 / (BOHR_TO_NM ** 2)
        r0_nm = r0_au * BOHR_TO_NM
        return k_kjmol_nm2, r0_nm

    np.random.seed(seed)
    box_nm = box_au * BOHR_TO_NM
    rcut_nm = RCUT_AU * BOHR_TO_NM

    system = openmm.System()
    system.setDefaultPeriodicBoxVectors(
        openmm.Vec3(box_nm, 0, 0),
        openmm.Vec3(0, box_nm, 0),
        openmm.Vec3(0, 0, box_nm),
    )

    bond_force = openmm.HarmonicBondForce()

    coulomb_force = openmm.NonbondedForce()
    coulomb_force.setNonbondedMethod(openmm.NonbondedForce.PME)
    coulomb_force.setCutoffDistance(rcut_nm)
    coulomb_force.setUseDispersionCorrection(False)

    eps_aa, sig_aa = _au_to_openmm_lj(EPS_AA_AU, SIG_AA_AU)
    eps_bb, sig_bb = _au_to_openmm_lj(EPS_BB_AU, SIG_BB_AU)
    eps_ab, sig_ab = _au_to_openmm_lj(EPS_AB_AU, SIG_AB_AU)

    # 3x3 LJ tables: type 0=A, 1=B, 2=photon (non-interacting)
    # Row-major: eps[t1 + 3*t2]
    eps_table_3x3 = [
        eps_aa, eps_ab, 0.0,  # type 0 with 0, 1, 2
        eps_ab, eps_bb, 0.0,  # type 1 with 0, 1, 2
        0.0, 0.0, 0.0,        # type 2 with 0, 1, 2 (photon has no LJ)
    ]
    sig_table_3x3 = [
        sig_aa, sig_ab, 1.0,  # type 0 with 0, 1, 2 (sig irrelevant when eps=0)
        sig_ab, sig_bb, 1.0,  # type 1 with 0, 1, 2
        1.0, 1.0, 1.0,        # type 2 with 0, 1, 2
    ]

    lj_force = openmm.CustomNonbondedForce(
        "lj - ljcut;"
        "lj = 4*eps*((sig/r)^12 - (sig/r)^6);"
        "ljcut = 4*eps*((sig/rc)^12 - (sig/rc)^6);"
        "eps = epsfun(type1, type2);"
        "sig = sigfun(type1, type2)"
    )
    lj_force.addPerParticleParameter("type")
    lj_force.addGlobalParameter("rc", rcut_nm)
    lj_force.addTabulatedFunction(
        "epsfun", openmm.Discrete2DFunction(3, 3, eps_table_3x3)
    )
    lj_force.addTabulatedFunction(
        "sigfun", openmm.Discrete2DFunction(3, 3, sig_table_3x3)
    )
    lj_force.setNonbondedMethod(openmm.CustomNonbondedForce.CutoffPeriodic)
    lj_force.setCutoffDistance(rcut_nm)
    lj_force.setUseLongRangeCorrection(False)

    positions = []
    num_aa = int(frac_aa * num_molecules)
    side = int(np.ceil(num_molecules ** (1.0 / 3.0)))
    spacing = box_nm / side

    k_aa, r0_aa = _au_to_openmm_bond(K_AA_AU, R0_AA_AU)
    k_bb, r0_bb = _au_to_openmm_bond(K_BB_AU, R0_BB_AU)

    bonded_pairs = []

    mol_idx = 0
    for i in range(side):
        for j in range(side):
            for kk in range(side):
                if mol_idx >= num_molecules:
                    break
                is_aa = mol_idx < num_aa

                cx = (i + 0.5) * spacing
                cy = (j + 0.5) * spacing
                cz = (kk + 0.5) * spacing

                theta = np.random.rand() * 2 * np.pi
                phi = np.arccos(2 * np.random.rand() - 1)
                d = np.array([
                    np.sin(phi) * np.cos(theta),
                    np.sin(phi) * np.sin(theta),
                    np.cos(phi),
                ])

                if is_aa:
                    mass, r0, k_bond = MASS_A, r0_aa, k_aa
                    atom_type = 0
                else:
                    mass, r0, k_bond = MASS_B, r0_bb, k_bb
                    atom_type = 1

                center = np.array([cx, cy, cz])
                if sample_bonds_at_T is not None and sample_bonds_at_T > 0:
                    from openmm.cavitymd import Units
                    sigma_r = np.sqrt(
                        Units.kelvin_to_kT_kjmol(sample_bonds_at_T) / k_bond
                    )
                    r_bond = max(r0 + np.random.normal(0.0, sigma_r), 0.05 * r0)
                else:
                    r_bond = r0

                r1 = center - 0.5 * r_bond * d
                r2 = center + 0.5 * r_bond * d

                idx1 = system.addParticle(mass)
                idx2 = system.addParticle(mass)
                positions.append(openmm.Vec3(*r1) * unit.nanometer)
                positions.append(openmm.Vec3(*r2) * unit.nanometer)

                bond_force.addBond(idx1, idx2, r0, k_bond)

                q1, q2 = -CHARGE_MAG, +CHARGE_MAG
                coulomb_force.addParticle(q1, 1.0, 0.0)
                coulomb_force.addParticle(q2, 1.0, 0.0)
                lj_force.addParticle([float(atom_type)])
                lj_force.addParticle([float(atom_type)])
                bonded_pairs.append((idx1, idx2))

                mol_idx += 1
            if mol_idx >= num_molecules:
                break
        if mol_idx >= num_molecules:
            break

    # Bonded exclusions (must be identical between NonbondedForce and CustomNonbondedForce)
    for idx1, idx2 in bonded_pairs:
        coulomb_force.addException(idx1, idx2, 0.0, 1.0, 0.0)
        lj_force.addExclusion(idx1, idx2)

    # NO interaction group - photon will be type 2 with zero LJ interactions
    # Need matching exceptions for photon in both forces
    print(f"LJ: CustomNonbondedForce (KA non-additive, shifted, r_cut={rcut_nm:.4f} nm), "
          f"{len(bonded_pairs)} bonded exclusions, NO interaction group, 3x3 table")

    system.addForce(bond_force)
    system.addForce(coulomb_force)
    system.addForce(lj_force)

    num_mol_particles = system.getNumParticles()
    print(f"Built mKA system: {num_mol_particles} atoms "
          f"({num_aa} AA + {num_molecules - num_aa} BB dimers), "
          f"box = {box_nm:.4f} nm")

    return system, positions, num_mol_particles


def add_cavity_particle_fixed(system, positions):
    """Add photon with type=2 (non-interacting in 3x3 LJ table)."""
    cavity_index = system.addParticle(PHOTON_MASS_AMU)
    positions.append(openmm.Vec3(0, 0, 0) * unit.nanometer)

    for force_idx in range(system.getNumForces()):
        force = system.getForce(force_idx)
        if isinstance(force, openmm.NonbondedForce):
            force.addParticle(0.0, 0.1, 0.0)
            for i in range(cavity_index):
                force.addException(cavity_index, i, 0.0, 0.1, 0.0)
        elif isinstance(force, openmm.CustomNonbondedForce):
            # Photon is type 2 (non-interacting in 3x3 LJ table)
            force.addParticle([2.0])
            # Add exclusions matching NonbondedForce exceptions for identical exception check
            for i in range(cavity_index):
                force.addExclusion(cavity_index, i)
    return cavity_index


def gpu_mem_mib():
    pid = os.getpid()
    try:
        out = subprocess.check_output([
            "nvidia-smi", "--query-compute-apps=pid,used_memory",
            "--format=csv,noheader,nounits"
        ], text=True)
        for line in out.strip().splitlines():
            p, m = [x.strip() for x in line.split(",")]
            if int(p) == pid:
                return float(m)
    except Exception:
        pass
    return float('nan')


def probe_fixed(n_mol):
    """Test with fixed (no interaction group) version."""
    density = 0.0078125
    box_au = (2 * n_mol / density) ** (1 / 3)
    system, positions, n_mol_p = build_mka_system_fixed(num_molecules=n_mol, box_au=box_au, seed=42)
    cav_idx = add_cavity_particle_fixed(system, positions)
    system.addForce(openmm.CavityForce(cav_idx, OMEGAC_AU, 0.03, PHOTON_MASS_AMU))
    DualThermostat.setup_bussi_for_system(system, list(range(n_mol_p)), 100.0, 1.0)
    assign_force_groups(system)
    integrator = openmm.VerletIntegrator(0.001 * unit.picosecond)
    ctx = openmm.Context(system, integrator, openmm.Platform.getPlatformByName("CUDA"))
    ctx.setPositions(positions)
    ctx.setVelocitiesToTemperature(100 * unit.kelvin)
    for _ in range(100):
        integrator.step(100)
    mem = gpu_mem_mib()
    nat = n_mol_p + 1
    box = box_au * BOHR_TO_NM
    del ctx, integrator, system
    gc.collect()
    return mem, nat, box


def probe_original(n_mol):
    """Test with original (interaction group) version."""
    from run_c2f import add_cavity_particle
    density = 0.0078125
    box_au = (2 * n_mol / density) ** (1 / 3)
    system, positions, n_mol_p = build_mka_system_original(num_molecules=n_mol, box_au=box_au, seed=42)
    cav_idx = add_cavity_particle(system, positions)
    system.addForce(openmm.CavityForce(cav_idx, OMEGAC_AU, 0.03, PHOTON_MASS_AMU))
    DualThermostat.setup_bussi_for_system(system, list(range(n_mol_p)), 100.0, 1.0)
    assign_force_groups(system)
    integrator = openmm.VerletIntegrator(0.001 * unit.picosecond)
    ctx = openmm.Context(system, integrator, openmm.Platform.getPlatformByName("CUDA"))
    ctx.setPositions(positions)
    ctx.setVelocitiesToTemperature(100 * unit.kelvin)
    for _ in range(100):
        integrator.step(100)
    mem = gpu_mem_mib()
    nat = n_mol_p + 1
    box = box_au * BOHR_TO_NM
    del ctx, integrator, system
    gc.collect()
    return mem, nat, box


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--n-mol", type=int, default=16000)
    parser.add_argument("--use-fixed", action="store_true", help="Use fixed version (no interaction group)")
    args = parser.parse_args()

    print(f"\n{'='*60}")
    print(f"Testing N_mol={args.n_mol} with {'FIXED' if args.use_fixed else 'ORIGINAL'} code")
    print(f"{'='*60}\n")

    try:
        if args.use_fixed:
            mem, nat, box = probe_fixed(args.n_mol)
        else:
            mem, nat, box = probe_original(args.n_mol)
        print(f"\nOK  N_mol={args.n_mol:6d}  atoms={nat:7d}  box={box:.1f}nm  GPU={mem:.0f} MiB")
    except Exception as e:
        print(f"\nFAIL N_mol={args.n_mol:6d}  {type(e).__name__}: {str(e)[:200]}")
        import traceback
        traceback.print_exc()
