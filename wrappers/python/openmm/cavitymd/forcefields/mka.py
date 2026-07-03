"""Modified Kob-Andersen (mKA) molecular system builder."""

from __future__ import annotations

import numpy as np
import openmm
from openmm import unit

from openmm.cavitymd.constants import Units

BOHR_TO_NM = Units.BOHR_TO_NM
HARTREE_TO_KJMOL = Units.HARTREE_TO_KJMOL

MASS_A = 16.0
MASS_B = 14.0

K_AA_AU = 0.73204
R0_AA_AU = 2.281655158
K_BB_AU = 1.4325
R0_BB_AU = 2.0743522177

EPS_AA_AU = 1.6685e-4
SIG_AA_AU = 6.2304
EPS_BB_AU = 8.3426e-5
SIG_BB_AU = 5.4828
EPS_AB_AU = 2.5028e-4
SIG_AB_AU = 4.9832
RCUT_AU = 15.0

CHARGE_MAG = 0.3
FKT_KMAG_PAPER_AU = 6.0
FKT_KMAG_AU = 2.0 * np.pi / SIG_AA_AU
BOX_AU = 40.0
NUM_MOL = 250
FRAC_AA = 0.8
REFERENCE_ATOM_DENSITY_AU3 = 2 * NUM_MOL / (BOX_AU ** 3)

OMEGA_C_CM1 = 1560.0
PHOTON_MASS_AMU = 1.0 / 1822.888
HARTREE_TO_CM1 = Units.HARTREE_TO_CM1


def box_au_for_num_molecules(num_molecules: int) -> float:
    """Return cubic box edge (Bohr) at fixed reference number density."""
    return (2 * num_molecules / REFERENCE_ATOM_DENSITY_AU3) ** (1.0 / 3.0)


def _au_to_openmm_bond(k_au, r0_au):
    k_kjmol_nm2 = k_au * HARTREE_TO_KJMOL / (BOHR_TO_NM ** 2)
    r0_nm = r0_au * BOHR_TO_NM
    return k_kjmol_nm2, r0_nm


def _au_to_openmm_lj(eps_au, sig_au):
    eps_kjmol = eps_au * HARTREE_TO_KJMOL
    sig_nm = sig_au * BOHR_TO_NM
    return eps_kjmol, sig_nm


def build_mka_system(
    num_molecules=NUM_MOL,
    frac_aa=FRAC_AA,
    box_au=None,
    seed=42,
    sample_bonds_at_T=None,
):
    """Build the modified Kob-Andersen dipole system without a cavity particle."""
    np.random.seed(seed)

    if box_au is None:
        box_au = box_au_for_num_molecules(num_molecules)

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
    lj_force = openmm.CustomNonbondedForce(
        "lj - ljcut;"
        "lj = 4*eps*((sig/r)^12 - (sig/r)^6);"
        "ljcut = 4*eps*((sig/rc)^12 - (sig/rc)^6);"
        "eps = epsfun(type1, type2);"
        "sig = sigfun(type1, type2)"
    )
    lj_force.addPerParticleParameter("type")
    lj_force.addGlobalParameter("rc", rcut_nm)
    eps_table_3x3 = [
        eps_aa, eps_ab, 0.0,
        eps_ab, eps_bb, 0.0,
        0.0, 0.0, 0.0,
    ]
    sig_table_3x3 = [
        sig_aa, sig_ab, 1.0,
        sig_ab, sig_bb, 1.0,
        1.0, 1.0, 1.0,
    ]
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
                d = np.array(
                    [
                        np.sin(phi) * np.cos(theta),
                        np.sin(phi) * np.sin(theta),
                        np.cos(phi),
                    ]
                )

                if is_aa:
                    mass, r0, k_bond = MASS_A, r0_aa, k_aa
                    atom_type = 0
                else:
                    mass, r0, k_bond = MASS_B, r0_bb, k_bb
                    atom_type = 1

                center = np.array([cx, cy, cz])
                if sample_bonds_at_T is not None and sample_bonds_at_T > 0:
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

    for idx1, idx2 in bonded_pairs:
        coulomb_force.addException(idx1, idx2, 0.0, 1.0, 0.0)
        lj_force.addExclusion(idx1, idx2)

    system.addForce(bond_force)
    system.addForce(coulomb_force)
    system.addForce(lj_force)

    num_mol_particles = system.getNumParticles()
    return system, positions, num_mol_particles


def add_cavity_particle(system, positions):
    """Add a single cavity (photon) particle with no nonbonded interactions."""
    cavity_index = system.addParticle(PHOTON_MASS_AMU)
    positions.append(openmm.Vec3(0, 0, 0) * unit.nanometer)

    for force_idx in range(system.getNumForces()):
        force = system.getForce(force_idx)
        if isinstance(force, openmm.NonbondedForce):
            force.addParticle(0.0, 0.1, 0.0)
        elif isinstance(force, openmm.CustomNonbondedForce):
            force.addParticle([2.0])
    return cavity_index
