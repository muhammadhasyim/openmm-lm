"""Modified Kob-Andersen (mKA) dipole system builder for C2F examples."""

from __future__ import annotations

import os
from pathlib import Path

import numpy as np
import openmm

from openmm.cavitymd.constants import Units
from openmm.cavitymd.forcefields import CavityParams, build_system
from openmm.cavitymd.forcefields.mka import (
    BOX_AU,
    CHARGE_MAG,
    FRAC_AA,
    FKT_KMAG_AU,
    FKT_KMAG_PAPER_AU,
    HARTREE_TO_CM1,
    NUM_MOL,
    OMEGA_C_CM1,
    PHOTON_MASS_AMU,
    REFERENCE_ATOM_DENSITY_AU3,
    add_cavity_particle,
    box_au_for_num_molecules,
    build_mka_system,
)

BOHR_TO_NM = Units.BOHR_TO_NM
HARTREE_TO_KJMOL = Units.HARTREE_TO_KJMOL
KB_HARTREE_PER_K = Units.KB_HARTREE_PER_K

# Masses (legacy aliases)
MASS_A = 16.0
MASS_B = 14.0

# Harmonic bonds
K_AA_AU = 0.73204
R0_AA_AU = 2.281655158
K_BB_AU = 1.4325
R0_BB_AU = 2.0743522177

# LJ parameters
EPS_AA_AU = 1.6685e-4
SIG_AA_AU = 6.2304
EPS_BB_AU = 8.3426e-5
SIG_BB_AU = 5.4828
EPS_AB_AU = 2.5028e-4
SIG_AB_AU = 4.9832
RCUT_AU = 15.0

BUSSI_TAU_PS = 1.0
_PLATFORM_PREFERENCE = ("CUDA", "CPU", "Reference")
_GPU_PLATFORMS = frozenset({"CUDA", "OpenCL"})
_DEFAULT_GPU_PRECISION = "mixed"


def _configure_platform_precision(platform) -> None:
    """Set GPU precision (mixed default) so adaptive dt stays numerically stable."""
    if platform.getName() not in _GPU_PLATFORMS:
        return
    precision = os.environ.get("OPENMM_PRECISION", _DEFAULT_GPU_PRECISION)
    try:
        platform.setPropertyDefaultValue("Precision", precision)
    except Exception:
        return
    print(f"  CUDA/OpenCL precision: {precision}")


def _select_platform(platform_name=None):
    """Return an OpenMM Platform, honoring OPENMM_PLATFORM and --platform."""
    name = platform_name or os.environ.get("OPENMM_PLATFORM")
    if name:
        platform = openmm.Platform.getPlatformByName(name)
        print(f"Using OpenMM platform: {platform.getName()}")
        _configure_platform_precision(platform)
        return platform

    for candidate in _PLATFORM_PREFERENCE:
        try:
            platform = openmm.Platform.getPlatformByName(candidate)
            print(f"Using OpenMM platform: {platform.getName()} (auto)")
            _configure_platform_precision(platform)
            return platform
        except Exception:
            continue

    raise RuntimeError("No usable OpenMM platform found (tried CUDA, CPU, Reference)")


def build_mka_cavity_system(
    *,
    omegac_au: float,
    lambda_coupling: float = 0.0,
    photon_mass_amu: float = PHOTON_MASS_AMU,
    include_dse: bool = True,
    **kwargs,
):
    """Build mKA system with cavity particle and native CavityForce attached."""
    cavity = CavityParams(
        omegac=omegac_au,
        lambda_coupling=lambda_coupling,
        photon_mass=photon_mass_amu,
        include_dse=include_dse,
    )
    return build_system("mka", cavity=cavity, **kwargs)


def _load_initial_state(path: Path) -> tuple[np.ndarray, np.ndarray | None]:
    """Load positions (and optional velocities) from a final-state npz."""
    data = np.load(path)
    if "positions_nm" not in data:
        raise KeyError(f"{path} missing positions_nm")
    positions = np.asarray(data["positions_nm"], dtype=float)
    velocities = None
    if "velocities_nm_per_ps" in data:
        velocities = np.asarray(data["velocities_nm_per_ps"], dtype=float)
    return positions, velocities


__all__ = [
    "BOHR_TO_NM",
    "HARTREE_TO_KJMOL",
    "KB_HARTREE_PER_K",
    "HARTREE_TO_CM1",
    "MASS_A",
    "MASS_B",
    "K_AA_AU",
    "R0_AA_AU",
    "K_BB_AU",
    "R0_BB_AU",
    "FKT_KMAG_PAPER_AU",
    "FKT_KMAG_AU",
    "EPS_AA_AU",
    "SIG_AA_AU",
    "EPS_BB_AU",
    "SIG_BB_AU",
    "EPS_AB_AU",
    "SIG_AB_AU",
    "RCUT_AU",
    "CHARGE_MAG",
    "BOX_AU",
    "NUM_MOL",
    "FRAC_AA",
    "REFERENCE_ATOM_DENSITY_AU3",
    "OMEGA_C_CM1",
    "PHOTON_MASS_AMU",
    "BUSSI_TAU_PS",
    "box_au_for_num_molecules",
    "build_mka_system",
    "build_mka_cavity_system",
    "add_cavity_particle",
    "_select_platform",
    "_load_initial_state",
]
