"""Dipole moment and Born effective charge (BEC) helpers for CavityMD coupling."""

from __future__ import annotations

from typing import Optional, Sequence

import numpy as np
import openmm
from openmm import unit

from .types import DipoleResponse

__all__ = [
    "DipoleResponse",
    "dipole_from_charges",
    "bec_from_scalar_charges",
    "validate_dipole_response",
    "extract_dipole_and_bec_from_model",
    "ChargeDipoleProvider",
    "SystemChargeDipoleProvider",
]


def dipole_from_charges(
    positions_nm: np.ndarray,
    charges: np.ndarray,
    atom_indices: np.ndarray,
) -> np.ndarray:
    """Compute molecular dipole mu = sum_i q_i r_i in e·nm."""
    positions_nm = np.asarray(positions_nm, dtype=np.float64)
    charges = np.asarray(charges, dtype=np.float64).reshape(-1)
    atom_indices = np.asarray(atom_indices, dtype=int).reshape(-1)
    coords = positions_nm[atom_indices]
    return np.sum(charges[:, None] * coords, axis=0)


def bec_from_scalar_charges(
    charges: np.ndarray,
    atom_indices: np.ndarray,
) -> np.ndarray:
    """Return Z with Z_{i,alpha,beta} = q_i delta_{alpha,beta} for point charges."""
    charges = np.asarray(charges, dtype=np.float64).reshape(-1)
    atom_indices = np.asarray(atom_indices, dtype=int).reshape(-1)
    n_atoms = len(atom_indices)
    bec = np.zeros((n_atoms, 3, 3), dtype=np.float64)
    for local_i, charge in enumerate(charges):
        bec[local_i] = charge * np.eye(3)
    return bec


def validate_dipole_response(response: DipoleResponse) -> None:
    """Validate shapes and finiteness of a DipoleResponse."""
    dipole = np.asarray(response.dipole_enm, dtype=np.float64)
    bec = np.asarray(response.bec, dtype=np.float64)
    atom_indices = np.asarray(response.atom_indices, dtype=int)

    if dipole.shape != (3,):
        raise ValueError(f"dipole_enm must have shape (3,), got {dipole.shape}")
    if bec.ndim != 3 or bec.shape[1:] != (3, 3):
        raise ValueError(f"bec must have shape (n_atoms, 3, 3), got {bec.shape}")
    if bec.shape[0] != atom_indices.shape[0]:
        raise ValueError(
            f"bec rows ({bec.shape[0]}) must match atom_indices ({atom_indices.shape[0]})"
        )
    if not np.all(np.isfinite(dipole)):
        raise ValueError("dipole_enm contains non-finite values")
    if not np.all(np.isfinite(bec)):
        raise ValueError("bec contains non-finite values")


def extract_dipole_and_bec_from_model(
    model_output: dict,
    positions_nm: np.ndarray,
    *,
    dipole_unit: str = "debye",
    box_vectors_nm: Optional[np.ndarray] = None,
    cavity_index: Optional[int] = None,
) -> tuple[np.ndarray, np.ndarray]:
    """Wrap openmmml.cavity_coupling.extract_dipole_and_bec when available."""
    try:
        from openmmml.cavity_coupling import extract_dipole_and_bec
    except ImportError as exc:
        raise ImportError(
            "openmmml is required for ML dipole extraction. "
            "Install with: pixi run -e ml install-ml"
        ) from exc
    return extract_dipole_and_bec(
        model_output,
        positions_nm,
        dipole_unit=dipole_unit,
        box_vectors_nm=box_vectors_nm,
        cavity_index=cavity_index,
    )


class ChargeDipoleProvider:
    """Evaluate mu and Z from fixed partial charges and a subset of atom indices."""

    def __init__(
        self,
        charges: Sequence[float],
        atom_indices: np.ndarray,
    ) -> None:
        self._charges = np.asarray(charges, dtype=np.float64)
        self._atom_indices = np.asarray(atom_indices, dtype=int)

    def evaluate_dipole_response(self, state: openmm.State) -> DipoleResponse:
        positions = state.getPositions(asNumpy=True).value_in_unit(unit.nanometer)
        positions_nm = np.asarray(positions, dtype=np.float64)
        dipole_enm = dipole_from_charges(
            positions_nm, self._charges, self._atom_indices
        )
        bec = bec_from_scalar_charges(self._charges, self._atom_indices)
        response = DipoleResponse(
            dipole_enm=dipole_enm,
            bec=bec,
            atom_indices=self._atom_indices.copy(),
        )
        validate_dipole_response(response)
        return response


class SystemChargeDipoleProvider(ChargeDipoleProvider):
    """Charge dipole provider that reads charges from the first NonbondedForce."""

    def __init__(
        self,
        system: openmm.System,
        atom_indices: np.ndarray,
    ) -> None:
        atom_indices = np.asarray(atom_indices, dtype=int)
        charges = []
        for force in system.getForces():
            if isinstance(force, openmm.NonbondedForce):
                for idx in atom_indices:
                    charge, _, _ = force.getParticleParameters(int(idx))
                    charges.append(charge.value_in_unit(unit.elementary_charge))
                break
        else:
            raise ValueError("System has no NonbondedForce for charge dipole provider")
        super().__init__(charges, atom_indices)
