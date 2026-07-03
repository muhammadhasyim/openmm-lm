"""Core datatypes for the unified CavityMD force field interface."""

from __future__ import annotations

from dataclasses import dataclass, field
from enum import Enum
from typing import Any, Callable, Optional, Protocol, Sequence

import numpy as np
import openmm
from openmm import app


class CavityBackend(Enum):
    """How cavity forces are applied after DipoleResponse is available."""

    NATIVE_CAVITY_FORCE = "native"
    ML_PYTHONFORCE = "pythonforce"
    ML_CUDA_BRIDGE = "cuda_bridge"


@dataclass(frozen=True)
class DipoleResponse:
    """Universal cavity coupling input: dipole mu and BEC tensor Z = d mu / d r."""

    dipole_enm: np.ndarray
    bec: np.ndarray
    atom_indices: np.ndarray


class DipoleProvider(Protocol):
    def evaluate_dipole_response(self, state: openmm.State) -> DipoleResponse:
        ...


@dataclass(frozen=True)
class CavityParams:
    omegac: float
    lambda_coupling: float
    photon_mass: float = 1.0
    include_dse: bool = True


@dataclass
class BuiltCavitySystem:
    system: openmm.System
    positions: Sequence[openmm.Vec3]
    cavity_backend: CavityBackend
    dipole_provider: DipoleProvider
    cavity_particle_index: int
    real_atom_indices: np.ndarray
    topology: Optional[app.Topology]
    metadata: dict = field(default_factory=dict)
    cavity_force: Optional[openmm.CavityForce] = None


@dataclass(frozen=True)
class ForceFieldSpec:
    name: str
    cavity_backend: CavityBackend
    provides_dipole_jacobian: bool
    requires_topology: bool
    optional_deps: tuple[str, ...]
    builder: Callable[..., BuiltCavitySystem]
    validate_dipole: Callable[[], None]
