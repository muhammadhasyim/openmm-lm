"""Unified force field registry for CavityMD (classical + ML dipole backends)."""

from __future__ import annotations

from .dipole import (
    ChargeDipoleProvider,
    SystemChargeDipoleProvider,
    bec_from_scalar_charges,
    dipole_from_charges,
    extract_dipole_and_bec_from_model,
    validate_dipole_response,
)
from .registry import (
    CAVITY_INCOMPATIBLE_MSG,
    build_system,
    evaluate_dipole,
    get_spec,
    list_forcefields,
)
from .types import (
    BuiltCavitySystem,
    CavityBackend,
    CavityParams,
    DipoleProvider,
    DipoleResponse,
    ForceFieldSpec,
)

__all__ = [
    "CAVITY_INCOMPATIBLE_MSG",
    "BuiltCavitySystem",
    "CavityBackend",
    "CavityParams",
    "ChargeDipoleProvider",
    "SystemChargeDipoleProvider",
    "DipoleProvider",
    "DipoleResponse",
    "ForceFieldSpec",
    "bec_from_scalar_charges",
    "build_system",
    "dipole_from_charges",
    "evaluate_dipole",
    "extract_dipole_and_bec_from_model",
    "get_spec",
    "list_forcefields",
    "validate_dipole_response",
]
