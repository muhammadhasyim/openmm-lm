"""Force field registry and build_system entry point."""

from __future__ import annotations

from typing import Any, Callable

import openmm

from .classical import (
    _build_amber_tip4pew_protein,
    _build_dimer_xml,
    _build_mka,
    _build_tip4pew_flex,
    validate_mka_dipole,
)
from .ml_dipole import (
    _build_aimnet2,
    _build_cace_les_bec,
    _build_cace_les_bec_batch,
    _build_mace_polar_1,
    _build_mbpol_2023,
    validate_ml_registry_entry,
)
from .types import BuiltCavitySystem, CavityBackend, CavityParams, ForceFieldSpec

CAVITY_INCOMPATIBLE_MSG = "Force field incompatible with cavity MD"

__all__ = [
    "CAVITY_INCOMPATIBLE_MSG",
    "ForceFieldRegistry",
    "register_forcefield",
    "list_forcefields",
    "get_spec",
    "build_system",
    "evaluate_dipole",
]


class ForceFieldRegistry:
    def __init__(self) -> None:
        self._specs: dict[str, ForceFieldSpec] = {}

    def register(self, spec: ForceFieldSpec) -> None:
        if not spec.provides_dipole_jacobian:
            raise ValueError(
                f"Force field '{spec.name}' must provide dipole moment and "
                f"position Jacobian (BEC); {CAVITY_INCOMPATIBLE_MSG}"
            )
        self._specs[spec.name] = spec

    def get(self, name: str) -> ForceFieldSpec:
        if name not in self._specs:
            raise KeyError(f"Unknown force field '{name}'")
        return self._specs[name]

    def names(self) -> list[str]:
        return sorted(self._specs.keys())


def register_forcefield(
    registry: ForceFieldRegistry,
    *,
    name: str,
    cavity_backend: CavityBackend,
    provides_dipole_jacobian: bool,
    requires_topology: bool,
    builder: Callable[..., BuiltCavitySystem],
    validate_dipole: Callable[[], None],
    optional_deps: tuple[str, ...] = (),
) -> None:
    registry.register(
        ForceFieldSpec(
            name=name,
            cavity_backend=cavity_backend,
            provides_dipole_jacobian=provides_dipole_jacobian,
            requires_topology=requires_topology,
            optional_deps=optional_deps,
            builder=builder,
            validate_dipole=validate_dipole,
        )
    )


_DEFAULT_REGISTRY = ForceFieldRegistry()


def _register_defaults(registry: ForceFieldRegistry) -> None:
    register_forcefield(
        registry,
        name="mka",
        cavity_backend=CavityBackend.NATIVE_CAVITY_FORCE,
        provides_dipole_jacobian=True,
        requires_topology=False,
        builder=_build_mka,
        validate_dipole=validate_mka_dipole,
    )
    register_forcefield(
        registry,
        name="dimer-xml",
        cavity_backend=CavityBackend.NATIVE_CAVITY_FORCE,
        provides_dipole_jacobian=True,
        requires_topology=False,
        builder=_build_dimer_xml,
        validate_dipole=validate_mka_dipole,
    )
    register_forcefield(
        registry,
        name="tip4pew-flex",
        cavity_backend=CavityBackend.NATIVE_CAVITY_FORCE,
        provides_dipole_jacobian=True,
        requires_topology=False,
        builder=_build_tip4pew_flex,
        optional_deps=("openmmforcefields",),
        validate_dipole=validate_mka_dipole,
    )
    register_forcefield(
        registry,
        name="amber-tip4pew-protein",
        cavity_backend=CavityBackend.NATIVE_CAVITY_FORCE,
        provides_dipole_jacobian=True,
        requires_topology=False,
        builder=_build_amber_tip4pew_protein,
        optional_deps=("openmmforcefields", "pdbfixer"),
        validate_dipole=validate_mka_dipole,
    )
    register_forcefield(
        registry,
        name="cace-les-bec",
        cavity_backend=CavityBackend.ML_PYTHONFORCE,
        provides_dipole_jacobian=True,
        requires_topology=True,
        builder=_build_cace_les_bec,
        optional_deps=("openmmml", "cace"),
        validate_dipole=lambda: validate_ml_registry_entry("cace-les-bec"),
    )
    register_forcefield(
        registry,
        name="cace-les-bec-batch",
        cavity_backend=CavityBackend.ML_PYTHONFORCE,
        provides_dipole_jacobian=True,
        requires_topology=True,
        builder=_build_cace_les_bec_batch,
        optional_deps=("openmmml", "cace"),
        validate_dipole=lambda: validate_ml_registry_entry("cace-les-bec-batch"),
    )
    register_forcefield(
        registry,
        name="aimnet2",
        cavity_backend=CavityBackend.ML_PYTHONFORCE,
        provides_dipole_jacobian=True,
        requires_topology=True,
        builder=_build_aimnet2,
        optional_deps=("openmmml", "aimnet"),
        validate_dipole=lambda: validate_ml_registry_entry("aimnet2"),
    )
    register_forcefield(
        registry,
        name="mace-polar-1",
        cavity_backend=CavityBackend.ML_CUDA_BRIDGE,
        provides_dipole_jacobian=True,
        requires_topology=True,
        builder=_build_mace_polar_1,
        optional_deps=("openmmml", "mace-torch"),
        validate_dipole=lambda: validate_ml_registry_entry("mace-polar-1"),
    )
    register_forcefield(
        registry,
        name="mbpol-2023",
        cavity_backend=CavityBackend.ML_PYTHONFORCE,
        provides_dipole_jacobian=True,
        requires_topology=True,
        builder=_build_mbpol_2023,
        optional_deps=("openmmml",),
        validate_dipole=lambda: validate_ml_registry_entry("mbpol-2023"),
    )


_register_defaults(_DEFAULT_REGISTRY)


def list_forcefields() -> list[str]:
    return _DEFAULT_REGISTRY.names()


def get_spec(name: str) -> ForceFieldSpec:
    return _DEFAULT_REGISTRY.get(name)


def build_system(name: str, *, cavity: CavityParams, **kwargs: Any) -> BuiltCavitySystem:
    spec = get_spec(name)
    if spec.requires_topology and kwargs.get("topology") is None:
        raise ValueError(f"Force field '{name}' requires topology=...")
    return spec.builder(cavity=cavity, **kwargs)


def evaluate_dipole(
    built: BuiltCavitySystem,
    state: openmm.State,
):
    """Evaluate DipoleResponse from a BuiltCavitySystem."""
    return built.dipole_provider.evaluate_dipole_response(state)
