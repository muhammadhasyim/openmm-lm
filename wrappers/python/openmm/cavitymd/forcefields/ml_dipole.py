"""ML dipole force field builders delegating to openmmml."""

from __future__ import annotations

import os
from typing import Any, Optional

import numpy as np
import openmm
from openmm import app, unit

from .dipole import validate_dipole_response
from .types import BuiltCavitySystem, CavityBackend, CavityParams, DipoleResponse


def _require_openmmml():
    try:
        from openmmml.cavity_coupling import CavitySpec
        from openmmml.mlpotential import MLPotential
    except ImportError as exc:
        raise ImportError(
            "openmmml is required for ML dipole backends. "
            "Install with: pixi run -e ml install-ml"
        ) from exc
    return MLPotential, CavitySpec


class MLDipoleProvider:
    """Dipole provider that runs the ML model forward pass for mu and Z."""

    def __init__(
        self,
        potential,
        topology: app.Topology,
        real_atom_indices: np.ndarray,
        *,
        dipole_unit: str = "debye",
        model_path: Optional[str] = None,
        cuda_bridge_key: Optional[str] = None,
    ) -> None:
        self._potential = potential
        self._topology = topology
        self._real_atom_indices = np.asarray(real_atom_indices, dtype=int)
        self._dipole_unit = dipole_unit
        self._model_path = model_path
        self._impl = potential._impl
        self._cuda_bridge_key = cuda_bridge_key

    def evaluate_dipole_response(self, state: openmm.State) -> DipoleResponse:
        if self._cuda_bridge_key is not None:
            from openmmml.cuda_bridge import get_dipole_cache

            cached = get_dipole_cache(self._cuda_bridge_key)
            if cached is not None:
                dipole_enm, bec, atom_indices = cached
                response = DipoleResponse(
                    dipole_enm=np.asarray(dipole_enm, dtype=np.float64),
                    bec=np.asarray(bec, dtype=np.float64),
                    atom_indices=np.asarray(atom_indices, dtype=int),
                )
                validate_dipole_response(response)
                return response

        from openmmml.cavity_coupling import (
            box_vectors_nm_from_state,
            extract_dipole_and_bec,
        )

        positions = state.getPositions(asNumpy=True).value_in_unit(unit.nanometer)
        positions_nm = np.asarray(positions, dtype=np.float64)
        ml_positions = positions_nm[self._real_atom_indices]
        box_vectors_nm = box_vectors_nm_from_state(state)

        if hasattr(self._impl, "evaluate_model_output"):
            model_output = self._impl.evaluate_model_output(
                self._topology,
                ml_positions,
                box_vectors_nm=box_vectors_nm,
            )
        elif hasattr(self._impl, "predict"):
            model_output = self._impl.predict(
                self._topology,
                ml_positions,
                box_vectors_nm=box_vectors_nm,
            )
        else:
            raise NotImplementedError(
                f"{type(self._impl).__name__} has no evaluate_model_output/predict"
            )

        dipole_enm, bec = extract_dipole_and_bec(
            model_output,
            ml_positions,
            dipole_unit=self._dipole_unit,
            box_vectors_nm=box_vectors_nm,
        )
        response = DipoleResponse(
            dipole_enm=np.asarray(dipole_enm, dtype=np.float64),
            bec=np.asarray(bec, dtype=np.float64),
            atom_indices=self._real_atom_indices.copy(),
        )
        validate_dipole_response(response)
        return response


def _ml_real_atom_indices(system: openmm.System, cavity_index: int) -> np.ndarray:
    return np.array(
        [
            i
            for i in range(system.getNumParticles())
            if i != cavity_index and not system.isVirtualSite(i)
        ],
        dtype=int,
    )


def _to_cavity_spec(cavity: CavityParams, cavity_index: int) -> Any:
    _, CavitySpec = _require_openmmml()
    return CavitySpec(
        omegac=cavity.omegac,
        lambda_coupling=cavity.lambda_coupling,
        photon_mass=cavity.photon_mass,
        include_dse=cavity.include_dse,
        cavity_particle_index=cavity_index,
    )


def _build_ml_system(
    *,
    registry_name: str,
    ml_name: str,
    cavity: CavityParams,
    topology: app.Topology,
    dipole_unit: str = "debye",
    model_path: Optional[str] = None,
    ml_kwargs: Optional[dict] = None,
    cavity_backend: CavityBackend = CavityBackend.ML_PYTHONFORCE,
    cuda_bridge_key: Optional[str] = None,
) -> BuiltCavitySystem:
    MLPotential, CavitySpec = _require_openmmml()
    ml_kwargs = dict(ml_kwargs or {})
    if model_path is not None:
        ml_kwargs.setdefault("modelPath", model_path)

    potential = MLPotential(ml_name, **ml_kwargs)
    if not potential.supportsDipole():
        from openmmml.cavity_coupling import CAVITY_INCOMPATIBLE_MSG

        raise openmm.OpenMMException(CAVITY_INCOMPATIBLE_MSG)

    cavity_spec = CavitySpec(
        omegac=cavity.omegac,
        lambda_coupling=cavity.lambda_coupling,
        photon_mass=cavity.photon_mass,
        include_dse=cavity.include_dse,
    )
    system = potential.createCavitySystem(topology, cavity_spec, **ml_kwargs)
    cavity_index = system.getNumParticles() - 1
    positions = _default_ml_positions(topology, system)
    real_atom_indices = _ml_real_atom_indices(system, cavity_index)
    dipole_provider = MLDipoleProvider(
        potential,
        topology,
        real_atom_indices,
        dipole_unit=dipole_unit,
        model_path=model_path,
        cuda_bridge_key=cuda_bridge_key,
    )
    return BuiltCavitySystem(
        system=system,
        positions=positions,
        cavity_backend=cavity_backend,
        dipole_provider=dipole_provider,
        cavity_particle_index=cavity_index,
        real_atom_indices=real_atom_indices,
        topology=topology,
        metadata={
            "backend": registry_name,
            "ml_name": ml_name,
            "model_path": model_path,
            "use_cavitymd_simulation": False,
            "use_cuda_bridge": cuda_bridge_key is not None,
            "cuda_bridge_key": cuda_bridge_key,
        },
        cavity_force=None,
    )


def _default_ml_positions(topology: app.Topology, system: openmm.System) -> list:
    positions = []
    for atom in topology.atoms():
        positions.append(openmm.Vec3(0, 0, 0) * unit.nanometer)
    positions.append(openmm.Vec3(0, 0, 0) * unit.nanometer)
    if len(positions) != system.getNumParticles():
        positions = [openmm.Vec3(0, 0, 0) * unit.nanometer] * system.getNumParticles()
    return positions


def resolve_mace_polar_model_path(model_path: Optional[str] = None) -> Optional[str]:
    if model_path and os.path.isfile(model_path):
        return model_path
    for env_name in ("OPENMMML_MACE_POLAR_MODEL", "OPENMMML_MACE_MODEL"):
        candidate = os.environ.get(env_name, "").strip()
        if candidate and os.path.isfile(candidate):
            return candidate
    return model_path


def _build_cace_les_bec(
    *,
    cavity: CavityParams,
    topology: app.Topology,
    model_path: Optional[str] = None,
    use_cuda_bridge: bool = True,
    device: Optional[str] = None,
    **kwargs,
) -> BuiltCavitySystem:
    del kwargs
    from openmmml.cavity_coupling import resolve_cace_model_path

    resolved = resolve_cace_model_path(model_path or "")
    ml_kwargs: dict[str, Any] = {"use_cuda_bridge": use_cuda_bridge}
    if device is not None:
        ml_kwargs["device"] = device
    try:
        from openmmml.cuda_bridge import CACE_BRIDGE_KEY
    except ImportError:
        CACE_BRIDGE_KEY = None
    bridge_key = CACE_BRIDGE_KEY if use_cuda_bridge else None
    backend = (
        CavityBackend.ML_CUDA_BRIDGE if use_cuda_bridge else CavityBackend.ML_PYTHONFORCE
    )
    return _build_ml_system(
        registry_name="cace-les-bec",
        ml_name="cace-lr",
        cavity=cavity,
        topology=topology,
        dipole_unit="e_ang",
        model_path=resolved,
        ml_kwargs=ml_kwargs,
        cavity_backend=backend,
        cuda_bridge_key=bridge_key,
    )


def _build_cace_les_bec_batch(
    *,
    cavity: CavityParams,
    topology: app.Topology,
    model_path: Optional[str] = None,
    use_cuda_bridge: bool = True,
    device: Optional[str] = None,
    **kwargs,
) -> BuiltCavitySystem:
    del kwargs
    from openmmml.cavity_coupling import resolve_cace_model_path

    resolved = resolve_cace_model_path(model_path or "")
    ml_kwargs: dict[str, Any] = {"use_cuda_bridge": use_cuda_bridge}
    if device is not None:
        ml_kwargs["device"] = device
    try:
        from openmmml.cuda_bridge import CACE_BRIDGE_KEY
    except ImportError:
        CACE_BRIDGE_KEY = None
    bridge_key = CACE_BRIDGE_KEY if use_cuda_bridge else None
    backend = (
        CavityBackend.ML_CUDA_BRIDGE if use_cuda_bridge else CavityBackend.ML_PYTHONFORCE
    )
    return _build_ml_system(
        registry_name="cace-les-bec-batch",
        ml_name="cace-pythonforce-batch",
        cavity=cavity,
        topology=topology,
        dipole_unit="e_ang",
        model_path=resolved,
        ml_kwargs=ml_kwargs,
        cavity_backend=backend,
        cuda_bridge_key=bridge_key,
    )


def _build_aimnet2(
    *,
    cavity: CavityParams,
    topology: app.Topology,
    charge: int = 0,
    multiplicity: int = 1,
    use_cuda_bridge: bool = True,
    device: Optional[str] = None,
    **kwargs,
) -> BuiltCavitySystem:
    del kwargs
    ml_kwargs: dict[str, Any] = {
        "charge": charge,
        "multiplicity": multiplicity,
        "use_cuda_bridge": use_cuda_bridge,
    }
    if device is not None:
        ml_kwargs["device"] = device
    try:
        from openmmml.cuda_bridge import AIMNET2_BRIDGE_KEY
    except ImportError:
        AIMNET2_BRIDGE_KEY = None
    # PythonForce always writes dipole/BEC to this cache key after each evaluation.
    bridge_key = AIMNET2_BRIDGE_KEY
    backend = (
        CavityBackend.ML_CUDA_BRIDGE if use_cuda_bridge else CavityBackend.ML_PYTHONFORCE
    )
    return _build_ml_system(
        registry_name="aimnet2",
        ml_name="aimnet2",
        cavity=cavity,
        topology=topology,
        dipole_unit="e_ang",
        ml_kwargs=ml_kwargs,
        cavity_backend=backend,
        cuda_bridge_key=bridge_key,
    )


def _build_mace_polar_1(
    *,
    cavity: CavityParams,
    topology: app.Topology,
    model_path: Optional[str] = None,
    charge: int = 0,
    multiplicity: int = 1,
    precision: Optional[str] = "single",
    use_cuda_bridge: bool = True,
    **kwargs,
) -> BuiltCavitySystem:
    del kwargs
    resolved = resolve_mace_polar_model_path(model_path)
    if resolved is None or not os.path.isfile(resolved):
        raise FileNotFoundError(
            "MACE-POLAR checkpoint not found. Run scripts/fetch_mace_polar_model.sh "
            "and set OPENMMML_MACE_POLAR_MODEL."
        )
    ml_kwargs = {
        "polar": True,
        "charge": charge,
        "multiplicity": multiplicity,
        "use_cuda_bridge": use_cuda_bridge,
    }
    if precision is not None:
        ml_kwargs["precision"] = precision
    try:
        from openmmml.cuda_bridge import MACE_POLAR_BRIDGE_KEY
    except ImportError:
        MACE_POLAR_BRIDGE_KEY = None
    return _build_ml_system(
        registry_name="mace-polar-1",
        ml_name="mace-polar",
        cavity=cavity,
        topology=topology,
        dipole_unit="debye",
        model_path=resolved,
        ml_kwargs=ml_kwargs,
        cavity_backend=CavityBackend.ML_CUDA_BRIDGE,
        cuda_bridge_key=MACE_POLAR_BRIDGE_KEY,
    )


def _build_mbpol_2023(
    *,
    cavity: CavityParams,
    topology: app.Topology,
    json_path: Optional[str] = None,
    bec_method: str = "analytic",
    **kwargs,
) -> BuiltCavitySystem:
    del kwargs
    ml_kwargs: dict[str, Any] = {
        "use_cuda_bridge": False,
        "becMethod": bec_method,
    }
    if json_path is not None:
        ml_kwargs["jsonPath"] = json_path
    return _build_ml_system(
        registry_name="mbpol-2023",
        ml_name="mbpol-2023",
        cavity=cavity,
        topology=topology,
        dipole_unit="e_ang",
        ml_kwargs=ml_kwargs,
        cavity_backend=CavityBackend.ML_PYTHONFORCE,
        cuda_bridge_key=None,
    )


def validate_ml_registry_entry(name: str) -> None:
    """Smoke validation hook; full tests require checkpoints and are env-gated."""
    del name
