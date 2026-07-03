"""Classical force field builders with charge-based dipole + BEC."""

from __future__ import annotations

import importlib.util
from pathlib import Path
from typing import Optional

import numpy as np
import openmm
from openmm import app, unit

from .dipole import SystemChargeDipoleProvider
from .mka import (
    add_cavity_particle as add_mka_cavity_particle,
    build_mka_system,
)
from .types import BuiltCavitySystem, CavityBackend, CavityParams


def repo_root() -> Path:
    return Path(__file__).resolve().parents[5]


def _attach_native_cavity_force(
    system: openmm.System,
    positions: list,
    cavity: CavityParams,
) -> tuple[int, openmm.CavityForce]:
    """Add photon particle and native CavityForce to a classical system."""
    cavity_index = add_mka_cavity_particle(system, positions)
    cavity_force = openmm.CavityForce(
        cavity_index,
        cavity.omegac,
        cavity.lambda_coupling,
        cavity.photon_mass,
    )
    cavity_force.setIncludeDipoleSelfEnergy(cavity.include_dse)
    system.addForce(cavity_force)
    return cavity_index, cavity_force


def _build_mka(
    *,
    cavity: CavityParams,
    num_molecules: int = 250,
    frac_aa: float = 0.8,
    box_au: Optional[float] = None,
    seed: int = 42,
    sample_bonds_at_T: Optional[float] = None,
    **kwargs,
) -> BuiltCavitySystem:
    del kwargs
    system, positions, num_mol_particles = build_mka_system(
        num_molecules=num_molecules,
        frac_aa=frac_aa,
        box_au=box_au,
        seed=seed,
        sample_bonds_at_T=sample_bonds_at_T,
    )
    real_atom_indices = np.arange(num_mol_particles, dtype=int)
    cavity_index, cavity_force = _attach_native_cavity_force(system, positions, cavity)
    dipole_provider = SystemChargeDipoleProvider(system, real_atom_indices)
    return BuiltCavitySystem(
        system=system,
        positions=positions,
        cavity_backend=CavityBackend.NATIVE_CAVITY_FORCE,
        dipole_provider=dipole_provider,
        cavity_particle_index=cavity_index,
        real_atom_indices=real_atom_indices,
        topology=None,
        metadata={
            "backend": "mka",
            "num_molecules": num_molecules,
            "num_mol_particles": num_mol_particles,
            "use_cavitymd_simulation": True,
        },
        cavity_force=cavity_force,
    )


def _load_example_module(relative_path: str, module_name: str):
    path = repo_root() / relative_path
    spec = importlib.util.spec_from_file_location(module_name, path)
    if spec is None or spec.loader is None:
        raise ImportError(f"Cannot load module from {path}")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def _build_dimer_xml(
    *,
    cavity: CavityParams,
    num_molecules: int = 50,
    fraction_OO: float = 0.8,
    box_size_nm: float = 2.0,
    seed: int = 42,
    **kwargs,
) -> BuiltCavitySystem:
    del kwargs
    dimer_mod = _load_example_module(
        "examples/cavity/dimer_system/run_simulation.py",
        "cavity_dimer_run",
    )
    ff_dir = repo_root() / "examples/cavity/dimer_system"
    system, positions, topology, cavity_index = (
        dimer_mod.create_diamer_system_from_forcefield(
            num_molecules=num_molecules,
            fraction_OO=fraction_OO,
            box_size_nm=box_size_nm,
            seed=seed,
            ff_dir=ff_dir,
            include_cavity=True,
        )
    )
    real_atom_indices = np.arange(2 * num_molecules, dtype=int)
    cavity_force = openmm.CavityForce(
        cavity_index,
        cavity.omegac,
        cavity.lambda_coupling,
        cavity.photon_mass,
    )
    cavity_force.setIncludeDipoleSelfEnergy(cavity.include_dse)
    system.addForce(cavity_force)
    dipole_provider = SystemChargeDipoleProvider(system, real_atom_indices)
    return BuiltCavitySystem(
        system=system,
        positions=positions,
        cavity_backend=CavityBackend.NATIVE_CAVITY_FORCE,
        dipole_provider=dipole_provider,
        cavity_particle_index=cavity_index,
        real_atom_indices=real_atom_indices,
        topology=topology,
        metadata={
            "backend": "dimer-xml",
            "num_molecules": num_molecules,
            "use_cavitymd_simulation": True,
        },
        cavity_force=cavity_force,
    )


def _create_tip4pew_flex_water_box(
    num_molecules: int = 64,
    box_size_nm: float = 2.0,
    seed: int = 42,
):
    """Build flexible TIP4P-Ew water box without importing the example script."""
    import tempfile

    pdb_lines = [
        "CRYST1{:9.3f}{:9.3f}{:9.3f}  90.00  90.00  90.00 P 1           1".format(
            box_size_nm * 10, box_size_nm * 10, box_size_nm * 10
        )
    ]
    oh_bond = 0.09572
    hoh_angle = 104.52 * np.pi / 180.0
    side = int(np.ceil(num_molecules ** (1 / 3)))
    spacing = box_size_nm / side
    atom_idx = 1
    mol_count = 0
    rng = np.random.RandomState(seed)

    for i in range(side):
        for j in range(side):
            for k in range(side):
                if mol_count >= num_molecules:
                    break
                cx = (i + 0.5) * spacing
                cy = (j + 0.5) * spacing
                cz = (k + 0.5) * spacing
                theta = rng.rand() * 2 * np.pi
                phi = rng.rand() * np.pi
                psi = rng.rand() * 2 * np.pi
                h1_local = np.array(
                    [oh_bond * np.sin(hoh_angle / 2), 0, oh_bond * np.cos(hoh_angle / 2)]
                )
                h2_local = np.array(
                    [-oh_bond * np.sin(hoh_angle / 2), 0, oh_bond * np.cos(hoh_angle / 2)]
                )
                o_local = np.array([0.0, 0.0, 0.0])
                rz = np.array(
                    [
                        [np.cos(theta), -np.sin(theta), 0.0],
                        [np.sin(theta), np.cos(theta), 0.0],
                        [0.0, 0.0, 1.0],
                    ]
                )
                ry = np.array(
                    [
                        [np.cos(phi), 0.0, np.sin(phi)],
                        [0.0, 1.0, 0.0],
                        [-np.sin(phi), 0.0, np.cos(phi)],
                    ]
                )
                rx = np.array(
                    [
                        [1.0, 0.0, 0.0],
                        [0.0, np.cos(psi), -np.sin(psi)],
                        [0.0, np.sin(psi), np.cos(psi)],
                    ]
                )
                rotation = rz @ ry @ rx
                center = np.array([cx, cy, cz])
                o_pos = (rotation @ o_local + center) * 10
                h1_pos = (rotation @ h1_local + center) * 10
                h2_pos = (rotation @ h2_local + center) * 10
                resnum = mol_count + 1
                pdb_lines.append(
                    f"HETATM{atom_idx:5d}  O   HOH  {resnum:4d}    "
                    f"{o_pos[0]:8.3f}{o_pos[1]:8.3f}{o_pos[2]:8.3f}  1.00  0.00           O"
                )
                atom_idx += 1
                pdb_lines.append(
                    f"HETATM{atom_idx:5d}  H1  HOH  {resnum:4d}    "
                    f"{h1_pos[0]:8.3f}{h1_pos[1]:8.3f}{h1_pos[2]:8.3f}  1.00  0.00           H"
                )
                atom_idx += 1
                pdb_lines.append(
                    f"HETATM{atom_idx:5d}  H2  HOH  {resnum:4d}    "
                    f"{h2_pos[0]:8.3f}{h2_pos[1]:8.3f}{h2_pos[2]:8.3f}  1.00  0.00           H"
                )
                atom_idx += 1
                mol_count += 1
            if mol_count >= num_molecules:
                break
        if mol_count >= num_molecules:
            break
    pdb_lines.append("END")

    with tempfile.NamedTemporaryFile(mode="w", suffix=".pdb", delete=False) as handle:
        handle.write("\n".join(pdb_lines))
        pdb_file = handle.name
    pdb = app.PDBFile(pdb_file)
    Path(pdb_file).unlink()

    flexible_xml = repo_root() / "examples/cavity/water_system/tip4pew_flexible.xml"
    if not flexible_xml.exists():
        raise FileNotFoundError(f"Flexible TIP4P-Ew XML not found: {flexible_xml}")
    forcefield = app.ForceField(str(flexible_xml))
    modeller = app.Modeller(pdb.topology, pdb.positions)
    modeller.addExtraParticles(forcefield)
    system = forcefield.createSystem(
        modeller.topology,
        nonbondedMethod=app.PME,
        nonbondedCutoff=0.9 * unit.nanometer,
        constraints=None,
        rigidWater=False,
        ewaldErrorTolerance=0.0005,
    )
    num_molecular = len(pdb.positions)
    return system, modeller.topology, modeller.getPositions(), num_molecular


def _build_tip4pew_flex(
    *,
    cavity: CavityParams,
    num_molecules: int = 64,
    box_size_nm: float = 2.0,
    temperature_K: float = 300.0,
    **kwargs,
) -> BuiltCavitySystem:
    del kwargs, temperature_K
    system, topology, positions, num_molecular = _create_tip4pew_flex_water_box(
        num_molecules=num_molecules,
        box_size_nm=box_size_nm,
    )
    positions = list(positions)
    real_atom_indices = np.arange(num_molecular, dtype=int)
    cavity_index, cavity_force = _attach_native_cavity_force(
        system, positions, cavity
    )
    dipole_provider = SystemChargeDipoleProvider(system, real_atom_indices)
    return BuiltCavitySystem(
        system=system,
        positions=positions,
        cavity_backend=CavityBackend.NATIVE_CAVITY_FORCE,
        dipole_provider=dipole_provider,
        cavity_particle_index=cavity_index,
        real_atom_indices=real_atom_indices,
        topology=topology,
        metadata={
            "backend": "tip4pew-flex",
            "num_molecules": num_molecules,
            "num_molecular_atoms": num_molecular,
            "use_cavitymd_simulation": True,
        },
        cavity_force=cavity_force,
    )


def _build_amber_tip4pew_protein(
    *,
    cavity: CavityParams,
    pdb_id: str = "3utl",
    padding_nm: float = 1.0,
    **kwargs,
) -> BuiltCavitySystem:
    del kwargs
    protein_mod = _load_example_module(
        "examples/cavity/protein_system/run_simulation.py",
        "cavity_protein_run",
    )
    try:
        import openmmforcefields  # noqa: F401
    except ImportError as exc:
        raise ImportError(
            "openmmforcefields is required for amber-tip4pew-protein. "
            "Install with: pixi run -e ff-classical install-ff-classical"
        ) from exc
    system, topology, positions, charges, real_indices = (
        protein_mod.prepare_protein_system(
            pdb_id=pdb_id,
            padding_nm=padding_nm,
            add_solvent=True,
            verbose=False,
        )
    )
    positions = list(positions)
    cavity_index, cavity_force = _attach_native_cavity_force(
        system, positions, cavity
    )
    dipole_provider = SystemChargeDipoleProvider(system, real_indices)
    return BuiltCavitySystem(
        system=system,
        positions=positions,
        cavity_backend=CavityBackend.NATIVE_CAVITY_FORCE,
        dipole_provider=dipole_provider,
        cavity_particle_index=cavity_index,
        real_atom_indices=real_indices,
        topology=topology,
        metadata={
            "backend": "amber-tip4pew-protein",
            "pdb_id": pdb_id,
            "use_cavitymd_simulation": True,
        },
        cavity_force=cavity_force,
    )


def validate_mka_dipole() -> None:
    built = _build_mka(
        cavity=CavityParams(omegac=0.01, lambda_coupling=0.001),
        num_molecules=1,
        box_au=40.0,
        seed=0,
    )
    platform = openmm.Platform.getPlatformByName("Reference")
    integrator = openmm.VerletIntegrator(0.001)
    simulation = app.Simulation(
        app.Topology(),
        built.system,
        integrator,
        platform,
    )
    simulation.context.setPositions(built.positions)
    state = simulation.context.getState(getPositions=True)
    built.dipole_provider.evaluate_dipole_response(state)
