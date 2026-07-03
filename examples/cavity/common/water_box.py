"""Shared water box topology helpers for ML cavity MD examples."""

from __future__ import annotations

import math

import numpy as np
import openmm
from openmm import app, unit


OH_BOND_NM = 0.09572
HOH_ANGLE_RAD = math.radians(104.52)


def box_size_nm_for_molecules(
    num_molecules: int,
    ref_molecules: int = 64,
    ref_box_nm: float = 2.0,
) -> float:
    """Scale cubic box edge at fixed number density."""
    return ref_box_nm * (num_molecules / ref_molecules) ** (1.0 / 3.0)


def make_water_topology(
    num_molecules: int,
    box_size_nm: float,
    seed: int = 42,
) -> tuple[app.Topology, np.ndarray]:
    """Build periodic HOH topology (no virtual sites) and positions in nm."""
    rng = np.random.RandomState(seed)
    topology = app.Topology()
    chain = topology.addChain()
    positions = []

    side = int(math.ceil(num_molecules ** (1.0 / 3.0)))
    spacing = box_size_nm / side
    half_angle = HOH_ANGLE_RAD / 2.0

    mol_count = 0
    for i in range(side):
        for j in range(side):
            for k in range(side):
                if mol_count >= num_molecules:
                    break
                cx = (i + 0.5) * spacing
                cy = (j + 0.5) * spacing
                cz = (k + 0.5) * spacing

                center = np.array([cx, cy, cz], dtype=np.float64)

                z_axis = rng.normal(size=3)
                z_axis /= np.linalg.norm(z_axis)
                x_axis = rng.normal(size=3)
                x_axis -= np.dot(x_axis, z_axis) * z_axis
                x_axis /= np.linalg.norm(x_axis)

                h1_dir = math.sin(half_angle) * x_axis + math.cos(half_angle) * z_axis
                h2_dir = math.sin(half_angle) * x_axis - math.cos(half_angle) * z_axis
                h1 = center + OH_BOND_NM * h1_dir
                h2 = center + OH_BOND_NM * h2_dir

                residue = topology.addResidue("HOH", chain)
                oxygen = topology.addAtom("O", app.Element.getBySymbol("O"), residue)
                hydrogen1 = topology.addAtom("H1", app.Element.getBySymbol("H"), residue)
                hydrogen2 = topology.addAtom("H2", app.Element.getBySymbol("H"), residue)
                topology.addBond(oxygen, hydrogen1)
                topology.addBond(oxygen, hydrogen2)

                positions.extend([center, h1, h2])
                mol_count += 1
            if mol_count >= num_molecules:
                break
        if mol_count >= num_molecules:
            break

    pos_nm = np.asarray(positions, dtype=np.float64)
    vectors = (
        openmm.Vec3(box_size_nm, 0, 0) * unit.nanometer,
        openmm.Vec3(0, box_size_nm, 0) * unit.nanometer,
        openmm.Vec3(0, 0, box_size_nm) * unit.nanometer,
    )
    topology.setPeriodicBoxVectors(vectors)
    return topology, pos_nm
