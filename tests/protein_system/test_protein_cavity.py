#!/usr/bin/env python3
"""
Unit tests for the protein cavity simulation scaffold.
"""

import importlib.util
from pathlib import Path

import numpy as np
import pytest

REPO_ROOT = Path(__file__).resolve().parents[2]
PROTEIN_EXAMPLE = REPO_ROOT / "examples" / "cavity" / "protein_system" / "run_simulation.py"

pytest.importorskip("pdbfixer")


def _load_protein_sim():
    spec = importlib.util.spec_from_file_location("protein_run_simulation", PROTEIN_EXAMPLE)
    if spec is None or spec.loader is None:
        raise ImportError(f"Could not load {PROTEIN_EXAMPLE}")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def _write_minimal_ala_pdb(tmp_path: Path) -> Path:
    pdb_text = "\n".join(
        [
            "ATOM      1  N   ALA A   1      -0.417   1.204   0.000  1.00  0.00           N",
            "ATOM      2  CA  ALA A   1       0.000   0.000   0.000  1.00  0.00           C",
            "ATOM      3  C   ALA A   1       1.524   0.000   0.000  1.00  0.00           C",
            "ATOM      4  O   ALA A   1       2.046  -1.143   0.000  1.00  0.00           O",
            "ATOM      5  CB  ALA A   1      -0.540  -0.781  -1.204  1.00  0.00           C",
            "TER",
            "END",
        ]
    )
    pdb_path = tmp_path / "minimal_ala.pdb"
    pdb_path.write_text(pdb_text)
    return pdb_path


def test_prepare_protein_system_minimal(tmp_path):
    protein_sim = _load_protein_sim()

    pdb_path = _write_minimal_ala_pdb(tmp_path)
    system, topology, positions, charges, real_indices = protein_sim.prepare_protein_system(
        pdb_path=pdb_path,
        pdb_id=None,
        add_solvent=False,
        fix_missing=False,
        add_hydrogens=True,
        ph=7.0,
    )

    assert system.getNumParticles() > 0
    assert len(positions) == system.getNumParticles()
    assert len(charges) == len(real_indices)
    assert isinstance(real_indices, np.ndarray)
