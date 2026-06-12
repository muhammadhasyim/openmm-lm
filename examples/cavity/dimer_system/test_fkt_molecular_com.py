#!/usr/bin/env python3
"""Tests for molecular COM vs atomic F(k,t) site selection."""

from pathlib import Path
import sys
import tempfile

import numpy as np

_here = Path(__file__).resolve().parent
if str(_here) not in sys.path:
    sys.path.insert(0, str(_here))

from fkt_tracker import (  # noqa: E402
    FKTTracker,
    compute_fkt,
    compute_rhok,
    fibonacci_sphere,
    fkt_positions_nm,
    molecular_com_positions_nm,
)


def test_molecular_com_shape():
    atoms = np.arange(36, dtype=float).reshape(6, 2, 3)
    flat = atoms.reshape(12, 3)
    com = molecular_com_positions_nm(flat, num_molecules=6)
    np.testing.assert_allclose(com, atoms.mean(axis=1))


def test_fkt_frozen_com_positions():
    """COM frozen: F(k,1 ps) equals F(k,0) when using molecular COM sites."""
    np.random.seed(42)
    n_mol = 10
    box_nm = 2.0
    com = np.random.uniform(0, box_nm, size=(n_mol, 3))
    atoms = np.stack([com - 0.01, com + 0.01], axis=1).reshape(2 * n_mol, 3)

    with tempfile.TemporaryDirectory() as tmpdir:
        prefix = str(Path(tmpdir) / "fkt_com_frozen")
        tracker = FKTTracker(
            kmag_nm_inv=113.4,
            num_wavevectors=50,
            reference_interval_ps=100.0,
            max_references=1,
            output_period_ps=1.0,
            output_prefix=prefix,
        )
        tracker.update(1.0, com)
        tracker.update(2.0, com)
        tracker.finalize()

        out_file = Path(prefix + "_fkt_ref_000.txt")
        data_lines = [
            line
            for line in out_file.read_text().splitlines()
            if not line.startswith("#") and line.strip()
        ]
        f0 = float(data_lines[0].split()[1])
        f1 = float(data_lines[1].split()[1])
        np.testing.assert_allclose(f1, f0, rtol=1e-10, atol=1e-10)


def test_atomic_dephases_faster_than_com_on_bond_vibration():
    """Bond vibration at fixed COM: atomic rho_k decorrelates faster than COM rho_k."""
    kmag = 113.4
    wavevectors = fibonacci_sphere(50) * kmag
    com = np.array([[0.5, 0.5, 0.5]])
    bond = 0.04
    r_ref_a = com - np.array([[0.5 * bond, 0.0, 0.0]])
    r_ref_b = com + np.array([[0.5 * bond, 0.0, 0.0]])
    atoms_ref = np.vstack([r_ref_a, r_ref_b])

    r_vib_a = com - np.array([[0.5 * (bond + 0.01), 0.0, 0.0]])
    r_vib_b = com + np.array([[0.5 * (bond + 0.01), 0.0, 0.0]])
    atoms_vib = np.vstack([r_vib_a, r_vib_b])

    rhok0_r, rhok0_i = compute_rhok(fkt_positions_nm(atoms_ref, 1, "atomic"), wavevectors)
    rhok1_r, rhok1_i = compute_rhok(fkt_positions_nm(atoms_vib, 1, "atomic"), wavevectors)
    f_atomic = compute_fkt(rhok0_r, rhok0_i, rhok1_r, rhok1_i)
    f0_atomic = compute_fkt(rhok0_r, rhok0_i, rhok0_r, rhok0_i)

    rhok0_r, rhok0_i = compute_rhok(com, wavevectors)
    f_com = compute_fkt(rhok0_r, rhok0_i, rhok0_r, rhok0_i)
    f0_com = f_com

    phi_atomic = abs(f_atomic / f0_atomic)
    phi_com = abs(f_com / f0_com)
    assert phi_com > phi_atomic
    assert phi_com > 0.99


if __name__ == "__main__":
    test_molecular_com_shape()
    test_fkt_frozen_com_positions()
    test_atomic_dephases_faster_than_com_on_bond_vibration()
    print("PASS: test_fkt_molecular_com")
