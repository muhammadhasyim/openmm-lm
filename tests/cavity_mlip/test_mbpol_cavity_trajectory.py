"""
tests/cavity_mlip/test_mbpol_cavity_trajectory.py
=================================================
End-to-end validation for MBPol(2023) / MBX + cavity MD.

Requires:
  - Built MBX (bash scripts/build_mbx.sh)
  - MBX_HOME set, libmbx.so on LD_LIBRARY_PATH
  - openmmml installed / on PYTHONPATH
"""

from __future__ import annotations

import os
import sys

import numpy as np
import pytest

ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
if ROOT not in sys.path:
    sys.path.insert(0, ROOT)
WRAPPERS = os.path.join(ROOT, "wrappers", "python")
if WRAPPERS not in sys.path:
    sys.path.insert(0, WRAPPERS)
OPENMMML = os.path.join(ROOT, "third_party", "openmm-ml")
if OPENMMML not in sys.path:
    sys.path.insert(0, OPENMMML)


def _mbx_ready() -> bool:
    mbx_home = os.environ.get("MBX_HOME", "").strip()
    if not mbx_home:
        mbx_home = os.path.join(ROOT, "third_party", "MBX")
    lib = os.path.join(mbx_home, "lib", "libmbx.so")
    return os.path.isfile(lib)


pytestmark = pytest.mark.skipif(
    not _mbx_ready(),
    reason="MBX not built. Run: bash scripts/build_mbx.sh && export MBX_HOME=third_party/MBX",
)


def _reference_platform():
    import openmm

    for name in ("Reference", "CPU"):
        try:
            return openmm.Platform.getPlatformByName(name)
        except openmm.OpenMMException:
            continue
    raise RuntimeError("No Reference/CPU OpenMM platform available")


def _three_water_topology():
    from openmm import app, unit

    sys.path.insert(
        0,
        os.path.join(ROOT, "examples", "cavity", "common"),
    )
    from water_box import make_water_topology

    return make_water_topology(3, box_size_nm=1.2)


@pytest.fixture(scope="module")
def mbx_env():
    mbx_home = os.environ.get("MBX_HOME", os.path.join(ROOT, "third_party", "MBX"))
    os.environ["MBX_HOME"] = mbx_home
    fftw_lib = os.path.join(ROOT, "third_party", "fftw", "install", "lib")
    ld = os.environ.get("LD_LIBRARY_PATH", "")
    paths = [os.path.join(mbx_home, "lib")]
    if os.path.isdir(fftw_lib):
        paths.append(fftw_lib)
    os.environ["LD_LIBRARY_PATH"] = ":".join(paths + ([ld] if ld else []))
    return mbx_home


def test_mbx_library_energy_dipole(mbx_env, tmp_path):
    from openmmml.mbx_binding import MBXLibrary, write_mbx_json
    from openmmml.mbx_monomers import infer_mbx_monomers

    topology, pos_nm = _three_water_topology()
    symbols = [a.element.symbol for a in topology.atoms()]
    monomer_names, nat_monomers = infer_mbx_monomers(symbols)
    coords_ang = MBXLibrary.positions_nm_to_ang(pos_nm)

    json_path = str(tmp_path / "mbx.json")
    write_mbx_json(json_path, periodic=True)

    mbx = MBXLibrary(mbx_home=mbx_env)
    mbx.initialize_system(coords_ang, nat_monomers, symbols, monomer_names, json_path)
    box_ang = np.diag([12.0, 12.0, 12.0]).ravel()
    energy, grads = mbx.get_energy_forces_pbc(coords_ang, box_ang)
    dipole = mbx.get_total_dipole_ang()

    assert np.isfinite(energy)
    assert grads.shape == coords_ang.shape
    assert dipole.shape == (3,)
    assert np.all(np.isfinite(dipole))
    mbx.finalize()


def test_bec_tier_a_vs_analytic(mbx_env, tmp_path):
    from openmmml.mbx_bec import compute_bec
    from openmmml.mbx_binding import MBXLibrary, write_mbx_json
    from openmmml.mbx_monomers import infer_mbx_monomers

    topology, pos_nm = _three_water_topology()
    symbols = [a.element.symbol for a in topology.atoms()]
    monomer_names, nat_monomers = infer_mbx_monomers(symbols)
    coords_ang = MBXLibrary.positions_nm_to_ang(pos_nm)
    n_sites = 4 * len(monomer_names)

    json_path = str(tmp_path / "mbx.json")
    write_mbx_json(json_path, periodic=False)

    mbx = MBXLibrary(mbx_home=mbx_env)
    mbx.initialize_system(coords_ang, nat_monomers, symbols, monomer_names, json_path)
    mbx.get_energy_forces(coords_ang)

    bec_fd = compute_bec(
        mbx, coords_ang, monomer_names, n_sites=n_sites, method="fd", box_ang=None
    )
    bec_analytic = compute_bec(
        mbx, coords_ang, monomer_names, n_sites=n_sites, method="analytic", box_ang=None
    )

    assert bec_fd.shape == bec_analytic.shape == (9, 3, 3)
    assert np.all(np.isfinite(bec_fd))
    assert np.all(np.isfinite(bec_analytic))
    max_diff = np.max(np.abs(bec_fd - bec_analytic))
    assert max_diff < 0.5, f"BEC tier mismatch max diff {max_diff}"
    mbx.finalize()


def test_registry_build_and_trajectory(mbx_env):
    import openmm
    from openmm import app, unit

    from openmm.cavitymd.forcefields import CavityParams, build_system, evaluate_dipole

    topology, pos_nm = _three_water_topology()
    cavity = CavityParams(
        omegac=0.01,
        lambda_coupling=0.001,
        photon_mass=1.0 / 1822.888,
        include_dse=True,
    )

    built = build_system(
        "mbpol-2023",
        cavity=cavity,
        topology=topology,
        bec_method="analytic",
    )
    system = built.system
    assert system.getNumParticles() == topology.getNumAtoms() + 1

    positions = list(pos_nm) + [openmm.Vec3(0, 0, 0)]
    positions = [p * unit.nanometer for p in positions]

    integrator = openmm.VerletIntegrator(1.0 * unit.femtoseconds)
    platform = _reference_platform()
    context = openmm.Context(system, integrator, platform)
    context.setPositions(positions)
    if topology.getPeriodicBoxVectors() is not None:
        context.setPeriodicBoxVectors(*topology.getPeriodicBoxVectors())

    state0 = context.getState(getEnergy=True, getPositions=True)
    e0 = state0.getPotentialEnergy().value_in_unit(unit.kilojoules_per_mole)
    assert np.isfinite(e0)

    response0 = evaluate_dipole(built, state0)
    assert response0.dipole_enm.shape == (3,)
    assert response0.bec.shape == (9, 3, 3)

    energies = [e0]
    for _ in range(50):
        integrator.step(1)
        st = context.getState(getEnergy=True, getPositions=True)
        energies.append(st.getPotentialEnergy().value_in_unit(unit.kilojoules_per_mole))

    spread = max(energies) - min(energies)
    assert spread < 5000.0, f"Energy drift too large over 50 steps: {spread} kJ/mol"

    del context
