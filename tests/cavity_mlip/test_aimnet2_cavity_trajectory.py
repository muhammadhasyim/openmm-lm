"""End-to-end AIMNet2 + cavity MD tests (small cluster; bulk water optional)."""

from __future__ import annotations

import os

import numpy as np
import pytest

pytestmark = pytest.mark.skipif(
    os.environ.get("OPENMMML_AIMNET2_TESTS", "").strip() != "1",
    reason="Set OPENMMML_AIMNET2_TESTS=1 to run AIMNet2 cavity trajectory tests.",
)


def _require_aimnet2():
    try:
        import torch  # noqa: F401
    except ImportError:
        pytest.skip("PyTorch not installed")
    try:
        from aimnet2calc.models import get_model_path  # noqa: F401
    except ImportError:
        pytest.skip("aimnet2calc not installed (pip install -e third_party/aimnet2)")


def _reference_platform():
    import openmm

    for name in ("Reference", "CPU"):
        try:
            return openmm.Platform.getPlatformByName(name)
        except openmm.OpenMMException:
            continue
    raise RuntimeError("No Reference/CPU OpenMM platform available")


def _make_3water_topology():
    from openmm import app

    topology = app.Topology()
    chain = topology.addChain()
    positions = []
    centers = [(0.0, 0.0, 0.0), (0.25, 0.0, 0.0), (0.0, 0.25, 0.0)]
    oh = 0.09572
    for cx, cy, cz in centers:
        center = np.array([cx, cy, cz])
        h1 = center + np.array([oh, 0.0, 0.0])
        h2 = center + np.array([0.0, oh, 0.0])
        residue = topology.addResidue("HOH", chain)
        o = topology.addAtom("O", app.Element.getBySymbol("O"), residue)
        h_a = topology.addAtom("H1", app.Element.getBySymbol("H"), residue)
        h_b = topology.addAtom("H2", app.Element.getBySymbol("H"), residue)
        topology.addBond(o, h_a)
        topology.addBond(o, h_b)
        positions.extend([center, h1, h2])
    return topology, np.asarray(positions, dtype=np.float64)


class TestAIMNet2CavityTrajectory:
    def test_three_water_cpu_smoke(self):
        _require_aimnet2()
        import openmm
        from openmm.cavitymd.forcefields import CavityParams, build_system, evaluate_dipole

        topology, pos_nm = _make_3water_topology()
        n_atoms = pos_nm.shape[0]
        cavity = CavityParams(
            omegac=0.005,
            lambda_coupling=0.0,
            photon_mass=1.0,
            include_dse=True,
        )
        built = build_system(
            "aimnet2",
            topology=topology,
            cavity=cavity,
            use_cuda_bridge=False,
        )
        integrator = openmm.LangevinMiddleIntegrator(
            300 * openmm.unit.kelvin,
            1.0 / openmm.unit.picosecond,
            1.0 * openmm.unit.femtoseconds,
        )
        platform = _reference_platform()
        simulation = openmm.app.Simulation(
            topology, built.system, integrator, platform
        )
        photon = np.zeros((1, 3))
        all_pos = np.vstack([pos_nm, photon])
        simulation.context.setPositions(all_pos * openmm.unit.nanometer)
        simulation.context.setVelocitiesToTemperature(300 * openmm.unit.kelvin)
        simulation.step(10)
        state = simulation.context.getState(getEnergy=True, getPositions=True)
        resp = evaluate_dipole(built, state)
        assert resp.bec.shape == (n_atoms, 3, 3)
        assert np.all(np.isfinite(resp.dipole_enm))
        assert np.all(np.isfinite(resp.bec))
        assert np.isfinite(state.getPotentialEnergy().value_in_unit(openmm.unit.kilojoules_per_mole))
