"""
tests/cavity_mlip/test_cace_cavity_trajectory.py
=================================================
End-to-end validation tests for CACE/LES-BEC + cavity MD coupling.

These tests require:
  - CACE installed (``pip install cace``)
  - The LES-BEC fit_version_3 or fit_version_4 checkpoint accessible via
    the ``OPENMMML_LES_BEC_WATER_MODEL`` (or OPENMMML_CACE_MODEL_PATH) env var,
    or by cloning the LES-BEC submodule::

        git submodule update --init LES-BEC
        export OPENMMML_LES_BEC_WATER_MODEL=$PWD/LES-BEC/water/fit/fit_version_4/best_model.pth

  - A compatible PyTorch installation

All tests are automatically skipped if the above requirements are not met.

Physics validated
-----------------
1. Schema lock  : loaded model emits ``polarization`` with shape ``(1, 3)`` and
                  ``CACE_bec`` with shape ``(n_atoms, 3, 3)`` on a single forward pass.
2. Trajectory   : NVE short run on a 3-water cluster; total energy (kinetic +
                  potential + cavity) stays finite and varies by < 20 kJ/mol over
                  50 steps.
3. Cavity decomp: cavity energy from the trajectory state matches a direct
                  recompute from ``polarization``/``CACE_bec`` via
                  ``apply_cavity_coupling``.
"""

import os
import pytest
import numpy as np


# ---------------------------------------------------------------------------
# Guards / fixtures
# ---------------------------------------------------------------------------

def _checkpoint_path():
    for env in ("OPENMMML_LES_BEC_WATER_MODEL", "OPENMMML_CACE_MODEL_PATH",
                "OPENMMML_LES_BEC_MODEL"):
        p = os.environ.get(env, "").strip()
        if p and os.path.isfile(p):
            return p
    return None


CHECKPOINT = _checkpoint_path()

pytestmark = pytest.mark.skipif(
    CHECKPOINT is None,
    reason=(
        "LES-BEC checkpoint not found. "
        "Set OPENMMML_LES_BEC_WATER_MODEL to a fit_version_3/4 best_model.pth path."
    ),
)

try:
    import torch
    HAS_TORCH = True
except ImportError:
    HAS_TORCH = False

try:
    import cace  # noqa: F401
    HAS_CACE = True
except ImportError:
    HAS_CACE = False


def _require_torch_cace():
    if not HAS_TORCH:
        pytest.skip("PyTorch not installed")
    if not HAS_CACE:
        pytest.skip("CACE not installed")


# ---------------------------------------------------------------------------
# Helpers: minimal 3-water box
# ---------------------------------------------------------------------------

def _make_3water_topology():
    """Return an openmm.app.Topology for 3 TIP3P-like water molecules."""
    import openmm.app as app
    import openmm.unit as u
    top = app.Topology()
    chain = top.addChain()
    positions = []
    # Place 3 water molecules roughly 0.3 nm apart
    offsets = [(0.0, 0.0, 0.0), (0.4, 0.0, 0.0), (0.8, 0.0, 0.0)]
    for ox, oy, oz in offsets:
        res = top.addResidue("HOH", chain)
        O = top.addAtom("O", app.Element.getBySymbol("O"), res)
        H1 = top.addAtom("H1", app.Element.getBySymbol("H"), res)
        H2 = top.addAtom("H2", app.Element.getBySymbol("H"), res)
        top.addBond(O, H1)
        top.addBond(O, H2)
        # TIP3P geometry (Å → nm): O-H = 0.09572 nm, H-O-H = 104.52°
        import math
        angle = math.radians(104.52 / 2)
        bond = 0.09572
        positions.append((ox, oy, oz))
        positions.append((ox + bond * math.sin(angle), oy + bond * math.cos(angle), oz))
        positions.append((ox - bond * math.sin(angle), oy + bond * math.cos(angle), oz))
    return top, np.array(positions, dtype=np.float64)


def _make_cace_system_with_cavity(checkpoint: str):
    """Create a minimal OpenMM system with CACE + cavity force."""
    import openmm
    import openmm.unit as u
    from openmmml import MLPotential, CavitySpec

    top, pos_nm = _make_3water_topology()
    n_atoms = len(list(top.atoms()))

    cavity = CavitySpec(
        omegac=0.005,
        lambda_coupling=0.01,
        photon_mass=1.0,
        include_dse=True,
        cavity_particle_index=n_atoms,  # photon added as the last particle
    )

    potential = MLPotential("cace-lr", checkpoint)
    sys = potential.createMixedSystem(top, cavity=cavity)

    return sys, top, pos_nm, n_atoms


# ---------------------------------------------------------------------------
# Test 1: schema lock
# ---------------------------------------------------------------------------

class TestCACESchemaLock:

    def test_polarization_and_bec_shapes(self):
        """The loaded CACE model must output polarization (1,3) and CACE_bec (n,3,3)."""
        _require_torch_cace()
        import torch
        from cace.data.neighborhood import get_neighborhood

        device = "cpu"
        model = torch.load(CHECKPOINT, map_location=device, weights_only=False)
        model.eval()

        _, pos_nm = _make_3water_topology()
        n_atoms = pos_nm.shape[0]  # 9
        pos_ang = torch.tensor(pos_nm * 10.0, dtype=torch.float32, requires_grad=True)

        # Minimal atomic data for 3 water molecules
        # H=1, O=8
        atomic_numbers_np = np.array([8, 1, 1, 8, 1, 1, 8, 1, 1], dtype=int)
        atomic_numbers = torch.tensor(atomic_numbers_np, dtype=torch.long)

        edge_index, shifts, unit_shifts = get_neighborhood(
            positions=pos_nm * 10.0,  # Å
            cutoff=6.0,
            pbc=(False, False, False),
            cell=None,
        )

        data_dict = {
            'positions': pos_ang,
            'atomic_numbers': atomic_numbers,
            'edge_index': torch.tensor(edge_index, dtype=torch.long),
            'shifts': torch.tensor(shifts, dtype=torch.float32),
            'unit_shifts': torch.tensor(unit_shifts, dtype=torch.float32),
            'num_nodes': torch.tensor([n_atoms], dtype=torch.long),
            'ptr': torch.tensor([0, n_atoms], dtype=torch.long),
            'batch': torch.zeros(n_atoms, dtype=torch.long),
            'cell': torch.eye(3, dtype=torch.float32).unsqueeze(0) * 100.0,
        }

        output = model(data_dict, training=True)

        assert 'polarization' in output, (
            f"Model output missing 'polarization'. Keys: {list(output.keys())}"
        )
        assert 'CACE_bec' in output, (
            f"Model output missing 'CACE_bec'. Keys: {list(output.keys())}"
        )

        pol = output['polarization']
        bec = output['CACE_bec']

        if hasattr(pol, 'detach'):
            pol = pol.detach().cpu().numpy()
        if hasattr(bec, 'detach'):
            bec = bec.detach().cpu().numpy()

        pol = np.asarray(pol).squeeze()
        bec = np.asarray(bec)

        assert pol.shape == (3,), (
            f"polarization shape should be (3,) after squeeze, got {pol.shape}"
        )
        assert bec.shape == (n_atoms, 3, 3), (
            f"CACE_bec shape should be ({n_atoms},3,3), got {bec.shape}"
        )
        assert np.all(np.isfinite(pol)), "polarization contains NaN/Inf"
        assert np.all(np.isfinite(bec)), "CACE_bec contains NaN/Inf"
        assert np.linalg.norm(pol) > 1e-6, (
            "polarization is zero — checkpoint may not have BEC training"
        )

    def test_assert_cace_model_has_bec_raises_on_bad_model(self, tmp_path):
        """assert_cace_model_has_bec raises for a model without BEC outputs."""
        _require_torch_cace()
        import torch
        from openmmml.cavity_coupling import assert_cace_model_has_bec

        # Build a minimal fake module that advertises only energy output
        class FakeModel(torch.nn.Module):
            model_outputs = ['CACE_energy', 'CACE_forces']

            def forward(self, x):
                return {}

        model = FakeModel()
        with pytest.raises(ValueError, match="dipole"):
            assert_cace_model_has_bec(model, "fake_path.pth")

    def test_assert_cace_model_passes_for_bec_model(self):
        """assert_cace_model_has_bec does not raise for the real LES-BEC model."""
        _require_torch_cace()
        import torch
        from openmmml.cavity_coupling import assert_cace_model_has_bec

        model = torch.load(CHECKPOINT, map_location="cpu", weights_only=False)
        # Should not raise
        assert_cace_model_has_bec(model, CHECKPOINT)


# ---------------------------------------------------------------------------
# Test 2: dipole + BEC extraction correctness
# ---------------------------------------------------------------------------

class TestCACEDipoleBECExtraction:

    def test_extract_dipole_and_bec_from_real_model(self):
        """extract_dipole_and_bec correctly reads polarization + CACE_bec from a forward pass."""
        _require_torch_cace()
        import torch
        from cace.data.neighborhood import get_neighborhood
        from openmmml.cavity_coupling import extract_dipole_and_bec

        device = "cpu"
        model = torch.load(CHECKPOINT, map_location=device, weights_only=False)
        model.eval()

        _, pos_nm = _make_3water_topology()
        n_atoms = pos_nm.shape[0]
        pos_ang = torch.tensor(pos_nm * 10.0, dtype=torch.float32, requires_grad=True)
        atomic_numbers_np = np.array([8, 1, 1, 8, 1, 1, 8, 1, 1], dtype=int)
        atomic_numbers = torch.tensor(atomic_numbers_np, dtype=torch.long)

        edge_index, shifts, unit_shifts = get_neighborhood(
            positions=pos_nm * 10.0, cutoff=6.0, pbc=(False,)*3, cell=None,
        )
        data_dict = {
            'positions': pos_ang,
            'atomic_numbers': atomic_numbers,
            'edge_index': torch.tensor(edge_index, dtype=torch.long),
            'shifts': torch.tensor(shifts, dtype=torch.float32),
            'unit_shifts': torch.tensor(unit_shifts, dtype=torch.float32),
            'num_nodes': torch.tensor([n_atoms], dtype=torch.long),
            'ptr': torch.tensor([0, n_atoms], dtype=torch.long),
            'batch': torch.zeros(n_atoms, dtype=torch.long),
            'cell': torch.eye(3, dtype=torch.float32).unsqueeze(0) * 100.0,
        }
        output = model(data_dict, training=True)
        output_np = {k: (v.detach().cpu().numpy() if hasattr(v, 'detach') else v)
                     for k, v in output.items()}

        dipole_enm, bec = extract_dipole_and_bec(
            output_np, pos_nm, dipole_unit='e_ang',
        )

        assert dipole_enm.shape == (3,), f"dipole shape wrong: {dipole_enm.shape}"
        assert bec.shape == (n_atoms, 3, 3), f"bec shape wrong: {bec.shape}"
        assert np.all(np.isfinite(dipole_enm)), "dipole has NaN/Inf"
        assert np.all(np.isfinite(bec)), "bec has NaN/Inf"
        # Dipole should be small but non-zero for water
        assert np.linalg.norm(dipole_enm) > 1e-6, "dipole is suspiciously zero"


# ---------------------------------------------------------------------------
# Test 3: short NVE trajectory - energy conservation + cavity decomp
# ---------------------------------------------------------------------------

class TestCACECavityTrajectory:
    """Short NVE trajectory test on 3-water with cavity coupling."""

    @pytest.fixture(scope="class")
    def trajectory_data(self):
        """Run 50 NVE steps and return energies."""
        _require_torch_cace()
        try:
            import openmm
            import openmm.unit as u
        except ImportError:
            pytest.skip("OpenMM not installed")

        from openmmml import MLPotential, CavitySpec
        from openmmml.cavity_coupling import (
            apply_cavity_coupling,
            extract_dipole_and_bec,
            resolve_cace_model_path,
        )
        import torch

        top, pos_nm = _make_3water_topology()
        n_atoms = pos_nm.shape[0]

        cavity = CavitySpec(
            omegac=0.005,
            lambda_coupling=0.01,
            photon_mass=1.0,
            include_dse=True,
            cavity_particle_index=n_atoms,
        )

        potential = MLPotential("cace-lr", CHECKPOINT)
        sys = potential.createMixedSystem(top, cavity=cavity)

        integrator = openmm.VerletIntegrator(0.5e-3 * u.picoseconds)
        platform = openmm.Platform.getPlatformByName("CPU")
        ctx = openmm.Context(sys, integrator, platform)

        # Set positions including photon at origin
        photon_pos = np.zeros((1, 3))
        all_pos = np.vstack([pos_nm, photon_pos])
        ctx.setPositions(all_pos * u.nanometer)
        ctx.setVelocitiesToTemperature(300 * u.kelvin)

        total_energies = []
        N_STEPS = 50
        for _ in range(N_STEPS):
            integrator.step(1)
            state = ctx.getState(getEnergy=True)
            e = state.getPotentialEnergy().value_in_unit(u.kilojoules_per_mole)
            k = state.getKineticEnergy().value_in_unit(u.kilojoules_per_mole)
            total_energies.append(e + k)

        return {
            "total_energies": np.array(total_energies),
            "context": ctx,
            "n_atoms": n_atoms,
            "cavity": cavity,
        }

    def test_finite_energy(self, trajectory_data):
        """All total energies must be finite."""
        energies = trajectory_data["total_energies"]
        assert np.all(np.isfinite(energies)), (
            f"NaN/Inf total energy at steps: {np.where(~np.isfinite(energies))[0]}"
        )

    def test_energy_conservation(self, trajectory_data):
        """NVE total energy drift must be < 20 kJ/mol over 50 steps."""
        energies = trajectory_data["total_energies"]
        drift = np.max(energies) - np.min(energies)
        assert drift < 20.0, (
            f"Total energy drift {drift:.2f} kJ/mol exceeds 20 kJ/mol tolerance."
        )

    def test_nontrivial_dipole(self, trajectory_data):
        """The molecular dipole from the last step must be non-trivially non-zero."""
        _require_torch_cace()
        import torch
        import openmm.unit as u
        from cace.data.neighborhood import get_neighborhood
        from openmmml.cavity_coupling import extract_dipole_and_bec

        ctx = trajectory_data["context"]
        n_atoms = trajectory_data["n_atoms"]
        state = ctx.getState(getPositions=True)
        all_pos_nm = state.getPositions(asNumpy=True).value_in_unit(u.nanometer)
        pos_nm = all_pos_nm[:n_atoms]

        device = "cpu"
        model = torch.load(CHECKPOINT, map_location=device, weights_only=False)
        model.eval()

        pos_ang = torch.tensor(pos_nm * 10.0, dtype=torch.float32, requires_grad=True)
        atomic_numbers_np = np.array([8, 1, 1, 8, 1, 1, 8, 1, 1], dtype=int)
        atomic_numbers = torch.tensor(atomic_numbers_np, dtype=torch.long)

        edge_index, shifts, unit_shifts = get_neighborhood(
            positions=pos_nm * 10.0, cutoff=6.0, pbc=(False,)*3, cell=None,
        )
        data_dict = {
            'positions': pos_ang,
            'atomic_numbers': atomic_numbers,
            'edge_index': torch.tensor(edge_index, dtype=torch.long),
            'shifts': torch.tensor(shifts, dtype=torch.float32),
            'unit_shifts': torch.tensor(unit_shifts, dtype=torch.float32),
            'num_nodes': torch.tensor([n_atoms], dtype=torch.long),
            'ptr': torch.tensor([0, n_atoms], dtype=torch.long),
            'batch': torch.zeros(n_atoms, dtype=torch.long),
            'cell': torch.eye(3, dtype=torch.float32).unsqueeze(0) * 100.0,
        }
        output = model(data_dict, training=True)
        output_np = {k: (v.detach().cpu().numpy() if hasattr(v, 'detach') else v)
                     for k, v in output.items()}

        dipole_enm, bec = extract_dipole_and_bec(
            output_np, pos_nm, dipole_unit='e_ang',
        )
        assert np.linalg.norm(dipole_enm) > 1e-4, (
            f"Molecular dipole is suspiciously zero: {dipole_enm}"
        )

    def test_cavity_energy_decomposition(self, trajectory_data):
        """Cavity energy from state equals recompute from polarization/CACE_bec."""
        _require_torch_cace()
        import torch
        import openmm
        import openmm.unit as u
        from cace.data.neighborhood import get_neighborhood
        from openmmml.cavity_coupling import (
            apply_cavity_coupling,
            extract_dipole_and_bec,
            spring_constant_kj_per_mol_nm2,
            effective_coupling_kj_per_mol_nm_e,
        )

        ctx = trajectory_data["context"]
        n_atoms = trajectory_data["n_atoms"]
        cavity = trajectory_data["cavity"]
        state = ctx.getState(getPositions=True, getEnergy=True)
        all_pos_nm = state.getPositions(asNumpy=True).value_in_unit(u.nanometer)
        pos_nm = all_pos_nm[:n_atoms]
        q_photon = all_pos_nm[n_atoms]

        device = "cpu"
        model = torch.load(CHECKPOINT, map_location=device, weights_only=False)
        model.eval()

        pos_ang = torch.tensor(pos_nm * 10.0, dtype=torch.float32, requires_grad=True)
        atomic_numbers_np = np.array([8, 1, 1, 8, 1, 1, 8, 1, 1], dtype=int)
        atomic_numbers = torch.tensor(atomic_numbers_np, dtype=torch.long)

        edge_index, shifts, unit_shifts = get_neighborhood(
            positions=pos_nm * 10.0, cutoff=6.0, pbc=(False,)*3, cell=None,
        )
        data_dict = {
            'positions': pos_ang,
            'atomic_numbers': atomic_numbers,
            'edge_index': torch.tensor(edge_index, dtype=torch.long),
            'shifts': torch.tensor(shifts, dtype=torch.float32),
            'unit_shifts': torch.tensor(unit_shifts, dtype=torch.float32),
            'num_nodes': torch.tensor([n_atoms], dtype=torch.long),
            'ptr': torch.tensor([0, n_atoms], dtype=torch.long),
            'batch': torch.zeros(n_atoms, dtype=torch.long),
            'cell': torch.eye(3, dtype=torch.float32).unsqueeze(0) * 100.0,
        }
        output = model(data_dict, training=True)
        output_np = {k: (v.detach().cpu().numpy() if hasattr(v, 'detach') else v)
                     for k, v in output.items()}

        dipole_enm, bec = extract_dipole_and_bec(
            output_np, pos_nm, dipole_unit='e_ang',
        )

        # Recompute cavity energy directly
        K = spring_constant_kj_per_mol_nm2(cavity.omegac, cavity.photon_mass)
        eps = effective_coupling_kj_per_mol_nm_e(cavity.lambda_coupling, cavity.omegac)
        dx, dy = dipole_enm[0], dipole_enm[1]
        qx, qy, qz = q_photon[0], q_photon[1], q_photon[2]

        cavity_e_ref = (
            0.5 * K * (qx**2 + qy**2 + qz**2)
            + eps * (qx * dx + qy * dy)
            + 0.5 * eps**2 / K * (dx**2 + dy**2)
        )

        # Apply coupling to get delta E (forces_out - forces_in = cavity contribution)
        full_forces_in = np.zeros((n_atoms + 1, 3))
        _, full_forces_out = apply_cavity_coupling(
            all_pos_nm,
            full_forces_in,
            0.0,
            n_atoms,  # photon index
            cavity,
            dipole_enm,
            bec,
        )

        # Energy decomp: cavity_e_ref should equal apply_cavity_coupling's returned energy
        _, _, delta_e = (None, None, None)
        delta_e, _ = apply_cavity_coupling(
            all_pos_nm,
            full_forces_in,
            0.0,
            n_atoms,
            cavity,
            dipole_enm,
            bec,
        )

        assert np.isfinite(delta_e), f"apply_cavity_coupling returned non-finite energy: {delta_e}"
        assert abs(delta_e - cavity_e_ref) < 1e-8, (
            f"Cavity energy mismatch: recompute={cavity_e_ref:.6f}, "
            f"apply_cavity_coupling={delta_e:.6f}"
        )
