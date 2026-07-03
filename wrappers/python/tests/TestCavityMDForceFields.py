"""Tests for openmm.cavitymd.forcefields dipole contract and registry."""

from __future__ import annotations

import unittest

import numpy as np
import openmm
from openmm import unit

from openmm.cavitymd.forcefields import (
    CavityBackend,
    CavityParams,
    DipoleResponse,
    bec_from_scalar_charges,
    build_system,
    dipole_from_charges,
    get_spec,
    list_forcefields,
    validate_dipole_response,
)
from openmm.cavitymd.forcefields.registry import (
    CAVITY_INCOMPATIBLE_MSG,
    ForceFieldRegistry,
    register_forcefield,
)


class TestDipoleHelpers(unittest.TestCase):
    def test_bec_from_scalar_charges_diagonal(self):
        charges = np.array([0.3, -0.3])
        atom_indices = np.array([0, 1], dtype=int)
        bec = bec_from_scalar_charges(charges, atom_indices)
        self.assertEqual(bec.shape, (2, 3, 3))
        np.testing.assert_allclose(bec[0], 0.3 * np.eye(3))
        np.testing.assert_allclose(bec[1], -0.3 * np.eye(3))

    def test_dipole_from_charges(self):
        positions_nm = np.array([[0.0, 0.0, 0.0], [0.1, 0.0, 0.0]])
        charges = np.array([0.3, -0.3])
        atom_indices = np.array([0, 1], dtype=int)
        mu = dipole_from_charges(positions_nm, charges, atom_indices)
        np.testing.assert_allclose(mu, [-0.03, 0.0, 0.0], rtol=1e-12)

    def test_validate_dipole_response(self):
        response = DipoleResponse(
            dipole_enm=np.zeros(3),
            bec=np.zeros((1, 3, 3)),
            atom_indices=np.array([0], dtype=int),
        )
        validate_dipole_response(response)


class TestMkaForceField(unittest.TestCase):
    def test_list_forcefields_includes_mka(self):
        names = list_forcefields()
        self.assertIn("mka", names)

    def test_build_mka_dipole_response(self):
        built = build_system(
            "mka",
            num_molecules=2,
            box_au=40.0,
            seed=42,
            cavity=CavityParams(omegac=0.01, lambda_coupling=0.001),
        )
        self.assertEqual(built.cavity_backend, CavityBackend.NATIVE_CAVITY_FORCE)
        self.assertIsNotNone(built.cavity_force)
        self.assertIsNotNone(built.dipole_provider)

        platform = openmm.Platform.getPlatformByName("Reference")
        integrator = openmm.VerletIntegrator(0.001)
        simulation = openmm.app.Simulation(
            openmm.app.Topology(),
            built.system,
            integrator,
            platform,
        )
        simulation.context.setPositions(built.positions)
        state = simulation.context.getState(getPositions=True)
        response = built.dipole_provider.evaluate_dipole_response(state)
        validate_dipole_response(response)

        positions_nm = state.getPositions(asNumpy=True).value_in_unit(unit.nanometer)
        charges = []
        for force in built.system.getForces():
            if isinstance(force, openmm.NonbondedForce):
                for idx in response.atom_indices:
                    q, _, _ = force.getParticleParameters(int(idx))
                    charges.append(q.value_in_unit(unit.elementary_charge))
                break
        charges = np.asarray(charges, dtype=float)
        mu_expected = dipole_from_charges(
            positions_nm, charges, response.atom_indices
        )
        np.testing.assert_allclose(response.dipole_enm, mu_expected, rtol=1e-10)

        bec_expected = bec_from_scalar_charges(charges, response.atom_indices)
        np.testing.assert_allclose(response.bec, bec_expected, rtol=1e-12)

    def test_finite_difference_bec_mka_dimer(self):
        built = build_system(
            "mka",
            num_molecules=1,
            box_au=40.0,
            seed=7,
            cavity=CavityParams(omegac=0.01, lambda_coupling=0.001),
        )
        platform = openmm.Platform.getPlatformByName("Reference")
        integrator = openmm.VerletIntegrator(0.001)
        simulation = openmm.app.Simulation(
            openmm.app.Topology(),
            built.system,
            integrator,
            platform,
        )
        simulation.context.setPositions(built.positions)
        state = simulation.context.getState(getPositions=True)
        response = built.dipole_provider.evaluate_dipole_response(state)
        positions_nm = state.getPositions(asNumpy=True).value_in_unit(unit.nanometer)
        charges = []
        for force in built.system.getForces():
            if isinstance(force, openmm.NonbondedForce):
                for idx in response.atom_indices:
                    q, _, _ = force.getParticleParameters(int(idx))
                    charges.append(q.value_in_unit(unit.elementary_charge))
                break
        charges = np.asarray(charges, dtype=float)

        delta = 1e-5
        for local_i, atom_idx in enumerate(response.atom_indices):
            for beta in range(3):
                pos_plus = positions_nm.copy()
                pos_minus = positions_nm.copy()
                pos_plus[atom_idx, beta] += delta
                pos_minus[atom_idx, beta] -= delta
                mu_plus = dipole_from_charges(
                    pos_plus,
                    charges,
                    response.atom_indices,
                )
                mu_minus = dipole_from_charges(
                    pos_minus,
                    charges,
                    response.atom_indices,
                )
                fd_col = (mu_plus - mu_minus) / (2.0 * delta)
                np.testing.assert_allclose(
                    response.bec[local_i, :, beta],
                    fd_col,
                    rtol=1e-4,
                    atol=1e-6,
                    err_msg=f"atom {atom_idx} beta {beta}",
                )


class TestRegistryRejection(unittest.TestCase):
    def test_rejects_backend_without_dipole_jacobian(self):
        registry = ForceFieldRegistry()

        def bad_builder(**kwargs):
            raise NotImplementedError

        with self.assertRaises(ValueError) as ctx:
            register_forcefield(
                registry,
                name="bad-ff",
                cavity_backend=CavityBackend.NATIVE_CAVITY_FORCE,
                provides_dipole_jacobian=False,
                requires_topology=False,
                builder=bad_builder,
                validate_dipole=lambda: None,
            )
        self.assertIn("dipole", str(ctx.exception).lower())

    def test_get_spec_unknown_raises(self):
        with self.assertRaises(KeyError):
            get_spec("nonexistent-force-field-xyz")


class TestMlRegistryEntries(unittest.TestCase):
    def test_ml_names_registered(self):
        names = list_forcefields()
        bridge_backends = {
            "mace-polar-1": CavityBackend.ML_CUDA_BRIDGE,
            "cace-les-bec": CavityBackend.ML_CUDA_BRIDGE,
            "aimnet2": CavityBackend.ML_CUDA_BRIDGE,
        }
        for name in (
            "cace-les-bec",
            "cace-les-bec-batch",
            "aimnet2",
            "mace-polar-1",
        ):
            self.assertIn(name, names)
            spec = get_spec(name)
            expected = bridge_backends.get(name, CavityBackend.ML_PYTHONFORCE)
            self.assertEqual(spec.cavity_backend, expected)
            self.assertTrue(spec.provides_dipole_jacobian)


if __name__ == "__main__":
    unittest.main()
