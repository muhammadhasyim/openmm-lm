"""Pytest fixtures for RPMD integration tests."""

import pytest


def _extended_rpmd_api_available():
    """True if RPMDIntegrator exposes fork-only hybrid/thermostat extensions."""
    import openmm
    from openmm import unit

    try:
        integrator = openmm.RPMDIntegrator(
            2,
            300 * unit.kelvin,
            1.0 / unit.picosecond,
            0.001 * unit.picosecond,
        )
    except Exception:
        return False
    return hasattr(integrator, "setParticleType") or hasattr(
        integrator, "setThermostatType"
    )


@pytest.fixture
def extended_rpmd_api():
    """Skip when this build only has upstream (stock) RPMDIntegrator."""
    if not _extended_rpmd_api_available():
        pytest.skip(
            "Extended RPMDIntegrator API not in this build "
            "(requires setParticleType / setThermostatType / …)"
        )
