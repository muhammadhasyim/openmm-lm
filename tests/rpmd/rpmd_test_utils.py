"""Helpers for RPMD pytest modules (unit-aware API)."""

import math

from openmm import unit


def rpmd_total_energy_kjmol(integrator):
    """Return RPMD total energy as a plain float (kJ/mol)."""
    e = integrator.getTotalEnergy()
    if hasattr(e, "value_in_unit"):
        return e.value_in_unit(unit.kilojoule_per_mole)
    return float(e)


def is_finite_energy(integrator):
    return math.isfinite(rpmd_total_energy_kjmol(integrator))


def periodic_box_volume_nm3(state):
    """Scalar box volume (nm^3) from a Context/Integrator State."""
    vecs = state.getPeriodicBoxVectors()
    v = vecs[0][0] * vecs[1][1] * vecs[2][2]
    if hasattr(v, "value_in_unit"):
        return v.value_in_unit(unit.nanometer**3)
    return float(v)
