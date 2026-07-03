#!/usr/bin/env python
"""Validate the repaired mKA potential energy surface against cav-hoomd.

Equilibrates the refactored system (CustomNonbondedForce LJ + Coulomb-only PME)
at several temperatures and compares the per-component LJ / Coulomb / harmonic
energies (Hartree, total system) to reference_potential_energy_vs_T.txt.  Also
checks that the *reference* calibration inverts the 300 K equilibrium LJ+Coulomb
energy back to ~300 K (i.e. the +185 K T_s bias is gone).
"""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import openmm
from openmm import unit

_SCRIPT_DIR = Path(__file__).resolve().parent
if str(_SCRIPT_DIR) not in sys.path:
    sys.path.insert(0, str(_SCRIPT_DIR))

from run_c2f import (  # noqa: E402
    REFERENCE_CALIBRATION_FILE,
    build_mka_system,
    BUSSI_TAU_PS,
    HARTREE_TO_KJMOL,
    NUM_MOL,
)
from openmm.cavitymd import (  # noqa: E402
    EnergyTracker,
    TemperatureTracker,
    EmpiricalTemperatureData,
    assign_force_groups,
)

KJMOL_TO_HARTREE = 1.0 / HARTREE_TO_KJMOL


def equil_and_measure(T, equil_ps=100.0, prod_ps=40.0, dt_ps=0.001, seed=42,
                      sample_every_ps=0.5):
    system, positions, n_atoms = build_mka_system(seed=seed, sample_bonds_at_T=T)
    group_map = assign_force_groups(system)

    bussi = openmm.BussiThermostat(T, BUSSI_TAU_PS)
    bussi.setApplyToAllParticles(False)
    bussi.setSubtractCMMotion(True)
    for idx in range(n_atoms):
        bussi.addParticle(idx)
    system.addForce(bussi)
    group_map = assign_force_groups(system)

    integrator = openmm.VerletIntegrator(dt_ps * unit.picosecond)
    platform = openmm.Platform.getPlatformByName("CUDA")
    context = openmm.Context(system, integrator, platform)
    context.setPositions(positions)
    context.setVelocitiesToTemperature(T * unit.kelvin)
    openmm.LocalEnergyMinimizer.minimize(context, maxIterations=100)
    integrator.step(int(equil_ps / dt_ps))

    etrk = EnergyTracker(context, None, group_map, n_atoms)

    n_samp = int(prod_ps / sample_every_ps)
    steps = int(sample_every_ps / dt_ps)
    lj, coul, harm, T_kin = [], [], [], []
    for _ in range(n_samp):
        integrator.step(steps)
        etrk._cached = None
        etrk._cached_step = -1
        e = etrk.get_energies_hartree()
        lj.append(e.get("lj", 0.0))
        coul.append(e.get("coulombic", 0.0))
        harm.append(e.get("harmonic_bond", 0.0))
        ke = context.getState(getEnergy=True).getKineticEnergy().value_in_unit(
            unit.kilojoule_per_mole)
        T_kin.append(2.0 * ke / (3.0 * n_atoms * 0.0083145))
    del context, integrator
    return {
        "lj": float(np.mean(lj)), "coul": float(np.mean(coul)),
        "harm": float(np.mean(harm)), "T_kin": float(np.mean(T_kin)),
    }


def ref_at(T):
    """Linear-interpolate the reference table at temperature T."""
    import pandas as pd
    df = pd.read_csv(REFERENCE_CALIBRATION_FILE, sep=r"\s+", comment="#")
    df = df.sort_values("temperature")
    return {
        "lj": float(np.interp(T, df.temperature, df.lj_hartree)),
        "coul": float(np.interp(T, df.temperature, df.coulombic_hartree)),
        "harm": float(np.interp(T, df.temperature, df.harmonic_hartree)),
    }


def main():
    temps = [100.0, 200.0, 300.0]
    print("=== PES validation: OpenMM (repaired) vs cav-hoomd reference ===\n")
    print(f"{'T(K)':>6} {'src':>8} {'harm':>10} {'lj':>10} {'coul':>10} {'lj+coul':>10}")
    results = {}
    for T in temps:
        m = equil_and_measure(T)
        r = ref_at(T)
        results[T] = m
        print(f"{T:>6.0f} {'openmm':>8} {m['harm']:>10.4f} {m['lj']:>10.4f} "
              f"{m['coul']:>10.4f} {m['lj'] + m['coul']:>10.4f}  (T_kin={m['T_kin']:.0f})")
        print(f"{'':>6} {'ref':>8} {r['harm']:>10.4f} {r['lj']:>10.4f} "
              f"{r['coul']:>10.4f} {r['lj'] + r['coul']:>10.4f}")
        print(f"{'':>6} {'delta':>8} {m['harm'] - r['harm']:>10.4f} "
              f"{m['lj'] - r['lj']:>10.4f} {m['coul'] - r['coul']:>10.4f} "
              f"{(m['lj'] + m['coul']) - (r['lj'] + r['coul']):>10.4f}\n")

    # Inversion check at 300 K using the reference calibration.
    emp_s = EmpiricalTemperatureData(str(REFERENCE_CALIBRATION_FILE),
                                     energy_component="lj_coulombic")
    emp_h = EmpiricalTemperatureData(str(REFERENCE_CALIBRATION_FILE),
                                     energy_component="harmonic")
    m300 = results[300.0]
    E_struct = m300["lj"] + m300["coul"]
    T_s = emp_s.calculate_temperature(E_struct)
    T_v = emp_h.calculate_temperature(m300["harm"]) if m300["harm"] > 0 else 0.0
    print("=== Reference-calibration inversion at 300 K equilibrium ===")
    print(f"  LJ+Coul = {E_struct:.4f} Ha  ->  T_s = {T_s:.1f} K  (target ~300, bias {T_s - 300:+.1f} K)")
    print(f"  harm    = {m300['harm']:.4f} Ha  ->  T_v = {T_v:.1f} K  (target ~300, bias {T_v - 300:+.1f} K)")


if __name__ == "__main__":
    main()
