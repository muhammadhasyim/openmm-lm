#!/usr/bin/env python
"""Isolated ultrastrong-coupling kick diagnostic.

Applies a single square-wave-style lambda quench (0 -> 0.09) at t=10 ps with NO
feedback controller, comparing photon-displacement ON vs OFF.  The cav-hoomd
Figure 5 run is q=0 mode (no displacement); placing the photon at its
equilibrium q_eq = -(lambda/omega_c) d makes the molecular and photon forces
vanish at the pulse (Dq=0 with DSE on), cancelling the sudden-quench kick.

Metric: peak T_v (vibrational fictive) and bond energy in the first ~2 ps after
the pulse.  Expect displacement OFF to show a large spike, ON to be flat.
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
    add_cavity_particle,
    initialize_cavity_position,
    equilibrate_nvt,
    OMEGA_C_CM1,
    PHOTON_MASS_AMU,
    BUSSI_TAU_PS,
    HARTREE_TO_CM1,
    DT_MAX_PS,
    NUM_MOL,
)
from openmm.cavitymd import (  # noqa: E402
    EnergyTracker,
    TemperatureTracker,
    EmpiricalTemperatureData,
    DualThermostat,
    assign_force_groups,
    create_adaptive_integrator,
    setup_gpu_square_wave,
)


def run_quench(displace: bool, seed: int = 42, equil_ps: float = 100.0,
               quench_time_ps: float = 10.0, post_ps: float = 20.0,
               dt_ps: float = 0.001, calibration_file: str | None = None):
    """Single 0->0.09 quench at quench_time_ps; return time/T arrays."""
    T0 = 300.0
    lam = 0.09
    omegac_au = OMEGA_C_CM1 / HARTREE_TO_CM1
    np.random.seed(seed)

    cal = calibration_file or str(REFERENCE_CALIBRATION_FILE)

    equil_positions = equilibrate_nvt(
        seed, T0, equil_ps, dt_ps=dt_ps, sample_bonds_at_T=T0,
    )

    system, positions, n_atoms = build_mka_system(seed=seed, sample_bonds_at_T=T0)
    if equil_positions is not None:
        positions = equil_positions

    cavity_index = add_cavity_particle(system, positions)

    cavity_force = openmm.CavityForce(cavity_index, omegac_au, 0.0, PHOTON_MASS_AMU)
    cavity_force.setIncludeDipoleSelfEnergy(True)
    # Square wave: 100% duty after the quench so lambda stays ON (single quench).
    setup_gpu_square_wave(
        cavity_force, amplitude=lam, period_ps=1e9, duty_cycle=1.0,
        start_time_ps=quench_time_ps,
    )
    system.addForce(cavity_force)

    displacer = openmm.CavityParticleDisplacer(cavity_index, omegac_au, PHOTON_MASS_AMU)
    displacer.setSwitchOnLambda(lam)
    displacer.setSwitchOnStep(2**31 - 1)
    system.addForce(displacer)

    DualThermostat.setup_bussi_for_system(system, list(range(n_atoms)), T0, BUSSI_TAU_PS)
    group_map = assign_force_groups(system)

    integrator = create_adaptive_integrator(DT_MAX_PS)
    platform = openmm.Platform.getPlatformByName("CUDA")
    context = openmm.Context(system, integrator, platform)
    context.setPositions(positions)
    context.setVelocitiesToTemperature(T0 * unit.kelvin)
    initialize_cavity_position(context, cavity_index, T0, omegac_au)

    thermostat = DualThermostat(
        context, system, cavity_index,
        cavity_friction_ps_inv=0.5, cavity_temperature_K=T0,
    )

    energy_tracker = EnergyTracker(context, cavity_force, group_map, n_atoms, cavity_index)
    emp_s = EmpiricalTemperatureData(cal, energy_component="lj_coulombic")
    emp_h = EmpiricalTemperatureData(cal, energy_component="harmonic")
    temp_tracker = TemperatureTracker(
        energy_tracker, num_molecular_particles=n_atoms, num_molecules=NUM_MOL,
        empirical_structural=emp_s, empirical_harmonic=emp_h,
    )

    runtime_ps = quench_time_ps + post_ps
    sample_ps = 0.05
    n_samples = int(round(runtime_ps / sample_ps))

    times, T_v, T_s, E_bond, q_norm = [], [], [], [], []
    prev_on = False
    for s in range(n_samples):
        target = (s + 1) * sample_ps
        while True:
            t = context.getState().getTime().value_in_unit(unit.picosecond)
            if t >= target - 1e-12:
                break
            curr_on = t >= quench_time_ps
            if curr_on and not prev_on and displace:
                displacer.displaceToEquilibrium(context, lam)
            prev_on = curr_on
            integrator.step(1)
            dt_actual = integrator.getStepSize().value_in_unit(unit.picosecond)
            thermostat.apply_cavity_thermostat_step(dt_actual)

        energy_tracker._cached = None
        energy_tracker._cached_step = -1
        e = energy_tracker.get_energies()
        temps = temp_tracker.get_all()
        t = context.getState().getTime().value_in_unit(unit.picosecond)
        cav_state = context.getState(getPositions=True)
        qpos = np.array(cav_state.getPositions(asNumpy=True)[cavity_index]
                        .value_in_unit(unit.nanometer))
        times.append(t)
        T_v.append(temps.get("harmonic_equipartition", 0.0))
        T_s.append(temps.get("structural_fictive") or 0.0)
        E_bond.append(e.get("harmonic_bond", 0.0))
        q_norm.append(float(np.linalg.norm(qpos)))

    del context, integrator
    return {
        "t": np.array(times), "T_v": np.array(T_v), "T_s": np.array(T_s),
        "E_bond": np.array(E_bond), "q": np.array(q_norm),
    }


def main():
    quench = 10.0
    print("=== Kick diagnostic: single lambda 0->0.09 quench at 10 ps ===\n")
    res = {}
    for disp in (False, True):
        label = "ON" if disp else "OFF"
        print(f"\n--- Displacement {label} ---")
        res[disp] = run_quench(displace=disp, quench_time_ps=quench)

    print("\n\n=== Results (peak in first 2 ps after quench) ===")
    print(f"{'displace':>10} {'T_v(pre)':>10} {'T_v(peak)':>10} {'dT_v':>8} "
          f"{'E_bond(pre)':>12} {'E_bond(peak)':>13}")
    for disp in (False, True):
        r = res[disp]
        t = r["t"]
        pre = t < quench
        post = (t >= quench) & (t < quench + 2.0)
        tv_pre = float(np.mean(r["T_v"][pre][-10:])) if pre.any() else 0.0
        tv_peak = float(np.max(r["T_v"][post])) if post.any() else 0.0
        eb_pre = float(np.mean(r["E_bond"][pre][-10:])) if pre.any() else 0.0
        eb_peak = float(np.max(r["E_bond"][post])) if post.any() else 0.0
        print(f"{('ON' if disp else 'OFF'):>10} {tv_pre:>10.1f} {tv_peak:>10.1f} "
              f"{tv_peak - tv_pre:>8.1f} {eb_pre:>12.1f} {eb_peak:>13.1f}")

    out = _SCRIPT_DIR / "fig5_output" / "kick_diagnostic.npz"
    out.parent.mkdir(parents=True, exist_ok=True)
    np.savez(
        out,
        t_off=res[False]["t"], Tv_off=res[False]["T_v"], Ts_off=res[False]["T_s"],
        Ebond_off=res[False]["E_bond"], q_off=res[False]["q"],
        t_on=res[True]["t"], Tv_on=res[True]["T_v"], Ts_on=res[True]["T_s"],
        Ebond_on=res[True]["E_bond"], q_on=res[True]["q"],
    )
    print(f"\nSaved {out}")


if __name__ == "__main__":
    main()
