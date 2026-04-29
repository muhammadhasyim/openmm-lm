#!/usr/bin/env python
"""
Full C2F (Cavity Configurational Feedback) Protocol
====================================================
Reproduces the C2F cooling protocol from:
  "Non-Thermal Aging of Supercooled Liquids in Optical Cavities"
  Hasyim, Damiani, Hoffmann  (arXiv:2603.15693)

System: modified Kob-Andersen dipole model (mKA)
  - 250 diatomic molecules (200 A-A + 50 B-B), 500 atoms
  - Kob-Andersen LJ with non-additive cross-terms
  - Partial charges ±0.3e per molecule
  - Cavity frequency ω_c = 1560 cm⁻¹ (resonant with A-A vibration)

Protocol:
  Stage 1 — Build system on lattice
  Stage 2 — Equilibrate WITHOUT cavity at multiple temperatures (calibration)
  Stage 3 — Equilibrate at T=300K, then run C2F with adaptive square-wave coupling
"""

import argparse
import sys
import time as wall_time
from pathlib import Path

import numpy as np

try:
    import openmm
    from openmm import unit
except ImportError:
    sys.exit("OpenMM (cavity-md branch) required.  Build from "
             "https://github.com/muhammadhasyim/openmm  branch cavity-md")

from openmm.cavitymd import (
    Units,
    EmpiricalTemperatureData,
    EnergyTracker,
    TemperatureTracker,
    ElapsedTimeTracker,
    DualThermostat,
    assign_force_groups,
    setup_gpu_adaptive_square_wave,
)

# ---------------------------------------------------------------------------
#  Physical constants & unit conversions
# ---------------------------------------------------------------------------
BOHR_TO_NM = 0.0529177
HARTREE_TO_KJMOL = 2625.5
KB_HARTREE_PER_K = 3.16681e-6
HARTREE_TO_CM1 = 219474.63

# ---------------------------------------------------------------------------
#  mKA force-field parameters (Table 1 in the paper, atomic units)
# ---------------------------------------------------------------------------
# Masses
MASS_A = 16.0   # amu  ("O" in the existing OpenMM dimer code)
MASS_B = 14.0   # amu  ("N" in the existing code)

# Harmonic bonds:  V = 0.5*k*(r-r0)^2
K_AA_AU   = 0.73204       # Hartree/Bohr^2
R0_AA_AU  = 2.281655158   # Bohr   -->  omega ~ 1560 cm^-1
K_BB_AU   = 1.4325        # Hartree/Bohr^2
R0_BB_AU  = 2.0743522177  # Bohr   -->  omega ~ 2433 cm^-1

# LJ:  V = 4*eps*[(sigma/r)^12 - (sigma/r)^6],  shifted at r_cut
EPS_AA_AU   = 1.6685e-4;  SIG_AA_AU = 6.2304
EPS_BB_AU   = 8.3426e-5;  SIG_BB_AU = 5.4828
EPS_AB_AU   = 2.5028e-4;  SIG_AB_AU = 4.9832
RCUT_AU     = 15.0  # Bohr

# Charges
CHARGE_MAG = 0.3  # elementary charge

# Box
BOX_AU      = 40.0   # Bohr  (cubic)
NUM_MOL     = 250
FRAC_AA     = 0.8    # 200 AA + 50 BB

# Cavity
OMEGA_C_CM1 = 1560.0
PHOTON_MASS_AMU = 1.0 / 1822.888  # 1 a.u. mass in amu

# Thermostat
BUSSI_TAU_PS = 1.0  # Bussi time constant (paper: tau_b = 1 ps)


# ---------------------------------------------------------------------------
#  Helper: convert force-field parameters to OpenMM units
# ---------------------------------------------------------------------------
def _au_to_openmm_bond(k_au, r0_au):
    k_kjmol_nm2 = k_au * HARTREE_TO_KJMOL / (BOHR_TO_NM ** 2)
    r0_nm = r0_au * BOHR_TO_NM
    return k_kjmol_nm2, r0_nm

def _au_to_openmm_lj(eps_au, sig_au):
    eps_kjmol = eps_au * HARTREE_TO_KJMOL
    sig_nm = sig_au * BOHR_TO_NM
    return eps_kjmol, sig_nm


# ===================================================================
#  STAGE 1 — Build the mKA system
# ===================================================================
def build_mka_system(num_molecules=NUM_MOL, frac_aa=FRAC_AA,
                     box_au=BOX_AU, seed=42):
    """Build the modified Kob-Andersen dipole system from scratch."""
    np.random.seed(seed)

    box_nm = box_au * BOHR_TO_NM
    rcut_nm = RCUT_AU * BOHR_TO_NM

    system = openmm.System()
    system.setDefaultPeriodicBoxVectors(
        openmm.Vec3(box_nm, 0, 0),
        openmm.Vec3(0, box_nm, 0),
        openmm.Vec3(0, 0, box_nm),
    )

    bond_force = openmm.HarmonicBondForce()
    nb_force = openmm.NonbondedForce()
    nb_force.setNonbondedMethod(openmm.NonbondedForce.PME)
    nb_force.setCutoffDistance(rcut_nm)
    nb_force.setUseDispersionCorrection(False)  # shifted potential, no tail correction

    positions = []
    num_aa = int(frac_aa * num_molecules)
    side = int(np.ceil(num_molecules ** (1.0 / 3.0)))
    spacing = box_nm / side

    k_aa, r0_aa = _au_to_openmm_bond(K_AA_AU, R0_AA_AU)
    k_bb, r0_bb = _au_to_openmm_bond(K_BB_AU, R0_BB_AU)
    eps_aa, sig_aa = _au_to_openmm_lj(EPS_AA_AU, SIG_AA_AU)
    eps_bb, sig_bb = _au_to_openmm_lj(EPS_BB_AU, SIG_BB_AU)
    eps_ab, sig_ab = _au_to_openmm_lj(EPS_AB_AU, SIG_AB_AU)

    a_indices = []  # (particle_index, charge)
    b_indices = []

    mol_idx = 0
    for i in range(side):
        for j in range(side):
            for kk in range(side):
                if mol_idx >= num_molecules:
                    break
                is_aa = mol_idx < num_aa

                cx = (i + 0.5) * spacing
                cy = (j + 0.5) * spacing
                cz = (kk + 0.5) * spacing

                theta = np.random.rand() * 2 * np.pi
                phi = np.arccos(2 * np.random.rand() - 1)
                d = np.array([np.sin(phi) * np.cos(theta),
                              np.sin(phi) * np.sin(theta),
                              np.cos(phi)])

                if is_aa:
                    mass, r0, k_bond = MASS_A, r0_aa, k_aa
                    sig, eps = sig_aa, eps_aa
                else:
                    mass, r0, k_bond = MASS_B, r0_bb, k_bb
                    sig, eps = sig_bb, eps_bb

                r1 = np.array([cx, cy, cz]) - 0.5 * r0 * d
                r2 = np.array([cx, cy, cz]) + 0.5 * r0 * d

                idx1 = system.addParticle(mass)
                idx2 = system.addParticle(mass)
                positions.append(openmm.Vec3(*r1) * unit.nanometer)
                positions.append(openmm.Vec3(*r2) * unit.nanometer)

                bond_force.addBond(idx1, idx2, r0, k_bond)

                q1, q2 = -CHARGE_MAG, +CHARGE_MAG
                nb_force.addParticle(q1, sig, eps)
                nb_force.addParticle(q2, sig, eps)
                nb_force.addException(idx1, idx2, 0.0, 1.0, 0.0)

                if is_aa:
                    a_indices.append((idx1, q1))
                    a_indices.append((idx2, q2))
                else:
                    b_indices.append((idx1, q1))
                    b_indices.append((idx2, q2))

                mol_idx += 1
            if mol_idx >= num_molecules:
                break
        if mol_idx >= num_molecules:
            break

    # Kob-Andersen non-additive A-B cross-terms
    n_cross = 0
    for idx_a, q_a in a_indices:
        for idx_b, q_b in b_indices:
            nb_force.addException(idx_a, idx_b, q_a * q_b, sig_ab, eps_ab)
            n_cross += 1
    print(f"Added {n_cross} A-B cross-term exceptions")

    system.addForce(bond_force)
    system.addForce(nb_force)

    num_mol_particles = system.getNumParticles()
    print(f"Built mKA system: {num_mol_particles} atoms "
          f"({num_aa} AA + {num_molecules - num_aa} BB dimers), "
          f"box = {box_nm:.4f} nm")

    return system, positions, num_mol_particles


def add_cavity_particle(system, positions):
    """Add a single cavity (photon) particle with no nonbonded interactions."""
    cavity_index = system.addParticle(PHOTON_MASS_AMU)
    positions.append(openmm.Vec3(0, 0, 0) * unit.nanometer)

    for force_idx in range(system.getNumForces()):
        force = system.getForce(force_idx)
        if isinstance(force, openmm.NonbondedForce):
            force.addParticle(0.0, 0.1, 0.0)
            for i in range(cavity_index):
                force.addException(cavity_index, i, 0.0, 0.1, 0.0)
    return cavity_index


def initialize_cavity_position(context, cavity_index, temperature_K, omegac_au):
    """Sample cavity position from thermal distribution."""
    sigma_bohr = np.sqrt(KB_HARTREE_PER_K * temperature_K / (omegac_au ** 2))
    sigma_nm = sigma_bohr * BOHR_TO_NM
    new_pos = np.random.normal(0.0, sigma_nm, size=3)

    state = context.getState(getPositions=True)
    pos_list = list(state.getPositions())
    pos_list[cavity_index] = openmm.Vec3(*new_pos) * unit.nanometer
    context.setPositions(pos_list)


# ===================================================================
#  STAGE 2 — Calibration: short equilibrium runs at multiple T
# ===================================================================
def run_equilibrium_calibration(system_template_fn, temperatures,
                                run_ps=50.0, dt_ps=0.001,
                                output_file="calibration_data.txt"):
    """Run short NVT equilibrations at each T to collect <V_bond>(T) and <V_LJ+C>(T).

    Writes a whitespace-separated file with columns:
        temperature  harmonic_hartree  lj_hartree  coulombic_hartree
    """
    print("\n=== Stage 2: Equilibrium calibration ===")
    rows = []

    for T in temperatures:
        print(f"\n--- T = {T:.0f} K ---")
        system, positions, n_mol_particles = system_template_fn()

        # Assign force groups for decomposition
        group_map = {}
        for fi in range(system.getNumForces()):
            f = system.getForce(fi)
            if isinstance(f, openmm.HarmonicBondForce):
                f.setForceGroup(1)
                group_map["harmonic_bond"] = 1
            elif isinstance(f, openmm.NonbondedForce):
                f.setForceGroup(0)
                group_map["nonbonded"] = 0

        # Bussi thermostat on all particles (no cavity at calibration stage)
        bussi = openmm.BussiThermostat(T, BUSSI_TAU_PS)
        system.addForce(bussi)

        integrator = openmm.VerletIntegrator(dt_ps * unit.picosecond)
        try:
            platform = openmm.Platform.getPlatformByName("CUDA")
        except Exception:
            platform = openmm.Platform.getPlatformByName("Reference")

        context = openmm.Context(system, integrator, platform)
        context.setPositions(positions)
        context.setVelocitiesToTemperature(T * unit.kelvin)

        # Equilibrate 20% of run time
        equil_steps = int(0.2 * run_ps / dt_ps)
        integrator.step(equil_steps)

        # Production: collect averages
        prod_steps = int(0.8 * run_ps / dt_ps)
        sample_interval = max(1, int(0.1 / dt_ps))  # sample every 0.1 ps
        n_samples = prod_steps // sample_interval

        E_bond_samples = []
        E_nb_samples = []

        for _ in range(n_samples):
            integrator.step(sample_interval)
            s_bond = context.getState(getEnergy=True, groups={1 << 1})
            E_bond_kjmol = s_bond.getPotentialEnergy().value_in_unit(
                unit.kilojoule_per_mole)

            s_nb = context.getState(getEnergy=True, groups={1 << 0})
            E_nb_kjmol = s_nb.getPotentialEnergy().value_in_unit(
                unit.kilojoule_per_mole)

            E_bond_samples.append(E_bond_kjmol * Units.KJMOL_TO_HARTREE)
            E_nb_samples.append(E_nb_kjmol * Units.KJMOL_TO_HARTREE)

        E_bond_mean = float(np.mean(E_bond_samples))
        E_nb_mean = float(np.mean(E_nb_samples))

        # OpenMM PME lumps LJ + Coulomb together in NonbondedForce.
        # For the empirical fit this is fine — we use the combined value.
        rows.append((T, E_bond_mean, E_nb_mean, 0.0))
        print(f"  <V_bond> = {E_bond_mean:.6f} Ha,  <V_LJ+C> = {E_nb_mean:.6f} Ha")

        del context, integrator

    # Write calibration file
    with open(output_file, "w") as f:
        f.write("temperature  harmonic_hartree  lj_hartree  coulombic_hartree\n")
        for T, E_b, E_lj, E_c in rows:
            f.write(f"{T:.1f}  {E_b:.10f}  {E_lj:.10f}  {E_c:.10f}\n")

    print(f"\nCalibration data written to {output_file}")
    return output_file


# ===================================================================
#  STAGE 3 — C2F production run (GPU-native inner loop)
#
#  Architecture:
#    INNER LOOP (GPU, zero host sync):
#      The CavityForce kernel evaluates the adaptive square-wave
#      modulation every MD step.  It reads T_bath from the
#      "BussiTemperature" context parameter and adapts the amplitude
#      autonomously at each period boundary.
#
#    OUTER LOOP (Python, every feedback_interval_ps):
#      1. integrator.step(steps_per_interval)  — runs thousands of
#         uninterrupted MD steps on the GPU
#      2. getState(getEnergy=True) — single GPU→CPU transfer
#      3. Compute T_s (structural fictive temperature) from V_LJ+C
#      4. Update "BussiTemperature" via context.setParameter()
#      5. Log to CSV
#
#  This minimises host synchronisation to ~1 getState() call per
#  feedback interval (typically every 5-10 ps), compared to the old
#  design which called Python every 10 fs.
# ===================================================================
def run_c2f(
    calibration_file: str,
    initial_temperature_K: float = 300.0,
    cavity_freq_cm1: float = OMEGA_C_CM1,
    lambda_coupling: float = 0.03,
    square_wave_period_ps: float = 5.0,
    square_wave_duty_cycle: float = 0.5,
    coupling_start_ps: float = 20.0,
    runtime_ps: float = 200.0,
    dt_ps: float = 0.001,
    feedback_interval_ps: float = 5.0,
    feedback_method: str = "empirical",
    gd_time_constant_ps: float = 5.0,
    gd_target_temperature_K: float = 50.0,
    T_min: float = 1.0,
    T_max_factor: float = 1.5,
    output_prefix: str = "c2f",
    seed: int = 42,
):
    """Run the full C2F protocol with GPU-side adaptive square-wave."""
    print("\n=== Stage 3: C2F production run (GPU-native) ===")
    print(f"  T_initial        = {initial_temperature_K} K")
    print(f"  T_target (GD)    = {gd_target_temperature_K} K")
    print(f"  ω_c              = {cavity_freq_cm1} cm⁻¹")
    print(f"  λ                = {lambda_coupling}")
    print(f"  period           = {square_wave_period_ps} ps,  duty = {square_wave_duty_cycle}")
    print(f"  coupling on      = {coupling_start_ps} ps")
    print(f"  runtime          = {runtime_ps} ps")
    print(f"  feedback method  = {feedback_method}")
    print(f"  feedback interval= {feedback_interval_ps} ps")

    np.random.seed(seed)
    omegac_au = cavity_freq_cm1 / HARTREE_TO_CM1
    T_max = initial_temperature_K * T_max_factor

    # ---- Build system ----
    system, positions, n_mol = build_mka_system(seed=seed)
    cavity_index = add_cavity_particle(system, positions)

    # ---- Add CavityForce with GPU-side adaptive modulation ----
    cavity_force = openmm.CavityForce(cavity_index, omegac_au, 0.0, PHOTON_MASS_AMU)

    # Configure the kernel to handle square-wave modulation entirely on the GPU.
    # The kernel reads "BussiTemperature" each step and adapts the amplitude
    # per-period: g_next = g_target * sqrt(T_target / T_bath).
    setup_gpu_adaptive_square_wave(
        cavity_force,
        target_coupling=lambda_coupling,
        target_temperature_K=gd_target_temperature_K,
        period_ps=square_wave_period_ps,
        duty_cycle=square_wave_duty_cycle,
        start_time_ps=coupling_start_ps,
        stop_time_ps=-1.0,
        min_amplitude=1e-8,
        max_amplitude=0.15,
    )
    system.addForce(cavity_force)

    displacer = openmm.CavityParticleDisplacer(cavity_index, omegac_au, PHOTON_MASS_AMU)
    displacer.setSwitchOnLambda(lambda_coupling)
    displacer.setSwitchOnStep(int(coupling_start_ps / dt_ps))
    system.addForce(displacer)

    # ---- Bussi thermostat (molecular particles only) ----
    mol_indices = list(range(n_mol))
    DualThermostat.setup_bussi_for_system(
        system, mol_indices, initial_temperature_K, BUSSI_TAU_PS
    )

    # ---- Assign force groups ----
    group_map = assign_force_groups(system)

    # ---- Integrator & Context ----
    integrator = openmm.VerletIntegrator(dt_ps * unit.picosecond)
    try:
        platform = openmm.Platform.getPlatformByName("CUDA")
    except Exception:
        platform = openmm.Platform.getPlatformByName("Reference")

    context = openmm.Context(system, integrator, platform)
    context.setPositions(positions)
    context.setVelocitiesToTemperature(initial_temperature_K * unit.kelvin)
    initialize_cavity_position(context, cavity_index, initial_temperature_K, omegac_au)

    # ---- Energy / temperature trackers (used only at feedback cadence) ----
    energy_tracker = EnergyTracker(
        context, cavity_force, group_map, n_mol, cavity_index
    )
    empirical_structural = EmpiricalTemperatureData(
        calibration_file, energy_component="lj_coulombic"
    )
    temp_tracker = TemperatureTracker(energy_tracker, n_mol, empirical_structural)

    # ---- Thermostat wrapper ----
    thermostat = DualThermostat(
        context, system, cavity_index,
        cavity_friction_ps_inv=0.5,
        cavity_temperature_K=initial_temperature_K,
    )

    # ---- Feedback state ----
    # For "empirical": set T_bath = T_s (structural fictive temperature)
    # For "gradient_descent": T_bath -= alpha * 0.5 * (T_eff - T_target)
    current_bath_T = initial_temperature_K
    gd_alpha = feedback_interval_ps / gd_time_constant_ps

    # Sliding window for empirical feedback
    T_s_window = []
    T_s_window_max_size = 5  # keep last N measurements (~5 intervals)

    # ---- CSV output ----
    csv_path = f"{output_prefix}_energies.csv"
    csv_file = open(csv_path, "w")
    csv_file.write(
        "time_ps,T_bath_K,T_kinetic_K,T_v_fictive_K,T_s_fictive_K,"
        "E_bond_kjmol,E_nonbonded_kjmol,"
        "E_cav_harmonic_kjmol,E_cav_coupling_kjmol,E_cav_dse_kjmol\n"
    )

    # ---- Outer feedback loop ----
    steps_per_interval = max(1, int(feedback_interval_ps / dt_ps))
    n_intervals = int(runtime_ps / feedback_interval_ps)

    print(f"\n  {n_intervals} feedback intervals × {steps_per_interval} MD steps "
          f"= {n_intervals * steps_per_interval} total steps")
    print(f"  Host sync cadence: every {feedback_interval_ps} ps "
          f"({steps_per_interval} steps between getState calls)\n")

    t0 = wall_time.time()

    try:
        for interval in range(n_intervals):
            # ---- GPU runs uninterrupted for steps_per_interval ----
            integrator.step(steps_per_interval)

            # ---- Single GPU→CPU transfer: read energies ----
            time_ps = context.getState().getTime().value_in_unit(unit.picosecond)
            energies = energy_tracker.get_energies()
            energies_h = energy_tracker.get_energies_hartree()
            temps = temp_tracker.get_all()

            T_kin = temps.get("kinetic", 0.0)
            T_v = temps.get("harmonic_equipartition", 0.0)
            T_s = temps.get("structural_fictive")

            # ---- Feedback: update bath temperature ----
            new_bath_T = current_bath_T  # default: no change

            if time_ps >= coupling_start_ps:
                if feedback_method == "empirical" and T_s is not None:
                    # Set T_bath = windowed average of T_s
                    T_s_window.append(T_s)
                    if len(T_s_window) > T_s_window_max_size:
                        T_s_window.pop(0)
                    avg_T_s = float(np.mean(T_s_window))
                    new_bath_T = max(T_min, min(T_max, avg_T_s))

                elif feedback_method == "gradient_descent":
                    T_meas = T_v if T_v > 0 else T_kin
                    T_eff = (T_meas + current_bath_T) / 2.0
                    error = T_eff - gd_target_temperature_K
                    raw = current_bath_T - gd_alpha * 0.5 * error
                    new_bath_T = max(T_min, min(T_max, raw))

            if new_bath_T != current_bath_T:
                thermostat.set_molecular_temperature(new_bath_T)
                thermostat.set_cavity_temperature(new_bath_T)
                current_bath_T = new_bath_T

            # ---- Log ----
            T_s_str = f"{T_s:.4f}" if T_s is not None else ""
            csv_file.write(
                f"{time_ps:.6f},{current_bath_T:.4f},{T_kin:.4f},{T_v:.4f},{T_s_str},"
                f"{energies.get('harmonic_bond', 0.0):.6f},"
                f"{energies.get('nonbonded', 0.0):.6f},"
                f"{energies.get('cavity_harmonic', 0.0):.6f},"
                f"{energies.get('cavity_coupling', 0.0):.6f},"
                f"{energies.get('cavity_dipole_self', 0.0):.6f}\n"
            )
            csv_file.flush()

            if interval % max(1, n_intervals // 20) == 0:
                elapsed = wall_time.time() - t0
                rate = time_ps / elapsed if elapsed > 0 else 0
                print(f"  t={time_ps:8.2f} ps  T_bath={current_bath_T:7.2f} K  "
                      f"T_kin={T_kin:7.2f} K  T_v={T_v:7.2f} K  "
                      f"T_s={T_s if T_s else 0:7.2f} K  "
                      f"[{rate:.1f} ps/s]")
    finally:
        csv_file.close()

    elapsed = wall_time.time() - t0
    print(f"\nSimulation complete: {runtime_ps:.1f} ps in {elapsed:.1f} s "
          f"({elapsed / runtime_ps:.2f} s/ps)")

    # ---- Save final snapshot ----
    state = context.getState(getPositions=True, getVelocities=True)
    pos_final = state.getPositions(asNumpy=True).value_in_unit(unit.nanometer)
    vel_final = state.getVelocities(asNumpy=True).value_in_unit(
        unit.nanometer / unit.picosecond)

    np.savez(
        f"{output_prefix}_final_state.npz",
        positions_nm=pos_final,
        velocities_nm_per_ps=vel_final,
        temperature_K=current_bath_T,
        lambda_coupling=lambda_coupling,
        omega_c_cm1=cavity_freq_cm1,
    )
    print(f"Final state saved to {output_prefix}_final_state.npz")


# ===================================================================
#  Main
# ===================================================================
def main():
    parser = argparse.ArgumentParser(description="C2F protocol for mKA system")
    parser.add_argument("--skip-calibration", action="store_true",
                        help="Skip Stage 2 if calibration_data.txt already exists")
    parser.add_argument("--calibration-file", default="calibration_data.txt")
    parser.add_argument("--calibration-run-ps", type=float, default=50.0,
                        help="Duration of each calibration run (ps)")
    parser.add_argument("--initial-T", type=float, default=300.0,
                        help="Initial bath temperature for C2F (K)")
    parser.add_argument("--target-T", type=float, default=50.0,
                        help="GD feedback target temperature (K)")
    parser.add_argument("--lambda", dest="lam", type=float, default=0.03,
                        help="Cavity coupling strength (dimensionless)")
    parser.add_argument("--period-ps", type=float, default=5.0,
                        help="Square wave period (ps)")
    parser.add_argument("--duty-cycle", type=float, default=0.5)
    parser.add_argument("--coupling-start-ps", type=float, default=20.0,
                        help="When to activate cavity coupling (ps)")
    parser.add_argument("--runtime-ps", type=float, default=200.0)
    parser.add_argument("--dt-ps", type=float, default=0.001,
                        help="Integration timestep (ps)")
    parser.add_argument("--feedback-interval-ps", type=float, default=5.0,
                        help="How often Python reads T_s and updates T_bath (ps)")
    parser.add_argument("--feedback", choices=["gradient_descent", "empirical"],
                        default="empirical")
    parser.add_argument("--gd-tau-ps", type=float, default=5.0,
                        help="GD controller time constant (ps)")
    parser.add_argument("--output-prefix", default="c2f")
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    # --- Stage 2: calibration ---
    cal_path = Path(args.calibration_file)
    if not args.skip_calibration or not cal_path.exists():
        calibration_temps = np.array([
            30, 50, 75, 100, 125, 150, 200, 250, 300, 400, 500
        ], dtype=float)

        def _make_system():
            return build_mka_system(seed=args.seed)

        run_equilibrium_calibration(
            _make_system, calibration_temps,
            run_ps=args.calibration_run_ps,
            dt_ps=args.dt_ps,
            output_file=args.calibration_file,
        )
    else:
        print(f"Using existing calibration: {args.calibration_file}")

    # --- Stage 3: C2F production ---
    run_c2f(
        calibration_file=args.calibration_file,
        initial_temperature_K=args.initial_T,
        lambda_coupling=args.lam,
        square_wave_period_ps=args.period_ps,
        square_wave_duty_cycle=args.duty_cycle,
        coupling_start_ps=args.coupling_start_ps,
        runtime_ps=args.runtime_ps,
        dt_ps=args.dt_ps,
        feedback_interval_ps=args.feedback_interval_ps,
        feedback_method=args.feedback,
        gd_time_constant_ps=args.gd_tau_ps,
        gd_target_temperature_K=args.target_T,
        output_prefix=args.output_prefix,
        seed=args.seed,
    )


if __name__ == "__main__":
    main()
