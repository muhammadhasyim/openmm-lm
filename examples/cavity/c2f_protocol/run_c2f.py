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
import os
import sys
import time as wall_time
from pathlib import Path
from typing import Optional

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
    DiffEqController,
    SimpleSetpointController,
    PIDControl,
    GradientDescentFeedback,
    assign_force_groups,
    create_adaptive_integrator,
    advance_to_time,
    square_wave_on,
    DT_MAX_PS,
    compute_harmonic_bond_energy_kjmol,
    setup_gpu_adaptive_square_wave,
    setup_gpu_square_wave,
    setup_gpu_decaying_square_wave,
    setup_gpu_sinusoid,
    setup_gpu_exponential_wave,
    run_legacy_equilibrium_calibration as run_equilibrium_calibration,
    validate_calibration_file,
    crosscheck_calibration_against_reference,
)

_SCRIPT_DIR = Path(__file__).resolve().parent
REFERENCE_CALIBRATION_FILE = _SCRIPT_DIR / "reference_potential_energy_vs_T.txt"

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

# Platform preference order when auto-selecting
_PLATFORM_PREFERENCE = ("CUDA", "CPU", "Reference")


def _select_platform(platform_name=None):
    """Return an OpenMM Platform, honoring OPENMM_PLATFORM and --platform.

    When *platform_name* is None, uses ``OPENMM_PLATFORM`` if set, otherwise
    tries CUDA, then CPU, then Reference.
    """
    name = platform_name or os.environ.get("OPENMM_PLATFORM")
    if name:
        platform = openmm.Platform.getPlatformByName(name)
        print(f"Using OpenMM platform: {platform.getName()}")
        return platform

    for candidate in _PLATFORM_PREFERENCE:
        try:
            platform = openmm.Platform.getPlatformByName(candidate)
            print(f"Using OpenMM platform: {platform.getName()} (auto)")
            return platform
        except Exception:
            continue

    raise RuntimeError("No usable OpenMM platform found (tried CUDA, CPU, Reference)")


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
                     box_au=BOX_AU, seed=42, sample_bonds_at_T=None):
    """Build the modified Kob-Andersen dipole system from scratch.

    Parameters
    ----------
    sample_bonds_at_T : float or None
        If set, sample bond lengths from Boltzmann distribution at this
        temperature (K): r ~ N(r0, sqrt(k_B T / k)).
    """
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

    # Coulomb-only NonbondedForce (PME).  LJ is handled by a separate
    # CustomNonbondedForce below so that the A-B cross terms obey a proper
    # cutoff + energy shift (cav-hoomd uses pair.LJ(mode='shift')) rather than
    # the cutoff-free, PME-excluding NonbondedForce exceptions used previously.
    coulomb_force = openmm.NonbondedForce()
    coulomb_force.setNonbondedMethod(openmm.NonbondedForce.PME)
    coulomb_force.setCutoffDistance(rcut_nm)
    coulomb_force.setUseDispersionCorrection(False)

    # All Lennard-Jones (AA/BB/AB) via Kob-Andersen non-additive sigma/eps.
    # Energy is shifted to zero at r_cut to match cav-hoomd mode='shift'.
    eps_aa, sig_aa = _au_to_openmm_lj(EPS_AA_AU, SIG_AA_AU)
    eps_bb, sig_bb = _au_to_openmm_lj(EPS_BB_AU, SIG_BB_AU)
    eps_ab, sig_ab = _au_to_openmm_lj(EPS_AB_AU, SIG_AB_AU)
    lj_force = openmm.CustomNonbondedForce(
        "lj - ljcut;"
        "lj = 4*eps*((sig/r)^12 - (sig/r)^6);"
        "ljcut = 4*eps*((sig/rc)^12 - (sig/rc)^6);"
        "eps = epsfun(type1, type2);"
        "sig = sigfun(type1, type2)"
    )
    lj_force.addPerParticleParameter("type")
    lj_force.addGlobalParameter("rc", rcut_nm)
    # type index: 0 = A ("O"), 1 = B ("N"); table is row-major f(t1,t2)=v[t1+2*t2]
    lj_force.addTabulatedFunction(
        "epsfun", openmm.Discrete2DFunction(2, 2, [eps_aa, eps_ab, eps_ab, eps_bb])
    )
    lj_force.addTabulatedFunction(
        "sigfun", openmm.Discrete2DFunction(2, 2, [sig_aa, sig_ab, sig_ab, sig_bb])
    )
    lj_force.setNonbondedMethod(openmm.CustomNonbondedForce.CutoffPeriodic)
    lj_force.setCutoffDistance(rcut_nm)
    lj_force.setUseLongRangeCorrection(False)

    positions = []
    num_aa = int(frac_aa * num_molecules)
    side = int(np.ceil(num_molecules ** (1.0 / 3.0)))
    spacing = box_nm / side

    k_aa, r0_aa = _au_to_openmm_bond(K_AA_AU, R0_AA_AU)
    k_bb, r0_bb = _au_to_openmm_bond(K_BB_AU, R0_BB_AU)

    bonded_pairs = []  # (idx1, idx2) excluded from both LJ and Coulomb

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
                    atom_type = 0
                else:
                    mass, r0, k_bond = MASS_B, r0_bb, k_bb
                    atom_type = 1

                center = np.array([cx, cy, cz])
                if sample_bonds_at_T is not None and sample_bonds_at_T > 0:
                    sigma_r = np.sqrt(
                        Units.kelvin_to_kT_kjmol(sample_bonds_at_T) / k_bond
                    )
                    r_bond = max(r0 + np.random.normal(0.0, sigma_r), 0.05 * r0)
                else:
                    r_bond = r0

                r1 = center - 0.5 * r_bond * d
                r2 = center + 0.5 * r_bond * d

                idx1 = system.addParticle(mass)
                idx2 = system.addParticle(mass)
                positions.append(openmm.Vec3(*r1) * unit.nanometer)
                positions.append(openmm.Vec3(*r2) * unit.nanometer)

                bond_force.addBond(idx1, idx2, r0, k_bond)

                q1, q2 = -CHARGE_MAG, +CHARGE_MAG
                coulomb_force.addParticle(q1, 1.0, 0.0)
                coulomb_force.addParticle(q2, 1.0, 0.0)
                lj_force.addParticle([float(atom_type)])
                lj_force.addParticle([float(atom_type)])
                bonded_pairs.append((idx1, idx2))

                mol_idx += 1
            if mol_idx >= num_molecules:
                break
        if mol_idx >= num_molecules:
            break

    # Bonded exclusions (cav-hoomd nlist exclusions=('bond',)) on both LJ and
    # Coulomb.  A-B cross terms are now ordinary cutoff LJ pairs (the
    # CustomNonbondedForce tabulated rules), exactly as in cav-hoomd.
    for idx1, idx2 in bonded_pairs:
        coulomb_force.addException(idx1, idx2, 0.0, 1.0, 0.0)
        lj_force.addExclusion(idx1, idx2)

    # Confine LJ to molecular atoms; the photon (added later) stays
    # non-interacting because it never joins this interaction group.
    mol_atoms = set(range(system.getNumParticles()))
    lj_force.addInteractionGroup(mol_atoms, mol_atoms)
    print(f"LJ: CustomNonbondedForce (KA non-additive, shifted, r_cut={rcut_nm:.4f} nm), "
          f"{len(bonded_pairs)} bonded exclusions")

    system.addForce(bond_force)
    system.addForce(coulomb_force)
    system.addForce(lj_force)

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
        elif isinstance(force, openmm.CustomNonbondedForce):
            # Photon needs a per-particle parameter; it is excluded from LJ via
            # the molecular-only interaction group, so the value is irrelevant.
            force.addParticle([0.0])
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


def equilibrate_nvt(seed, temperature_K, equil_ps, dt_ps=0.001,
                    platform_name=None, minimize_steps=100,
                    sample_bonds_at_T=None, calibration_file=None,
                    ts_bias_max_K=10.0, max_equil_ps=None, chunk_ps=50.0):
    """NVT pre-equilibration on a standalone system (no cavity).

    Builds a fresh mKA system, equilibrates, and returns final positions.
    When *calibration_file* is set, extends equil in *chunk_ps* steps until
    |T_s − T_bath| ≤ *ts_bias_max_K* or *max_equil_ps* is reached.
    """
    if equil_ps <= 0 and max_equil_ps is None:
        return None

    target_ps = max_equil_ps if max_equil_ps is not None else equil_ps
    print(f"\n--- NVT pre-equilibration at T={temperature_K:.0f} K "
          f"(target {target_ps:.1f} ps) ---")

    system, positions, n_atoms = build_mka_system(
        seed=seed, sample_bonds_at_T=sample_bonds_at_T or temperature_K
    )
    mol_indices = list(range(n_atoms))

    bussi = openmm.BussiThermostat(temperature_K, BUSSI_TAU_PS)
    bussi.setApplyToAllParticles(False)
    bussi.setSubtractCMMotion(True)
    for idx in mol_indices:
        bussi.addParticle(idx)
    system.addForce(bussi)
    group_map = assign_force_groups(system)

    integrator = openmm.VerletIntegrator(dt_ps * unit.picosecond)
    platform = _select_platform(platform_name)
    context = openmm.Context(system, integrator, platform)
    context.setPositions(positions)
    context.setVelocitiesToTemperature(temperature_K * unit.kelvin)

    if minimize_steps > 0:
        openmm.LocalEnergyMinimizer.minimize(context, maxIterations=minimize_steps)

    temp_tracker = None
    energy_tracker = None
    if calibration_file is not None:
        empirical_structural = EmpiricalTemperatureData(
            calibration_file, energy_component="lj_coulombic"
        )
        empirical_harmonic = EmpiricalTemperatureData(
            calibration_file, energy_component="harmonic"
        )
        energy_tracker = EnergyTracker(context, None, group_map, n_atoms)
        temp_tracker = TemperatureTracker(
            energy_tracker,
            num_molecular_particles=n_atoms,
            num_molecules=NUM_MOL,
            empirical_structural=empirical_structural,
            empirical_harmonic=empirical_harmonic,
        )

    best_abs_bias = float("inf")
    best_positions = None
    equil_done = 0.0

    while equil_done < target_ps:
        run_ps = min(chunk_ps, target_ps - equil_done)
        if run_ps <= 0:
            break
        n_steps = max(1, int(run_ps / dt_ps))
        integrator.step(n_steps)
        equil_done += run_ps

        if temp_tracker is None:
            continue

        energy_tracker._cached = None
        energy_tracker._cached_step = -1
        T_s = temp_tracker.get_all().get("structural_fictive")
        if T_s is None:
            continue
        bias = T_s - temperature_K
        print(f"  After {equil_done:.1f} ps: T_s={T_s:.1f} K (bias {bias:+.1f} K vs bath)")
        if abs(bias) < best_abs_bias:
            best_abs_bias = abs(bias)
            best_positions = list(context.getState(getPositions=True).getPositions())
        if abs(bias) <= ts_bias_max_K:
            print(f"  Pre-equilibration converged (|bias| ≤ {ts_bias_max_K:.0f} K)")
            break

    if best_positions is not None and best_abs_bias > ts_bias_max_K:
        context.setPositions(best_positions)
        print(
            f"  Restored best pre-equil state (|T_s bias|={best_abs_bias:.1f} K "
            f"vs target < {ts_bias_max_K:.0f} K)"
        )

    equil_positions = list(context.getState(getPositions=True).getPositions())
    print(f"  Pre-equilibration complete ({equil_done:.1f} ps)")

    del context, integrator
    return equil_positions


def _post_cavity_equilibrate(context, integrator, thermostat, dt_ps, equil_ps):
    """NVT equilibration on the production system (λ=0, cavity present)."""
    if equil_ps <= 0:
        return
    n_steps = max(1, int(equil_ps / dt_ps))
    chunk = max(1, min(5000, n_steps))
    done = 0
    while done < n_steps:
        steps = min(chunk, n_steps - done)
        integrator.step(steps)
        thermostat.apply_cavity_thermostat_step(dt_ps * steps)
        done += steps
    print(f"  Post-cavity NVT equilibration complete ({n_steps} steps, {equil_ps:.1f} ps)")


def _mean_structural_T_s(
    context,
    integrator,
    thermostat,
    temp_tracker,
    energy_tracker,
    dt_ps,
    window_ps=10.0,
    sample_ps=1.0,
):
    """Average T_s over a short trailing window to reduce single-frame noise."""
    if window_ps <= 0:
        energy_tracker._cached = None
        energy_tracker._cached_step = -1
        return temp_tracker.get_all().get("structural_fictive")

    n_samples = max(1, int(round(window_ps / sample_ps)))
    step_chunk = max(1, int(sample_ps / dt_ps))
    values = []
    for _ in range(n_samples):
        integrator.step(step_chunk)
        thermostat.apply_cavity_thermostat_step(sample_ps)
        energy_tracker._cached = None
        energy_tracker._cached_step = -1
        T_s = temp_tracker.get_all().get("structural_fictive")
        if T_s is not None:
            values.append(T_s)
    return float(np.mean(values)) if values else None


# ===================================================================
#  STAGE 3 — C2F production run (adaptive integration + split-operator)
#
#  INNER LOOP (per MD step):
#    VariableVerletIntegrator with SI error-tolerance ramp on each λ edge,
#    BussiThermostat (molecules, system force), cavity Langevin (γ=0.5 ps⁻¹).
#
#  OUTER LOOP (every sample_interval_ps, typically 0.1 ps):
#    Read energies, compute T_s/T_v, apply DiffEqController bath feedback.
# ===================================================================
def run_c2f(
    calibration_file: str,
    initial_temperature_K: float = 300.0,
    cavity_freq_cm1: float = OMEGA_C_CM1,
    lambda_coupling: float = 0.09,
    square_wave_period_ps: float = 10.0,
    square_wave_duty_cycle: float = 0.10,
    coupling_start_ps: float = 20.0,
    runtime_ps: float = 200.0,
    dt_ps: float = 0.001,
    feedback_interval_ps: float = 0.01,
    sample_interval_ps: Optional[float] = None,
    feedback_method: str = "diffeq",
    lambda_profile: str = "square",
    gd_time_constant_ps: float = 5.0,
    gd_target_temperature_K: float = 50.0,
    diffeq_tau_ps: float = 1.0,
    pid_kp: float = 0.1,
    pid_ki: float = 0.01,
    pid_kd: float = 0.0,
    T_min: float = 0.1,
    T_max: Optional[float] = None,
    equil_ps: float = 20.0,
    post_cavity_equil_ps: float = 0.0,
    ts_bias_max_K: float = 10.0,
    max_post_equil_ps: float = 200.0,
    max_pre_equil_ps: Optional[float] = None,
    output_prefix: str = "c2f",
    seed: int = 42,
    include_dipole_self_energy: bool = True,
    platform_name=None,
    adaptive: bool = True,
    log_dt: bool = False,
    finite_q: bool = True,
    feedback_every_step: bool = False,
):
    """Run the full C2F protocol with SI-accurate adaptive integration.

    Parameters
    ----------
    finite_q : bool
        When True (default), the photon is displaced to its equilibrium
        ``q_eq = -(λ/ω_c)d`` on each rising λ edge.  For a faithful cav-hoomd
        Figure 5 reproduction set ``finite_q=False`` (q≈0 mode): the photon is
        never displaced so the sudden λ quench delivers the ultrastrong-coupling
        kick (a large T_v spike).
    feedback_every_step : bool
        When True, the bath feedback runs at every sample on the *instantaneous*
        T_s with no λ-off-window gating (matches cav-hoomd
        ``--diffeq-update-interval 0``).  When False, feedback updates once per
        period from the mean λ-off T_s.
    """
    print("\n=== Stage 3: C2F production run ===")
    print(f"  T_initial        = {initial_temperature_K} K")
    print(f"  ω_c              = {cavity_freq_cm1} cm⁻¹")
    print(f"  λ                = {lambda_coupling}")
    print(f"  period           = {square_wave_period_ps} ps,  duty = {square_wave_duty_cycle}")
    print(f"  coupling on      = {coupling_start_ps} ps")
    print(f"  runtime          = {runtime_ps} ps")
    print(f"  feedback method  = {feedback_method}")
    print(f"  lambda profile   = {lambda_profile}")
    print(f"  feedback interval = {feedback_interval_ps} ps")
    if sample_interval_ps is None:
        sample_interval_ps = feedback_interval_ps
    print(f"  sample interval  = {sample_interval_ps} ps")
    print(f"  pre-equil        = {equil_ps} ps (max {max_pre_equil_ps or equil_ps:.1f} ps)")
    print(f"  post-cavity equil= {post_cavity_equil_ps} ps (max {max_post_equil_ps} ps)")
    print(f"  adaptive dt      = {adaptive}")

    np.random.seed(seed)
    omegac_au = cavity_freq_cm1 / HARTREE_TO_CM1

    # ---- Build molecular system with Boltzmann bond sampling ----
    equil_positions = None
    if equil_ps > 0:
        equil_positions = equilibrate_nvt(
            seed, initial_temperature_K, equil_ps,
            dt_ps=dt_ps, platform_name=platform_name,
            sample_bonds_at_T=initial_temperature_K,
            calibration_file=calibration_file,
            ts_bias_max_K=ts_bias_max_K,
            max_equil_ps=max_pre_equil_ps,
        )

    system, positions, n_atoms = build_mka_system(
        seed=seed, sample_bonds_at_T=initial_temperature_K
    )
    if equil_positions is not None:
        positions = equil_positions

    cavity_index = add_cavity_particle(system, positions)

    # ---- Add CavityForce with GPU-side coupling modulation ----
    cavity_force = openmm.CavityForce(cavity_index, omegac_au, 0.0, PHOTON_MASS_AMU)
    cavity_force.setIncludeDipoleSelfEnergy(include_dipole_self_energy)
    print(f"  Dipole self-energy: {'ON' if include_dipole_self_energy else 'OFF'}")

    if lambda_profile == "adaptive_square":
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
    elif lambda_profile == "square":
        setup_gpu_square_wave(
            cavity_force,
            amplitude=lambda_coupling,
            period_ps=square_wave_period_ps,
            duty_cycle=square_wave_duty_cycle,
            start_time_ps=coupling_start_ps,
        )
    elif lambda_profile == "decaying_square":
        setup_gpu_decaying_square_wave(
            cavity_force,
            initial_amplitude=lambda_coupling,
            period_ps=square_wave_period_ps,
            duty_cycle=square_wave_duty_cycle,
            decay_rate_per_period=0.05,
            start_time_ps=coupling_start_ps,
        )
    elif lambda_profile == "sinusoid":
        setup_gpu_sinusoid(
            cavity_force,
            amplitude=lambda_coupling,
            period_ps=square_wave_period_ps,
            start_time_ps=coupling_start_ps,
        )
    elif lambda_profile == "exp_wave":
        setup_gpu_exponential_wave(
            cavity_force,
            amplitude=lambda_coupling,
            period_ps=square_wave_period_ps,
            decay_tau_ps=square_wave_period_ps * square_wave_duty_cycle,
            start_time_ps=coupling_start_ps,
        )
    else:
        raise ValueError(f"Unknown lambda profile: {lambda_profile}")
    system.addForce(cavity_force)

    # Displacer: auto step trigger disabled; Python calls displaceToEquilibrium
    # on each rising λ edge via advance_to_time().  In q≈0 mode (finite_q=False,
    # cav-hoomd Figure 5) the photon is never displaced so the sudden quench
    # delivers the ultrastrong-coupling kick.
    if finite_q:
        displacer = openmm.CavityParticleDisplacer(
            cavity_index, omegac_au, PHOTON_MASS_AMU
        )
        displacer.setSwitchOnLambda(lambda_coupling)
        displacer.setSwitchOnStep(2**31 - 1)
        system.addForce(displacer)
        print("  Photon displacement: ON (finite-q mode)")
    else:
        displacer = None
        print("  Photon displacement: OFF (q≈0 mode — ultrastrong-coupling kick)")

    mol_indices = list(range(n_atoms))
    DualThermostat.setup_bussi_for_system(
        system, mol_indices, initial_temperature_K, BUSSI_TAU_PS
    )

    group_map = assign_force_groups(system)

    if adaptive:
        integrator = create_adaptive_integrator(DT_MAX_PS)
    else:
        integrator = openmm.VerletIntegrator(dt_ps * unit.picosecond)
    platform = _select_platform(platform_name)

    context = openmm.Context(system, integrator, platform)
    context.setPositions(positions)
    context.setVelocitiesToTemperature(initial_temperature_K * unit.kelvin)
    initialize_cavity_position(context, cavity_index, initial_temperature_K, omegac_au)

    energy_tracker = EnergyTracker(
        context, cavity_force, group_map, n_atoms, cavity_index
    )
    empirical_structural = EmpiricalTemperatureData(
        calibration_file, energy_component="lj_coulombic"
    )
    empirical_harmonic = EmpiricalTemperatureData(
        calibration_file, energy_component="harmonic"
    )
    temp_tracker = TemperatureTracker(
        energy_tracker,
        num_molecular_particles=n_atoms,
        num_molecules=NUM_MOL,
        empirical_structural=empirical_structural,
        empirical_harmonic=empirical_harmonic,
    )

    thermostat = DualThermostat(
        context, system, cavity_index,
        cavity_friction_ps_inv=0.5,
        cavity_temperature_K=initial_temperature_K,
    )

    current_bath_T = initial_temperature_K

    feedback_controller = None
    if feedback_method == "diffeq":
        feedback_controller = DiffEqController(
            temp_tracker,
            time_constant_ps=diffeq_tau_ps,
            update_interval_ps=feedback_interval_ps,
            T_min=T_min,
            T_max=T_max,
            turn_on_time_ps=coupling_start_ps,
        )
    elif feedback_method == "setpoint":
        feedback_controller = SimpleSetpointController(
            temp_tracker,
            time_constant_ps=diffeq_tau_ps,
            update_interval_ps=feedback_interval_ps,
            T_min=T_min,
            T_max=T_max,
            turn_on_time_ps=coupling_start_ps,
        )
    elif feedback_method == "pid":
        pid_Ti = (pid_kp / pid_ki) if pid_ki > 0 else 1e10
        feedback_controller = PIDControl(
            temp_tracker,
            target_temperature=gd_target_temperature_K,
            Kp=pid_kp,
            Ti=pid_Ti,
            Td=pid_kd,
            update_interval_ps=feedback_interval_ps,
            T_min=T_min,
            T_max=T_max,
            turn_on_time_ps=coupling_start_ps,
        )
    elif feedback_method == "gradient_descent":
        feedback_controller = GradientDescentFeedback(
            temperature_method="harmonic_equipartition",
            time_constant_ps=gd_time_constant_ps,
            target_temperature_K=gd_target_temperature_K,
            temperature_tracker=temp_tracker,
            update_interval_ps=feedback_interval_ps,
            T_min=T_min,
            T_max=T_max,
            turn_on_time_ps=coupling_start_ps,
        )

    T_s_window = []
    T_s_window_max_size = 5
    ts_off_samples: list = []
    prev_lambda_on_sample = False

    # Optional brief post-cavity NVT at λ=0 (disabled by default — long equil
    # can redistribute bond/nonbonded energy and destabilize T_s at t=0).
    if post_cavity_equil_ps > 0 and max_post_equil_ps > 0:
        post_equil_done = 0.0
        chunk_ps = max(post_cavity_equil_ps, 10.0)
        best_abs_bias = float("inf")
        best_equil_state = None
        while post_equil_done < max_post_equil_ps:
            run_ps = min(chunk_ps, max_post_equil_ps - post_equil_done)
            if run_ps <= 0:
                break
            print(f"\n--- Post-cavity equilibration ({run_ps:.1f} ps, "
                  f"{post_equil_done + run_ps:.1f}/{max_post_equil_ps:.1f} ps total) ---")
            _post_cavity_equilibrate(context, integrator, thermostat, dt_ps, run_ps)
            post_equil_done += run_ps
            T_s_check = _mean_structural_T_s(
                context, integrator, thermostat, temp_tracker, energy_tracker, dt_ps,
                window_ps=min(10.0, run_ps * 0.2),
            )
            if T_s_check is not None:
                bias = T_s_check - initial_temperature_K
                print(f"  After equil: T_s={T_s_check:.1f} K (bias {bias:+.1f} K vs bath)")
                if abs(bias) < best_abs_bias:
                    best_abs_bias = abs(bias)
                    best_equil_state = context.getState(getPositions=True, getVelocities=True)
                if abs(bias) <= ts_bias_max_K:
                    break
            if post_equil_done >= max_post_equil_ps:
                break

        if best_equil_state is not None:
            context.setState(best_equil_state)
            energy_tracker._cached = None
            energy_tracker._cached_step = -1
            if best_abs_bias > ts_bias_max_K:
                print(
                    f"  Restored best post-equil state (|T_s bias|={best_abs_bias:.1f} K "
                    f"vs target < {ts_bias_max_K:.0f} K)"
                )

    # Post-equil MD advanced the context clock; production must start at t=0.
    context.setTime(0.0 * unit.picosecond)
    energy_tracker._cached = None
    energy_tracker._cached_step = -1
    adaptive_state = {"ramp_t0": None, "prev_lambda_on": False}

    # t=0 structural fictive temperature bias guard (coupling still off)
    feedback_armed = True
    temps0 = temp_tracker.get_all()
    T_s0 = temps0.get("structural_fictive")
    if T_s0 is not None:
        ts_bias = T_s0 - initial_temperature_K
        if abs(ts_bias) > ts_bias_max_K:
            feedback_armed = False
            print(
                f"  WARNING: t=0 T_s={T_s0:.1f} K vs bath={initial_temperature_K:.1f} K "
                f"(bias {ts_bias:+.1f} K; expected |bias| < {ts_bias_max_K:.0f} K). "
                "Bath feedback deferred until λ-off T_s is within tolerance."
            )
        else:
            print(
                f"  t=0 T_s={T_s0:.1f} K (bias {ts_bias:+.1f} K vs bath) — OK, feedback enabled"
            )
    else:
        feedback_armed = False
        print("  WARNING: t=0 T_s unavailable — bath feedback deferred")

    csv_path = f"{output_prefix}_energies.csv"
    csv_file = open(csv_path, "w")
    csv_file.write(
        "time_ps,T_bath_K,T_kinetic_K,T_v_fictive_K,T_s_fictive_K,"
        "E_bond_kjmol,E_nonbonded_kjmol,"
        "E_cav_harmonic_kjmol,E_cav_coupling_kjmol,E_cav_dse_kjmol\n"
    )

    sample_interval_ps = (
        sample_interval_ps if sample_interval_ps is not None else feedback_interval_ps
    )
    n_samples = max(1, int(round(runtime_ps / sample_interval_ps)))
    dt_log = []

    print(f"\n  {n_samples} sample points every {sample_interval_ps} ps")
    print(f"  Integration: {'VariableVerlet (SI adaptive)' if adaptive else 'fixed Verlet'}\n")

    t0 = wall_time.time()

    try:
        for sample_idx in range(n_samples):
            target_time_ps = (sample_idx + 1) * sample_interval_ps
            if target_time_ps > runtime_ps:
                target_time_ps = runtime_ps

            if adaptive:
                dts = advance_to_time(
                    context, integrator, thermostat, displacer,
                    lambda_coupling, target_time_ps,
                    coupling_start_ps, square_wave_period_ps,
                    square_wave_duty_cycle, adaptive_state, log_dt=log_dt,
                )
                if log_dt:
                    dt_log.extend(dts)
            else:
                steps = max(1, int(sample_interval_ps / dt_ps))
                integrator.step(steps)
                thermostat.apply_cavity_thermostat_step(
                    sample_interval_ps if steps == 1 else dt_ps * steps
                )

            time_ps = context.getState().getTime().value_in_unit(unit.picosecond)
            energies = energy_tracker.get_energies()
            temps = temp_tracker.get_all()

            T_kin = temps.get("kinetic", 0.0)
            T_v = temps.get("harmonic_equipartition", 0.0)
            T_s = temps.get("structural_fictive")

            lambda_on = square_wave_on(
                time_ps, coupling_start_ps, square_wave_period_ps, square_wave_duty_cycle
            )

            new_bath_T = current_bath_T

            if feedback_every_step:
                # Faithful cav-hoomd C2F: update bath every sample from the
                # *instantaneous* T_s (no λ-off gating, --diffeq-update-interval 0).
                if time_ps >= coupling_start_ps and T_s is not None:
                    if not feedback_armed:
                        inst_bias = T_s - current_bath_T
                        if abs(inst_bias) <= ts_bias_max_K:
                            feedback_armed = True
                            print(
                                f"  Feedback armed at t={time_ps:.1f} ps "
                                f"(T_s={T_s:.1f} K, bias {inst_bias:+.1f} K)"
                            )
                    if feedback_armed:
                        if feedback_method == "empirical":
                            T_s_window.append(T_s)
                            if len(T_s_window) > T_s_window_max_size:
                                T_s_window.pop(0)
                            new_bath_T = max(T_min, float(np.mean(T_s_window)))
                            if T_max is not None:
                                new_bath_T = min(T_max, new_bath_T)
                        elif feedback_controller is not None:
                            result = feedback_controller.step(
                                time_ps,
                                current_bath_T,
                                signal_override=T_s,
                                force=True,
                                dt_ps_override=sample_interval_ps,
                            )
                            if result is not None:
                                new_bath_T = result
            else:
                # C2F feedback: measure T_s only during λ-off windows; update bath at
                # each OFF→ON edge using the mean structural signal from the off period.
                if time_ps >= coupling_start_ps:
                    if not lambda_on and prev_lambda_on_sample:
                        ts_off_samples = []

                    if not lambda_on and T_s is not None:
                        ts_off_samples.append(T_s)

                    if lambda_on and not prev_lambda_on_sample and ts_off_samples:
                        T_s_mean = float(np.mean(ts_off_samples))
                        off_window_ps = square_wave_period_ps * (
                            1.0 - square_wave_duty_cycle
                        )
                        if not feedback_armed:
                            mean_bias = T_s_mean - current_bath_T
                            if abs(mean_bias) <= ts_bias_max_K:
                                feedback_armed = True
                                print(
                                    f"  Feedback armed at t={time_ps:.1f} ps "
                                    f"(λ-off T_s={T_s_mean:.1f} K, bias {mean_bias:+.1f} K)"
                                )
                            else:
                                print(
                                    f"  Skipping feedback at t={time_ps:.1f} ps "
                                    f"(λ-off T_s={T_s_mean:.1f} K, bias {mean_bias:+.1f} K)"
                                )
                                ts_off_samples = []
                        if feedback_armed:
                            if feedback_method == "empirical":
                                T_s_window.append(T_s_mean)
                                if len(T_s_window) > T_s_window_max_size:
                                    T_s_window.pop(0)
                                new_bath_T = max(T_min, float(np.mean(T_s_window)))
                                if T_max is not None:
                                    new_bath_T = min(T_max, new_bath_T)
                            elif feedback_controller is not None:
                                result = feedback_controller.step(
                                    time_ps,
                                    current_bath_T,
                                    signal_override=T_s_mean,
                                    force=True,
                                    dt_ps_override=off_window_ps,
                                )
                                if result is not None:
                                    new_bath_T = result
                            ts_off_samples = []

            prev_lambda_on_sample = lambda_on

            if new_bath_T != current_bath_T:
                thermostat.set_molecular_temperature(new_bath_T)
                thermostat.set_cavity_temperature(new_bath_T)
                current_bath_T = new_bath_T

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

            if sample_idx % max(1, n_samples // 20) == 0:
                elapsed = wall_time.time() - t0
                rate = time_ps / elapsed if elapsed > 0 else 0
                dt_str = ""
                if adaptive and log_dt and dt_log:
                    dt_str = f"  dt=[{min(dt_log):.2e},{max(dt_log):.2e}] ps"
                print(f"  t={time_ps:8.2f} ps  T_bath={current_bath_T:7.2f} K  "
                      f"T_kin={T_kin:7.2f} K  T_v={T_v:7.2f} K  "
                      f"T_s={T_s if T_s else 0:7.2f} K  "
                      f"[{rate:.1f} ps/s]{dt_str}")
    finally:
        csv_file.close()

    elapsed = wall_time.time() - t0
    print(f"\nSimulation complete: {runtime_ps:.1f} ps in {elapsed:.1f} s "
          f"({elapsed / runtime_ps:.2f} s/ps)")

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
    parser.add_argument("--calibration-file", default=str(REFERENCE_CALIBRATION_FILE),
                        help="Empirical calibration for T_s/T_v inversion "
                             "(default: reference_potential_energy_vs_T.txt)")
    parser.add_argument("--run-self-calibration", action="store_true",
                        help="Run legacy self-generated calibration and cross-check vs reference")
    parser.add_argument("--calibration-run-ps", type=float, default=500.0,
                        help="Duration of each legacy calibration run (ps); "
                             "for paper-scale calibration use run_fictive_calibration.py")
    parser.add_argument("--equil-ps", type=float, default=20.0,
                        help="NVT pre-equilibration before C2F (ps)")
    parser.add_argument("--initial-T", type=float, default=300.0,
                        help="Initial bath temperature for C2F (K)")
    parser.add_argument("--target-T", type=float, default=50.0,
                        help="GD feedback target temperature (K)")
    parser.add_argument("--lambda", dest="lam", type=float, default=0.09,
                        help="Cavity coupling strength (dimensionless)")
    parser.add_argument("--period-ps", type=float, default=10.0,
                        help="Square wave period (ps)")
    parser.add_argument("--duty-cycle", type=float, default=0.10)
    parser.add_argument("--coupling-start-ps", type=float, default=20.0,
                        help="When to activate cavity coupling (ps)")
    parser.add_argument("--runtime-ps", type=float, default=200.0)
    parser.add_argument("--dt-ps", type=float, default=0.001,
                        help="Nominal integration timestep (ps)")
    parser.add_argument("--feedback-interval-ps", type=float, default=0.1,
                        help="Sample/feedback interval (ps)")
    parser.add_argument("--feedback", choices=[
        "empirical", "gradient_descent", "diffeq", "setpoint", "pid",
    ], default="diffeq")
    parser.add_argument("--lambda-profile", choices=[
        "adaptive_square", "square", "decaying_square", "sinusoid", "exp_wave",
    ], default="square",
                        help="GPU-side coupling modulation profile")
    parser.add_argument("--no-adaptive", action="store_true",
                        help="Use fixed VerletIntegrator instead of adaptive")
    parser.add_argument("--log-dt", action="store_true",
                        help="Log adaptive dt range during production")
    parser.add_argument("--diffeq-tau-ps", type=float, default=1.0,
                        help="DiffEq/setpoint controller time constant (ps)")
    parser.add_argument("--pid-kp", type=float, default=0.1)
    parser.add_argument("--pid-ki", type=float, default=0.01)
    parser.add_argument("--pid-kd", type=float, default=0.0)
    parser.add_argument("--gd-tau-ps", type=float, default=5.0,
                        help="GD controller time constant (ps)")
    parser.add_argument("--no-dipole-self-energy", action="store_true",
                        help="Disable dipole self-energy (self-polarization) term in CavityForce")
    parser.add_argument("--output-prefix", default="c2f")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--platform", default=None,
                        help="OpenMM platform (CUDA, CPU, Reference). "
                             "Default: auto (CUDA > CPU > Reference). "
                             "Also respects OPENMM_PLATFORM env var.")
    args = parser.parse_args()

    # --- Stage 2: calibration ---
    cal_path = Path(args.calibration_file)
    if args.run_self_calibration:
        self_cal_path = cal_path.parent / "self_calibration_data.txt"
        calibration_temps = np.array([
            30, 50, 75, 100, 125, 150, 200, 250, 300, 400, 500
        ], dtype=float)

        def _make_system(T=None):
            return build_mka_system(
                seed=args.seed,
                sample_bonds_at_T=T if T is not None else args.initial_T,
            )

        run_equilibrium_calibration(
            _make_system, calibration_temps,
            run_ps=args.calibration_run_ps,
            dt_ps=args.dt_ps,
            output_file=str(self_cal_path),
            platform_name=args.platform,
        )
        validate_calibration_file(self_cal_path)
        if REFERENCE_CALIBRATION_FILE.exists():
            crosscheck_calibration_against_reference(
                self_cal_path, REFERENCE_CALIBRATION_FILE
            )
    elif not args.skip_calibration and not cal_path.exists():
        print(f"WARNING: calibration file not found: {cal_path}")
        print(f"  Expected reference at {REFERENCE_CALIBRATION_FILE}")
    elif cal_path.exists():
        print(f"Using calibration: {cal_path}")
        validate_calibration_file(cal_path)

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
        lambda_profile=args.lambda_profile,
        gd_time_constant_ps=args.gd_tau_ps,
        gd_target_temperature_K=args.target_T,
        diffeq_tau_ps=args.diffeq_tau_ps,
        pid_kp=args.pid_kp,
        pid_ki=args.pid_ki,
        pid_kd=args.pid_kd,
        output_prefix=args.output_prefix,
        seed=args.seed,
        include_dipole_self_energy=not args.no_dipole_self_energy,
        platform_name=args.platform,
        equil_ps=args.equil_ps,
        adaptive=not args.no_adaptive,
        log_dt=args.log_dt,
    )


if __name__ == "__main__":
    main()
