"""Shared helpers for the mKA cavity MD tutorial and validation scripts."""

from __future__ import annotations

import numpy as np
import openmm
from openmm import unit

from openmm.cavitymd.constants import Units
from openmm.cavitymd.forcefields.mka import (
    CHARGE_MAG,
    EPS_AA_AU,
    EPS_AB_AU,
    EPS_BB_AU,
    K_AA_AU,
    K_BB_AU,
    MASS_A,
    MASS_B,
    OMEGA_C_CM1,
    PHOTON_MASS_AMU,
    R0_AA_AU,
    R0_BB_AU,
    SIG_AA_AU,
    SIG_AB_AU,
    SIG_BB_AU,
)
from openmm.cavitymd.thermostats import (
    DEFAULT_LANGEVIN_FRICTION_PS,
    DualThermostat,
    create_langevin_integrator,
)

B2NM = Units.BOHR_TO_NM
H2K = Units.HARTREE_TO_KJMOL
K_AA_OMM = K_AA_AU * H2K / B2NM**2
R0_AA_OMM = R0_AA_AU * B2NM


def select_platform(prefer_cuda: bool = True) -> openmm.Platform:
    """Return CUDA (mixed precision) when available, otherwise CPU/Reference."""
    names = []
    if prefer_cuda:
        names.append("CUDA")
    names.extend(["CPU", "Reference"])
    for name in names:
        try:
            platform = openmm.Platform.getPlatformByName(name)
            if name == "CUDA":
                platform.setPropertyDefaultValue("Precision", "mixed")
            return platform
        except Exception:
            continue
    raise RuntimeError("No OpenMM platform available")


def dipole_magnitude(
    pos: np.ndarray,
    charges: np.ndarray,
    indices: list[int],
) -> float:
    """Translation/rotation-invariant dipole magnitude (|Σ qᵢ rᵢ|)."""
    d = np.dot(charges, pos[indices])
    return float(np.linalg.norm(d))


def build_single_aa_dimer_charged_system(
    lambda_coupling: float,
    omegac_au: float | None = None,
    photon_index: int = 2,
) -> tuple[openmm.System, openmm.CavityParticleDisplacer, list]:
    """Single A–A dimer + photon with NonbondedForce charges for CavityForce."""
    if omegac_au is None:
        omegac_au = Units.cm1_to_au(OMEGA_C_CM1)

    system = openmm.System()
    system.addParticle(MASS_A)
    system.addParticle(MASS_A)
    system.addParticle(PHOTON_MASS_AMU)

    bond_force = openmm.HarmonicBondForce()
    bond_force.addBond(0, 1, R0_AA_OMM, K_AA_OMM)
    system.addForce(bond_force)

    nb = openmm.NonbondedForce()
    nb.setNonbondedMethod(openmm.NonbondedForce.NoCutoff)
    nb.addParticle(-CHARGE_MAG, 0.1, 0.0)
    nb.addParticle(+CHARGE_MAG, 0.1, 0.0)
    nb.addParticle(0.0, 0.1, 0.0)
    nb.addException(0, 1, 0.0, 1.0, 0.0)
    system.addForce(nb)

    cavity_force = openmm.CavityForce(
        photon_index, omegac_au, lambda_coupling, PHOTON_MASS_AMU
    )
    system.addForce(cavity_force)

    displacer = openmm.CavityParticleDisplacer(
        photon_index, omegac_au, PHOTON_MASS_AMU
    )
    displacer.setSwitchOnStep(2**31 - 1)
    system.addForce(displacer)

    half_r0 = R0_AA_OMM / 2.0
    positions = [
        openmm.Vec3(-half_r0, 0, 0) * unit.nanometer,
        openmm.Vec3(+half_r0, 0, 0) * unit.nanometer,
        openmm.Vec3(0, 0, 0) * unit.nanometer,
    ]
    return system, displacer, positions


def build_two_dimer_system(
    lambda_coupling: float,
    omegac_au: float | None = None,
    separation_bohr: float = 15.0,
    photon_index: int = 4,
) -> tuple[openmm.System, openmm.CavityParticleDisplacer, list]:
    """Two dimers (A–A + B–B) + LJ + Coulomb + photon."""
    if omegac_au is None:
        omegac_au = Units.cm1_to_au(OMEGA_C_CM1)

    k_bb_omm = K_BB_AU * H2K / B2NM**2
    r0_bb_omm = R0_BB_AU * B2NM

    system = openmm.System()
    for mass in (MASS_A, MASS_A, MASS_B, MASS_B, PHOTON_MASS_AMU):
        system.addParticle(mass)

    bond_force = openmm.HarmonicBondForce()
    bond_force.addBond(0, 1, R0_AA_OMM, K_AA_OMM)
    bond_force.addBond(2, 3, r0_bb_omm, k_bb_omm)
    system.addForce(bond_force)

    coulomb = openmm.NonbondedForce()
    coulomb.setNonbondedMethod(openmm.NonbondedForce.NoCutoff)
    for q in (-CHARGE_MAG, +CHARGE_MAG, -CHARGE_MAG, +CHARGE_MAG, 0.0):
        coulomb.addParticle(q, 0.1, 0.0)
    coulomb.addException(0, 1, 0.0, 1.0, 0.0)
    coulomb.addException(2, 3, 0.0, 1.0, 0.0)
    system.addForce(coulomb)

    eps_aa, sig_aa = EPS_AA_AU * H2K, SIG_AA_AU * B2NM
    eps_ab, sig_ab = EPS_AB_AU * H2K, SIG_AB_AU * B2NM
    eps_bb, sig_bb = EPS_BB_AU * H2K, SIG_BB_AU * B2NM

    lj = openmm.CustomNonbondedForce(
        "lj - ljcut;"
        "lj = 4*eps*((sig/r)^12 - (sig/r)^6);"
        "ljcut = 4*eps*((sig/rc)^12 - (sig/rc)^6);"
        "eps = epsfun(type1, type2);"
        "sig = sigfun(type1, type2)"
    )
    lj.addPerParticleParameter("type")
    lj.setNonbondedMethod(openmm.CustomNonbondedForce.NoCutoff)
    lj.addGlobalParameter("rc", 100.0)
    lj.addTabulatedFunction(
        "epsfun",
        openmm.Discrete2DFunction(
            3,
            3,
            [eps_aa, eps_ab, 0.0, eps_ab, eps_bb, 0.0, 0.0, 0.0, 0.0],
        ),
    )
    lj.addTabulatedFunction(
        "sigfun",
        openmm.Discrete2DFunction(
            3,
            3,
            [sig_aa, sig_ab, 1.0, sig_ab, sig_bb, 1.0, 1.0, 1.0, 1.0],
        ),
    )
    for particle_type in (0.0, 0.0, 1.0, 1.0, 2.0):
        lj.addParticle([particle_type])
    lj.addExclusion(0, 1)
    lj.addExclusion(2, 3)
    system.addForce(lj)

    cavity_force = openmm.CavityForce(
        photon_index, omegac_au, lambda_coupling, PHOTON_MASS_AMU
    )
    system.addForce(cavity_force)

    displacer = openmm.CavityParticleDisplacer(
        photon_index, omegac_au, PHOTON_MASS_AMU
    )
    displacer.setSwitchOnStep(2**31 - 1)
    system.addForce(displacer)

    sep_nm = separation_bohr * B2NM
    half_r0 = R0_AA_OMM / 2.0
    positions = [
        openmm.Vec3(-half_r0, 0, 0) * unit.nanometer,
        openmm.Vec3(+half_r0, 0, 0) * unit.nanometer,
        openmm.Vec3(-r0_bb_omm / 2, sep_nm, 0) * unit.nanometer,
        openmm.Vec3(+r0_bb_omm / 2, sep_nm, 0) * unit.nanometer,
        openmm.Vec3(0, 0, 0) * unit.nanometer,
    ]
    return system, displacer, positions


def build_single_aa_dimer_system(
    lambda_coupling: float,
    omegac_au: float | None = None,
) -> tuple[openmm.System, openmm.CavityParticleDisplacer, list]:
    """Build a single A-A dimer + photon system (no integrator/thermostat)."""
    if omegac_au is None:
        omegac_au = Units.cm1_to_au(OMEGA_C_CM1)

    system = openmm.System()
    system.addParticle(MASS_A)
    system.addParticle(MASS_A)
    system.addParticle(PHOTON_MASS_AMU)

    bond_force = openmm.HarmonicBondForce()
    bond_force.addBond(0, 1, R0_AA_OMM, K_AA_OMM)
    system.addForce(bond_force)

    cavity_idx = 2
    cavity_force = openmm.CavityForce(
        cavity_idx, omegac_au, lambda_coupling, PHOTON_MASS_AMU
    )
    system.addForce(cavity_force)

    displacer = openmm.CavityParticleDisplacer(
        cavity_idx, omegac_au, PHOTON_MASS_AMU
    )
    displacer.setSwitchOnStep(2**31 - 1)
    system.addForce(displacer)

    half_r0 = R0_AA_OMM / 2.0
    positions = [
        openmm.Vec3(-half_r0, 0, 0) * unit.nanometer,
        openmm.Vec3(+half_r0, 0, 0) * unit.nanometer,
        openmm.Vec3(0, 0, 0) * unit.nanometer,
    ]
    return system, displacer, positions


def create_context(
    system: openmm.System,
    dt_fs: float,
    temperature_K: float,
    seed: int,
    platform_name: str | None = None,
    *,
    use_langevin: bool = False,
    friction_ps_inv: float = DEFAULT_LANGEVIN_FRICTION_PS,
    positions: list | None = None,
) -> openmm.Context:
    """Create a Context with thermal velocities on the selected platform."""
    dt_ps = dt_fs * 1e-3
    if use_langevin:
        integrator = create_langevin_integrator(
            temperature_K, dt_ps, friction_ps_inv=friction_ps_inv
        )
    else:
        integrator = openmm.VerletIntegrator(dt_ps)
    if platform_name is None:
        platform = select_platform(prefer_cuda=False)
    else:
        platform = openmm.Platform.getPlatformByName(platform_name)
    context = openmm.Context(system, integrator, platform)
    if positions is not None:
        context.setPositions(positions)
    context.setVelocitiesToTemperature(temperature_K, seed)
    return context


def molecular_kinetic_energy(
    state: openmm.State,
    molecular_indices: list[int],
    system: openmm.System,
) -> float:
    """Kinetic energy of selected particles in kJ/mol."""
    vel = state.getVelocities(asNumpy=True).value_in_unit(
        unit.nanometer / unit.picosecond
    )
    ke = 0.0
    for idx in molecular_indices:
        mass = system.getParticleMass(idx).value_in_unit(unit.dalton)
        ke += 0.5 * mass * float(np.sum(vel[idx] ** 2))
    return ke


def molecular_kinetic_temperature(
    state: openmm.State,
    system: openmm.System,
    molecular_indices: list[int] | None = None,
    subtract_com: bool = False,
) -> float:
    """Molecular kinetic temperature from translational DOFs.

    Use subtract_com=False (3N DOF) with LangevinMiddleIntegrator, which
    thermostats each particle independently.  Use subtract_com=True (3N-3)
    when a BussiThermostat with setSubtractCMMotion(True) enforces the bath.
    """
    if molecular_indices is None:
        molecular_indices = [0, 1]
    ke = molecular_kinetic_energy(state, molecular_indices, system)
    dof = 3 * len(molecular_indices)
    if subtract_com:
        dof -= 3
    return 2.0 * ke / (dof * Units.KB_KJMOL_PER_K)


def system_kinetic_temperature(
    state: openmm.State,
    system: openmm.System,
) -> float:
    """Total kinetic temperature (3 DOF per particle) from an OpenMM State."""
    ke = state.getKineticEnergy().value_in_unit(unit.kilojoule_per_mole)
    dof = 3 * system.getNumParticles()
    return 2.0 * ke / (dof * Units.KB_KJMOL_PER_K)


def photon_kinetic_temperature(
    state: openmm.State,
    system: openmm.System,
    photon_index: int = 2,
    dof: int = 2,
    in_plane_components: tuple[int, int] = (1, 2),
) -> float:
    """Photon kinetic temperature from selected translational DOFs.

    With LangevinMiddleIntegrator all particles (including the photon) are
    thermostatted at T_bath; use dof=3 for the canonical kinetic temperature.
    Use dof=2 for the cavity plane (y, z when the dimer lies along x) as a
    supplementary in-plane diagnostic.
    """
    vel = state.getVelocities(asNumpy=True).value_in_unit(
        unit.nanometer / unit.picosecond
    )
    mass = system.getParticleMass(photon_index).value_in_unit(unit.dalton)
    if dof == 2:
        ke = 0.5 * mass * float(
            vel[photon_index, in_plane_components[0]] ** 2
            + vel[photon_index, in_plane_components[1]] ** 2
        )
    else:
        ke = 0.5 * mass * float(np.sum(vel[photon_index] ** 2))
    return 2.0 * ke / (dof * Units.KB_KJMOL_PER_K)


def collect_dipole_trajectory(
    context: openmm.Context,
    integrator: openmm.Integrator,
    n_steps: int,
    sample_stride: int = 1,
) -> np.ndarray:
    """Collect molecular dipole d(t) for the single A-A dimer."""
    charges = np.array([-CHARGE_MAG, +CHARGE_MAG])
    dipoles = []
    for step in range(n_steps):
        integrator.step(1)
        if step % sample_stride != 0:
            continue
        state = context.getState(getPositions=True)
        pos = state.getPositions(asNumpy=True)
        dipoles.append(charges[0] * pos[0] + charges[1] * pos[1])
    return np.asarray(dipoles)


def dipole_spectrum_cm1(
    dipoles: np.ndarray,
    dt_fs: float,
    component: int = 0,
    min_freq_cm1: float = 500.0,
    max_freq_cm1: float = 3000.0,
) -> tuple[np.ndarray, np.ndarray, float]:
    """Compute dipole power spectrum (direct FFT) and return peak frequency in cm^-1."""
    d_x = dipoles[:, component].copy()
    n_sig = len(d_x)
    d_x -= d_x.mean()

    dt_s = dt_fs * 1e-15
    freqs_hz = np.fft.rfftfreq(n_sig, d=dt_s)
    freqs_cm1 = freqs_hz / 3e10
    spectrum = np.abs(np.fft.rfft(d_x)) ** 2

    mask = (freqs_cm1 > min_freq_cm1) & (freqs_cm1 < max_freq_cm1)
    peak_idx = int(np.argmax(spectrum[mask]))
    peak_cm1 = float(freqs_cm1[mask][peak_idx])
    return freqs_cm1, spectrum, peak_cm1


def run_nvt_single_dimer(
    lambda_coupling: float = 0.01,
    temperature_K: float = 100.0,
    dt_fs: float = 1.0,
    n_steps: int = 5000,
    seed: int = 42,
    sample_stride: int = 1,
    platform_name: str | None = None,
    friction_ps_inv: float = DEFAULT_LANGEVIN_FRICTION_PS,
) -> dict:
    """Run tutorial Section 2 (Langevin NVT) and return diagnostics."""
    omegac_au = Units.cm1_to_au(OMEGA_C_CM1)
    system, displacer, positions = build_single_aa_dimer_charged_system(
        lambda_coupling=lambda_coupling,
        omegac_au=omegac_au,
    )

    dt_ps = dt_fs * 1e-3
    integrator = create_langevin_integrator(
        temperature_K, dt_ps, friction_ps_inv=friction_ps_inv
    )
    platform = (
        select_platform(prefer_cuda=False)
        if platform_name is None
        else openmm.Platform.getPlatformByName(platform_name)
    )
    context = openmm.Context(system, integrator, platform)
    context.setPositions(positions)
    context.setVelocitiesToTemperature(temperature_K, seed)
    displacer.displaceToEquilibrium(context, lambda_coupling)

    system_temperatures = []
    molecular_temperatures = []
    photon_temperatures = []
    dipoles = []
    charges = np.array([-CHARGE_MAG, +CHARGE_MAG])

    for step in range(n_steps):
        integrator.step(1)
        if step % sample_stride != 0:
            continue
        state = context.getState(
            getPositions=True, getVelocities=True, getEnergy=True
        )
        pos = state.getPositions(asNumpy=True).value_in_unit(unit.nanometer)
        dipoles.append(dipole_magnitude(pos, charges, [0, 1]))
        system_temperatures.append(system_kinetic_temperature(state, system))
        molecular_temperatures.append(
            molecular_kinetic_temperature(
                state, system, molecular_indices=[0, 1], subtract_com=False
            )
        )
        photon_temperatures.append(
            photon_kinetic_temperature(state, system, dof=3)
        )

    dipoles_arr = np.asarray(dipoles)
    freqs, spectrum, _ = dipole_acf_ir_spectrum(
        dipoles_arr, dt_fs * sample_stride, temperature_K=temperature_K
    )
    peak_cm1 = find_dominant_peak_cm1(freqs, spectrum)

    return {
        "mean_system_temperature_K": float(np.mean(system_temperatures)),
        "mean_temperature_K": float(np.mean(molecular_temperatures)),
        "mean_photon_temperature_K": float(np.mean(photon_temperatures)),
        "peak_frequency_cm1": peak_cm1,
        "omega_c_cm1": OMEGA_C_CM1,
        "dipoles": dipoles_arr,
        "system_temperatures": np.asarray(system_temperatures),
        "temperatures": np.asarray(molecular_temperatures),
        "photon_temperatures": np.asarray(photon_temperatures),
        "freqs_cm1": freqs,
        "spectrum": spectrum,
    }


def dipole_acf_ir_spectrum(
    dipole_signal: np.ndarray,
    dt_fs: float,
    temperature_K: float = 100.0,
    fraction_acf: float = 0.25,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """IR spectrum from dipole ACF + DCT (tutorial method)."""
    from scipy import fftpack
    from scipy import signal as sp_signal

    signal = dipole_signal.copy() - dipole_signal.mean()
    n_sig = len(signal)
    n_acf = max(3, int(n_sig * fraction_acf))

    if n_sig % 2 == 0:
        shifted = np.zeros(2 * n_sig)
    else:
        shifted = np.zeros(2 * n_sig - 1)
    shifted[n_sig // 2 : n_sig // 2 + n_sig] = signal
    acf_full = sp_signal.fftconvolve(shifted, signal[::-1], mode="same")[
        -n_sig:
    ] / np.arange(n_sig, 0, -1)
    autocorr = acf_full[:n_acf]

    timestep = dt_fs * 1e-15
    lineshape = fftpack.dct(autocorr, type=1)[1:]
    freqs_hz = np.linspace(0, 0.5 / timestep, len(autocorr))[1:]
    freqs_cm1 = freqs_hz / (100.0 * 299792458.0)

    boltz = 1.38064852e-23
    hbar = 1.05457180013e-34
    field = freqs_hz * (1.0 - np.exp(-hbar * freqs_hz / (boltz * temperature_K)))
    spectrum = lineshape * field
    return freqs_cm1, spectrum, autocorr


def find_local_peaks_cm1(
    freqs_cm1: np.ndarray,
    spectrum: np.ndarray,
    min_freq_cm1: float = 500.0,
    max_freq_cm1: float = 3000.0,
    relative_height: float = 0.12,
) -> list[tuple[float, float]]:
    """Return local maxima (frequency, intensity) sorted by intensity descending."""
    mask = (freqs_cm1 > min_freq_cm1) & (freqs_cm1 < max_freq_cm1)
    freqs = freqs_cm1[mask]
    spec = spectrum[mask]
    if spec.size == 0 or spec.max() <= 0:
        return []
    threshold = relative_height * spec.max()
    peaks: list[tuple[float, float]] = []
    for idx in range(1, len(spec) - 1):
        if spec[idx] > spec[idx - 1] and spec[idx] > spec[idx + 1]:
            if spec[idx] >= threshold:
                peaks.append((float(freqs[idx]), float(spec[idx])))
    peaks.sort(key=lambda item: item[1], reverse=True)
    return peaks


def find_dominant_peak_cm1(
    freqs_cm1: np.ndarray,
    spectrum: np.ndarray,
    min_freq_cm1: float = 500.0,
    max_freq_cm1: float = 3000.0,
) -> float:
    """Return the strongest spectral peak in the given window."""
    peaks = find_local_peaks_cm1(
        freqs_cm1, spectrum, min_freq_cm1, max_freq_cm1, relative_height=0.0
    )
    if peaks:
        return peaks[0][0]
    mask = (freqs_cm1 > min_freq_cm1) & (freqs_cm1 < max_freq_cm1)
    if mask.any() and spectrum[mask].max() > 0:
        return float(freqs_cm1[mask][np.argmax(spectrum[mask])])
    return float(freqs_cm1[1:][np.argmax(spectrum[1:])]) if len(spectrum) > 1 else 0.0


def find_polariton_peaks_cm1(
    freqs_cm1: np.ndarray,
    spectrum: np.ndarray,
    omega_c_cm1: float = OMEGA_C_CM1,
    search_half_width_cm1: float = 400.0,
    min_split_cm1: float = 15.0,
) -> tuple[float | None, float | None]:
    """Find lower (LP) and upper (UP) polariton peaks straddling omega_c."""
    peaks = find_local_peaks_cm1(
        freqs_cm1,
        spectrum,
        omega_c_cm1 - search_half_width_cm1,
        omega_c_cm1 + search_half_width_cm1,
    )
    lower = [p for p in peaks if p[0] < omega_c_cm1 - 5.0]
    upper = [p for p in peaks if p[0] > omega_c_cm1 + 5.0]
    if not lower or not upper:
        return None, None
    lp_freq = max(lower, key=lambda item: item[1])[0]
    up_freq = max(upper, key=lambda item: item[1])[0]
    if up_freq - lp_freq < min_split_cm1:
        return None, None
    return lp_freq, up_freq


def compare_finite_q_energy_exchange(
    lambda_coupling: float = 0.01,
    n_steps: int = 500,
    dt_fs: float = 1.0,
    seed: int = 42,
    platform_name: str | None = None,
) -> dict:
    """Demonstrate energy exchange with q=0 vs finite-q equilibrium displacement."""
    omegac_au = Units.cm1_to_au(OMEGA_C_CM1)
    system, displacer, positions = build_single_aa_dimer_charged_system(
        lambda_coupling=lambda_coupling,
        omegac_au=omegac_au,
    )
    dt_ps = dt_fs * 1e-3
    platform = (
        select_platform(prefer_cuda=False)
        if platform_name is None
        else openmm.Platform.getPlatformByName(platform_name)
    )

    def run_case(use_shift: bool) -> dict:
        integrator = openmm.VerletIntegrator(dt_ps)
        context = openmm.Context(system, integrator, platform)
        context.setPositions(positions)
        context.setVelocities(
            np.zeros((3, 3)) * unit.nanometer / unit.picosecond
        )
        q_eq = 0.0
        if use_shift:
            displacer.displaceToEquilibrium(context, lambda_coupling)
            pos = context.getState(getPositions=True).getPositions(asNumpy=True)
            q_eq = float(pos[2, 0].value_in_unit(unit.nanometer))
        e0 = context.getState(getEnergy=True).getPotentialEnergy().value_in_unit(
            unit.kilojoule_per_mole
        )
        q_dev = []
        delta_pe = []
        for _ in range(n_steps):
            integrator.step(1)
            state = context.getState(getPositions=True, getEnergy=True)
            pos_nm = state.getPositions(asNumpy=True).value_in_unit(unit.nanometer)
            q_dev.append(abs(pos_nm[2, 0] - q_eq))
            pe = state.getPotentialEnergy().value_in_unit(unit.kilojoule_per_mole)
            delta_pe.append(pe - e0)
        q_dev_arr = np.asarray(q_dev)
        delta_pe_arr = np.asarray(delta_pe)
        return {
            "q_eq_nm": q_eq,
            "max_q_deviation_nm": float(q_dev_arr.max()),
            "mean_q_deviation_nm": float(q_dev_arr.mean()),
            "potential_energy_amplitude_kj_mol": float(
                delta_pe_arr.max() - delta_pe_arr.min()
            ),
        }

    no_shift = run_case(use_shift=False)
    with_shift = run_case(use_shift=True)
    return {
        "no_shift": no_shift,
        "with_shift": with_shift,
        "exchange_ratio": (
            no_shift["potential_energy_amplitude_kj_mol"]
            / max(with_shift["potential_energy_amplitude_kj_mol"], 1e-12)
        ),
    }


def run_nve_single_dimer(
    lambda_coupling: float = 0.01,
    temperature_K: float = 100.0,
    dt_fs: float = 1.0,
    n_steps: int = 12000,
    seed: int = 42,
    platform_name: str | None = None,
    apply_finite_q_shift: bool = True,
) -> dict:
    """Run tutorial 01 (NVE single dimer) and return diagnostics."""
    omegac_au = Units.cm1_to_au(OMEGA_C_CM1)
    system, displacer, positions = build_single_aa_dimer_charged_system(
        lambda_coupling=lambda_coupling,
        omegac_au=omegac_au,
    )
    dt_ps = dt_fs * 1e-3
    integrator = openmm.VerletIntegrator(dt_ps)
    platform = (
        select_platform(prefer_cuda=False)
        if platform_name is None
        else openmm.Platform.getPlatformByName(platform_name)
    )
    context = openmm.Context(system, integrator, platform)
    context.setPositions(positions)
    context.setVelocitiesToTemperature(temperature_K, seed)
    if apply_finite_q_shift:
        displacer.displaceToEquilibrium(context, lambda_coupling)

    charges = np.array([-CHARGE_MAG, +CHARGE_MAG])
    dipoles = []
    for _ in range(n_steps):
        integrator.step(1)
        pos = context.getState(getPositions=True).getPositions(asNumpy=True)
        pos_nm = pos.value_in_unit(unit.nanometer)
        dipoles.append(dipole_magnitude(pos_nm, charges, [0, 1]))

    dipoles_arr = np.asarray(dipoles)
    freqs, spectrum, _ = dipole_acf_ir_spectrum(
        dipoles_arr, dt_fs, temperature_K=temperature_K
    )
    peak_cm1 = find_dominant_peak_cm1(freqs, spectrum)
    local_peaks = find_local_peaks_cm1(freqs, spectrum)
    return {
        "peak_frequency_cm1": peak_cm1,
        "local_peaks_cm1": local_peaks,
        "omega_c_cm1": OMEGA_C_CM1,
        "dipoles": dipoles_arr,
        "freqs_cm1": freqs,
        "spectrum": spectrum,
    }


def run_nvt_bussi_single_dimer(
    lambda_coupling: float = 0.01,
    temperature_K: float = 100.0,
    dt_fs: float = 1.0,
    n_steps: int = 8000,
    equilibration_steps: int = 500,
    seed: int = 42,
    platform_name: str | None = None,
) -> dict:
    """Run tutorial 02 (Bussi NVT single dimer) and return diagnostics."""
    omegac_au = Units.cm1_to_au(OMEGA_C_CM1)
    system, displacer, positions = build_single_aa_dimer_charged_system(
        lambda_coupling=lambda_coupling,
        omegac_au=omegac_au,
    )
    DualThermostat.setup_bussi_for_system(
        system,
        molecular_indices=[0, 1],
        temperature_K=temperature_K,
        tau_ps=1.0,
        random_number_seed=seed,
    )
    dt_ps = dt_fs * 1e-3
    integrator = openmm.VerletIntegrator(dt_ps)
    platform = (
        select_platform(prefer_cuda=False)
        if platform_name is None
        else openmm.Platform.getPlatformByName(platform_name)
    )
    context = openmm.Context(system, integrator, platform)
    context.setPositions(positions)
    context.setVelocitiesToTemperature(temperature_K, seed)
    displacer.displaceToEquilibrium(context, lambda_coupling)

    charges = np.array([-CHARGE_MAG, +CHARGE_MAG])
    molecular_temperatures = []
    dipoles = []
    for step in range(n_steps):
        integrator.step(1)
        if step < equilibration_steps:
            continue
        state = context.getState(getPositions=True, getVelocities=True)
        pos_nm = state.getPositions(asNumpy=True).value_in_unit(unit.nanometer)
        dipoles.append(dipole_magnitude(pos_nm, charges, [0, 1]))
        molecular_temperatures.append(
            molecular_kinetic_temperature(
                state, system, molecular_indices=[0, 1], subtract_com=False
            )
        )

    dipoles_arr = np.asarray(dipoles)
    freqs, spectrum, _ = dipole_acf_ir_spectrum(
        dipoles_arr, dt_fs, temperature_K=temperature_K
    )
    peak_cm1 = find_dominant_peak_cm1(freqs, spectrum)
    temps = np.asarray(molecular_temperatures)
    return {
        "mean_temperature_K": float(temps.mean()),
        "std_temperature_K": float(temps.std()),
        "peak_frequency_cm1": peak_cm1,
        "omega_c_cm1": OMEGA_C_CM1,
        "dipoles": dipoles_arr,
        "temperatures": temps,
        "freqs_cm1": freqs,
        "spectrum": spectrum,
    }


def run_nvt_bussi_two_dimers(
    lambda_coupling: float = 0.01,
    temperature_K: float = 100.0,
    dt_fs: float = 1.0,
    n_steps: int = 8000,
    equilibration_steps: int = 0,
    seed: int = 42,
    platform_name: str | None = None,
    separation_bohr: float = 15.0,
) -> dict:
    """Run tutorial 03 (two dimers + LJ + Coulomb) and return diagnostics."""
    omegac_au = Units.cm1_to_au(OMEGA_C_CM1)
    system, displacer, positions = build_two_dimer_system(
        lambda_coupling=lambda_coupling,
        omegac_au=omegac_au,
        separation_bohr=separation_bohr,
    )
    DualThermostat.setup_bussi_for_system(
        system,
        molecular_indices=[0, 1, 2, 3],
        temperature_K=temperature_K,
        tau_ps=1.0,
        random_number_seed=seed,
    )
    dt_ps = dt_fs * 1e-3
    integrator = openmm.VerletIntegrator(dt_ps)
    platform = (
        select_platform(prefer_cuda=False)
        if platform_name is None
        else openmm.Platform.getPlatformByName(platform_name)
    )
    context = openmm.Context(system, integrator, platform)
    context.setPositions(positions)
    context.setVelocitiesToTemperature(temperature_K, seed)
    displacer.displaceToEquilibrium(context, lambda_coupling)

    charges = np.array([-CHARGE_MAG, +CHARGE_MAG])
    dipoles = []
    for step in range(n_steps):
        integrator.step(1)
        if step < equilibration_steps:
            continue
        pos = context.getState(getPositions=True).getPositions(asNumpy=True)
        pos_nm = pos.value_in_unit(unit.nanometer)
        d_aa = dipole_magnitude(pos_nm, charges, [0, 1])
        d_bb = dipole_magnitude(pos_nm, charges, [2, 3])
        dipoles.append(d_aa + d_bb)

    dipoles_arr = np.asarray(dipoles)
    freqs, spectrum, _ = dipole_acf_ir_spectrum(
        dipoles_arr, dt_fs, temperature_K=temperature_K
    )
    peak_cm1 = find_dominant_peak_cm1(freqs, spectrum)
    lp_cm1, up_cm1 = find_polariton_peaks_cm1(freqs, spectrum)
    local_peaks = find_local_peaks_cm1(
        freqs,
        spectrum,
        OMEGA_C_CM1 - 400.0,
        OMEGA_C_CM1 + 400.0,
    )
    return {
        "peak_frequency_cm1": peak_cm1,
        "lp_frequency_cm1": lp_cm1,
        "up_frequency_cm1": up_cm1,
        "polariton_split_cm1": (
            None if lp_cm1 is None or up_cm1 is None else up_cm1 - lp_cm1
        ),
        "local_peaks_cm1": local_peaks,
        "omega_c_cm1": OMEGA_C_CM1,
        "dipoles": dipoles_arr,
        "freqs_cm1": freqs,
        "spectrum": spectrum,
    }
