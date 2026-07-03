"""NVT equilibrium calibration for fictive temperature energy inversion."""

from __future__ import annotations

import json
import os
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Callable, Dict, List, Optional, Sequence, Tuple, Union

import numpy as np

from .constants import Units
from .correlated_stats import CorrelatedSummary, summarize_correlated
from .empirical import EmpiricalTemperatureData
from .simulation import assign_force_groups
from .trackers import compute_harmonic_bond_energy_kjmol

BUSSI_TAU_PS = 1.0
_PLATFORM_PREFERENCE = ("CUDA", "CPU", "Reference")

FULL_HEADER = (
    "temperature avg_temperature n_samples n_frames_production "
    "total_PE_hartree total_PE_std_hartree "
    "harmonic_hartree harmonic_std_hartree "
    "intramolecular_hartree intramolecular_std_hartree "
    "lj_hartree lj_std_hartree coulombic_hartree coulombic_std_hartree"
)

SLIM_HEADER = "temperature  harmonic_hartree  lj_hartree  coulombic_hartree"


@dataclass
class CalibrationRunSpec:
    """Per-temperature runtime overrides (picoseconds)."""

    temperature_K: float
    equil_ps: float
    prod_ps: float
    n_samples: int = 1000


@dataclass
class CalibrationRow:
    temperature_K: float
    avg_temperature_K: float
    n_samples: int
    total_PE_hartree: float
    total_PE_std_hartree: float
    harmonic_hartree: float
    harmonic_std_hartree: float
    lj_hartree: float
    lj_std_hartree: float
    coulombic_hartree: float = 0.0
    coulombic_std_hartree: float = 0.0
    harmonic_n_eff: float = 0.0
    lj_n_eff: float = 0.0
    harmonic_block_size: int = 1
    lj_block_size: int = 1

    @property
    def intramolecular_hartree(self) -> float:
        return self.harmonic_hartree

    @property
    def intramolecular_std_hartree(self) -> float:
        return self.harmonic_std_hartree


def _select_platform(platform_name: Optional[str] = None):
    import openmm

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
        except openmm.OpenMMException:
            continue

    raise RuntimeError("No usable OpenMM platform found (tried CUDA, CPU, Reference)")


def _kinetic_temperature_K(context, n_atoms: int) -> float:
    import openmm
    from openmm import unit

    if n_atoms <= 0:
        return 0.0
    state = context.getState(getEnergy=True)
    ke = state.getKineticEnergy().value_in_unit(unit.kilojoule_per_mole)
    return 2.0 * ke / (3.0 * n_atoms * Units.KB_KJMOL_PER_K)


def _sample_energy_snapshot(context, system) -> Tuple[float, float, float, float]:
    """Return (harmonic, total_PE, lj, coulombic) in Hartree."""
    import openmm
    from openmm import unit

    e_bond_kj = compute_harmonic_bond_energy_kjmol(context, system)
    total_pe_kj = context.getState(getEnergy=True).getPotentialEnergy().value_in_unit(
        unit.kilojoule_per_mole
    )
    e_nb_kj = total_pe_kj - e_bond_kj

    hartree = Units.KJMOL_TO_HARTREE
    return (
        e_bond_kj * hartree,
        total_pe_kj * hartree,
        e_nb_kj * hartree,
        0.0,
    )


def check_calibration_sanity(rows: Sequence[CalibrationRow]) -> bool:
    """Warn if calibration data looks unphysical."""
    issues = []
    for row in rows:
        if row.harmonic_hartree <= 0:
            issues.append(
                f"T={row.temperature_K:.0f} K: <V_bond>={row.harmonic_hartree:.6f} Ha "
                "(expected > 0)"
            )
        if row.temperature_K <= 100 and row.lj_hartree >= 0:
            issues.append(
                f"T={row.temperature_K:.0f} K: <V_LJ+C>={row.lj_hartree:.6f} Ha "
                "(expected < 0 at low T)"
            )

    if issues:
        print("WARNING: Calibration sanity check failed:")
        for issue in issues:
            print(f"  {issue}")
        return False
    print("Calibration sanity check passed.")
    return True


def load_completed_temperatures(output_file: Union[str, Path]) -> set:
    """Return set of target temperatures already present in a calibration file."""
    path = Path(output_file)
    if not path.exists():
        return set()

    completed = set()
    with path.open() as f:
        for line in f:
            line = line.strip()
            if not line or line.startswith("#") or line.startswith("temperature"):
                continue
            parts = line.split()
            if parts:
                try:
                    completed.add(float(parts[0]))
                except ValueError:
                    continue
    return completed


def write_full_calibration_file(
    rows: Sequence[CalibrationRow], output_file: Union[str, Path]
) -> None:
    path = Path(output_file)
    with path.open("w") as f:
        f.write("# Potential energy component calibration for fictive temperatures\n")
        f.write("# NVT equilibration + production; energies averaged over production samples\n")
        f.write("# *_std_hartree columns are reblocked SEM (Flyvbjerg/pyblock), not naive std\n")
        f.write("# Units: Hartree. intramolecular == harmonic bond energy (mKA diatomics).\n")
        f.write("# lj_hartree holds combined LJ+Coulomb (OpenMM NonbondedForce+PME).\n")
        f.write("#\n")
        f.write(f"# {FULL_HEADER}\n")
        f.write(f"{FULL_HEADER}\n")
        for row in rows:
            f.write(
                f"{row.temperature_K:.6f} {row.avg_temperature_K:.6f} {row.n_samples} "
                f"{row.n_samples} "
                f"{row.total_PE_hartree:.10f} {row.total_PE_std_hartree:.10f} "
                f"{row.harmonic_hartree:.10f} {row.harmonic_std_hartree:.10f} "
                f"{row.intramolecular_hartree:.10f} {row.intramolecular_std_hartree:.10f} "
                f"{row.lj_hartree:.10f} {row.lj_std_hartree:.10f} "
                f"{row.coulombic_hartree:.10f} {row.coulombic_std_hartree:.10f}\n"
            )


def write_slim_calibration_file(
    rows: Sequence[CalibrationRow], output_file: Union[str, Path]
) -> None:
    path = Path(output_file)
    with path.open("w") as f:
        f.write(f"{SLIM_HEADER}\n")
        for row in rows:
            f.write(
                f"{row.temperature_K:.1f}  {row.harmonic_hartree:.10f}  "
                f"{row.lj_hartree:.10f}  {row.coulombic_hartree:.10f}\n"
            )


def validate_calibration_file(
    full_output: Union[str, Path],
    slim_output: Optional[Union[str, Path]] = None,
) -> bool:
    """Load calibration with EmpiricalTemperatureData and print fit quality."""
    slim_path = slim_output or full_output
    ok = True
    for component, label in (("harmonic", "T_v"), ("lj_coulombic", "T_s")):
        try:
            emp = EmpiricalTemperatureData(str(slim_path), energy_component=component)
            params = emp.fit_params
            if component == "harmonic" and emp.has_extended_harmonic_fit:
                r2 = params["extended_harmonic"].get("r2", float("nan"))
                print(f"  {label} extended harmonic fit R² = {r2:.4f}")
            elif component == "lj_coulombic" and emp.has_extended_t35_fit:
                r2 = params["extended_t35"].get("r2", float("nan"))
                print(f"  {label} extended T^(3/5) fit R² = {r2:.4f}")
            else:
                print(f"  {label}: loaded {len(emp.temperatures or [])} temperature points")
            ok = emp.check_inversion_at_calibration_points(tol_K=15.0) and ok
        except Exception as exc:
            print(f"  WARNING: could not validate {label}: {exc}")
            ok = False
    return ok


def crosscheck_calibration_against_reference(
    candidate_file: Union[str, Path],
    reference_file: Union[str, Path],
    *,
    energy_tol_hartree: float = 0.05,
    temp_tol_K: float = 25.0,
) -> bool:
    """Compare self-generated calibration energies to the cav-hoomd reference table."""
    ref = EmpiricalTemperatureData(str(reference_file), energy_component="lj_coulombic")
    cand = EmpiricalTemperatureData(str(candidate_file), energy_component="lj_coulombic")

    if ref.temperatures is None or cand.temperatures is None:
        print("WARNING: cross-check skipped (missing temperature columns)")
        return False

    issues = []
    for T_ref, E_ref in zip(ref.temperatures, ref.energies):
        # Match nearest candidate temperature
        idx = int(np.argmin(np.abs(cand.temperatures - T_ref)))
        T_c = float(cand.temperatures[idx])
        E_c = float(cand.energies[idx])
        if abs(T_c - T_ref) > temp_tol_K:
            continue
        if abs(E_c - E_ref) > energy_tol_hartree:
            issues.append(
                f"T≈{T_ref:.0f} K: candidate E={E_c:.4f} Ha vs reference E={E_ref:.4f} Ha "
                f"(Δ={E_c - E_ref:+.4f} Ha)"
            )

    if issues:
        print("WARNING: calibration cross-check vs reference found mismatches:")
        for issue in issues[:10]:
            print(f"  {issue}")
        if len(issues) > 10:
            print(f"  ... and {len(issues) - 10} more")
        return False

    print(
        f"Calibration cross-check vs reference passed "
        f"({len(ref.temperatures)} reference temperatures compared)."
    )
    return True


def load_calibration_manifest(manifest_path: Union[str, Path]) -> List[CalibrationRunSpec]:
    """Load per-temperature run specs from plan_fictive_calibration JSON."""
    path = Path(manifest_path)
    payload = json.loads(path.read_text())
    entries = payload.get("entries", payload)
    specs: List[CalibrationRunSpec] = []
    for entry in entries:
        specs.append(
            CalibrationRunSpec(
                temperature_K=float(entry["temperature_K"]),
                equil_ps=float(entry["equil_ps"]),
                prod_ps=float(entry["prod_ps"]),
                n_samples=int(entry.get("n_samples", 1000)),
            )
        )
    return specs


def write_n_eff_report(
    diagnostics: Sequence[Dict],
    output_file: Union[str, Path],
) -> None:
    path = Path(output_file)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps({"entries": list(diagnostics)}, indent=2) + "\n")


def _summarize_component(samples: Sequence[float]) -> CorrelatedSummary:
    return summarize_correlated(np.asarray(samples, dtype=float))


def _check_n_eff_gates(
    temperature_K: float,
    harmonic: CorrelatedSummary,
    lj: CorrelatedSummary,
    *,
    warn_threshold: float = 10.0,
    fail_threshold: float = 5.0,
) -> bool:
    ok = True
    for label, summary in (("harmonic", harmonic), ("lj", lj)):
        if summary.n_eff < warn_threshold:
            print(
                f"  WARNING: T={temperature_K:.1f} K {label} N_eff={summary.n_eff:.1f} "
                f"(target >= {warn_threshold:.0f})"
            )
        if summary.n_eff < fail_threshold:
            print(
                f"  FAIL: T={temperature_K:.1f} K {label} N_eff={summary.n_eff:.1f} "
                f"(minimum {fail_threshold:.0f})"
            )
            ok = False
        if summary.sem > 0 and summary.sem_pymbar > 0:
            ratio = abs(summary.sem - summary.sem_pymbar) / summary.sem
            if ratio > 2.0:
                print(
                    f"  WARNING: T={temperature_K:.1f} K {label} pyblock/pymbar SEM "
                    f"disagree by factor {ratio:.1f}"
                )
    return ok


def run_nvt_energy_calibration(
    system_template_fn: Callable[[float], Tuple],
    temperatures_K: Sequence[float],
    *,
    prod_ns: float = 100.0,
    equil_ns: float = 10.0,
    n_samples: int = 1000,
    run_specs: Optional[Sequence[CalibrationRunSpec]] = None,
    dt_ps: float = 0.001,
    output_file: Union[str, Path] = "potential_energy_components_vs_temperature.txt",
    slim_output_file: Optional[Union[str, Path]] = None,
    write_full_output: bool = True,
    platform_name: Optional[str] = None,
    minimize_steps: int = 100,
    timeseries_dir: Optional[Union[str, Path]] = None,
    n_eff_report_file: Optional[Union[str, Path]] = None,
    resume: bool = False,
    bussi_tau_ps: float = BUSSI_TAU_PS,
) -> Path:
    """Run NVT calibration sweeps and write energy-vs-temperature tables.

    Parameters
    ----------
    system_template_fn : callable
        ``fn(T_K) -> (system, positions, n_atoms)`` — no cavity force.
    temperatures_K : sequence of float
        Bath temperatures to simulate (K).
    prod_ns : float
        Default production duration (nanoseconds) when ``run_specs`` is absent.
    equil_ns : float
        Default equilibration duration (nanoseconds).
    n_samples : int
        Default number of evenly spaced energy samples during production.
    run_specs : sequence of CalibrationRunSpec, optional
        Per-temperature overrides (from ``plan_fictive_calibration`` manifest).
    """
    import openmm
    from openmm import unit

    output_path = Path(output_file)
    existing_rows: List[CalibrationRow] = []
    existing_diagnostics: List[Dict] = []

    if n_eff_report_file is not None and Path(n_eff_report_file).exists():
        try:
            existing_diagnostics = json.loads(
                Path(n_eff_report_file).read_text()
            ).get("entries", [])
        except json.JSONDecodeError:
            existing_diagnostics = []

    if run_specs is not None:
        spec_by_T = {round(s.temperature_K, 6): s for s in run_specs}
        temperatures = [s.temperature_K for s in run_specs]
    else:
        spec_by_T = {}
        temperatures = list(temperatures_K)

    if resume and write_full_output and output_path.exists():
        completed = load_completed_temperatures(output_path)
        if completed:
            print(f"Resume: skipping {len(completed)} temperatures already in {output_path}")
        temperatures = [T for T in temperatures if T not in completed]
        with output_path.open() as f:
            for line in f:
                line = line.strip()
                if not line or line.startswith("#") or line.startswith("temperature"):
                    continue
                parts = line.split()
                if len(parts) < 15:
                    continue
                existing_rows.append(
                    CalibrationRow(
                        temperature_K=float(parts[0]),
                        avg_temperature_K=float(parts[1]),
                        n_samples=int(parts[2]),
                        total_PE_hartree=float(parts[4]),
                        total_PE_std_hartree=float(parts[5]),
                        harmonic_hartree=float(parts[6]),
                        harmonic_std_hartree=float(parts[7]),
                        lj_hartree=float(parts[10]),
                        lj_std_hartree=float(parts[11]),
                        coulombic_hartree=float(parts[12]),
                        coulombic_std_hartree=float(parts[13]),
                    )
                )
        existing_diagnostics = [
            d for d in existing_diagnostics
            if any(abs(d["temperature_K"] - t) < 0.05 for t in completed)
        ]

    if timeseries_dir is not None:
        Path(timeseries_dir).mkdir(parents=True, exist_ok=True)

    print("\n=== NVT energy calibration ===")
    print(f"  Temperatures remaining: {len(temperatures)}")

    new_rows: List[CalibrationRow] = []
    new_diagnostics: List[Dict] = []

    for T in temperatures:
        spec = spec_by_T.get(round(float(T), 6))
        if spec is not None:
            equil_ps = spec.equil_ps
            prod_ps = spec.prod_ps
            n_samples_T = spec.n_samples
        else:
            equil_ps = equil_ns * 1000.0
            prod_ps = prod_ns * 1000.0
            n_samples_T = n_samples

        prod_steps = max(1, int(prod_ps / dt_ps))
        equil_steps = max(0, int(equil_ps / dt_ps))
        sample_interval = max(1, prod_steps // max(1, n_samples_T))

        print(f"\n--- T = {T:.2f} K ---")
        print(
            f"  Production: {prod_ps / 1000.0:.4f} ns ({prod_steps} steps), "
            f"equil: {equil_ps / 1000.0:.4f} ns"
        )
        print(f"  Samples: {n_samples_T} (interval {sample_interval * dt_ps:.3f} ps)")

        system, positions, n_atoms = system_template_fn(T)

        bussi = openmm.BussiThermostat(T, bussi_tau_ps)
        system.addForce(bussi)
        assign_force_groups(system)

        integrator = openmm.VerletIntegrator(dt_ps * unit.picosecond)
        platform = _select_platform(platform_name)
        context = openmm.Context(system, integrator, platform)
        context.setPositions(positions)
        context.setVelocitiesToTemperature(T * unit.kelvin)

        if minimize_steps > 0:
            openmm.LocalEnergyMinimizer.minimize(context, maxIterations=minimize_steps)

        if equil_steps > 0:
            integrator.step(equil_steps)
            print(f"  Equilibrated {equil_ps:.1f} ps ({equil_steps} steps)")

        harmonic_samples: List[float] = []
        total_samples: List[float] = []
        lj_samples: List[float] = []
        coulomb_samples: List[float] = []
        temp_samples: List[float] = []

        for _ in range(n_samples_T):
            integrator.step(sample_interval)
            e_h, e_tot, e_lj, e_c = _sample_energy_snapshot(context, system)
            harmonic_samples.append(e_h)
            total_samples.append(e_tot)
            lj_samples.append(e_lj)
            coulomb_samples.append(e_c)
            temp_samples.append(_kinetic_temperature_K(context, n_atoms))

        if timeseries_dir is not None:
            ts_path = Path(timeseries_dir) / f"calibration_T{T:.1f}K.csv"
            with ts_path.open("w") as tf:
                tf.write(
                    "sample_index,time_ps,harmonic_hartree,total_PE_hartree,"
                    "intramolecular_hartree,lj_hartree,coulombic_hartree,T_kinetic_K\n"
                )
                t0 = equil_ps
                for i in range(n_samples_T):
                    t_ps = t0 + (i + 1) * sample_interval * dt_ps
                    tf.write(
                        f"{i},{t_ps:.6f},{harmonic_samples[i]:.10f},{total_samples[i]:.10f},"
                        f"{harmonic_samples[i]:.10f},{lj_samples[i]:.10f},"
                        f"{coulomb_samples[i]:.10f},{temp_samples[i]:.6f}\n"
                    )

        total_summary = _summarize_component(total_samples)
        harmonic_summary = _summarize_component(harmonic_samples)
        lj_summary = _summarize_component(lj_samples)
        coulomb_summary = _summarize_component(coulomb_samples)

        _check_n_eff_gates(float(T), harmonic_summary, lj_summary)

        row = CalibrationRow(
            temperature_K=float(T),
            avg_temperature_K=float(np.mean(temp_samples)),
            n_samples=n_samples_T,
            total_PE_hartree=total_summary.mean,
            total_PE_std_hartree=total_summary.sem,
            harmonic_hartree=harmonic_summary.mean,
            harmonic_std_hartree=harmonic_summary.sem,
            lj_hartree=lj_summary.mean,
            lj_std_hartree=lj_summary.sem,
            coulombic_hartree=coulomb_summary.mean,
            coulombic_std_hartree=coulomb_summary.sem,
            harmonic_n_eff=harmonic_summary.n_eff,
            lj_n_eff=lj_summary.n_eff,
            harmonic_block_size=harmonic_summary.block_size,
            lj_block_size=lj_summary.block_size,
        )
        new_rows.append(row)
        new_diagnostics.append(
            {
                "temperature_K": float(T),
                "harmonic": asdict(harmonic_summary),
                "lj": asdict(lj_summary),
                "total_PE": asdict(total_summary),
            }
        )
        print(
            f"  <V_bond> = {row.harmonic_hartree:.6f} ± {row.harmonic_std_hartree:.6f} Ha "
            f"(N_eff={harmonic_summary.n_eff:.1f}, block={harmonic_summary.block_size})"
        )
        print(
            f"  <V_LJ+C> = {row.lj_hartree:.6f} ± {row.lj_std_hartree:.6f} Ha "
            f"(N_eff={lj_summary.n_eff:.1f}, block={lj_summary.block_size})"
        )
        print(f"  <T_kin>  = {row.avg_temperature_K:.1f} K (target {T:.1f} K)")

        del context, integrator

        all_rows = existing_rows + new_rows
        if write_full_output:
            write_full_calibration_file(all_rows, output_path)
        if slim_output_file is not None:
            write_slim_calibration_file(all_rows, slim_output_file)
        if n_eff_report_file is not None:
            write_n_eff_report(existing_diagnostics + new_diagnostics, n_eff_report_file)

    all_rows = existing_rows + new_rows
    if not all_rows:
        print("No calibration rows produced.")
        return output_path if write_full_output else Path(slim_output_file or output_path)

    if write_full_output:
        write_full_calibration_file(all_rows, output_path)
    if slim_output_file is not None:
        write_slim_calibration_file(all_rows, slim_output_file)
    if n_eff_report_file is not None and (existing_diagnostics or new_diagnostics):
        write_n_eff_report(existing_diagnostics + new_diagnostics, n_eff_report_file)

    check_calibration_sanity(all_rows)
    result_path = output_path if write_full_output else Path(slim_output_file)
    print(f"\nCalibration written to {result_path}")
    if slim_output_file and write_full_output:
        print(f"Slim calibration written to {slim_output_file}")
    if n_eff_report_file is not None:
        print(f"N_eff diagnostics written to {n_eff_report_file}")

    return result_path


def run_legacy_equilibrium_calibration(
    system_template_fn: Callable[[float], Tuple],
    temperatures: Sequence[float],
    run_ps: float = 500.0,
    dt_ps: float = 0.001,
    output_file: Union[str, Path] = "calibration_data.txt",
    platform_name: Optional[str] = None,
) -> Path:
    """Backward-compatible wrapper: 20/80 equil/prod split of *run_ps* (picoseconds)."""
    equil_ns = 0.2 * run_ps / 1000.0
    prod_ns = 0.8 * run_ps / 1000.0
    sample_interval_ps = 0.1
    prod_ps = prod_ns * 1000.0
    n_samples = max(1, int(prod_ps // sample_interval_ps))

    return run_nvt_energy_calibration(
        system_template_fn,
        temperatures,
        prod_ns=prod_ns,
        equil_ns=equil_ns,
        n_samples=n_samples,
        dt_ps=dt_ps,
        output_file=output_file,
        slim_output_file=output_file,
        write_full_output=False,
        platform_name=platform_name,
    )
