#!/usr/bin/env python3
"""Orchestrate OpenMM vs cav-hoomd adaptive parity diagnosis (plan phases 1-4)."""

from __future__ import annotations

import argparse
import json
import sys
import traceback
from dataclasses import asdict
from pathlib import Path
from typing import Any

import numpy as np

_SCRIPT_DIR = Path(__file__).resolve().parent
_C2F_ROOT = _SCRIPT_DIR.parent
if str(_C2F_ROOT) not in sys.path:
    sys.path.insert(0, str(_C2F_ROOT))

from run_cavity_equilibrium import run_cavity_equilibrium  # noqa: E402

from config import (  # noqa: E402
    DIPOLE_INTERVAL_PS,
    FKT_KMAG_NM_INV,
    FKT_MAX_REFS,
    FKT_NUM_WAVEVECTORS,
    FKT_OUTPUT_PERIOD_PS,
    FKT_REF_INTERVAL_PS,
    FREQUENCY_CM1,
    INITIAL_STATE,
    IR_WINDOWS,
    SWITCH_TIME_PS,
    TEMPERATURE_K,
)
from openmm.cavitymd.adaptive import (  # noqa: E402
    AdaptiveParityConfig,
    cavhoomd_runtime_parity_config,
    default_parity_config,
)

DIAG_DIR = _SCRIPT_DIR / "diagnose_fkt"
PRE_SWITCH_LAMBDAS = [0.01, 0.016667, 0.023333, 0.03]
ABLATION_LAMBDAS = [0.01, 0.03]
STABILITY_T_KIN_MAX_K = 5000.0
QUICK_RUNTIME_BY_LAM_PS: dict[float, float] = {0.03: 250.0, 0.01: 800.0}


def _runtime_for_lambda(lam: float, default_runtime_ps: float, *, quick: bool) -> float:
    if quick:
        return QUICK_RUNTIME_BY_LAM_PS.get(lam, default_runtime_ps)
    return default_runtime_ps


def _cuda_available() -> bool:
    try:
        import os
        import openmm

        plugin_dir = os.environ.get("OPENMM_PLUGIN_DIR")
        if plugin_dir:
            openmm.Platform.loadPluginsFromDirectory(plugin_dir)
        for candidate in ("CUDA", "CPU", "Reference"):
            try:
                openmm.Platform.getPlatformByName(candidate)
                if candidate == "CUDA":
                    return True
            except Exception:
                continue
        return False
    except Exception:
        return False


def _select_platform_name(requested: str | None) -> str | None:
    import os
    import openmm

    plugin_dir = os.environ.get("OPENMM_PLUGIN_DIR")
    if plugin_dir:
        openmm.Platform.loadPluginsFromDirectory(plugin_dir)
    if requested:
        try:
            openmm.Platform.getPlatformByName(requested)
            return requested
        except Exception:
            pass
    for candidate in ("CUDA", "CPU", "Reference"):
        try:
            openmm.Platform.getPlatformByName(candidate)
            return candidate
        except Exception:
            continue
    return None


def _read_energies_csv(path: Path) -> dict[str, np.ndarray]:
    data = np.genfromtxt(path, delimiter=",", names=True)
    return {name: np.atleast_1d(data[name]) for name in data.dtype.names}


def _trial_is_complete(output_prefix: Path, runtime_ps: float) -> bool:
    csv_path = Path(f"{output_prefix}_energies.csv")
    if not csv_path.is_file():
        return False
    try:
        csv = _read_energies_csv(csv_path)
        return float(csv["time_ps"][-1]) >= runtime_ps - 1.0
    except Exception:
        return False


def _record_from_csv(
    *,
    label: str,
    output_prefix: Path,
    lambda_coupling: float,
    runtime_ps: float,
    adaptive: bool,
    enable_fkt: bool,
    dipole_windows: list[tuple[float, float]] | None,
    parity_config: AdaptiveParityConfig | None,
) -> dict[str, Any]:
    csv = _read_energies_csv(Path(f"{output_prefix}_energies.csv"))
    t_kin = csv["T_kinetic_K"]
    times = csv["time_ps"]
    max_t = float(np.max(t_kin))
    final_t = float(times[-1])
    return {
        "label": label,
        "lambda": lambda_coupling,
        "runtime_ps": runtime_ps,
        "adaptive": adaptive,
        "enable_fkt": enable_fkt,
        "dipole": dipole_windows is not None,
        "parity_config": asdict(parity_config) if parity_config else None,
        "ok": (
            np.all(np.isfinite(t_kin))
            and max_t < STABILITY_T_KIN_MAX_K
            and final_t >= runtime_ps - 1.0
        ),
        "max_T_kin_K": max_t,
        "final_time_ps": final_t,
        "error": None,
        "skipped": True,
    }


def _run_trial(
    *,
    label: str,
    output_prefix: Path,
    seed: int,
    lambda_coupling: float,
    runtime_ps: float,
    adaptive: bool,
    enable_fkt: bool,
    dipole_windows: list[tuple[float, float]] | None,
    parity_config: AdaptiveParityConfig | None = None,
    platform_name: str | None = "CUDA",
    skip_complete: bool = True,
) -> dict[str, Any]:
    output_prefix.parent.mkdir(parents=True, exist_ok=True)
    if skip_complete and _trial_is_complete(output_prefix, runtime_ps):
        print(f"  Skip complete: {label} ({output_prefix}_energies.csv)", flush=True)
        return _record_from_csv(
            label=label,
            output_prefix=output_prefix,
            lambda_coupling=lambda_coupling,
            runtime_ps=runtime_ps,
            adaptive=adaptive,
            enable_fkt=enable_fkt,
            dipole_windows=dipole_windows,
            parity_config=parity_config,
        )
    record: dict[str, Any] = {
        "label": label,
        "lambda": lambda_coupling,
        "runtime_ps": runtime_ps,
        "adaptive": adaptive,
        "enable_fkt": enable_fkt,
        "dipole": dipole_windows is not None,
        "parity_config": asdict(parity_config) if parity_config else None,
        "ok": False,
        "max_T_kin_K": None,
        "final_time_ps": None,
        "error": None,
    }
    try:
        run_cavity_equilibrium(
            temperature_K=TEMPERATURE_K,
            runtime_ps=runtime_ps,
            lambda_coupling=lambda_coupling,
            include_dipole_self_energy=True,
            output_prefix=str(output_prefix),
            seed=seed,
            sample_interval_ps=1.0,
            initial_state=INITIAL_STATE,
            platform_name=platform_name,
            finite_q=False,
            omega_c_cm1=FREQUENCY_CM1,
            snapshot_interval_ps=0.0,
            coupling_start_ps=SWITCH_TIME_PS,
            resample_velocities=True,
            enable_fkt=enable_fkt,
            fkt_kmag_nm_inv=FKT_KMAG_NM_INV,
            fkt_num_wavevectors=FKT_NUM_WAVEVECTORS,
            fkt_ref_interval_ps=FKT_REF_INTERVAL_PS,
            fkt_output_period_ps=FKT_OUTPUT_PERIOD_PS,
            fkt_max_refs=FKT_MAX_REFS,
            fkt_start_ps=SWITCH_TIME_PS,
            fkt_sites="atomic",
            dipole_windows=dipole_windows,
            dipole_interval_ps=DIPOLE_INTERVAL_PS,
            num_molecules=250,
            adaptive=adaptive,
            adaptive_parity_config=parity_config,
            no_resume=True,
        )
        csv = _read_energies_csv(Path(f"{output_prefix}_energies.csv"))
        t_kin = csv["T_kinetic_K"]
        times = csv["time_ps"]
        record["max_T_kin_K"] = float(np.max(t_kin))
        record["final_time_ps"] = float(times[-1])
        record["ok"] = (
            np.all(np.isfinite(t_kin))
            and record["max_T_kin_K"] < STABILITY_T_KIN_MAX_K
            and record["final_time_ps"] >= runtime_ps - 1.0
        )
    except Exception as exc:
        record["error"] = f"{type(exc).__name__}: {exc}"
        record["traceback"] = traceback.format_exc()
    return record


def phase_pre_switch_determinism(
    *,
    seed: int,
    runtime_ps: float,
    out_dir: Path,
    platform_name: str | None,
) -> dict[str, Any]:
    """Phase 1: serial λ sweep; compare pre-switch energy parity."""
    results: dict[str, Any] = {"phase": "pre_switch_determinism", "runs": [], "parity": {}}
    csv_by_lam: dict[float, dict[str, np.ndarray]] = {}

    for lam in PRE_SWITCH_LAMBDAS:
        prefix = out_dir / f"pre_switch_lam{str(lam).replace('.', 'p')}_seed{seed:04d}"
        rec = _run_trial(
            label=f"pre_switch_lam{lam:g}",
            output_prefix=prefix,
            seed=seed,
            lambda_coupling=lam,
            runtime_ps=runtime_ps,
            adaptive=True,
            enable_fkt=False,
            dipole_windows=None,
            parity_config=default_parity_config(),
            platform_name=platform_name,
        )
        results["runs"].append(rec)
        if rec.get("ok"):
            csv_by_lam[lam] = _read_energies_csv(Path(f"{prefix}_energies.csv"))

    if len(csv_by_lam) >= 2:
        ref_lam = PRE_SWITCH_LAMBDAS[0]
        ref = csv_by_lam.get(ref_lam)
        if ref is not None:
            mask = ref["time_ps"] <= SWITCH_TIME_PS + 1e-9
            for lam, data in csv_by_lam.items():
                if lam == ref_lam:
                    continue
                m = mask & (data["time_ps"] <= SWITCH_TIME_PS + 1e-9)
                n = min(int(np.sum(mask)), int(np.sum(m)))
                if n == 0:
                    continue
                t_kin_diff = np.max(
                    np.abs(ref["T_kinetic_K"][:n] - data["T_kinetic_K"][:n])
                )
                e_kin_diff = np.max(
                    np.abs(ref["E_kinetic_kjmol"][:n] - data["E_kinetic_kjmol"][:n])
                )
                results["parity"][str(lam)] = {
                    "max_dT_kin_K_pre_switch": float(t_kin_diff),
                    "max_dE_kin_pre_switch": float(e_kin_diff),
                    "lambda_independent_pre_switch": t_kin_diff < 1e-3 and e_kin_diff < 1e-6,
                }
    return results


def phase_ablation_matrix(
    *,
    seed: int,
    runtime_ps: float,
    out_dir: Path,
    platform_name: str | None,
    quick: bool = False,
) -> dict[str, Any]:
    """Phase 2: 2×2 FKT × dipole grid for adaptive vs fixed control."""
    results: dict[str, Any] = {"phase": "ablation_matrix", "runs": []}
    grid = [
        ("adapt_no_fkt_no_dipole", True, False, None),
        ("adapt_fkt_no_dipole", True, True, None),
        ("adapt_no_fkt_dipole", True, False, IR_WINDOWS),
        ("adapt_prod", True, True, IR_WINDOWS),
        ("fixed_no_fkt_no_dipole", False, False, None),
        ("fixed_prod", False, True, IR_WINDOWS),
    ]
    for lam in ABLATION_LAMBDAS:
        trial_runtime = _runtime_for_lambda(lam, runtime_ps, quick=quick)
        for tag, adaptive, fkt, dipole in grid:
            prefix = out_dir / f"abl_{tag}_lam{str(lam).replace('.', 'p')}_seed{seed:04d}"
            rec = _run_trial(
                label=f"{tag}_lam{lam:g}",
                output_prefix=prefix,
                seed=seed,
                lambda_coupling=lam,
                runtime_ps=trial_runtime,
                adaptive=adaptive,
                enable_fkt=fkt,
                dipole_windows=dipole,
                parity_config=default_parity_config() if adaptive else None,
                platform_name=platform_name,
            )
            results["runs"].append(rec)
    return results


def phase_parity_knob_sweep(
    *,
    seed: int,
    runtime_ps: float,
    out_dir: Path,
    platform_name: str | None,
    quick: bool = False,
) -> dict[str, Any]:
    """Phase 4: toggle one parity knob at a time on λ=0.01 and λ=0.03."""
    results: dict[str, Any] = {"phase": "parity_knob_sweep", "runs": []}
    knobs: list[tuple[str, AdaptiveParityConfig]] = [
        ("baseline", default_parity_config()),
        ("no_pre_switch_guard", AdaptiveParityConfig(pre_switch_guard_ps=0.0)),
        ("dt_slew_only", AdaptiveParityConfig(dt_slew_threshold=0.1)),
        (
            "runtime_strict_et001",
            cavhoomd_runtime_parity_config(error_tolerance=0.01, initial_fraction=1e-5),
        ),
        (
            "runtime_strict_et100",
            cavhoomd_runtime_parity_config(error_tolerance=1.0, initial_fraction=1e-5),
        ),
        (
            "runtime_strict_et500",
            cavhoomd_runtime_parity_config(error_tolerance=5.0, initial_fraction=1e-5),
        ),
        (
            "no_guard_dt_slew",
            AdaptiveParityConfig(pre_switch_guard_ps=0.0, dt_slew_threshold=0.1),
        ),
        (
            "paper_f0_legacy",
            AdaptiveParityConfig(
                f0=1e-3,
                pre_switch_guard_ps=5.0,
                dt_slew_threshold=0.0,
                absolute_error_tolerance=None,
            ),
        ),
        (
            "full_runtime_parity",
            AdaptiveParityConfig(
                f0=1e-5,
                pre_switch_guard_ps=0.0,
                dt_slew_threshold=0.1,
                max_timestep_change_factor=1.5,
                absolute_error_tolerance=1.0,
            ),
        ),
    ]

    for lam in ABLATION_LAMBDAS:
        trial_runtime = _runtime_for_lambda(lam, runtime_ps, quick=quick)
        for tag, cfg in knobs:
            prefix = out_dir / f"knob_{tag}_lam{str(lam).replace('.', 'p')}_seed{seed:04d}"
            rec = _run_trial(
                label=f"{tag}_lam{lam:g}",
                output_prefix=prefix,
                seed=seed,
                lambda_coupling=lam,
                runtime_ps=trial_runtime,
                adaptive=True,
                enable_fkt=True,
                dipole_windows=IR_WINDOWS,
                parity_config=cfg,
                platform_name=platform_name,
            )
            results["runs"].append(rec)
    return results


def phase_hoomd_reference_note(out_dir: Path) -> dict[str, Any]:
    """Phase 3: document HOOMD comparison command (requires HOOMD env)."""
    script = _C2F_ROOT / "diagnose_adaptive_switch.py"
    note = {
        "phase": "hoomd_short_run",
        "status": "manual_or_separate_env",
        "openmm_diagnostic_script": str(script),
        "recommended_commands": [
            (
                f"{sys.executable} {script} --seed 42 --lambda 0.03 "
                f"--window-before-ps 110 --window-after-ps 60 "
                f"--sample-interval-ps 0.1 "
                f"--output {out_dir}/openmm_dt_lam003_turnon.csv --platform CUDA"
            ),
            (
                f"{sys.executable} {script} --seed 42 --lambda 0.01 "
                f"--coupling-start-ps 200 --window-before-ps 10 --window-after-ps 560 "
                f"--sample-interval-ps 1.0 "
                f"--output {out_dir}/openmm_dt_lam001_aged.csv --platform CUDA"
            ),
        ],
        "hoomd_reference": (
            "Run cav-hoomd CavityMDSimulation with error_tolerance in {0.01,1.0,5.0}, "
            "initial_fraction=1e-5, switch_time_ps=200, dynamic coupling detection ON; "
            "compare Adaptive.timestep and Adaptive.error_tolerance logs to OpenMM CSV."
        ),
    }
    return note


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--pre-switch-runtime-ps", type=float, default=250.0)
    parser.add_argument("--ablation-runtime-ps", type=float, default=2500.0)
    parser.add_argument("--knob-runtime-ps", type=float, default=2500.0)
    parser.add_argument("--output-dir", type=Path, default=DIAG_DIR / "parity_runs")
    parser.add_argument("--platform", default="CUDA")
    parser.add_argument(
        "--phases",
        nargs="+",
        default=["pre_switch", "ablation", "knobs", "hoomd"],
        choices=["pre_switch", "ablation", "knobs", "hoomd", "all"],
    )
    parser.add_argument("--quick", action="store_true", help="Short runtimes for smoke testing")
    args = parser.parse_args()

    if args.quick:
        args.ablation_runtime_ps = 800.0
        args.knob_runtime_ps = 800.0

    platform = _select_platform_name(args.platform)
    if platform is None:
        raise RuntimeError("No OpenMM platform available (set OPENMM_PLUGIN_DIR)")

    out_dir = args.output_dir
    out_dir.mkdir(parents=True, exist_ok=True)
    phases = set(args.phases)
    if "all" in phases:
        phases = {"pre_switch", "ablation", "knobs", "hoomd"}

    report: dict[str, Any] = {
        "seed": args.seed,
        "platform": platform,
        "phases": {},
    }

    if "pre_switch" in phases:
        print("=== Phase 1: pre-switch determinism ===", flush=True)
        report["phases"]["pre_switch"] = phase_pre_switch_determinism(
            seed=args.seed,
            runtime_ps=args.pre_switch_runtime_ps,
            out_dir=out_dir / "phase1",
            platform_name=platform,
        )

    if "ablation" in phases:
        print("=== Phase 2: ablation matrix ===", flush=True)
        report["phases"]["ablation"] = phase_ablation_matrix(
            seed=args.seed,
            runtime_ps=args.ablation_runtime_ps,
            out_dir=out_dir / "phase2",
            platform_name=platform,
            quick=args.quick,
        )

    if "knobs" in phases:
        print("=== Phase 4: parity knob sweep ===", flush=True)
        report["phases"]["knobs"] = phase_parity_knob_sweep(
            seed=args.seed,
            runtime_ps=args.knob_runtime_ps,
            out_dir=out_dir / "phase4",
            platform_name=platform,
            quick=args.quick,
        )

    if "hoomd" in phases:
        print("=== Phase 3: HOOMD reference notes ===", flush=True)
        report["phases"]["hoomd"] = phase_hoomd_reference_note(out_dir / "phase3")
        for cmd in report["phases"]["hoomd"]["recommended_commands"]:
            print(f"  {cmd}")

    out_json = out_dir / "parity_diagnosis_report.json"
    out_json.write_text(json.dumps(report, indent=2), encoding="utf-8")
    print(f"\nWrote {out_json}")

    n_fail = 0
    for phase_name, phase_data in report["phases"].items():
        if phase_name == "hoomd":
            continue
        for run in phase_data.get("runs", []):
            if not run.get("ok"):
                n_fail += 1
                print(f"FAIL {run['label']}: {run.get('error') or run.get('max_T_kin_K')}")

    if n_fail:
        print(f"\n{n_fail} trial(s) failed — see {out_json}")
    else:
        print("\nAll trials passed.")


if __name__ == "__main__":
    main()
