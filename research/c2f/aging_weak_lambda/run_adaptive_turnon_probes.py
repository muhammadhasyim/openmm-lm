#!/usr/bin/env python3
"""500 ps turn-on probes: baseline parity, VariableVerlet RMS, production defaults."""

from __future__ import annotations

import argparse
import json
import sys
import traceback
from dataclasses import asdict
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import numpy as np

_SCRIPT_DIR = Path(__file__).resolve().parent
_C2F_ROOT = _SCRIPT_DIR.parent
if str(_C2F_ROOT) not in sys.path:
    sys.path.insert(0, str(_C2F_ROOT))

from run_cavity_equilibrium import run_cavity_equilibrium  # noqa: E402

from config import INITIAL_STATE, SWITCH_TIME_PS, TEMPERATURE_K  # noqa: E402
from openmm.cavitymd.adaptive import (  # noqa: E402
    AdaptiveParityConfig,
    cavhoomd_runtime_parity_config,
    default_parity_config,
)

STABILITY_T_KIN_MAX_K = 5000.0
PROBE_RUNTIME_PS = 500.0
PROBE_LAMBDAS = (0.01, 0.03)
PROBE_SEED = 42


def _utc_now() -> str:
    return datetime.now(timezone.utc).isoformat().replace("+00:00", "Z")


def _probe_configs() -> dict[str, dict[str, Any]]:
    return {
        "baseline_parity_eps1": {
            "parity_config": cavhoomd_runtime_parity_config(
                error_tolerance=1.0,
                initial_fraction=1e-5,
            ),
            "use_variable_verlet": False,
        },
        "variable_verlet_rms": {
            "parity_config": default_parity_config(),
            "use_variable_verlet": True,
        },
        "production_calibrated": {
            "parity_config": default_parity_config(),
            "use_variable_verlet": False,
        },
    }


def _run_probe(
    *,
    label: str,
    output_prefix: Path,
    lambda_coupling: float,
    parity_config: AdaptiveParityConfig,
    use_variable_verlet: bool,
    platform_name: str | None,
    runtime_ps: float,
) -> dict[str, Any]:
    record: dict[str, Any] = {
        "label": label,
        "lambda": lambda_coupling,
        "seed": PROBE_SEED,
        "runtime_ps": runtime_ps,
        "use_variable_verlet": use_variable_verlet,
        "parity_config": asdict(parity_config),
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
            seed=PROBE_SEED,
            sample_interval_ps=1.0,
            initial_state=INITIAL_STATE,
            platform_name=platform_name,
            finite_q=False,
            coupling_start_ps=SWITCH_TIME_PS,
            resample_velocities=True,
            enable_fkt=False,
            num_molecules=250,
            adaptive=True,
            adaptive_parity_config=parity_config,
            use_variable_verlet=use_variable_verlet,
            no_resume=True,
        )
        csv = np.genfromtxt(
            f"{output_prefix}_energies.csv", delimiter=",", names=True
        )
        t_kin = np.atleast_1d(csv["T_kinetic_K"])
        times = np.atleast_1d(csv["time_ps"])
        record["max_T_kin_K"] = float(np.max(t_kin))
        record["final_time_ps"] = float(times[-1])
        record["ok"] = (
            bool(np.all(np.isfinite(t_kin)))
            and record["max_T_kin_K"] < STABILITY_T_KIN_MAX_K
            and record["final_time_ps"] >= runtime_ps - 2.0
        )
    except Exception as exc:
        record["error"] = f"{type(exc).__name__}: {exc}"
        record["traceback"] = traceback.format_exc()
    return record


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--out-dir",
        type=Path,
        default=_SCRIPT_DIR / "diagnose_fkt" / "turnon_probes",
    )
    parser.add_argument("--platform", default=None)
    parser.add_argument("--runtime-ps", type=float, default=PROBE_RUNTIME_PS)
    parser.add_argument(
        "--configs",
        nargs="+",
        default=list(_probe_configs()),
        help="Probe config keys to run",
    )
    args = parser.parse_args()

    args.out_dir.mkdir(parents=True, exist_ok=True)
    configs = _probe_configs()
    results: dict[str, Any] = {
        "timestamp": _utc_now(),
        "runtime_ps": args.runtime_ps,
        "switch_time_ps": SWITCH_TIME_PS,
        "seed": PROBE_SEED,
        "probes": [],
    }

    for cfg_name in args.configs:
        if cfg_name not in configs:
            raise SystemExit(f"Unknown config: {cfg_name}")
        cfg = configs[cfg_name]
        for lam in PROBE_LAMBDAS:
            lam_tag = str(lam).replace(".", "p")
            prefix = args.out_dir / f"{cfg_name}_lam{lam_tag}_seed{PROBE_SEED:04d}"
            rec = _run_probe(
                label=f"{cfg_name}_lam{lam:g}",
                output_prefix=prefix,
                lambda_coupling=lam,
                parity_config=cfg["parity_config"],
                use_variable_verlet=bool(cfg["use_variable_verlet"]),
                platform_name=args.platform,
                runtime_ps=args.runtime_ps,
            )
            results["probes"].append(rec)
            status = "OK" if rec["ok"] else "FAIL"
            print(
                f"{status} {rec['label']}: max_T_kin={rec.get('max_T_kin_K')} "
                f"t_end={rec.get('final_time_ps')} err={rec.get('error')}"
            )

    out_json = args.out_dir / "turnon_probe_results.json"
    out_json.write_text(json.dumps(results, indent=2) + "\n", encoding="utf-8")
    print(f"Wrote {out_json}")
    failed = sum(1 for p in results["probes"] if not p["ok"])
    if failed:
        raise SystemExit(f"{failed} probe(s) failed")


if __name__ == "__main__":
    main()
