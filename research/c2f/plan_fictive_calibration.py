#!/usr/bin/env python
"""Plan τ-scaled fictive-temperature calibration runtimes."""

from __future__ import annotations

import argparse
import json
import sys
from dataclasses import asdict, dataclass
from pathlib import Path

import numpy as np

_SCRIPT_DIR = Path(__file__).resolve().parent
if str(_SCRIPT_DIR) not in sys.path:
    sys.path.insert(0, str(_SCRIPT_DIR))

from analyze_material_time import TauSModel  # noqa: E402

DEFAULT_RELAX = Path(
    "/scratch/mh7373/projects/cav-hoomd/relaxation_times_vs_temperature.txt"
)
DEFAULT_OUT = _SCRIPT_DIR / "calibration_output" / "plan.json"

PROD_TAU_FACTOR = 100.0
EQUIL_TAU_FACTOR = 10.0
N_SAMPLES = 1000


@dataclass
class CalibrationPlanEntry:
    temperature_K: float
    tau_s_ps: float
    equil_ps: float
    prod_ps: float
    n_samples: int
    sample_interval_ps: float

    @property
    def equil_ns(self) -> float:
        return self.equil_ps / 1000.0

    @property
    def prod_ns(self) -> float:
        return self.prod_ps / 1000.0


def uniform_temperature_grid(
    *,
    temperature_min: float = 65.0,
    temperature_max: float = 450.0,
    n_temperatures: int = 30,
) -> np.ndarray:
    """Uniform grid from T_min to T_max (inclusive endpoints)."""
    if n_temperatures < 2:
        raise ValueError("n_temperatures must be at least 2")
    return np.linspace(temperature_min, temperature_max, n_temperatures)


def build_plan(
    tau_model: TauSModel,
    temperatures_K: np.ndarray,
    *,
    prod_tau_factor: float = PROD_TAU_FACTOR,
    equil_tau_factor: float = EQUIL_TAU_FACTOR,
    n_samples: int = N_SAMPLES,
) -> list[CalibrationPlanEntry]:
    entries: list[CalibrationPlanEntry] = []
    for T in temperatures_K:
        tau = float(tau_model.tau_of_T(T))
        prod_ps = prod_tau_factor * tau
        equil_ps = equil_tau_factor * tau
        sample_interval = prod_ps / n_samples
        entries.append(
            CalibrationPlanEntry(
                temperature_K=float(T),
                tau_s_ps=tau,
                equil_ps=equil_ps,
                prod_ps=prod_ps,
                n_samples=n_samples,
                sample_interval_ps=sample_interval,
            )
        )
    return entries


def estimate_wall_time_s(
    entries: list[CalibrationPlanEntry],
    ps_per_s: float,
    n_workers: int = 8,
) -> float:
    if ps_per_s <= 0:
        return float("inf")
    total_ps = sum(e.equil_ps + e.prod_ps for e in entries)
    return total_ps / ps_per_s / max(1, n_workers)


def write_plan(entries: list[CalibrationPlanEntry], output: Path) -> None:
    output.parent.mkdir(parents=True, exist_ok=True)
    payload = {
        "prod_tau_factor": PROD_TAU_FACTOR,
        "equil_tau_factor": EQUIL_TAU_FACTOR,
        "n_samples": N_SAMPLES,
        "entries": [asdict(e) for e in entries],
    }
    output.write_text(json.dumps(payload, indent=2) + "\n")


def main() -> int:
    parser = argparse.ArgumentParser(description="Plan τ-scaled fictive calibration")
    parser.add_argument(
        "--relaxation-file",
        type=Path,
        default=DEFAULT_RELAX,
        help="cav-hoomd relaxation_times_vs_temperature.txt",
    )
    parser.add_argument("--temperature-min", type=float, default=65.0)
    parser.add_argument("--temperature-max", type=float, default=450.0)
    parser.add_argument("--n-temperatures", type=int, default=30)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUT)
    parser.add_argument(
        "--benchmark-ps-per-s",
        type=float,
        default=None,
        help="Measured simulation throughput for ETA (picoseconds per second)",
    )
    parser.add_argument("--n-workers", type=int, default=8)
    args = parser.parse_args()

    if not args.relaxation_file.exists():
        print(f"ERROR: relaxation file not found: {args.relaxation_file}", file=sys.stderr)
        return 1

    tau_model = TauSModel(args.relaxation_file)
    temperatures = uniform_temperature_grid(
        temperature_min=args.temperature_min,
        temperature_max=args.temperature_max,
        n_temperatures=args.n_temperatures,
    )
    entries = build_plan(tau_model, temperatures)
    slowest = max(entries, key=lambda e: e.prod_ps)
    write_plan(entries, args.output)

    print(f"Wrote {len(entries)} plan entries to {args.output}")
    print(
        f"  T range: {entries[0].temperature_K:.1f} → {entries[-1].temperature_K:.1f} K"
    )
    print(
        f"  Slowest point: T={slowest.temperature_K:.1f} K, "
        f"τ={slowest.tau_s_ps:.1f} ps, prod={slowest.prod_ns:.2f} ns"
    )
    if args.benchmark_ps_per_s is not None:
        eta_s = estimate_wall_time_s(entries, args.benchmark_ps_per_s, args.n_workers)
        print(
            f"  ETA ({args.n_workers} workers @ {args.benchmark_ps_per_s:.1f} ps/s): "
            f"{eta_s / 3600.0:.2f} h"
        )
    return 0


if __name__ == "__main__":
    sys.exit(main())
