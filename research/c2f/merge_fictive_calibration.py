#!/usr/bin/env python
"""Merge parallel fictive-calibration shard outputs into a single table."""

from __future__ import annotations

import argparse
import glob
import json
import sys
from pathlib import Path

from openmm.cavitymd.calibration import (
    CalibrationRow,
    SLIM_HEADER,
    write_full_calibration_file,
    write_n_eff_report,
    write_slim_calibration_file,
)

_SCRIPT_DIR = Path(__file__).resolve().parent
DEFAULT_OUT_DIR = _SCRIPT_DIR / "calibration_output"


def _parse_full_row(parts: list[str]) -> CalibrationRow:
    return CalibrationRow(
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


def load_shard_rows(shard_file: Path) -> list[CalibrationRow]:
    rows: list[CalibrationRow] = []
    with shard_file.open() as handle:
        for line in handle:
            line = line.strip()
            if not line or line.startswith("#") or line.startswith("temperature"):
                continue
            parts = line.split()
            if len(parts) < 14:
                continue
            rows.append(_parse_full_row(parts))
    return rows


def load_shard_diagnostics(shard_file: Path) -> list[dict]:
    diag_path = shard_file.parent / f"{shard_file.stem}_n_eff.json"
    if not diag_path.exists():
        return []
    try:
        return json.loads(diag_path.read_text()).get("entries", [])
    except json.JSONDecodeError:
        return []


def merge_rows(all_rows: list[CalibrationRow]) -> list[CalibrationRow]:
    by_T: dict[float, CalibrationRow] = {}
    for row in all_rows:
        by_T[row.temperature_K] = row
    return [by_T[T] for T in sorted(by_T)]


def merge_diagnostics(all_diag: list[dict]) -> list[dict]:
    by_T: dict[float, dict] = {}
    for entry in all_diag:
        by_T[float(entry["temperature_K"])] = entry
    return [by_T[T] for T in sorted(by_T)]


def main() -> int:
    parser = argparse.ArgumentParser(description="Merge fictive calibration shard files")
    parser.add_argument(
        "--shard-glob",
        default=str(DEFAULT_OUT_DIR / "shard_*.txt"),
        help="Glob for shard full-output files",
    )
    parser.add_argument(
        "--full-output",
        type=Path,
        default=DEFAULT_OUT_DIR / "potential_energy_components_vs_temperature.txt",
    )
    parser.add_argument(
        "--slim-output",
        type=Path,
        default=DEFAULT_OUT_DIR / "calibration_data.txt",
    )
    parser.add_argument(
        "--n-eff-report",
        type=Path,
        default=DEFAULT_OUT_DIR / "n_eff_report.json",
    )
    args = parser.parse_args()

    shard_files = sorted(Path(p) for p in glob.glob(args.shard_glob))
    if not shard_files:
        print(f"ERROR: no shard files match {args.shard_glob}", file=sys.stderr)
        return 1

    all_rows: list[CalibrationRow] = []
    all_diag: list[dict] = []
    for shard in shard_files:
        rows = load_shard_rows(shard)
        all_rows.extend(rows)
        all_diag.extend(load_shard_diagnostics(shard))
        print(f"  {shard.name}: {len(rows)} temperatures")

    merged = merge_rows(all_rows)
    if not merged:
        print("ERROR: no calibration rows found in shards", file=sys.stderr)
        return 1

    args.full_output.parent.mkdir(parents=True, exist_ok=True)
    write_full_calibration_file(merged, args.full_output)
    write_slim_calibration_file(merged, args.slim_output)
    if all_diag:
        write_n_eff_report(merge_diagnostics(all_diag), args.n_eff_report)

    print(f"Merged {len(merged)} temperatures → {args.full_output}")
    print(f"Slim table → {args.slim_output}")
    if all_diag:
        print(f"N_eff report → {args.n_eff_report}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
