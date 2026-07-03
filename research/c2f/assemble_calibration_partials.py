#!/usr/bin/env python
"""Assemble per-temperature partial calibration outputs into shard files."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

from openmm.cavitymd.calibration import (
    CalibrationRow,
    write_full_calibration_file,
    write_n_eff_report,
    write_slim_calibration_file,
)

_SCRIPT_DIR = Path(__file__).resolve().parent
DEFAULT_OUT = _SCRIPT_DIR / "calibration_output"


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


def load_rows(path: Path) -> list[CalibrationRow]:
    rows: list[CalibrationRow] = []
    with path.open() as handle:
        for line in handle:
            line = line.strip()
            if not line or line.startswith("#") or line.startswith("temperature"):
                continue
            parts = line.split()
            if len(parts) < 14:
                continue
            rows.append(_parse_full_row(parts))
    return rows


def load_diagnostics(path: Path) -> list[dict]:
    if not path.exists():
        return []
    try:
        return json.loads(path.read_text()).get("entries", [])
    except json.JSONDecodeError:
        return []


def assemble_shard(
    partial_dir: Path,
    shard_id: int,
    output_dir: Path,
) -> None:
    prefix = f"T*"
    partials = sorted(partial_dir.glob("T*/run.txt"))
    rows: list[CalibrationRow] = []
    diagnostics: list[dict] = []
    for partial in partials:
        meta_path = partial.parent / "meta.json"
        if meta_path.exists():
            meta = json.loads(meta_path.read_text())
            if int(meta.get("shard_id", -1)) != shard_id:
                continue
        rows.extend(load_rows(partial))
        diagnostics.extend(load_diagnostics(partial.parent / "n_eff.json"))

    if not rows:
        print(f"Shard {shard_id}: no partial rows found in {partial_dir}")
        return

    rows.sort(key=lambda row: row.temperature_K)
    diagnostics.sort(key=lambda entry: entry["temperature_K"])

    full_path = output_dir / f"shard_{shard_id}.txt"
    slim_path = output_dir / f"shard_{shard_id}_slim.txt"
    n_eff_path = output_dir / f"shard_{shard_id}_n_eff.json"

    write_full_calibration_file(rows, full_path)
    write_slim_calibration_file(rows, slim_path)
    write_n_eff_report(diagnostics, n_eff_path)
    print(f"Shard {shard_id}: wrote {len(rows)} temperatures → {full_path}")


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--partial-dir",
        type=Path,
        default=DEFAULT_OUT / "partial",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=DEFAULT_OUT,
    )
    parser.add_argument(
        "--shards",
        type=int,
        nargs="+",
        default=[0, 2],
        help="Shard ids to assemble from partial runs",
    )
    args = parser.parse_args()

    for shard_id in args.shards:
        assemble_shard(args.partial_dir, shard_id, args.output_dir)
    return 0


if __name__ == "__main__":
    sys.exit(main())
