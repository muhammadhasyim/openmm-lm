#!/usr/bin/env python3
"""Single-frame F(0) parity: OpenMM fkt_tracker vs HOOMD-style autocorr."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np

_C2F_ROOT = Path(__file__).resolve().parent.parent
if str(_C2F_ROOT) not in sys.path:
    sys.path.insert(0, str(_C2F_ROOT))

from config import BOHR_TO_NM, FKT_KMAG_AU, FKT_KMAG_NM_INV, INITIAL_STATE  # noqa: E402
from fkt_physics import compute_f0_from_positions_nm, compute_f0_hoomd_style  # noqa: E402


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--initial-state", type=Path, default=INITIAL_STATE)
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path(__file__).resolve().parent / "diagnose_fkt",
    )
    args = parser.parse_args()

    positions_nm = np.asarray(np.load(args.initial_state)["positions_nm"], dtype=np.float64)
    positions_nm = positions_nm[:500]
    positions_bohr = positions_nm / BOHR_TO_NM

    f0_openmm = compute_f0_from_positions_nm(
        positions_nm, FKT_KMAG_NM_INV, site_mode="atomic"
    )
    f0_hoomd = compute_f0_hoomd_style(positions_bohr, FKT_KMAG_AU)

    hoomd_fkt_path = (
        Path(__file__).resolve().parents[4]
        / "third_party/cav-hoomd"
        / "aging_weak_lambda"
        / "step_lambda0_nocontrol"
        / "prod-0_fkt_ref_000.txt"
    )
    hoomd_f0_file = None
    if hoomd_fkt_path.exists():
        for line in hoomd_fkt_path.read_text().splitlines():
            if line.startswith("0.") and not line.startswith("#"):
                hoomd_f0_file = float(line.split()[1])
                break

    rel_diff = abs(f0_openmm - f0_hoomd) / abs(f0_hoomd)
    report = {
        "F0_openmm_atomic": f0_openmm,
        "F0_hoomd_style_same_frame": f0_hoomd,
        "relative_diff": rel_diff,
        "pass_parity": rel_diff < 1e-10,
        "F0_hoomd_aging_file_lag0": hoomd_f0_file,
        "note": "Aging file F0 is from different MD state, not same IC frame",
    }

    args.output_dir.mkdir(parents=True, exist_ok=True)
    out_path = args.output_dir / "fkt_parity.json"
    out_path.write_text(json.dumps(report, indent=2), encoding="utf-8")
    print(json.dumps(report, indent=2))
    print(f"Wrote {out_path}")


if __name__ == "__main__":
    main()
