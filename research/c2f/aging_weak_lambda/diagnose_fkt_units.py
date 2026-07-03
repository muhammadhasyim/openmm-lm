#!/usr/bin/env python3
"""Audit F(k,t) unit chain, tracker round-trip, and position-scale sanity."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np

_C2F_ROOT = Path(__file__).resolve().parent.parent
if str(_C2F_ROOT) not in sys.path:
    sys.path.insert(0, str(_C2F_ROOT))

from openmm import unit

from config import (  # noqa: E402
    BOHR_TO_NM,
    FKT_KMAG_AU,
    FKT_KMAG_NM_INV,
    INITIAL_STATE,
)
from fkt_physics import (  # noqa: E402
    bond_lengths_nm,
    compute_f0_from_positions_nm,
    compute_f0_hoomd_style,
    dimensionless_products,
    kmag_nm_from_au,
)
from run_c2f import (  # noqa: E402
    BOX_AU,
    NUM_MOL,
    R0_AA_AU,
    R0_BB_AU,
    SIG_AA_AU,
    SIG_AB_AU,
    SIG_BB_AU,
    build_mka_system,
)


def _load_ic_positions_nm(path: Path) -> np.ndarray:
    data = np.load(path)
    positions = np.asarray(data["positions_nm"], dtype=np.float64)
    return positions[: 2 * NUM_MOL]


def _openmm_box_nm() -> float:
    system, _, _ = build_mka_system(seed=42)
    box = system.getDefaultPeriodicBoxVectors()
    return float(box[0][0].value_in_unit(unit.nanometer))


def run_audit(initial_state: Path) -> dict:
    kmag_exact = kmag_nm_from_au(FKT_KMAG_AU)
    rel_diff = abs(FKT_KMAG_NM_INV - kmag_exact) / kmag_exact

    positions_nm = _load_ic_positions_nm(initial_state)
    positions_bohr = positions_nm / BOHR_TO_NM

    f0_nm = compute_f0_from_positions_nm(positions_nm, FKT_KMAG_NM_INV, site_mode="atomic")
    f0_bohr_path = compute_f0_from_positions_nm(
        positions_bohr * BOHR_TO_NM, kmag_exact, site_mode="atomic"
    )
    f0_hoomd = compute_f0_hoomd_style(positions_bohr, FKT_KMAG_AU)

    bonds = bond_lengths_nm(positions_nm)
    box_edge_nm = _openmm_box_nm()

    report = {
        "k_conversion": {
            "FKT_KMAG_AU": FKT_KMAG_AU,
            "FKT_KMAG_NM_INV_config": FKT_KMAG_NM_INV,
            "FKT_KMAG_NM_INV_exact": kmag_exact,
            "relative_diff": rel_diff,
            "pass": rel_diff < 2e-4,
        },
        "force_field_lengths": {
            "sigma_AA_nm": SIG_AA_AU * BOHR_TO_NM,
            "sigma_BB_nm": SIG_BB_AU * BOHR_TO_NM,
            "sigma_AB_nm": SIG_AB_AU * BOHR_TO_NM,
            "r0_AA_nm": R0_AA_AU * BOHR_TO_NM,
            "r0_BB_nm": R0_BB_AU * BOHR_TO_NM,
            "box_AU": BOX_AU,
            "box_nm_expected": BOX_AU * BOHR_TO_NM,
            "box_nm_openmm": box_edge_nm,
            "box_match": abs(box_edge_nm - BOX_AU * BOHR_TO_NM) < 1e-4,
        },
        "dimensionless_k_products": dimensionless_products(),
        "tracker_round_trip": {
            "F0_openmm_nm_tracker": f0_nm,
            "F0_openmm_exact_k": f0_bohr_path,
            "F0_hoomd_bohr_style": f0_hoomd,
            "F0_nm_vs_hoomd_rel_diff": abs(f0_nm - f0_hoomd) / abs(f0_hoomd),
            "pass": abs(f0_nm - f0_hoomd) / abs(f0_hoomd) < 1e-10,
        },
        "position_sanity": {
            "n_atoms": int(positions_nm.shape[0]),
            "bond_length_nm_mean": float(np.mean(bonds)),
            "bond_length_nm_std": float(np.std(bonds)),
            "bond_length_nm_min": float(np.min(bonds)),
            "bond_length_nm_max": float(np.max(bonds)),
            "r0_AA_nm": R0_AA_AU * BOHR_TO_NM,
            "min_pair_distance_nm": float(
                np.partition(
                    np.linalg.norm(
                        positions_nm[:, None, :] - positions_nm[None, :, :], axis=-1
                    ).reshape(-1),
                    1,
                )[1]
            ),
        },
        "overall_pass": True,
    }
    report["overall_pass"] = all(
        report[key]["pass"]
        for key in ("k_conversion", "force_field_lengths", "tracker_round_trip")
        if "pass" in report[key]
    )
    report["force_field_lengths"]["pass"] = report["force_field_lengths"]["box_match"]
    return report


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--initial-state", type=Path, default=INITIAL_STATE)
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path(__file__).resolve().parent / "diagnose_fkt",
    )
    args = parser.parse_args()

    args.output_dir.mkdir(parents=True, exist_ok=True)
    report = run_audit(args.initial_state)
    out_path = args.output_dir / "diagnose_fkt_units.json"
    out_path.write_text(json.dumps(report, indent=2), encoding="utf-8")
    print(json.dumps(report, indent=2))
    print(f"Wrote {out_path}")


if __name__ == "__main__":
    main()
