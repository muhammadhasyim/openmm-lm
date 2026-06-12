#!/usr/bin/env python3
"""Write consolidated FKT diagnostic conclusion from JSON artifacts."""

from __future__ import annotations

import json
from pathlib import Path


def main() -> None:
    base = Path(__file__).resolve().parent / "diagnose_fkt"
    artifacts = {
        "units": base / "diagnose_fkt_units.json",
        "parity": base / "fkt_parity.json",
        "normalization": base / "diagnose_normalization.json",
        "kscan": base / "compute_sk_kscan.json",
        "sk_ic": base / "sk_ic_only.json",
    }
    loaded = {}
    for key, path in artifacts.items():
        if path.exists():
            loaded[key] = json.loads(path.read_text())

    conclusion = {
        "primary_root_cause": "analysis_calibration_mismatch",
        "ruled_out": [
            "OpenMM nm/Bohr k conversion error",
            "OpenMM vs HOOMD tracker mismatch on same IC",
            "Molecular COM as production fix",
            "Replacing paper k=6 with S(k)-peak k for production",
        ],
        "confirmed": {
            "paper_k_au": 6.0,
            "production_sites": "atomic",
            "units_audit_pass": loaded.get("units", {}).get("overall_pass"),
            "tracker_parity_pass": loaded.get("parity", {}).get("pass_parity"),
        },
        "open_issue": (
            "relaxation_times_vs_temperature.txt reports tau_s≈105 ps at 100 K but "
            "atomic k=6 aging FKT (OpenMM and HOOMD) gives tau_block≈15 ps and phi(1)≈0.05. "
            "Calibration generator not in repo."
        ),
        "production_action": loaded.get("normalization", {}).get(
            "recommended_production",
            {
                "fkt_sites": "atomic",
                "kmag_au": 6.0,
                "tau_method": "tau_F_over_F0_block",
            },
        ),
        "kscan_diagnostic_only": (
            "compute_sk_kscan.json compares tau at multiple |k| for hypothesis testing; "
            "production remains k=6 per paper."
        ),
    }
    if "kscan" in loaded:
        k6 = loaded["kscan"].get("kscan", {}).get("k_au_6", {})
        conclusion["k6_replay_tau_block_ps"] = k6.get("tau_s_block")

    out = base / "diagnostic_conclusion.json"
    out.write_text(json.dumps(conclusion, indent=2), encoding="utf-8")
    print(json.dumps(conclusion, indent=2))
    print(f"Wrote {out}")


if __name__ == "__main__":
    main()
