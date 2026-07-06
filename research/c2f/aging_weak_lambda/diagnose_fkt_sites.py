#!/usr/bin/env python3
"""Compare atomic vs molecular-COM FKT on short equilibrium runs."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np

_C2F_ROOT = Path(__file__).resolve().parent.parent
if str(_C2F_ROOT) not in sys.path:
    sys.path.insert(0, str(_C2F_ROOT))

from run_cavity_equilibrium import run_cavity_equilibrium  # noqa: E402

from fkt_utils import (  # noqa: E402
    block_average_abs_phi,
    extract_tau_s,
    normalize_fkt_to_phi,
    parse_fkt_file,
)

DEFAULT_IC = _C2F_ROOT / "equilibrium_output" / "eq10ns100K_lam0_final_state.npz"


def _summarize(prefix: str) -> dict:
    _, lags, vals = parse_fkt_file(Path(f"{prefix}_fkt_ref_000.txt"))
    norm = normalize_fkt_to_phi(lags, vals)
    if norm[0] is None:
        return {"error": "normalization failed"}
    lags_n, phi = norm
    block_lags, block_phi = block_average_abs_phi(lags_n, phi, 10.0, 10.0)
    idx_block = int(np.argmin(np.abs(block_lags - 15.0))) if block_lags.size else 0
    out = {"F0": float(vals[np.argsort(lags)[0]])}
    for target in (1, 10, 100):
        idx = int(np.argmin(np.abs(lags_n - target)))
        out[f"phi_at_{target}ps"] = float(phi[idx])
    out["phi_block_15ps"] = float(block_phi[idx_block]) if block_phi.size else None
    out["tau_s_ps"] = extract_tau_s(
        lags,
        vals,
        threshold=0.1,
        min_lag_ps=10.0,
        use_block_average=True,
        block_window_ps=10.0,
    )
    return out


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--runtime-ps", type=float, default=600.0)
    parser.add_argument("--output-dir", type=Path, default=Path(__file__).resolve().parent / "diagnose_fkt")
    parser.add_argument("--initial-state", type=Path, default=DEFAULT_IC)
    args = parser.parse_args()

    args.output_dir.mkdir(parents=True, exist_ok=True)
    results: dict[str, dict] = {}
    for mode in ("atomic", "molecular_com"):
        prefix = str(args.output_dir / f"eq100K_{mode}")
        run_cavity_equilibrium(
            temperature_K=100.0,
            runtime_ps=args.runtime_ps,
            lambda_coupling=0.0,
            include_dipole_self_energy=True,
            output_prefix=prefix,
            seed=42,
            initial_state=args.initial_state,
            finite_q=False,
            resample_velocities=False,
            enable_fkt=True,
            fkt_ref_interval_ps=200.0,
            fkt_max_refs=3,
            fkt_output_period_ps=1.0,
            fkt_start_ps=0.0,
            fkt_sites=mode,
        )
        results[mode] = _summarize(prefix)

    out_path = args.output_dir / "diagnose_fkt_sites.json"
    out_path.write_text(json.dumps(results, indent=2), encoding="utf-8")
    print(json.dumps(results, indent=2))
    print(f"Wrote {out_path}")


if __name__ == "__main__":
    main()
