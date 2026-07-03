#!/usr/bin/env python3
"""Check pilot energy CSVs for T_kin blowups and completion to target runtime."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np

_SCRIPT_DIR = Path(__file__).resolve().parent
if str(_SCRIPT_DIR) not in sys.path:
    sys.path.insert(0, str(_SCRIPT_DIR))

from config import job_dir_path  # noqa: E402
from fkt_utils import build_energy_csv_index, resolve_energy_csv  # noqa: E402

T_KIN_LIMIT = 5000.0


def check_replica(
    job_dir: Path,
    lam: float,
    replica: int,
    *,
    runtime_ps: float,
    index: dict,
) -> dict:
    csv_path = resolve_energy_csv(job_dir, lam, replica, index)
    if csv_path is None:
        return {"ok": False, "reason": "missing_csv"}
    data = np.genfromtxt(csv_path, delimiter=",", names=True)
    t = np.asarray(data["time_ps"], dtype=float)
    t_kin = np.asarray(data["T_kinetic_K"], dtype=float)
    tmax = float(np.nanmax(t)) if t.size else 0.0
    blow = t_kin > T_KIN_LIMIT
    return {
        "ok": tmax >= runtime_ps - 5.0 and not np.any(blow),
        "tmax": tmax,
        "blowup": bool(np.any(blow)),
        "max_t_kin": float(np.nanmax(t_kin)) if t_kin.size else float("nan"),
        "csv": str(csv_path),
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--lambda", dest="lams", type=float, nargs="+", required=True)
    parser.add_argument("--replica", type=int, default=42)
    parser.add_argument("--runtime-ps", type=float, default=1500.0)
    parser.add_argument("--campaign-dir", type=Path, default=_SCRIPT_DIR)
    args = parser.parse_args()

    all_ok = True
    for lam in args.lams:
        job_dir = job_dir_path(lam, campaign_root=args.campaign_dir)
        index = build_energy_csv_index(job_dir, lam)
        stats = check_replica(
            job_dir, lam, args.replica, runtime_ps=args.runtime_ps, index=index
        )
        if stats.get("reason") == "missing_csv":
            print(f"λ={lam:g} rep={args.replica}: MISSING CSV")
            all_ok = False
            continue
        status = "PASS" if stats["ok"] else "FAIL"
        print(
            f"λ={lam:g} rep={args.replica}: {status} "
            f"tmax={stats['tmax']:.1f} ps max_T_kin={stats['max_t_kin']:.1f} K "
            f"blowup={stats['blowup']}"
        )
        if not stats["ok"]:
            all_ok = False
    sys.exit(0 if all_ok else 1)


if __name__ == "__main__":
    main()
