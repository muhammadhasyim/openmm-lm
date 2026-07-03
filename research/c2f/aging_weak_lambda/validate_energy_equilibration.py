#!/usr/bin/env python3
"""Check late-time ΔE_bond / ΔE_nonbonded residuals for energy equilibration."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np

_SCRIPT_DIR = Path(__file__).resolve().parent
if str(_SCRIPT_DIR) not in sys.path:
    sys.path.insert(0, str(_SCRIPT_DIR))

from config import SWITCH_TIME_PS, job_dir_path  # noqa: E402
from fkt_utils import build_energy_csv_index, list_available_energy_replicas, resolve_energy_csv  # noqa: E402


def late_time_residuals(
    job_dir: Path,
    lam: float,
    *,
    t_late_lo: float = 1800.0,
    t_late_hi: float = 2000.0,
    min_tmax_ps: float = 2000.0,
) -> dict[str, float | int]:
    """Return ensemble late-time mean residuals (kJ/mol) and replica count."""
    replicas = list_available_energy_replicas(job_dir, lam, min_tmax_ps=min_tmax_ps)
    index = build_energy_csv_index(job_dir, lam)
    late_bond: list[float] = []
    late_nb: list[float] = []

    for replica in replicas:
        csv_path = resolve_energy_csv(job_dir, lam, replica, index)
        if csv_path is None:
            continue
        try:
            data = np.genfromtxt(csv_path, delimiter=",", names=True)
            t = np.asarray(data["time_ps"], dtype=float)
            bond = np.asarray(data["E_bond_kjmol"], dtype=float)
            nb = np.asarray(data["E_nonbonded_kjmol"], dtype=float)
        except Exception:
            continue

        pre = t < SWITCH_TIME_PS
        late = (t >= t_late_lo) & (t < t_late_hi)
        if not np.any(pre) or not np.any(late):
            continue
        b0 = float(np.mean(bond[pre]))
        n0 = float(np.mean(nb[pre]))
        late_bond.append(float(np.mean(bond[late]) - b0))
        late_nb.append(float(np.mean(nb[late]) - n0))

    if not late_bond:
        return {"n_replicas": 0}

    lb = np.asarray(late_bond)
    ln = np.asarray(late_nb)
    n = len(lb)
    sem_b = float(np.std(lb) / np.sqrt(n))
    sem_n = float(np.std(ln) / np.sqrt(n))
    return {
        "n_replicas": n,
        "bond_mean": float(np.mean(lb)),
        "bond_sem": sem_b,
        "nb_mean": float(np.mean(ln)),
        "nb_sem": sem_n,
        "mol_total_mean": float(np.mean(lb + ln)),
        "mol_total_sem": float(np.sqrt(sem_b**2 + sem_n**2)),
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--lambda", dest="lam", type=float, nargs="+", required=True)
    parser.add_argument("--campaign-dir", type=Path, default=_SCRIPT_DIR)
    parser.add_argument("--t-late-lo", type=float, default=1800.0)
    parser.add_argument("--t-late-hi", type=float, default=2000.0)
    parser.add_argument("--min-tmax-ps", type=float, default=2000.0)
    args = parser.parse_args()

    print(f"Late window t=[{args.t_late_lo}, {args.t_late_hi}) ps (baseline t<{SWITCH_TIME_PS})")
    for lam in args.lam:
        job_dir = job_dir_path(lam, campaign_root=args.campaign_dir)
        stats = late_time_residuals(
            job_dir,
            lam,
            t_late_lo=args.t_late_lo,
            t_late_hi=args.t_late_hi,
            min_tmax_ps=args.min_tmax_ps,
        )
        if stats.get("n_replicas", 0) == 0:
            print(f"λ={lam:g}: no replicas")
            continue
        print(
            f"λ={lam:g} n={stats['n_replicas']} "
            f"Δbond={stats['bond_mean']:+.2f}±{stats['bond_sem']:.2f} "
            f"Δnb={stats['nb_mean']:+.2f}±{stats['nb_sem']:.2f} "
            f"Δmol={stats['mol_total_mean']:+.2f}±{stats['mol_total_sem']:.2f} kJ/mol"
        )


if __name__ == "__main__":
    main()
