#!/usr/bin/env python3
"""Fig 3c: fictive and kinetic temperatures vs time from CSV."""

from __future__ import annotations

import argparse
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

from config import FIGURES_DIR, N_REPLICAS, SWITCH_TIME_PS, TEMPERATURE_K, job_dir_path, run_prefix


def load_temperature_series(job_dir: Path, lam: float, replicas: list[int]) -> dict[str, np.ndarray]:
    buckets: dict[str, list[np.ndarray]] = {
        "time": [],
        "T_v": [],
        "T_s": [],
        "T_k": [],
    }
    for replica in replicas:
        csv_path = job_dir / f"{run_prefix(lam, replica)}_energies.csv"
        if not csv_path.exists():
            continue
        data = np.genfromtxt(
            csv_path, delimiter=",", names=True, missing_values="", usemask=False
        )
        t = np.asarray(data["time_ps"], dtype=float)
        T_v = np.asarray(data["T_v_fictive_K"], dtype=float)
        T_s_raw = np.asarray(data["T_s_fictive_K"], dtype=float)
        T_s = np.where(np.isfinite(T_s_raw), T_s_raw, np.nan)
        T_k = np.asarray(data["T_kinetic_K"], dtype=float)
        buckets["time"].append(t)
        buckets["T_v"].append(T_v)
        buckets["T_s"].append(T_s)
        buckets["T_k"].append(T_k)

    if not buckets["time"]:
        return {}

    t_ref = buckets["time"][0]
    out: dict[str, np.ndarray] = {"time": t_ref}
    for key in ("T_v", "T_s", "T_k"):
        stack = np.vstack(
            [np.interp(t_ref, buckets["time"][i], buckets[key][i]) for i in range(len(buckets["time"]))]
        )
        out[key] = np.nanmean(stack, axis=0)
    return out


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--lambda", dest="lam", type=float, default=0.03)
    parser.add_argument("--output-dir", type=Path, default=FIGURES_DIR)
    parser.add_argument("--replicas", type=int, nargs="+", default=list(range(N_REPLICAS)))
    args = parser.parse_args()

    from fkt_utils import list_available_replicas

    job_dir = job_dir_path(args.lam)
    replicas = [r for r in args.replicas if r in list_available_replicas(job_dir, args.lam)]
    data = load_temperature_series(job_dir, args.lam, replicas)
    if not data:
        raise SystemExit(f"No temperature CSV data in {job_dir}")

    t = data["time"]
    fig, ax = plt.subplots(figsize=(8, 5))
    ax.plot(t, data["T_v"], label="$T_v$ (harmonic fictive)", color="#1f77b4")
    ax.plot(t, data["T_s"], label="$T_s$ (structural fictive)", color="#d62728")
    ax.plot(t, data["T_k"], label="$T_k$ (kinetic)", color="#ff7f0e")
    ax.axhline(TEMPERATURE_K, color="gray", ls=":", lw=1.0, label=f"bath {TEMPERATURE_K:.0f} K")
    ax.axvline(SWITCH_TIME_PS, color="k", ls="--", lw=1.0, alpha=0.7)
    ax.set_xlabel("time (ps)")
    ax.set_ylabel("temperature (K)")
    ax.set_title(f"Fictive temperatures ($\\lambda$={args.lam:g})")
    ax.legend(fontsize=8)
    fig.tight_layout()
    out_path = args.output_dir / f"fig3c_fictive_temperatures_lam{args.lam:g}.png"
    fig.savefig(out_path, dpi=150)
    plt.close(fig)
    print(f"Wrote {out_path}")


if __name__ == "__main__":
    main()
