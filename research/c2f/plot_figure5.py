#!/usr/bin/env python
"""
Plot Figure 5b — C2F cooling panels
====================================
Top:    square-wave lambda(t)  (peak lambda = 0.09 a.u.)
Middle: structural T_s and bath T converging to T_g ~ 32 K
Bottom: vibrational T_v oscillations

Run via pixi:
    pixi run -e test python research/c2f/plot_figure5.py
    pixi run -e test python research/c2f/plot_figure5.py \\
        --input research/c2f/fig5_output/fig5_averaged.csv
"""

from __future__ import annotations

import argparse
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

_SCRIPT_DIR = Path(__file__).resolve().parent
DEFAULT_INPUT = _SCRIPT_DIR / "fig5_output" / "fig5_averaged.csv"
DEFAULT_META = _SCRIPT_DIR / "fig5_output" / "fig5_meta.txt"
DEFAULT_REFERENCE_DIR = Path.home() / "GitRepos/cav-hoomd/examples/square_lambda0.09_diffeq_recalib"

# Paper reference values
TG_APPROX_K = 32.0


def _load_meta(meta_path: Path) -> dict:
    """Parse key=value metadata written by reproduce_figure5.py."""
    defaults = {
        "lambda": 0.09,
        "period_ps": 10.0,
        "duty_cycle": 0.10,
        "coupling_start_ps": 20.0,
        "initial_T_K": 300.0,
        "runtime_ps": 150.0,
    }
    if not meta_path.exists():
        return defaults

    meta = dict(defaults)
    with open(meta_path) as f:
        for line in f:
            line = line.strip()
            if not line or "=" not in line:
                continue
            key, val = line.split("=", 1)
            key = key.strip()
            val = val.strip()
            try:
                meta[key] = float(val)
            except ValueError:
                meta[key] = val
    return meta


def square_wave_lambda(
    time_ps: np.ndarray,
    amplitude: float,
    period_ps: float,
    duty_cycle: float,
    start_time_ps: float,
) -> np.ndarray:
    """Reconstruct fixed-peak square-wave coupling lambda(t)."""
    lam = np.zeros_like(time_ps, dtype=float)
    active = time_ps >= start_time_ps
    if not np.any(active):
        return lam

    dt = time_ps[active] - start_time_ps
    phase = (dt / period_ps) % 1.0
    lam[active] = np.where(phase < duty_cycle, amplitude, 0.0)
    return lam


def load_reference_trajectory_average(
    ref_dir: Path,
    runtime_ps: float,
    dt_ps: float = 0.01,
) -> dict | None:
    """Average cav-hoomd temperature_tracker_replica_*.csv files."""
    csv_files = sorted(ref_dir.glob("temperature_tracker_replica_*.csv"))
    if not csv_files:
        return None

    time_grid = np.arange(0.0, runtime_ps + 0.5 * dt_ps, dt_ps)
    cols = {
        "T_bath_K": "molecular_bath_K",
        "T_s_fictive_K": "lj_coul_fictive_K",
        "T_v_fictive_K": "harmonic_equipartition_K",
    }
    stacked = {k: [] for k in cols}

    try:
        from scipy.interpolate import interp1d
    except ImportError:
        return None

    for csv_path in csv_files:
        try:
            data = np.genfromtxt(
                csv_path, delimiter=",", names=True, missing_values="", usemask=False
            )
        except OSError:
            continue
        t = np.asarray(data["time_ps"], dtype=float)
        if t.size == 0:
            continue
        for out_col, src_col in cols.items():
            y = np.asarray(data[src_col], dtype=float)
            y = np.where(np.isfinite(y), y, np.nan)
            valid = np.isfinite(y)
            if np.sum(valid) < 2:
                stacked[out_col].append(np.full_like(time_grid, np.nan))
                continue
            f = interp1d(
                t[valid], y[valid],
                kind="linear", bounds_error=False, fill_value=np.nan,
            )
            stacked[out_col].append(f(time_grid))

    if not stacked["T_s_fictive_K"]:
        return None

    result = {"time_ps": time_grid}
    for col in cols:
        arr = np.array(stacked[col])
        result[col] = np.nanmean(arr, axis=0)
    result["n_replicas"] = len(csv_files)
    return result


def plot_figure5b(
    csv_path: Path,
    meta: dict,
    output_prefix: Path,
    show_std: bool = True,
    reference_dir: Path | None = None,
) -> tuple[Path, Path]:
    data = np.genfromtxt(csv_path, delimiter=",", names=True)

    time_ps = np.asarray(data["time_ps"], dtype=float)
    T_bath = np.asarray(data["T_bath_K"], dtype=float)
    T_v = np.asarray(data["T_v_fictive_K"], dtype=float)
    T_s = np.asarray(data["T_s_fictive_K"], dtype=float)

    lam_amp = float(meta.get("lambda", 0.09))
    period_ps = float(meta.get("period_ps", 5.0))
    duty = float(meta.get("duty_cycle", 0.5))
    start_ps = float(meta.get("coupling_start_ps", 20.0))

    lam_t = square_wave_lambda(time_ps, lam_amp, period_ps, duty, start_ps)

    ref_data = None
    ref_dir = reference_dir
    if ref_dir is None and "reference_traj_dir" in meta:
        ref_dir = Path(str(meta["reference_traj_dir"]))
    if ref_dir is None:
        ref_dir = DEFAULT_REFERENCE_DIR if DEFAULT_REFERENCE_DIR.exists() else None
    if ref_dir is not None and Path(ref_dir).exists():
        runtime_ps = float(meta.get("runtime_ps", time_ps[-1] if time_ps.size else 150.0))
        ref_data = load_reference_trajectory_average(Path(ref_dir), runtime_ps)
        if ref_data is not None:
            print(
                f"Overlaying reference mean from {ref_dir} "
                f"({ref_data['n_replicas']} replicas)"
            )

    fig, axes = plt.subplots(3, 1, figsize=(8, 7), sharex=True)
    fig.subplots_adjust(hspace=0.08)

    # --- Top: lambda(t) ---
    ax0 = axes[0]
    ax0.plot(time_ps, lam_t, color="#1f77b4", linewidth=1.2)
    ax0.set_ylabel(r"$\lambda$ (a.u.)")
    ax0.set_ylim(-0.01, lam_amp * 1.15)
    ax0.axhline(lam_amp, color="gray", ls="--", lw=0.8, alpha=0.5)
    ax0.text(
        0.02, 0.92, rf"peak $\lambda = {lam_amp:.2f}$ a.u.",
        transform=ax0.transAxes, fontsize=9,
    )

    # --- Middle: T_s and bath T ---
    ax1 = axes[1]
    if show_std and "T_s_fictive_K_std" in data.dtype.names:
        T_s_std = np.asarray(data["T_s_fictive_K_std"], dtype=float)
        T_bath_std = np.asarray(data["T_bath_K_std"], dtype=float)
        ax1.fill_between(
            time_ps, T_s - T_s_std, T_s + T_s_std,
            color="#d62728", alpha=0.15, linewidth=0,
        )
        ax1.fill_between(
            time_ps, T_bath - T_bath_std, T_bath + T_bath_std,
            color="#2c2c2c", alpha=0.10, linewidth=0,
        )

    ax1.plot(time_ps, T_s, color="#d62728", linewidth=1.5, label=r"$T_\mathrm{s}$")
    ax1.plot(time_ps, T_bath, color="#2c2c2c", linewidth=1.5, label=r"$T$ (bath)")
    if ref_data is not None:
        rt = ref_data["time_ps"]
        ax1.plot(
            rt, ref_data["T_s_fictive_K"], color="#d62728", ls="--", lw=1.0, alpha=0.7,
            label=r"$T_\mathrm{s}$ (ref)",
        )
        ax1.plot(
            rt, ref_data["T_bath_K"], color="#2c2c2c", ls="--", lw=1.0, alpha=0.7,
            label=r"$T$ bath (ref)",
        )
    ax1.axhline(TG_APPROX_K, color="#888888", ls=":", lw=1.0)
    ax1.text(
        time_ps[-1] * 0.98, TG_APPROX_K + 5,
        rf"$T_\mathrm{{g}} \approx {TG_APPROX_K:.0f}$ K",
        ha="right", fontsize=9, color="#666666",
    )
    ax1.set_ylabel("Temperature (K)")
    temp_vals = np.concatenate([
        T_s[np.isfinite(T_s)],
        T_bath[np.isfinite(T_bath)],
    ])
    if ref_data is not None:
        temp_vals = np.concatenate([
            temp_vals,
            ref_data["T_s_fictive_K"][np.isfinite(ref_data["T_s_fictive_K"])],
            ref_data["T_bath_K"][np.isfinite(ref_data["T_bath_K"])],
        ])
    y_max = float(np.nanmax(temp_vals)) if temp_vals.size else 350.0
    ax1.set_ylim(0, max(350.0, y_max * 1.05))
    ax1.legend(loc="upper right", frameon=False, fontsize=9)

    # --- Bottom: T_v ---
    ax2 = axes[2]
    if show_std and "T_v_fictive_K_std" in data.dtype.names:
        T_v_std = np.asarray(data["T_v_fictive_K_std"], dtype=float)
        ax2.fill_between(
            time_ps, T_v - T_v_std, T_v + T_v_std,
            color="#1f77b4", alpha=0.15, linewidth=0,
        )
    ax2.plot(time_ps, T_v, color="#1f77b4", linewidth=1.2, label=r"$T_\mathrm{v}$")
    if ref_data is not None:
        ax2.plot(
            ref_data["time_ps"], ref_data["T_v_fictive_K"],
            color="#1f77b4", ls="--", lw=1.0, alpha=0.7, label=r"$T_\mathrm{v}$ (ref)",
        )
    ax2.set_ylabel(r"$T_\mathrm{v}$ (K)")
    ax2.set_xlabel("Time (ps)")
    ax2.legend(loc="upper right", frameon=False, fontsize=9)

    fig.suptitle(
        r"C$^2$F cooling: room-temperature liquid $\rightarrow$ $T_\mathrm{g}$",
        fontsize=11, y=0.98,
    )

    png_path = output_prefix.with_suffix(".png")
    pdf_path = output_prefix.with_suffix(".pdf")
    fig.savefig(png_path, dpi=200, bbox_inches="tight")
    fig.savefig(pdf_path, bbox_inches="tight")
    plt.close(fig)

    print(f"Saved {png_path}")
    print(f"Saved {pdf_path}")
    return png_path, pdf_path


def main():
    parser = argparse.ArgumentParser(description="Plot Figure 5b from ensemble CSV")
    parser.add_argument(
        "--input", type=Path, default=DEFAULT_INPUT,
        help="Ensemble-averaged CSV (fig5_averaged.csv)",
    )
    parser.add_argument(
        "--meta", type=Path, default=None,
        help="Metadata file (default: alongside input)",
    )
    parser.add_argument(
        "--output", type=Path, default=None,
        help="Output prefix without extension (default: Figure5b_reproduced)",
    )
    parser.add_argument(
        "--reference-dir", type=Path, default=None,
        help="cav-hoomd temperature_tracker CSV directory for dashed overlay",
    )
    parser.add_argument("--no-std-bands", action="store_true")
    args = parser.parse_args()

    csv_path = args.input.resolve()
    if not csv_path.exists():
        raise SystemExit(f"Input not found: {csv_path}\n"
                         "Run reproduce_figure5.py first.")

    meta_path = args.meta or csv_path.parent / "fig5_meta.txt"
    meta = _load_meta(meta_path.resolve())

    out_prefix = args.output or (csv_path.parent / "Figure5b_reproduced")
    plot_figure5b(
        csv_path, meta, out_prefix,
        show_std=not args.no_std_bands,
        reference_dir=args.reference_dir,
    )


if __name__ == "__main__":
    main()
