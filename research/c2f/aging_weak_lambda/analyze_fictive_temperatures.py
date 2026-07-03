#!/usr/bin/env python3
"""Fig 3c: fictive and kinetic temperatures vs time from CSV.

Temperature inversion uses the equilibrium U(T) calibration table from
cav-hoomd (potential_energy_vs_T.txt) to properly map E_bond → T_v and
E_nonbonded → T_s.  The calibration is anchored at T_bath via the
pre-switch mean energies so that both fictive temperatures converge to
the bath temperature in the uncoupled equilibrium.
"""

from __future__ import annotations

import argparse
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

from config import (
    CSV_INTERVAL_PS,
    FIG3_SHOWCASE_LAMBDA,
    FIGURES_DIR,
    POTENTIAL_ENERGY_VS_T,
    SWITCH_TIME_PS,
    TEMPERATURE_K,
    job_dir_path,
)
from paper_style import (
    COLOR_HARMONIC,
    COLOR_KINETIC,
    COLOR_LJ_COULOMB,
    apply_paper_style,
    paper_legend,
    save_figure,
    style_axes,
)

HARTREE_TO_KJMOL: float = 2625.499639


def smooth_uniform(arr: np.ndarray, window: int) -> np.ndarray:
    """Centered rolling mean with edge padding (avoids boundary undershoot)."""
    if window <= 1:
        return arr
    pad = window // 2
    padded = np.pad(arr, pad, mode="edge")
    return np.convolve(padded, np.ones(window) / window, mode="valid")


# ---------------------------------------------------------------------------
# Calibration helpers
# ---------------------------------------------------------------------------

def load_calibration(cal_file: Path) -> dict[str, np.ndarray]:
    """Load the cav-hoomd U(T) calibration table.

    Parameters
    ----------
    cal_file:
        Path to ``potential_energy_vs_T.txt``.

    Returns
    -------
    dict with keys ``T_K``, ``harmonic_ha``, ``nb_ha``
        where ``nb_ha`` = coulombic + LJ in Hartree (whole simulation box).
    """
    # Robust parse: skip comment lines (#) and blank lines; treat first
    # non-empty non-comment line as the column-name header.
    lines = cal_file.read_text().splitlines()
    header: list[str] = []
    rows: list[list[float]] = []
    for line in lines:
        stripped = line.strip()
        if not stripped or stripped.startswith("#"):
            continue
        if not header:
            header = stripped.split()
            continue
        try:
            rows.append([float(v) for v in stripped.split()])
        except ValueError:
            continue

    arr = np.array(rows, dtype=float)
    col = {name: arr[:, i] for i, name in enumerate(header)}

    T_K = col["temperature"]
    harmonic_ha = col["harmonic_hartree"]
    nb_ha = col["coulombic_hartree"] + col["lj_hartree"]

    # Sort by temperature (defensive; data may have scattered extra points)
    order = np.argsort(T_K)
    return {
        "T_K": T_K[order],
        "harmonic_ha": harmonic_ha[order],
        "nb_ha": nb_ha[order],
    }


def build_inverters(
    cal: dict[str, np.ndarray],
    E_bond_ref_kjmol: float,
    E_nb_ref_kjmol: float,
    T_bath: float,
) -> tuple[object, object]:
    """Build T_v and T_s inverter functions anchored at T_bath.

    The calibration energies (Hartree, extensive in N_hoomd) are scaled so
    that at T_bath they equal the OpenMM pre-switch reference energies.
    Polynomial fits to the calibration are used rather than raw interpolation
    to avoid non-monotone artefacts from statistical scatter in the data.

    - Harmonic energy E_harm(T): linear fit (classical equipartition).
    - Nonbonded energy E_nb(T): quadratic fit (captures LJ/Coulomb curvature).

    Parameters
    ----------
    cal:
        Output of :func:`load_calibration`.
    E_bond_ref_kjmol:
        Mean E_bond from OpenMM at pre-switch equilibrium (kJ/mol, total).
    E_nb_ref_kjmol:
        Mean E_nonbonded from OpenMM at pre-switch equilibrium (kJ/mol, total).
    T_bath:
        Thermostat temperature (K) used to anchor the normalization.

    Returns
    -------
    invert_v, invert_s
        Callable ``T_fictive = invert(E_obs_kjmol)`` for harmonic and
        nonbonded channels.
    """
    T_cal = cal["T_K"]

    # Calibration energies in kJ/mol (extensive in N_hoomd)
    harm_kjmol = cal["harmonic_ha"] * HARTREE_TO_KJMOL
    nb_kjmol = cal["nb_ha"] * HARTREE_TO_KJMOL

    # --- Fit smooth polynomial models to remove statistical noise ---
    # Harmonic: linear (E_harm = a·T + b) — exact for classical equipartition
    p_harm = np.polyfit(T_cal, harm_kjmol, 1)
    # Nonbonded: quadratic to capture LJ + Coulomb curvature
    p_nb = np.polyfit(T_cal, nb_kjmol, 2)

    harm_at_Tbath = float(np.polyval(p_harm, T_bath))
    nb_at_Tbath = float(np.polyval(p_nb, T_bath))

    # --- Scale calibration so E_cal(T_bath) == E_ref_openmm ---
    scale_v = E_bond_ref_kjmol / harm_at_Tbath
    scale_s = E_nb_ref_kjmol / nb_at_Tbath

    # Scaled polynomial coefficients
    p_harm_s = np.array([p_harm[0] * scale_v, p_harm[1] * scale_v])
    p_nb_s = np.array([p_nb[0] * scale_s, p_nb[1] * scale_s, p_nb[2] * scale_s])

    # Dense T grid for numerical inversion (extrapolate well beyond cal range)
    T_min = max(1.0, T_cal[0] * 0.3)
    T_max = T_cal[-1] * 2.0
    T_fine = np.linspace(T_min, T_max, 10_000)

    harm_fine = np.polyval(p_harm_s, T_fine)  # monotone increasing
    nb_fine = np.polyval(p_nb_s, T_fine)       # monotone in fitted range

    def _invert(E_obs: np.ndarray, E_fine: np.ndarray) -> np.ndarray:
        """Invert E_obs against E_fine(T_fine) by linear interpolation."""
        return np.interp(E_obs, E_fine, T_fine)

    def invert_v(E_obs: np.ndarray) -> np.ndarray:
        return _invert(np.asarray(E_obs, dtype=float), harm_fine)

    def invert_s(E_obs: np.ndarray) -> np.ndarray:
        return _invert(np.asarray(E_obs, dtype=float), nb_fine)

    return invert_v, invert_s


# ---------------------------------------------------------------------------
# Data loading
# ---------------------------------------------------------------------------

def load_fictive_temperature_ensemble(
    job_dir: Path,
    lam: float,
    replicas: list[int],
    cal: dict[str, np.ndarray],
    t_grid: np.ndarray,
) -> dict[str, np.ndarray]:
    """Load replicas, invert E→T per replica, then ensemble-average on *t_grid*."""
    from fkt_utils import build_energy_csv_index, resolve_energy_csv

    index = build_energy_csv_index(job_dir, lam)
    T_v_stack: list[np.ndarray] = []
    T_s_stack: list[np.ndarray] = []
    T_k_stack: list[np.ndarray] = []
    used_replicas: list[int] = []

    for replica in replicas:
        csv_path = resolve_energy_csv(job_dir, lam, replica, index)
        if csv_path is None:
            continue
        data = np.genfromtxt(
            csv_path, delimiter=",", names=True, missing_values="", usemask=False
        )
        t = np.asarray(data["time_ps"], dtype=float)
        E_bond = np.asarray(data["E_bond_kjmol"], dtype=float)
        E_nb = np.asarray(data["E_nonbonded_kjmol"], dtype=float)
        T_k = np.asarray(data["T_kinetic_K"], dtype=float)

        pre = t < SWITCH_TIME_PS
        if not pre.any():
            continue
        E_bond_ref = float(np.mean(E_bond[pre]))
        E_nb_ref = float(np.mean(E_nb[pre]))
        invert_v, invert_s = build_inverters(cal, E_bond_ref, E_nb_ref, TEMPERATURE_K)

        T_v_stack.append(np.interp(t_grid, t, invert_v(E_bond), left=np.nan, right=np.nan))
        T_s_stack.append(np.interp(t_grid, t, invert_s(E_nb), left=np.nan, right=np.nan))
        T_k_stack.append(np.interp(t_grid, t, T_k, left=np.nan, right=np.nan))
        used_replicas.append(replica)

    if not T_v_stack:
        return {}

    return {
        "time": t_grid,
        "T_v": np.nanmean(np.vstack(T_v_stack), axis=0),
        "T_s": np.nanmean(np.vstack(T_s_stack), axis=0),
        "T_k": np.nanmean(np.vstack(T_k_stack), axis=0),
        "n_replicas": len(used_replicas),
    }


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--lambda", dest="lam", type=float, default=FIG3_SHOWCASE_LAMBDA)
    parser.add_argument(
        "--campaign-dir",
        type=Path,
        default=None,
        help="Campaign root (default: aging_weak_lambda/).",
    )
    parser.add_argument("--output-dir", type=Path, default=FIGURES_DIR)
    parser.add_argument(
        "--replicas",
        type=int,
        nargs="+",
        default=None,
        help="Replica indices (default: all with energy CSVs reaching --tmax-ps).",
    )
    parser.add_argument(
        "--calibration-file",
        type=Path,
        default=POTENTIAL_ENERGY_VS_T,
        help="Path to cav-hoomd potential_energy_vs_T.txt calibration table.",
    )
    parser.add_argument(
        "--tmin-ps",
        type=float,
        default=50.0,
        help="Plot window start (ps); trims rolling-average edge artefacts.",
    )
    parser.add_argument(
        "--tmax-ps",
        type=float,
        default=2000.0,
        help="Plot window end (ps).",
    )
    parser.add_argument(
        "--smooth-window",
        type=int,
        default=25,
        metavar="N",
        help="Uniform-window rolling average for display (points); 1 = no smoothing.",
    )
    args = parser.parse_args()

    from fkt_utils import list_available_energy_replicas

    job_dir = job_dir_path(args.lam, campaign_root=args.campaign_dir)
    if args.replicas is None:
        replicas = list_available_energy_replicas(job_dir, args.lam, min_tmax_ps=args.tmax_ps)
    else:
        replicas = args.replicas

    if not replicas:
        raise SystemExit(f"No energy CSV replicas in {job_dir} (tmax >= {args.tmax_ps} ps)")

    if not args.calibration_file.exists():
        raise SystemExit(f"Calibration file not found: {args.calibration_file}")
    cal = load_calibration(args.calibration_file)

    t_grid = np.arange(args.tmin_ps, args.tmax_ps + CSV_INTERVAL_PS, CSV_INTERVAL_PS)
    data = load_fictive_temperature_ensemble(job_dir, args.lam, replicas, cal, t_grid)
    if not data:
        raise SystemExit(f"No temperature data loaded from {job_dir}")

    t = data["time"]
    T_v = data["T_v"]
    T_s = data["T_s"]
    T_k = data["T_k"]

    w = max(1, args.smooth_window)
    T_v_plot = smooth_uniform(T_v, w)
    T_s_plot = smooth_uniform(T_s, w)
    T_k_plot = smooth_uniform(T_k, w)

    apply_paper_style(grid=False)

    # --- Plot ---
    fig, ax = plt.subplots(figsize=(8, 5), facecolor="white")
    ax.set_facecolor("white")

    ax.plot(
        t,
        T_v_plot,
        label=r"$T_{\mathrm{v}}$ Harmonic",
        color=COLOR_HARMONIC,
        lw=2.0,
    )
    ax.plot(
        t,
        T_s_plot,
        label=r"$T_{\mathrm{s}}$ LJ+Coulomb",
        color=COLOR_LJ_COULOMB,
        lw=2.0,
    )
    ax.plot(
        t,
        T_k_plot,
        label=r"$T_{\mathrm{k}}$ Kinetic",
        color=COLOR_KINETIC,
        lw=1.5,
        alpha=0.9,
    )
    ax.axhline(
        TEMPERATURE_K,
        color="gray",
        ls="--",
        lw=1.0,
        label=f"{TEMPERATURE_K:.0f} K",
    )
    ax.axvline(SWITCH_TIME_PS, color="k", ls=":", lw=1.0, alpha=0.7)

    n_rep = data.get("n_replicas", len(replicas))
    ax.set_xlim(args.tmin_ps, args.tmax_ps)
    ax.set_xlabel(r"$t$ (ps)")
    ax.set_ylabel(r"$T$ (K)")
    style_axes(ax, grid=False)
    paper_legend(ax, loc="best", fontsize=10)
    fig.tight_layout()

    args.output_dir.mkdir(parents=True, exist_ok=True)
    out_stem = args.output_dir / f"fig3c_fictive_temperatures_lam{args.lam:g}"
    save_figure(fig, out_stem)
    plt.close(fig)
    print(f"Ensemble over {n_rep} replicas, t=[{args.tmin_ps:.0f}, {args.tmax_ps:.0f}] ps")


if __name__ == "__main__":
    main()
