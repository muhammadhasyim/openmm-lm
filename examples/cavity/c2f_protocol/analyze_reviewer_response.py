#!/usr/bin/env python
"""Reviewer-response analyses for the non-thermal aging paper.

Consumes the NVT cavity-equilibrium CSV logs produced by
``run_cavity_equilibrium.py`` (columns documented in that script) and produces
the figures and summary tables that answer the referee comments:

  Calc 1  -- weak-coupling DSE on/off comparison (R1: not a PF/DSE artifact)
  Calc 3  -- steady-state DSE vs bilinear energy decomposition (R1/R2)
  Calc 5  -- scaling of the cavity energy terms with lambda (R1/R2)
  Fig 3b  -- harmonic-curve decomposition: cavity mode vs dipole vibration (R2)

Each equilibrium run is auto-discovered from ``*_final_state.npz`` (which stores
lambda, temperature, DSE flag, finite-q flag, omega_c) and paired with its
``*_energies.csv``.  Steady-state averages discard a burn-in fraction.

Usage:
  python analyze_reviewer_response.py [--burn-in 0.2] [--runtime-tag eq10ns100K]
"""

from __future__ import annotations

import argparse
import json
from dataclasses import dataclass, asdict
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

_SCRIPT_DIR = Path(__file__).resolve().parent
EQ_DIR = _SCRIPT_DIR / "equilibrium_output"
OUT_DIR = _SCRIPT_DIR / "reviewer_response"

# mKA system constants (match run_c2f.py / plot_equilibrium.py)
NUM_MOL = 250
KB_HARTREE_PER_K = 3.16681153e-6
KJMOL_TO_HARTREE = 4.184 / (6.02214076e23 * 1e3) * 2625.49962


@dataclass
class Run:
    """One equilibrium run: metadata + steady-state energy/temperature means."""

    name: str
    lam: float
    temperature_K: float
    dse: bool
    finite_q: object
    omega_c_cm1: float
    runtime_tag: str
    n_samples: int
    # steady-state means (kJ/mol unless noted)
    E_bond: float
    E_nonbonded: float
    E_cav_harmonic: float
    E_cav_coupling: float
    E_cav_dse: float
    T_kinetic_K: float
    T_v_fictive_K: float
    T_s_fictive_K: float


def _load_csv(path: Path) -> dict[str, np.ndarray]:
    data = np.genfromtxt(path, delimiter=",", names=True)
    return {name: np.asarray(data[name], dtype=float) for name in data.dtype.names}


def _steady_mean(arr: np.ndarray, burn_in: float) -> float:
    a = np.asarray(arr, dtype=float)
    a = a[np.isfinite(a)]
    if a.size == 0:
        return float("nan")
    start = int(round(burn_in * a.size))
    return float(np.mean(a[start:])) if a.size - start > 0 else float(np.mean(a))


def discover_runs(burn_in: float) -> list[Run]:
    runs: list[Run] = []
    for npz_path in sorted(EQ_DIR.glob("*_final_state.npz")):
        prefix = npz_path.name[: -len("_final_state.npz")]
        csv_path = EQ_DIR / f"{prefix}_energies.csv"
        if not csv_path.exists():
            continue
        try:
            meta = np.load(npz_path)
        except Exception:
            continue
        try:
            data = _load_csv(csv_path)
        except Exception:
            continue
        if "time_ps" not in data:
            continue
        # runtime tag: text before the first "_lam" or "_dse"
        runtime_tag = prefix.split("_lam")[0].split("_dse")[0]
        runs.append(
            Run(
                name=prefix,
                lam=float(meta.get("lambda_coupling", np.nan)),
                temperature_K=float(meta.get("temperature_K", np.nan)),
                dse=bool(meta.get("include_dipole_self_energy", False)),
                finite_q=(None if "finite_q" not in meta else bool(meta["finite_q"])),
                omega_c_cm1=float(meta.get("omega_c_cm1", np.nan)),
                runtime_tag=runtime_tag,
                n_samples=int(len(data["time_ps"])),
                E_bond=_steady_mean(data.get("E_bond_kjmol", np.array([])), burn_in),
                E_nonbonded=_steady_mean(
                    data.get("E_nonbonded_kjmol", np.array([])), burn_in
                ),
                E_cav_harmonic=_steady_mean(
                    data.get("E_cav_harmonic_kjmol", np.array([])), burn_in
                ),
                E_cav_coupling=_steady_mean(
                    data.get("E_cav_coupling_kjmol", np.array([])), burn_in
                ),
                E_cav_dse=_steady_mean(
                    data.get("E_cav_dse_kjmol", np.array([])), burn_in
                ),
                T_kinetic_K=_steady_mean(
                    data.get("T_kinetic_K", np.array([])), burn_in
                ),
                T_v_fictive_K=_steady_mean(
                    data.get("T_v_fictive_K", np.array([])), burn_in
                ),
                T_s_fictive_K=_steady_mean(
                    data.get("T_s_fictive_K", np.array([])), burn_in
                ),
            )
        )
    return runs


def _find(runs, runtime_tag, lam, dse, temperature_K=100.0, tol=1e-6):
    for r in runs:
        if (
            r.runtime_tag == runtime_tag
            and abs(r.lam - lam) < tol
            and r.dse == dse
            and abs(r.temperature_K - temperature_K) < 1e-3
        ):
            return r
    return None


# --------------------------------------------------------------------------
# Calc 1 -- weak-coupling DSE on/off comparison
# --------------------------------------------------------------------------
def calc1_dse_onoff(runtime_tag: str, lam: float, burn_in: float) -> dict | None:
    on_csv = EQ_DIR / f"{runtime_tag}_lam{lam}_dse_on_energies.csv"
    off_csv = EQ_DIR / f"{runtime_tag}_lam{lam}_dse_off_energies.csv"
    if not (on_csv.exists() and off_csv.exists()):
        print(f"[Calc1] missing CSVs for {runtime_tag} lam={lam}")
        return None
    on, off = _load_csv(on_csv), _load_csv(off_csv)

    fig, axes = plt.subplots(2, 2, figsize=(11, 7.5))
    fig.suptitle(
        f"Calc 1 (R1): weak coupling $\\lambda={lam}$ -- DSE on vs off "
        f"({runtime_tag}, 100 K)",
        fontsize=12,
    )

    def _plot(ax, key, label):
        ax.plot(on["time_ps"], on[key], lw=0.8, color="#d62728", label="DSE on")
        ax.plot(off["time_ps"], off[key], lw=0.8, color="#1f77b4", label="DSE off")
        ax.set_xlabel("time (ps)")
        ax.set_ylabel(label)
        ax.legend(fontsize=8)
        ax.grid(alpha=0.3)

    _plot(axes[0, 0], "T_v_fictive_K", r"$T_\mathrm{v}$ fictive (K)")
    _plot(axes[0, 1], "T_s_fictive_K", r"$T_\mathrm{s}$ fictive (K)")
    _plot(axes[1, 0], "E_cav_coupling_kjmol", "bilinear $E_\\mathrm{coup}$ (kJ/mol)")
    _plot(axes[1, 1], "E_cav_dse_kjmol", "DSE $E_\\mathrm{dse}$ (kJ/mol)")
    fig.tight_layout(rect=(0, 0, 1, 0.96))
    out = OUT_DIR / f"calc1_dse_onoff_lam{lam}_{runtime_tag}.png"
    fig.savefig(out, dpi=180)
    plt.close(fig)

    def _m(d, k):
        return _steady_mean(d[k], burn_in)

    summary = {
        "lambda": lam,
        "runtime_tag": runtime_tag,
        "T_v_fictive_K_dse_on": _m(on, "T_v_fictive_K"),
        "T_v_fictive_K_dse_off": _m(off, "T_v_fictive_K"),
        "T_s_fictive_K_dse_on": _m(on, "T_s_fictive_K"),
        "T_s_fictive_K_dse_off": _m(off, "T_s_fictive_K"),
        "E_cav_coupling_dse_on": _m(on, "E_cav_coupling_kjmol"),
        "E_cav_dse_dse_on": _m(on, "E_cav_dse_kjmol"),
        "figure": str(out),
    }
    print(f"[Calc1] wrote {out}")
    return summary


# --------------------------------------------------------------------------
# Calc 3 + Calc 5 -- DSE vs bilinear decomposition and scaling vs lambda
# --------------------------------------------------------------------------
def calc3_calc5_scaling(runs: list[Run], runtime_tag: str) -> dict:
    sel = sorted(
        [r for r in runs if r.runtime_tag == runtime_tag and r.dse],
        key=lambda r: r.lam,
    )
    lam = np.array([r.lam for r in sel])
    e_coup = np.array([abs(r.E_cav_coupling) for r in sel])
    e_dse = np.array([abs(r.E_cav_dse) for r in sel])

    rows = []
    for r in sel:
        ratio = abs(r.E_cav_coupling) / abs(r.E_cav_dse) if r.E_cav_dse else float("nan")
        rows.append(
            {
                "lambda": r.lam,
                "E_cav_coupling_kjmol": r.E_cav_coupling,
                "E_cav_dse_kjmol": r.E_cav_dse,
                "abs_ratio_bilinear_over_dse": ratio,
                "E_cav_harmonic_kjmol": r.E_cav_harmonic,
                "E_bond_kjmol": r.E_bond,
            }
        )

    # Power-law fits |E| = A * lambda^p  (log-log linear fit)
    def _fit(y):
        m = (lam > 0) & (y > 0)
        if m.sum() < 2:
            return float("nan"), float("nan")
        p, logA = np.polyfit(np.log(lam[m]), np.log(y[m]), 1)
        return float(p), float(np.exp(logA))

    p_coup, A_coup = _fit(e_coup)
    p_dse, A_dse = _fit(e_dse)

    fig, ax = plt.subplots(figsize=(7.5, 6))
    ax.loglog(lam, e_coup, "o-", color="#7f7f7f",
              label=f"bilinear $|E_\\mathrm{{coup}}|$  (fit $\\propto\\lambda^{{{p_coup:.2f}}}$)")
    ax.loglog(lam, e_dse, "s-", color="#bcbd22",
              label=f"DSE $|E_\\mathrm{{dse}}|$  (fit $\\propto\\lambda^{{{p_dse:.2f}}}$)")
    if len(lam):
        xx = np.linspace(lam.min(), lam.max(), 50)
        if np.isfinite(p_coup):
            ax.loglog(xx, A_coup * xx ** p_coup, "--", color="#7f7f7f", alpha=0.5)
        if np.isfinite(p_dse):
            ax.loglog(xx, A_dse * xx ** p_dse, "--", color="#bcbd22", alpha=0.5)
    ax.set_xlabel(r"coupling $\lambda$ (a.u.)")
    ax.set_ylabel("steady-state |energy| (kJ/mol)")
    ax.set_title(f"Calc 3/5 (R1,R2): bilinear vs DSE scaling ({runtime_tag}, 100 K)")
    ax.legend()
    ax.grid(alpha=0.3, which="both")
    out = OUT_DIR / f"calc5_scaling_{runtime_tag}.png"
    fig.tight_layout()
    fig.savefig(out, dpi=180)
    plt.close(fig)
    print(f"[Calc3/5] wrote {out}")

    return {
        "runtime_tag": runtime_tag,
        "rows": rows,
        "bilinear_power": p_coup,
        "dse_power": p_dse,
        "figure": str(out),
    }


# --------------------------------------------------------------------------
# Fig 3b -- harmonic-curve decomposition (cavity mode vs dipole vibration)
# --------------------------------------------------------------------------
def fig3b_decomposition(runs: list[Run], runtime_tag: str) -> dict:
    sel = sorted(
        [r for r in runs if r.runtime_tag == runtime_tag and r.dse],
        key=lambda r: r.lam,
    )
    lam = np.array([r.lam for r in sel])
    e_cav_harm = np.array([r.E_cav_harmonic for r in sel])  # cavity-mode harmonic
    e_bond = np.array([r.E_bond for r in sel])  # dipole vibrational

    fig, ax = plt.subplots(figsize=(7.5, 6))
    width = 0.35
    x = np.arange(len(lam))
    ax.bar(x - width / 2, e_bond, width, label="dipole vibration ($E_\\mathrm{bond}$)",
           color="#9467bd")
    ax.bar(x + width / 2, e_cav_harm, width,
           label="cavity mode ($E_\\mathrm{cav,harm}$)", color="#e377c2")
    ax.set_xticks(x)
    ax.set_xticklabels([f"{l:g}" for l in lam])
    ax.set_xlabel(r"coupling $\lambda$ (a.u.)")
    ax.set_ylabel("steady-state energy (kJ/mol)")
    ax.set_title(f'Fig 3b decomposition (R2): "Harmonic" = vibration + cavity mode'
                 f"\n({runtime_tag}, 100 K)")
    ax.legend()
    ax.grid(alpha=0.3, axis="y")
    out = OUT_DIR / f"fig3b_decomposition_{runtime_tag}.png"
    fig.tight_layout()
    fig.savefig(out, dpi=180)
    plt.close(fig)
    print(f"[Fig3b] wrote {out}")

    return {
        "runtime_tag": runtime_tag,
        "lambda": lam.tolist(),
        "E_bond_dipole_vibration": e_bond.tolist(),
        "E_cav_harmonic_cavity_mode": e_cav_harm.tolist(),
        "figure": str(out),
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--burn-in", type=float, default=0.2,
                        help="fraction of each trajectory discarded as burn-in")
    parser.add_argument("--runtime-tag", default="eq10ns100K",
                        help="preferred runtime tag for scaling/decomposition")
    args = parser.parse_args()

    OUT_DIR.mkdir(parents=True, exist_ok=True)
    runs = discover_runs(args.burn_in)
    print(f"Discovered {len(runs)} equilibrium runs.")

    summary: dict = {"burn_in": args.burn_in, "runs": [asdict(r) for r in runs]}

    # Calc 1: prefer long runs, fall back to 1 ns
    summary["calc1"] = (
        calc1_dse_onoff(args.runtime_tag, 0.01, args.burn_in)
        or calc1_dse_onoff("eq100K", 0.01, args.burn_in)
    )

    # Calc 3 + 5: scaling at the chosen runtime tag (fall back to eq100K)
    tag = args.runtime_tag if any(r.runtime_tag == args.runtime_tag for r in runs) else "eq100K"
    summary["calc3_calc5"] = calc3_calc5_scaling(runs, tag)
    summary["fig3b"] = fig3b_decomposition(runs, tag)

    out_json = OUT_DIR / "reviewer_response_summary.json"
    out_json.write_text(json.dumps(summary, indent=2))
    print(f"\nWrote summary -> {out_json}")

    # Human-readable table
    lines = ["# Steady-state energy decomposition (DSE-on runs)\n",
             f"# burn-in fraction = {args.burn_in}\n",
             "runtime_tag  lambda  E_coupling(kJ/mol)  E_dse(kJ/mol)  |coup|/|dse|  E_cav_harm  E_bond\n"]
    for r in sorted(runs, key=lambda r: (r.runtime_tag, r.lam, r.dse)):
        if not r.dse:
            continue
        ratio = abs(r.E_cav_coupling) / abs(r.E_cav_dse) if r.E_cav_dse else float("nan")
        lines.append(
            f"{r.runtime_tag:12s}  {r.lam:6.3f}  {r.E_cav_coupling:18.4f}  "
            f"{r.E_cav_dse:13.4f}  {ratio:11.3f}  {r.E_cav_harmonic:10.4f}  {r.E_bond:8.4f}\n"
        )
    (OUT_DIR / "energy_decomposition_table.txt").write_text("".join(lines))
    print(f"Wrote table -> {OUT_DIR / 'energy_decomposition_table.txt'}")


if __name__ == "__main__":
    main()
