#!/usr/bin/env python
"""Material-time / structural-relaxation analysis for the aging paper.

Implements two reviewer-relevant quantities:

1.  Material time h(t) along a cooling trajectory (MTTI, observables.rst):
        dh/dt = tau_ref / tau_s(T_s(t)),   h(t) = \\int_0^t dh
    where tau_s(T) is the structural relaxation time from the F(k,t)/F(k,0)=0.1
    threshold (cav-hoomd/relaxation_times_vs_temperature.txt).  The aging rate
    dh/dt > 1 means the system ages faster than real time.

2.  Effective structural relaxation time per coupling / cavity frequency:
    map the steady-state structural fictive temperature T_s of each equilibrium
    run onto tau_s(T_s).  A flat tau_s vs cavity frequency supports the referee
    point (R2 / Fig S3) that polariton formation is not the relevant process.

tau_s(T) is modelled as log10(tau) linear in 1/T (Arrhenius interpolation with
linear extrapolation), which captures the table over the relevant range.

Usage:
  python analyze_material_time.py \
      --relaxation-file ../../../cav-hoomd/relaxation_times_vs_temperature.txt \
      --cooling-csv fig5_output/fig5_averaged.csv
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

_SCRIPT_DIR = Path(__file__).resolve().parent
EQ_DIR = _SCRIPT_DIR / "equilibrium_output"
OUT_DIR = _SCRIPT_DIR / "reviewer_response"
DEFAULT_RELAX = _SCRIPT_DIR.parents[2] / "cav-hoomd" / "relaxation_times_vs_temperature.txt"


class TauSModel:
    """log10(tau_s) linear in 1/T, fit to the F(k,t)=0.1 relaxation table."""

    def __init__(self, relaxation_file: Path):
        data = np.loadtxt(relaxation_file, comments="#", usecols=(0, 1, 2))
        self.T = data[:, 0]
        self.tau = data[:, 2]
        order = np.argsort(self.T)
        self.T, self.tau = self.T[order], self.tau[order]
        inv_T = 1.0 / self.T
        self.slope, self.intercept = np.polyfit(inv_T, np.log10(self.tau), 1)
        self.T_min, self.T_max = float(self.T.min()), float(self.T.max())

    def tau_of_T(self, T_K):
        T = np.clip(np.asarray(T_K, dtype=float), 1e-6, None)
        return 10.0 ** (self.slope / T + self.intercept)


def material_time(tau_model, time_ps, T_s_K, tau_ref_ps):
    """h(t) = int_0^t tau_ref/tau_s(T_s) dt';  also return aging rate dh/dt."""
    rate = tau_ref_ps / tau_model.tau_of_T(T_s_K)
    h = np.concatenate([[0.0], np.cumsum(0.5 * (rate[1:] + rate[:-1]) * np.diff(time_ps))])
    return h, rate


def analyze_cooling(tau_model, cooling_csv: Path) -> dict:
    d = np.genfromtxt(cooling_csv, delimiter=",", names=True)
    t = np.asarray(d["time_ps"], dtype=float)
    T_s = np.asarray(d["T_s_fictive_K"], dtype=float)
    good = np.isfinite(t) & np.isfinite(T_s)
    t, T_s = t[good], T_s[good]
    if t.size < 2:
        print("[material-time] cooling trajectory has too few finite points")
        return {}
    tau_s = tau_model.tau_of_T(T_s)
    tau_ref = float(np.nanmin(tau_s))  # fastest relaxation along the run
    h, rate = material_time(tau_model, t, T_s, tau_ref)

    fig, axes = plt.subplots(3, 1, figsize=(8, 9), sharex=True)
    axes[0].plot(t, T_s, color="#d62728")
    axes[0].set_ylabel(r"$T_\mathrm{s}$ fictive (K)")
    axes[0].grid(alpha=0.3)
    axes[1].semilogy(t, tau_s, color="#1f77b4")
    axes[1].set_ylabel(r"$\tau_\mathrm{s}(T_\mathrm{s})$ (ps)")
    axes[1].grid(alpha=0.3, which="both")
    axes[2].plot(t, h, color="#2ca02c")
    axes[2].set_ylabel(r"material time $h(t)$")
    axes[2].set_xlabel("time (ps)")
    axes[2].grid(alpha=0.3)
    fig.suptitle("Material time along the cooling protocol (R2 structural-cooling metric)")
    fig.tight_layout(rect=(0, 0, 1, 0.97))
    out = OUT_DIR / "material_time_cooling.png"
    fig.savefig(out, dpi=180)
    plt.close(fig)
    print(f"[material-time] wrote {out}")
    return {
        "tau_ref_ps": tau_ref,
        "h_final": float(h[-1]),
        "mean_aging_rate": float(np.mean(rate)),
        "max_aging_rate": float(np.max(rate)),
        "figure": str(out),
    }


def effective_tau_per_run(tau_model, burn_in: float) -> dict:
    """Map steady-state T_s of each equilibrium run -> effective tau_s."""
    rows = []
    for npz_path in sorted(EQ_DIR.glob("*_final_state.npz")):
        prefix = npz_path.name[: -len("_final_state.npz")]
        csv_path = EQ_DIR / f"{prefix}_energies.csv"
        if not csv_path.exists():
            continue
        try:
            meta = np.load(npz_path)
            d = np.genfromtxt(csv_path, delimiter=",", names=True)
        except Exception:
            continue
        if "T_s_fictive_K" not in (d.dtype.names or ()):
            continue
        ts = np.asarray(d["T_s_fictive_K"], dtype=float)
        ts = ts[np.isfinite(ts)]
        if ts.size == 0:
            continue
        start = int(round(burn_in * ts.size))
        ts_ss = float(np.mean(ts[start:])) if ts.size - start > 0 else float(np.mean(ts))
        rows.append(
            {
                "name": prefix,
                "lambda": float(meta.get("lambda_coupling", np.nan)),
                "temperature_K": float(meta.get("temperature_K", np.nan)),
                "dse": bool(meta.get("include_dipole_self_energy", False)),
                "omega_c_cm1": float(meta.get("omega_c_cm1", np.nan)),
                "T_s_steady_K": ts_ss,
                "tau_s_eff_ps": float(tau_model.tau_of_T(ts_ss)),
            }
        )

    # Frequency-sweep plot (R2): tau_s vs omega_c at fixed weak coupling.
    # One consistent (1 ns) point per frequency: freqsweep runs + the eq100K 1 ns
    # reference at 1560; exclude the 10 ns and smoke duplicates.
    by_wc: dict[float, dict] = {}
    for r in rows:
        if not (abs(r["lambda"] - 0.01) < 1e-6 and r["dse"] and abs(r["temperature_K"] - 100) < 1):
            continue
        if r["name"].startswith("freqsweep") or r["name"] == "eq100K_lam0.01_dse_on":
            by_wc[r["omega_c_cm1"]] = r
    fsweep = [by_wc[k] for k in sorted(by_wc)]
    if len(fsweep) >= 2:
        wc = np.array([r["omega_c_cm1"] for r in fsweep])
        tau = np.array([r["tau_s_eff_ps"] for r in fsweep])
        fig, ax = plt.subplots(figsize=(7.5, 5))
        ax.semilogy(wc, tau, "o-", color="#9467bd")
        ax.axvline(1560, ls="--", color="gray", alpha=0.6, label="A-A resonance (1560)")
        ax.set_xlabel(r"cavity frequency $\omega_c$ (cm$^{-1}$)")
        ax.set_ylabel(r"effective $\tau_\mathrm{s}$ (ps)")
        ax.set_title("Calc 2 / R2: structural relaxation vs cavity frequency ($\\lambda=0.01$)")
        ax.legend()
        ax.grid(alpha=0.3, which="both")
        out = OUT_DIR / "calc2_tau_vs_frequency.png"
        fig.tight_layout()
        fig.savefig(out, dpi=180)
        plt.close(fig)
        print(f"[material-time] wrote {out}")
    return {"rows": rows}


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--relaxation-file", type=Path, default=DEFAULT_RELAX)
    parser.add_argument("--cooling-csv", type=Path,
                        default=_SCRIPT_DIR / "fig5_output" / "fig5_averaged.csv")
    parser.add_argument("--burn-in", type=float, default=0.2)
    args = parser.parse_args()

    OUT_DIR.mkdir(parents=True, exist_ok=True)
    if not args.relaxation_file.exists():
        raise SystemExit(f"relaxation file not found: {args.relaxation_file}")
    tau_model = TauSModel(args.relaxation_file)
    print(f"tau_s(T) fit: log10(tau_ps) = {tau_model.slope:.2f}/T + {tau_model.intercept:.3f}"
          f"  (table T range {tau_model.T_min:.0f}-{tau_model.T_max:.0f} K)")

    summary = {
        "tau_model": {"slope_K": tau_model.slope, "intercept": tau_model.intercept,
                      "T_min": tau_model.T_min, "T_max": tau_model.T_max},
    }
    if args.cooling_csv.exists():
        summary["cooling"] = analyze_cooling(tau_model, args.cooling_csv)
    else:
        print(f"[material-time] cooling CSV not found: {args.cooling_csv}")
    summary["effective_tau"] = effective_tau_per_run(tau_model, args.burn_in)

    out_json = OUT_DIR / "material_time_summary.json"
    out_json.write_text(json.dumps(summary, indent=2))
    print(f"Wrote summary -> {out_json}")


if __name__ == "__main__":
    main()
