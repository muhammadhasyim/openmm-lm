#!/usr/bin/env python
"""Consolidated 'regenerated main-text' overview for the reviewer response.

Reads the steady-state energy table and material-time summary produced by
``analyze_reviewer_response.py`` / ``analyze_material_time.py`` and builds a
single overview figure across the confirmed main-text coupling set
(lambda = 0.01, 0.03, 0.042, 0.07) at 100 K and 50 K, with the ultrastrong
cases (0.09, 0.141) marked as SI-only.

Panels:
  (a) bilinear |E_coup| and DSE |E_dse| vs lambda (log-log), 100 K vs 50 K
  (b) net cavity energy (coup + DSE + harmonic) vs lambda -> finite-q shift ~ 0
  (c) effective structural relaxation time tau_s vs lambda
"""

from __future__ import annotations

import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

_SCRIPT_DIR = Path(__file__).resolve().parent
OUT_DIR = _SCRIPT_DIR / "reviewer_response"
MAIN_TEXT = [0.01, 0.03, 0.042, 0.07]
SI_ONLY = [0.09, 0.141]


def parse_table(path: Path) -> dict:
    """tag -> {lambda: (E_coup, E_dse, E_harm, E_bond)}."""
    out: dict[str, dict[float, tuple]] = {}
    for line in path.read_text().splitlines():
        if line.startswith("#") or line.strip().startswith("runtime_tag") or not line.strip():
            continue
        p = line.split()
        if len(p) < 7:
            continue
        tag = p[0]
        lam = float(p[1])
        out.setdefault(tag, {})[lam] = (
            float(p[2]), float(p[3]), float(p[5]), float(p[6])
        )
    return out


def eff_tau(summary_path: Path) -> dict:
    """(temperature_K rounded) -> {lambda: tau_s_eff_ps} for dse-on, default freq."""
    s = json.loads(summary_path.read_text())
    out: dict[int, dict[float, float]] = {}
    for r in s["effective_tau"]["rows"]:
        if not r["dse"]:
            continue
        if abs(r["omega_c_cm1"] - 1560.0) > 1.0:
            continue
        T = int(round(r["temperature_K"]))
        out.setdefault(T, {})[round(r["lambda"], 3)] = r["tau_s_eff_ps"]
    return out


def main() -> None:
    table = parse_table(OUT_DIR / "energy_decomposition_table.txt")
    taus = eff_tau(OUT_DIR / "material_time_summary.json")

    series = {"100 K": "eq100K", "50 K": "eq50K"}
    colors = {"100 K": "#1f77b4", "50 K": "#d62728"}

    fig, axes = plt.subplots(1, 3, figsize=(16, 5))

    # (a) bilinear vs DSE scaling
    ax = axes[0]
    for label, tag in series.items():
        if tag not in table:
            continue
        lams = sorted(l for l in table[tag] if l in MAIN_TEXT)
        if not lams:
            continue
        ec = [abs(table[tag][l][0]) for l in lams]
        ed = [abs(table[tag][l][1]) for l in lams]
        ax.loglog(lams, ec, "o-", color=colors[label], label=f"|E_coup| {label}")
        ax.loglog(lams, ed, "s--", color=colors[label], alpha=0.6, label=f"|E_dse| {label}")
        # SI-only ultrastrong points (faded)
        si = sorted(l for l in table[tag] if l in SI_ONLY)
        if si:
            ax.loglog(si, [abs(table[tag][l][0]) for l in si], "x",
                      color=colors[label], alpha=0.35)
    ax.set_xlabel(r"$\lambda$ (a.u.)")
    ax.set_ylabel("steady-state |energy| (kJ/mol)")
    ax.set_title("(a) bilinear vs DSE ($\\propto\\lambda^2$, ratio 2:1)")
    ax.legend(fontsize=8)
    ax.grid(alpha=0.3, which="both")

    # (b) net cavity energy ~ 0 (finite-q shift)
    ax = axes[1]
    for label, tag in series.items():
        if tag not in table:
            continue
        lams = sorted(l for l in table[tag] if l in MAIN_TEXT)
        net = [sum(table[tag][l][k] for k in (0, 1, 2)) for l in lams]
        ax.plot(lams, net, "o-", color=colors[label], label=label)
    ax.axhline(0, ls=":", color="gray")
    ax.set_xlabel(r"$\lambda$ (a.u.)")
    ax.set_ylabel("net cavity energy coup+DSE+harm (kJ/mol)")
    ax.set_title("(b) finite-q shift: net cavity energy $\\approx$ 0")
    ax.legend(fontsize=9)
    ax.grid(alpha=0.3)

    # (c) effective structural relaxation time
    ax = axes[2]
    for label, tag in series.items():
        T = 100 if "100" in label else 50
        if T not in taus:
            continue
        lams = sorted(l for l in taus[T] if l in MAIN_TEXT)
        if not lams:
            continue
        ax.semilogy(lams, [taus[T][round(l, 3)] for l in lams], "o-",
                    color=colors[label], label=label)
    ax.set_xlabel(r"$\lambda$ (a.u.)")
    ax.set_ylabel(r"effective $\tau_\mathrm{s}$ (ps)")
    ax.set_title(r"(c) structural relaxation vs $\lambda$")
    ax.legend(fontsize=9)
    ax.grid(alpha=0.3, which="both")

    fig.suptitle("Regenerated main-text overview (main-text set: "
                 r"$\lambda \in \{0.01, 0.03, 0.042, 0.07\}$; 0.09/0.141 -> SI)",
                 fontsize=13)
    fig.tight_layout(rect=(0, 0, 1, 0.95))
    out = OUT_DIR / "regen_main_text_overview.png"
    fig.savefig(out, dpi=180)
    plt.close(fig)
    print(f"wrote {out}")


if __name__ == "__main__":
    main()
