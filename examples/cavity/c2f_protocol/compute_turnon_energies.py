#!/usr/bin/env python
"""Evaluate turn-on vs relaxed bilinear and DSE energies from lambda=0 snapshots.

Reads position snapshots from a zero-coupling equilibrium run and, for each
coupling strength, evaluates:
  - E_coup_turnon: bilinear energy with the observed uncoupled photon q
  - E_coup_relaxed: bilinear energy after displacing q to finite-q equilibrium
  - E_dse: dipole self-energy (same for both cases)

Writes plain-text per-lambda timeseries and a summary table, plus a plot.
"""

from __future__ import annotations

import argparse
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

import openmm
from openmm import unit

from openmm.cavitymd import EnergyTracker, assign_force_groups, setup_gpu_step

from run_c2f import (
    HARTREE_TO_CM1,
    OMEGA_C_CM1,
    PHOTON_MASS_AMU,
    build_mka_system,
    add_cavity_particle,
    _select_platform,
)

_SCRIPT_DIR = Path(__file__).resolve().parent
DEFAULT_SNAPSHOTS = _SCRIPT_DIR / "equilibrium_output" / "eq10ns100K_lam0_snapshots.npz"
DEFAULT_OUT_DIR = _SCRIPT_DIR / "reviewer_response"
LAMBDAS = [0.01, 0.03, 0.042, 0.07, 0.09, 0.141]


def _lam_tag(lam: float) -> str:
    return f"{lam:g}".replace(".", "p")


def _build_eval_context(
    *,
    lam: float,
    seed: int,
    temperature_K: float,
    omega_c_cm1: float,
    platform_name: str | None,
):
    omegac_au = omega_c_cm1 / HARTREE_TO_CM1
    system, positions, n_atoms = build_mka_system(
        seed=seed,
        sample_bonds_at_T=temperature_K,
    )
    cavity_index = add_cavity_particle(system, positions)

    cavity_force = openmm.CavityForce(cavity_index, omegac_au, 0.0, PHOTON_MASS_AMU)
    cavity_force.setIncludeDipoleSelfEnergy(True)
    setup_gpu_step(cavity_force, lam, start_time_ps=0.0)
    system.addForce(cavity_force)

    displacer = openmm.CavityParticleDisplacer(
        cavity_index, omegac_au, PHOTON_MASS_AMU
    )
    displacer.setSwitchOnLambda(lam)
    displacer.setSwitchOnStep(2**31 - 1)
    system.addForce(displacer)

    group_map = assign_force_groups(system, include_dipole_self_energy=True)
    integrator = openmm.VerletIntegrator(0.001 * unit.picosecond)
    platform = _select_platform(platform_name)
    context = openmm.Context(system, integrator, platform)

    tracker = EnergyTracker(
        context, cavity_force, group_map, n_atoms, cavity_index
    )
    return context, displacer, tracker


def _positions_from_nm(frame_nm: np.ndarray) -> list:
    return [openmm.Vec3(*frame_nm[i]) * unit.nanometer for i in range(frame_nm.shape[0])]


def _fresh_energies(tracker: EnergyTracker) -> dict:
    """Bypass EnergyTracker step cache after setPositions/displacement."""
    tracker._cached = None
    tracker._cached_step = -1
    return tracker.get_energies()


def evaluate_turnon_energies(
    snapshots_path: Path,
    out_dir: Path,
    lambdas: list[float],
    burn_in_ps: float,
    platform_name: str | None,
) -> Path:
    data = np.load(snapshots_path)
    positions_nm = np.asarray(data["positions_nm"], dtype=float)
    times_ps = np.asarray(data["times_ps"], dtype=float)
    temperature_K = float(data.get("temperature_K", 100.0))
    omega_c_cm1 = float(data.get("omega_c_cm1", OMEGA_C_CM1))
    seed = int(data.get("seed", 42))

    mask = times_ps >= burn_in_ps
    positions_nm = positions_nm[mask]
    times_ps = times_ps[mask]
    if positions_nm.size == 0:
        raise ValueError(f"No frames after burn-in {burn_in_ps} ps")

    out_dir.mkdir(parents=True, exist_ok=True)
    summary_rows: list[str] = []

    for lam in lambdas:
        context, displacer, tracker = _build_eval_context(
            lam=lam,
            seed=seed,
            temperature_K=temperature_K,
            omega_c_cm1=omega_c_cm1,
            platform_name=platform_name,
        )

        coup_turnon = np.empty(len(times_ps), dtype=float)
        coup_relaxed = np.empty(len(times_ps), dtype=float)
        dse_vals = np.empty(len(times_ps), dtype=float)

        for i, frame_nm in enumerate(positions_nm):
            context.setPositions(_positions_from_nm(frame_nm))
            e_turn = _fresh_energies(tracker)
            coup_turnon[i] = e_turn.get("cavity_coupling", 0.0)
            dse_vals[i] = e_turn.get("cavity_dipole_self", 0.0)

            displacer.displaceToEquilibrium(context, lam)
            e_rel = _fresh_energies(tracker)
            coup_relaxed[i] = e_rel.get("cavity_coupling", 0.0)

        ts_path = out_dir / f"turnon_timeseries_lam{_lam_tag(lam)}.txt"
        with open(ts_path, "w", encoding="utf-8") as f:
            f.write(
                "# turn-on energies from lambda=0 equilibrium snapshots\n"
                f"# lambda_eval={lam}  T={temperature_K} K  burn_in_ps={burn_in_ps}\n"
                "time_ps  E_coup_turnon_kjmol  E_coup_relaxed_kjmol  E_dse_kjmol\n"
            )
            for t, et, er, ed in zip(times_ps, coup_turnon, coup_relaxed, dse_vals):
                f.write(f"{t:.6f}  {et:.8f}  {er:.8f}  {ed:.8f}\n")

        mean_turn = float(np.mean(coup_turnon))
        std_turn = float(np.std(coup_turnon))
        rms_turn = float(np.sqrt(np.mean(coup_turnon**2)))
        mean_rel = float(np.mean(coup_relaxed))
        std_rel = float(np.std(coup_relaxed))
        mean_dse = float(np.mean(dse_vals))
        std_dse = float(np.std(dse_vals))

        summary_rows.append(
            f"{lam:.6f}  {mean_turn:.8f}  {std_turn:.8f}  {rms_turn:.8f}  "
            f"{mean_rel:.8f}  {std_rel:.8f}  {mean_dse:.8f}  {std_dse:.8f}  "
            f"{len(times_ps)}\n"
        )
        print(
            f"lambda={lam:g}: mean_turn={mean_turn:.4f} rms_turn={rms_turn:.4f} "
            f"mean_rel={mean_rel:.4f} mean_dse={mean_dse:.4f} -> {ts_path.name}"
        )

    summary_path = out_dir / "turnon_energy_vs_lambda.txt"
    with open(summary_path, "w", encoding="utf-8") as f:
        f.write(
            "# Averages from lambda=0 equilibrium snapshots (turn-on vs relaxed bilinear)\n"
            f"# snapshots: {snapshots_path}\n"
            f"# burn_in_ps={burn_in_ps}\n"
            "lambda  mean_E_coup_turnon  std_E_coup_turnon  rms_E_coup_turnon  "
            "mean_E_coup_relaxed  std_E_coup_relaxed  mean_E_dse  std_E_dse  n_frames\n"
        )
        f.writelines(summary_rows)
    print(f"Wrote summary -> {summary_path}")
    return summary_path


def plot_turnon_energies(summary_path: Path, out_dir: Path) -> Path:
    rows = []
    with open(summary_path, encoding="utf-8") as f:
        for line in f:
            if line.startswith("#") or "lambda" in line and "mean_E" in line:
                continue
            if not line.strip():
                continue
            p = line.split()
            rows.append(
                {
                    "lam": float(p[0]),
                    "mean_turn": float(p[1]),
                    "std_turn": float(p[2]),
                    "rms_turn": float(p[3]),
                    "mean_rel": float(p[4]),
                    "mean_dse": float(p[6]),
                }
            )

    lam = np.array([r["lam"] for r in rows])
    mean_turn = np.array([r["mean_turn"] for r in rows])
    std_turn = np.array([r["std_turn"] for r in rows])
    rms_turn = np.array([r["rms_turn"] for r in rows])
    mean_rel = np.array([r["mean_rel"] for r in rows])
    mean_dse = np.array([r["mean_dse"] for r in rows])

    fig, axes = plt.subplots(1, 2, figsize=(13, 5.5))

    ax = axes[0]
    ax.errorbar(
        lam, mean_turn, yerr=std_turn, fmt="o-", color="#7f7f7f",
        label=r"turn-on $E_\mathrm{coup}$ (mean $\pm$ std)",
    )
    ax.plot(lam, mean_rel, "s-", color="#1f77b4",
            label=r"relaxed $E_\mathrm{coup}$ (finite-$q$)")
    ax.plot(lam, mean_dse, "^-", color="#bcbd22", label=r"$E_\mathrm{dse}$")
    ax.axhline(0, ls=":", color="gray", alpha=0.5)
    ax.set_xlabel(r"coupling $\lambda$ (a.u.)")
    ax.set_ylabel("energy (kJ/mol)")
    ax.set_title("(a) turn-on vs relaxed bilinear and DSE (linear)")
    ax.legend(fontsize=9)
    ax.grid(alpha=0.3)

    ax = axes[1]
    pos = lam > 0
    ax.loglog(lam[pos], np.abs(mean_rel[pos]), "s-", color="#1f77b4",
              label=r"$|E_\mathrm{coup}^\mathrm{relaxed}|$")
    ax.loglog(lam[pos], mean_dse[pos], "^-", color="#bcbd22",
              label=r"$E_\mathrm{dse}$")
    ax.loglog(lam[pos], rms_turn[pos], "o-", color="#7f7f7f",
              label=r"RMS $E_\mathrm{coup}^\mathrm{turnon}$")
    ax.set_xlabel(r"coupling $\lambda$ (a.u.)")
    ax.set_ylabel("|energy| (kJ/mol)")
    ax.set_title("(b) scaling crossover (log-log)")
    ax.legend(fontsize=9)
    ax.grid(alpha=0.3, which="both")

    fig.suptitle(
        "Turn-on bilinear vs DSE from $\\lambda=0$ equilibrium (100 K, 10 ns)",
        fontsize=12,
    )
    fig.tight_layout(rect=(0, 0, 1, 0.94))
    out = out_dir / "turnon_energy_vs_lambda.png"
    fig.savefig(out, dpi=180)
    plt.close(fig)
    print(f"Wrote plot -> {out}")
    return out


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--snapshots", type=Path, default=DEFAULT_SNAPSHOTS)
    parser.add_argument("--out-dir", type=Path, default=DEFAULT_OUT_DIR)
    parser.add_argument("--burn-in-ps", type=float, default=1000.0)
    parser.add_argument("--platform", default=None)
    parser.add_argument("--no-plot", action="store_true")
    args = parser.parse_args()

    if not args.snapshots.exists():
        raise SystemExit(f"Snapshots not found: {args.snapshots}")

    summary = evaluate_turnon_energies(
        snapshots_path=args.snapshots,
        out_dir=args.out_dir,
        lambdas=LAMBDAS,
        burn_in_ps=args.burn_in_ps,
        platform_name=args.platform,
    )
    if not args.no_plot:
        plot_turnon_energies(summary, args.out_dir)


if __name__ == "__main__":
    main()
