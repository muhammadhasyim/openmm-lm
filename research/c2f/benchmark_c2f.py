#!/usr/bin/env python
"""
Benchmark: Python macro-step loop vs GPU-native C2F feedback loop.

Profiles wall-clock time broken down by category:
  - integrator.step()     (GPU force computation + integration)
  - setLambdaCoupling()   (Python→C++ param push, old method only)
  - updateParametersInContext()  (full param sync, old method only)
  - getState()            (GPU→CPU energy transfer)
  - setParameter()        (scalar context param write)
  - Python overhead       (everything else: loop, numpy, etc.)

Reports per-category time, GPU utilization, host sync counts, and speedup.

Usage:
    python benchmark_c2f.py
    python benchmark_c2f.py --run-ps 50 --sizes 250 1000 4000
"""

import argparse
import sys
import time as wall_time
from collections import defaultdict

import numpy as np

try:
    import openmm
    from openmm import unit
except ImportError:
    sys.exit("OpenMM (cavity-md branch) required.")

from openmm.cavitymd import (
    Units,
    DualThermostat,
    assign_force_groups,
    setup_gpu_adaptive_square_wave,
)

# ---------------------------------------------------------------------------
BOHR_TO_NM = 0.0529177
HARTREE_TO_KJMOL = 2625.5
HARTREE_TO_CM1 = 219474.63
MASS_A, MASS_B = 16.0, 14.0
K_AA_AU, R0_AA_AU = 0.73204, 2.281655158
K_BB_AU, R0_BB_AU = 1.4325, 2.0743522177
EPS_AA_AU, SIG_AA_AU = 1.6685e-4, 6.2304
EPS_BB_AU, SIG_BB_AU = 8.3426e-5, 5.4828
EPS_AB_AU, SIG_AB_AU = 2.5028e-4, 4.9832
RCUT_AU = 15.0
CHARGE_MAG = 0.3
FRAC_AA = 0.8
OMEGA_C_CM1 = 1560.0
PHOTON_MASS_AMU = 1.0 / 1822.888
BUSSI_TAU_PS = 1.0
DT_PS = 0.001
OMEGAC_AU = OMEGA_C_CM1 / HARTREE_TO_CM1


class Timer:
    """Accumulating timer for profiling categories."""
    def __init__(self):
        self.totals = defaultdict(float)
        self.counts = defaultdict(int)
        self._start = None
        self._cat = None

    def start(self, category):
        self._cat = category
        self._start = wall_time.perf_counter()

    def stop(self):
        dt = wall_time.perf_counter() - self._start
        self.totals[self._cat] += dt
        self.counts[self._cat] += 1

    def total(self):
        return sum(self.totals.values())

    def report(self, label, run_ps):
        total = self.total()
        print(f"\n  {label}  ({total:.3f} s total, {total/run_ps:.4f} s/ps)")
        print(f"  {'Category':<30} {'Time(s)':>8} {'%':>6} {'Calls':>8} {'us/call':>8}")
        print(f"  {'-'*66}")
        for cat in sorted(self.totals.keys(), key=lambda c: -self.totals[c]):
            t = self.totals[cat]
            n = self.counts[cat]
            pct = 100.0 * t / total if total > 0 else 0
            us = 1e6 * t / n if n > 0 else 0
            print(f"  {cat:<30} {t:>8.4f} {pct:>5.1f}% {n:>8} {us:>8.1f}")


def _conv_bond(k_au, r0_au):
    return k_au * HARTREE_TO_KJMOL / (BOHR_TO_NM**2), r0_au * BOHR_TO_NM

def _conv_lj(eps_au, sig_au):
    return eps_au * HARTREE_TO_KJMOL, sig_au * BOHR_TO_NM


def build_system(n_mol, seed=42):
    np.random.seed(seed)
    density = 0.0078125
    n_atoms = 2 * n_mol
    box_nm = (n_atoms / density) ** (1.0/3.0) * BOHR_TO_NM
    rcut_nm = RCUT_AU * BOHR_TO_NM

    system = openmm.System()
    system.setDefaultPeriodicBoxVectors(
        openmm.Vec3(box_nm,0,0), openmm.Vec3(0,box_nm,0), openmm.Vec3(0,0,box_nm))

    bond_f = openmm.HarmonicBondForce()
    nb_f = openmm.NonbondedForce()
    nb_f.setNonbondedMethod(openmm.NonbondedForce.PME)
    nb_f.setCutoffDistance(rcut_nm)
    nb_f.setUseDispersionCorrection(False)

    positions = []
    k_aa, r0_aa = _conv_bond(K_AA_AU, R0_AA_AU)
    k_bb, r0_bb = _conv_bond(K_BB_AU, R0_BB_AU)
    eps_aa, sig_aa = _conv_lj(EPS_AA_AU, SIG_AA_AU)
    eps_bb, sig_bb = _conv_lj(EPS_BB_AU, SIG_BB_AU)
    eps_ab, sig_ab = _conv_lj(EPS_AB_AU, SIG_AB_AU)

    num_aa = int(FRAC_AA * n_mol)
    side = int(np.ceil(n_mol ** (1.0/3.0)))
    spacing = box_nm / side
    a_idx, b_idx = [], []

    mol = 0
    for i in range(side):
        for j in range(side):
            for kk in range(side):
                if mol >= n_mol: break
                is_aa = mol < num_aa
                cx, cy, cz = (i+.5)*spacing, (j+.5)*spacing, (kk+.5)*spacing
                d = np.random.randn(3); d /= np.linalg.norm(d)
                if is_aa:
                    mass, r0, k_b, sig, eps = MASS_A, r0_aa, k_aa, sig_aa, eps_aa
                else:
                    mass, r0, k_b, sig, eps = MASS_B, r0_bb, k_bb, sig_bb, eps_bb
                r1 = np.array([cx,cy,cz]) - 0.5*r0*d
                r2 = np.array([cx,cy,cz]) + 0.5*r0*d
                i1 = system.addParticle(mass); i2 = system.addParticle(mass)
                positions.append(openmm.Vec3(*r1)*unit.nanometer)
                positions.append(openmm.Vec3(*r2)*unit.nanometer)
                bond_f.addBond(i1, i2, r0, k_b)
                q1, q2 = -CHARGE_MAG, +CHARGE_MAG
                nb_f.addParticle(q1, sig, eps); nb_f.addParticle(q2, sig, eps)
                nb_f.addException(i1, i2, 0.0, 1.0, 0.0)
                (a_idx if is_aa else b_idx).append((i1,q1))
                (a_idx if is_aa else b_idx).append((i2,q2))
                mol += 1
            if mol >= n_mol: break
        if mol >= n_mol: break

    for ia, qa in a_idx:
        for ib, qb in b_idx:
            nb_f.addException(ia, ib, qa*qb, sig_ab, eps_ab)

    system.addForce(bond_f)
    system.addForce(nb_f)
    n_mol_p = system.getNumParticles()

    cav_idx = system.addParticle(PHOTON_MASS_AMU)
    positions.append(openmm.Vec3(0,0,0)*unit.nanometer)
    for fi in range(system.getNumForces()):
        f = system.getForce(fi)
        if isinstance(f, openmm.NonbondedForce):
            f.addParticle(0.0, 0.1, 0.0)
            for p in range(cav_idx):
                f.addException(cav_idx, p, 0.0, 0.1, 0.0)

    return system, positions, n_mol_p, cav_idx


# ---------------------------------------------------------------------------
def bench_python_macrostep(n_mol, run_ps, macro_step_ps, pname):
    """Old: Python drives coupling + feedback every macro-step."""
    system, positions, n_mol_p, cav_idx = build_system(n_mol)
    cavity_force = openmm.CavityForce(cav_idx, OMEGAC_AU, 0.0, PHOTON_MASS_AMU)
    system.addForce(cavity_force)
    mol_indices = list(range(n_mol_p))
    DualThermostat.setup_bussi_for_system(system, mol_indices, 300.0, BUSSI_TAU_PS)
    assign_force_groups(system)

    integrator = openmm.VerletIntegrator(DT_PS * unit.picosecond)
    platform = openmm.Platform.getPlatformByName(pname)
    context = openmm.Context(system, integrator, platform)
    context.setPositions(positions)
    context.setVelocitiesToTemperature(300.0 * unit.kelvin)

    steps_per_macro = max(1, round(macro_step_ps / DT_PS))
    total_macros = int(run_ps / macro_step_ps)
    lambda_val = 0.03

    integrator.step(200)  # warm-up

    tm = Timer()
    for _ in range(total_macros):
        tm.start("integrator.step")
        integrator.step(steps_per_macro)
        tm.stop()

        tm.start("setLambdaCoupling")
        cavity_force.setLambdaCoupling(lambda_val)
        tm.stop()

        tm.start("updateParametersInContext")
        cavity_force.updateParametersInContext(context)
        tm.stop()

        tm.start("getState")
        context.getState(getEnergy=True)
        tm.stop()

    del context, integrator
    return tm


def bench_gpu_native(n_mol, run_ps, feedback_interval_ps, pname):
    """New: GPU handles modulation; Python only reads T_s periodically."""
    system, positions, n_mol_p, cav_idx = build_system(n_mol)
    cavity_force = openmm.CavityForce(cav_idx, OMEGAC_AU, 0.0, PHOTON_MASS_AMU)
    setup_gpu_adaptive_square_wave(
        cavity_force, target_coupling=0.03, target_temperature_K=50.0,
        period_ps=5.0, duty_cycle=0.5, start_time_ps=0.0)
    system.addForce(cavity_force)
    mol_indices = list(range(n_mol_p))
    DualThermostat.setup_bussi_for_system(system, mol_indices, 300.0, BUSSI_TAU_PS)
    assign_force_groups(system)

    integrator = openmm.VerletIntegrator(DT_PS * unit.picosecond)
    platform = openmm.Platform.getPlatformByName(pname)
    context = openmm.Context(system, integrator, platform)
    context.setPositions(positions)
    context.setVelocitiesToTemperature(300.0 * unit.kelvin)

    steps_per_interval = max(1, int(feedback_interval_ps / DT_PS))
    n_intervals = int(run_ps / feedback_interval_ps)

    integrator.step(200)  # warm-up

    tm = Timer()
    for _ in range(n_intervals):
        tm.start("integrator.step")
        integrator.step(steps_per_interval)
        tm.stop()

        tm.start("getState")
        context.getState(getEnergy=True)
        tm.stop()

        tm.start("setParameter")
        context.setParameter("BussiTemperature", 200.0)
        tm.stop()

    del context, integrator
    return tm


# ---------------------------------------------------------------------------
def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--sizes", type=int, nargs="+", default=[250, 1000, 4000])
    parser.add_argument("--run-ps", type=float, default=10.0)
    parser.add_argument("--macro-step-ps", type=float, default=0.01)
    parser.add_argument("--feedback-interval-ps", type=float, default=5.0)
    parser.add_argument("--platform", default=None)
    args = parser.parse_args()

    pname = args.platform
    if not pname:
        try:
            openmm.Platform.getPlatformByName("CUDA"); pname = "CUDA"
        except Exception:
            pname = "Reference"
    print(f"Platform: {pname}\n")

    all_results = []

    for n_mol in args.sizes:
        n_atoms = 2 * n_mol + 1
        print(f"{'='*70}")
        print(f"  {n_mol} molecules ({n_atoms} atoms)")
        print(f"{'='*70}")

        print(f"\n  [OLD] Python macro-step ({args.macro_step_ps} ps interval)")
        try:
            tm_old = bench_python_macrostep(n_mol, args.run_ps, args.macro_step_ps, pname)
            tm_old.report("OLD", args.run_ps)
        except Exception as e:
            print(f"    FAILED: {e}")
            tm_old = None

        print(f"\n  [NEW] GPU-native ({args.feedback_interval_ps} ps interval)")
        try:
            tm_new = bench_gpu_native(n_mol, args.run_ps, args.feedback_interval_ps, pname)
            tm_new.report("NEW", args.run_ps)
        except Exception as e:
            print(f"    FAILED: {e}")
            tm_new = None

        if tm_old and tm_new:
            old_total = tm_old.total()
            new_total = tm_new.total()
            speedup = old_total / new_total if new_total > 0 else float('inf')
            old_gpu = tm_old.totals.get("integrator.step", 0) / old_total * 100 if old_total > 0 else 0
            new_gpu = tm_new.totals.get("integrator.step", 0) / new_total * 100 if new_total > 0 else 0
            old_overhead = old_total - tm_old.totals.get("integrator.step", 0)
            new_overhead = new_total - tm_new.totals.get("integrator.step", 0)
            overhead_reduction = old_overhead / new_overhead if new_overhead > 0 else float('inf')

            print(f"\n  COMPARISON ({n_mol} mol):")
            print(f"    Speedup:            {speedup:.1f}x")
            print(f"    GPU utilization:    {old_gpu:.1f}% -> {new_gpu:.1f}%")
            print(f"    Host overhead:      {old_overhead:.3f}s -> {new_overhead:.3f}s ({overhead_reduction:.0f}x reduction)")
            print(f"    getState calls:     {tm_old.counts.get('getState',0)} -> {tm_new.counts.get('getState',0)}")

            all_results.append({
                "n_mol": n_mol, "n_atoms": n_atoms,
                "old_s_per_ps": old_total / args.run_ps,
                "new_s_per_ps": new_total / args.run_ps,
                "speedup": speedup,
                "old_gpu_pct": old_gpu, "new_gpu_pct": new_gpu,
                "old_overhead_s": old_overhead, "new_overhead_s": new_overhead,
            })
        print()

    if all_results:
        print(f"\n{'='*70}")
        print(f"  SUMMARY")
        print(f"{'='*70}")
        print(f"  {'N_mol':>6} {'N_atoms':>7} {'Old s/ps':>9} {'New s/ps':>9} "
              f"{'Speedup':>8} {'Old GPU%':>8} {'New GPU%':>8}")
        print(f"  {'-'*66}")
        for r in all_results:
            print(f"  {r['n_mol']:>6} {r['n_atoms']:>7} "
                  f"{r['old_s_per_ps']:>9.4f} {r['new_s_per_ps']:>9.4f} "
                  f"{r['speedup']:>7.1f}x {r['old_gpu_pct']:>7.1f}% {r['new_gpu_pct']:>7.1f}%")

        np.savez("benchmark_results.npz", results=all_results)
        print(f"\nResults saved to benchmark_results.npz")


if __name__ == "__main__":
    main()
