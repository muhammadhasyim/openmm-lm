#!/usr/bin/env python3
"""Mode-resolved temperature diagnostic for the mKA cavity energy-recovery test.

Purpose
-------
The energy-recovery runs show the harmonic-bond energy climbing after the
coupling switch instead of "jump then decay back to baseline". This script
tests the hypothesis that the stiff intramolecular vibrations are *not* in
thermal equilibrium before the switch (they sit far below k_BT), so the
resonant cavity merely refills them toward equipartition rather than heating
an already-equilibrated system.

It rebuilds the *exact* system/protocol used by ``run_energy_recovery_test.py``
and, at each sample, decomposes the kinetic energy of every dimer into
translational (COM), rotational, and vibrational (bond-stretch) parts, then
reports a temperature for each channel:

    T_channel = 2 * <KE_channel> / (N_dof_channel * k_B)

Expectation if the hypothesis is correct:
  * pre-switch: T_trans ~ T_rot ~ 100 K   (Bussi holds these)
                T_vib(O-O), T_vib(N-N) << 100 K   (cold / frozen)
  * post-switch: T_vib(O-O) climbs toward 100 K (resonant cavity, omega_c = omega_OO)

NOTE: requires a working OpenMM build with the cavity plugin (CavityForce,
CavityParticleDisplacer, BussiThermostat). Run it where the plugin is built.

Usage
-----
  python diagnose_mode_temperatures.py \
      --dimers 250 --lambda 0.005 --switch-time 200 --total-time 1000 \
      --cavity-friction 1.0 --sample-interval 2.0

  # no cavity bath (photon NVE):
  python diagnose_mode_temperatures.py --cavity-friction 0.0
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
if str(HERE) not in sys.path:
    sys.path.insert(0, str(HERE))

from openmm import openmm, unit

from openmm.cavitymd.adaptive import DT_MAX_PS, EPS_STAR_NM, epsilon_tolerance
from openmm.cavitymd.thermostats import DualThermostat

import run_simulation as rs

KB_KJ_MOL_K = 0.00831446262  # Boltzmann constant, kJ/mol/K
GROUP_BONDS = 1
GROUP_NONBONDED = 2


def _assign_energy_groups(system: openmm.System) -> None:
    for force in system.getForces():
        if isinstance(force, openmm.HarmonicBondForce):
            force.setForceGroup(GROUP_BONDS)
        elif isinstance(force, (openmm.NonbondedForce, openmm.CustomNonbondedForce)):
            force.setForceGroup(GROUP_NONBONDED)


def _group_energy(context: openmm.Context, group: int) -> float:
    state = context.getState(getEnergy=True, groups=1 << group)
    return state.getPotentialEnergy().value_in_unit(unit.kilojoule_per_mole)


def _read_bonds(system: openmm.System):
    """Return list of (i, j, r0_nm, k, species) from the HarmonicBondForce.

    species is 'OO' or 'NN' inferred from particle mass (16 amu -> O, 14 -> N).
    """
    bond_force = None
    for force in system.getForces():
        if isinstance(force, openmm.HarmonicBondForce):
            bond_force = force
            break
    if bond_force is None:
        raise RuntimeError("No HarmonicBondForce found in system")

    masses = np.array([
        system.getParticleMass(i).value_in_unit(unit.dalton)
        for i in range(system.getNumParticles())
    ])
    bonds = []
    for b in range(bond_force.getNumBonds()):
        i, j, r0, k = bond_force.getBondParameters(b)
        r0_nm = r0.value_in_unit(unit.nanometer)
        species = "OO" if masses[i] > 15.0 else "NN"
        bonds.append((i, j, r0_nm, float(k.value_in_unit(
            unit.kilojoule_per_mole / unit.nanometer**2)), species))
    return bonds, masses


def _mode_temperatures(pos, vel, masses, bonds):
    """Decompose per-dimer KE into trans/rot/vib and return channel temperatures.

    pos, vel: (N,3) arrays in nm and nm/ps. masses: (N,) amu.
    Returns dict of temperatures (K) and the by-hand bond PE by species.
    """
    ke_trans = 0.0
    ke_rot = 0.0
    ke_vib = {"OO": 0.0, "NN": 0.0}
    pe_bond = {"OO": 0.0, "NN": 0.0}
    n_species = {"OO": 0, "NN": 0}

    for (i, j, r0, k, sp) in bonds:
        mi, mj = masses[i], masses[j]
        M = mi + mj
        mu = mi * mj / M
        vi, vj = vel[i], vel[j]

        v_com = (mi * vi + mj * vj) / M
        ke_trans += 0.5 * M * float(v_com @ v_com)

        dr = pos[i] - pos[j]
        r = float(np.linalg.norm(dr))
        r_hat = dr / r if r > 0 else np.zeros(3)

        v_rel = vi - vj
        v_str = float(v_rel @ r_hat)
        ke_vib[sp] += 0.5 * mu * v_str * v_str
        v_perp = v_rel - v_str * r_hat
        ke_rot += 0.5 * mu * float(v_perp @ v_perp)

        pe_bond[sp] += 0.5 * k * (r - r0) ** 2
        n_species[sp] += 1

    n_mol = len(bonds)
    dof_trans = 3 * n_mol - 3  # remove global COM
    dof_rot = 2 * n_mol

    def T(ke, dof):
        return 2.0 * ke / (dof * KB_KJ_MOL_K) if dof > 0 else float("nan")

    out = {
        "T_trans": T(ke_trans, dof_trans),
        "T_rot": T(ke_rot, dof_rot),
        "T_vib_OO_kin": T(ke_vib["OO"], n_species["OO"]),
        "T_vib_NN_kin": T(ke_vib["NN"], n_species["NN"]),
        # potential-side vibrational temperature: <U> = (1/2) k_B T per bond
        "T_vib_OO_pot": T(pe_bond["OO"], n_species["OO"]),
        "T_vib_NN_pot": T(pe_bond["NN"], n_species["NN"]),
        "pe_bond_total": pe_bond["OO"] + pe_bond["NN"],
    }
    return out


def run(args) -> None:
    box_size_nm = rs.box_size_nm_at_constant_density(args.dimers)
    omegac_au = 1560.0 / 219474.63
    photon_mass = 1.0 / 1822.888

    system, positions, topology, cavity_index = rs.create_diamer_system_from_forcefield(
        num_molecules=args.dimers,
        fraction_OO=0.8,
        box_size_nm=box_size_nm,
        seed=args.seed,
        include_cavity=True,
    )
    num_molecular = cavity_index
    system.setParticleMass(cavity_index, photon_mass)
    _assign_energy_groups(system)

    dt_ps = DT_MAX_PS
    switch_step = int(round(args.switch_time / dt_ps))

    cavity_force = openmm.CavityForce(cavity_index, omegac_au, 0.0, photon_mass)
    cavity_force.setCouplingOnStep(switch_step, args.lambda_coupling)
    system.addForce(cavity_force)

    displacer = openmm.CavityParticleDisplacer(cavity_index, omegac_au, photon_mass)
    displacer.setSwitchOnLambda(args.lambda_coupling)
    displacer.setSwitchOnStep(switch_step)
    system.addForce(displacer)

    bussi = openmm.BussiThermostat(args.temp, args.bussi_tau)
    bussi.setApplyToAllParticles(False)
    for i in range(num_molecular):
        bussi.addParticle(i)
    system.addForce(bussi)

    integrator = openmm.VariableVerletIntegrator(EPS_STAR_NM)
    integrator.setMaximumStepSize(dt_ps * unit.picosecond)

    try:
        platform = openmm.Platform.getPlatformByName("CUDA")
        platform.setPropertyDefaultValue("Precision", "mixed")
        platform_name = "CUDA"
    except Exception:
        platform = openmm.Platform.getPlatformByName("Reference")
        platform_name = "Reference"

    context = openmm.Context(system, integrator, platform)
    context.setPositions(positions)
    context.setVelocitiesToTemperature(args.temp * unit.kelvin, args.seed)

    cavity_thermostat = DualThermostat(
        context, system,
        cavity_particle_index=cavity_index,
        cavity_friction_ps_inv=args.cavity_friction,
        cavity_temperature_K=args.temp,
    )

    bonds, masses = _read_bonds(system)

    print("=" * 74)
    print("Mode-resolved temperature diagnostic")
    print("=" * 74)
    print(f"  dimers={args.dimers}  lambda={args.lambda_coupling}  switch={args.switch_time} ps"
          f"  total={args.total_time} ps")
    print(f"  cavity_friction={args.cavity_friction} /ps  platform={platform_name}")
    print(f"  bonds: {sum(1 for b in bonds if b[4]=='OO')} O-O, "
          f"{sum(1 for b in bonds if b[4]=='NN')} N-N")
    print("\n--- minimize ---")
    openmm.LocalEnergyMinimizer.minimize(context, maxIterations=100)

    rows = []
    next_sample = 0.0
    hdr = (f"{'t(ps)':>8} {'T_trans':>8} {'T_rot':>8} {'T_vibOO':>8} {'T_vibNN':>8} "
           f"{'TvibOOpe':>9} {'T_phot':>8} {'bondPE':>9} {'nbPE':>10}")
    print("\n" + hdr)

    while True:
        t = context.getState().getTime().value_in_unit(unit.picosecond)
        if t >= args.total_time - 1e-12:
            break
        integrator.setErrorTolerance(epsilon_tolerance(t, args.switch_time))
        integrator.step(1)
        step_dt = integrator.getStepSize().value_in_unit(unit.picosecond)
        cavity_thermostat.apply_cavity_thermostat_step(step_dt)
        t = context.getState().getTime().value_in_unit(unit.picosecond)

        if t + 1e-12 >= next_sample:
            state = context.getState(getPositions=True, getVelocities=True)
            pos = state.getPositions(asNumpy=True).value_in_unit(unit.nanometer)
            vel = state.getVelocities(asNumpy=True).value_in_unit(unit.nanometer / unit.picosecond)
            temps = _mode_temperatures(pos, vel, masses, bonds)

            m_ph = masses[cavity_index]
            v_ph = vel[cavity_index]
            ke_ph = 0.5 * m_ph * float(v_ph @ v_ph)
            T_phot = 2.0 * ke_ph / (3.0 * KB_KJ_MOL_K)

            bond_e = _group_energy(context, GROUP_BONDS)
            nb_e = _group_energy(context, GROUP_NONBONDED)

            rows.append({
                "t": t, **temps, "T_phot": T_phot,
                "bond_e": bond_e, "nb_e": nb_e,
            })
            print(f"{t:8.1f} {temps['T_trans']:8.1f} {temps['T_rot']:8.1f} "
                  f"{temps['T_vib_OO_kin']:8.1f} {temps['T_vib_NN_kin']:8.1f} "
                  f"{temps['T_vib_OO_pot']:9.1f} {T_phot:8.1f} "
                  f"{bond_e:9.1f} {nb_e:10.1f}")
            next_sample += args.sample_interval

    # Save + summarize
    arr = {k: np.array([r[k] for r in rows]) for k in rows[0]}
    tag = f"lambda{args.lambda_coupling}_t{int(args.switch_time)}_g{args.cavity_friction}"
    out_npz = HERE / f"mode_temperatures_{tag}.npz"
    np.savez(out_npz, **arr)

    t = arr["t"]
    pre = (t >= args.switch_time - 50) & (t < args.switch_time)
    print("\n" + "=" * 74)
    print("Pre-switch channel temperatures (target = %.0f K):" % args.temp)
    for key, label in [("T_trans", "translational"), ("T_rot", "rotational"),
                       ("T_vib_OO_kin", "O-O vibration (kin)"),
                       ("T_vib_OO_pot", "O-O vibration (pot)"),
                       ("T_vib_NN_kin", "N-N vibration (kin)")]:
        if pre.any():
            print(f"    {label:22s}: {arr[key][pre].mean():7.1f} K"
                  f"   ({100*arr[key][pre].mean()/args.temp:5.1f}% of target)")
    print(f"\n  Saved: {out_npz}")

    _plot(arr, args, HERE / f"mode_temperatures_{tag}.png")


def _plot(arr, args, out_png):
    try:
        import matplotlib.pyplot as plt
    except ImportError:
        print("matplotlib not available; skipping plot")
        return
    t = arr["t"]
    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(10, 7.5), sharex=True)
    fig.suptitle(
        f"Mode-resolved temperatures (lambda={args.lambda_coupling}, "
        f"cavity friction={args.cavity_friction}/ps)",
        fontsize=12, fontweight="bold")
    ax1.axhline(args.temp, color="k", ls="--", lw=1.0, label=f"target {args.temp:.0f} K")
    ax1.plot(t, arr["T_trans"], label="translational", color="#1f77b4")
    ax1.plot(t, arr["T_rot"], label="rotational", color="#2ca02c")
    ax1.plot(t, arr["T_vib_OO_kin"], label="O-O vibration (kin)", color="#d62728")
    ax1.plot(t, arr["T_vib_NN_kin"], label="N-N vibration (kin)", color="#9467bd")
    ax1.plot(t, arr["T_phot"], label="photon", color="#ff7f0e", alpha=0.6)
    ax1.axvline(args.switch_time, color="gray", ls=":", lw=1.0)
    ax1.set_ylabel("channel temperature (K)")
    ax1.legend(fontsize=8, ncol=2)
    ax1.grid(alpha=0.25)

    ax2.plot(t, arr["bond_e"], label="bond PE", color="#d62728")
    ax2.plot(t, arr["nb_e"], label="nonbonded PE", color="#1f77b4")
    ax2.axvline(args.switch_time, color="gray", ls=":", lw=1.0)
    ax2.set_xlabel("time (ps)")
    ax2.set_ylabel("energy (kJ/mol)")
    ax2.legend(fontsize=8)
    ax2.grid(alpha=0.25)
    fig.tight_layout(rect=[0, 0, 1, 0.96])
    fig.savefig(out_png, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"  Plot:  {out_png}")


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--dimers", type=int, default=250)
    p.add_argument("--lambda", type=float, default=0.005, dest="lambda_coupling")
    p.add_argument("--switch-time", type=float, default=200.0)
    p.add_argument("--total-time", type=float, default=1000.0)
    p.add_argument("--temp", type=float, default=100.0)
    p.add_argument("--bussi-tau", type=float, default=1.0)
    p.add_argument("--cavity-friction", type=float, default=1.0)
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--sample-interval", type=float, default=2.0)
    run(p.parse_args())


if __name__ == "__main__":
    main()
