#!/usr/bin/env python
"""
GPU memory scaling probe for mKA+cavity system.
Tests system sizes from 16k up to memory limit.
"""
import os
import subprocess
import gc
import sys
from pathlib import Path

sys.path.insert(0, "/scratch/mh7373/openmm/research/c2f")
import openmm
from openmm import unit
from run_c2f import (
    build_mka_system,
    add_cavity_particle,
    box_au_for_num_molecules,
    OMEGA_C_CM1,
    PHOTON_MASS_AMU,
    BOHR_TO_NM,
)
from openmm.cavitymd import DualThermostat, assign_force_groups

OMEGAC_AU = OMEGA_C_CM1 / 219474.63


def gpu_mem_mib():
    pid = os.getpid()
    try:
        out = subprocess.check_output(
            [
                "nvidia-smi",
                "--query-compute-apps=pid,used_memory",
                "--format=csv,noheader,nounits",
            ],
            text=True,
        )
        for line in out.strip().splitlines():
            p, m = [x.strip() for x in line.split(",")]
            if int(p) == pid:
                return float(m)
    except Exception:
        pass
    return float("nan")


def probe(n_mol):
    """Test system size and return GPU memory used."""
    box_au = box_au_for_num_molecules(n_mol)

    system, positions, n_mol_p = build_mka_system(
        num_molecules=n_mol, box_au=box_au, seed=42
    )
    cav_idx = add_cavity_particle(system, positions)
    system.addForce(
        openmm.CavityForce(cav_idx, OMEGAC_AU, 0.03, PHOTON_MASS_AMU)
    )
    DualThermostat.setup_bussi_for_system(system, list(range(n_mol_p)), 100.0, 1.0)
    assign_force_groups(system)

    integrator = openmm.VerletIntegrator(0.001 * unit.picosecond)
    ctx = openmm.Context(
        system, integrator, openmm.Platform.getPlatformByName("CUDA")
    )
    ctx.setPositions(positions)
    ctx.setVelocitiesToTemperature(100 * unit.kelvin)

    # Run 100 steps
    integrator.step(100)

    mem = gpu_mem_mib()
    nat = n_mol_p + 1
    box = box_au * BOHR_TO_NM

    del ctx, integrator, system
    gc.collect()

    return mem, nat, box


def format_number(n):
    """Format large numbers with k/M suffix."""
    if n >= 1_000_000:
        return f"{n/1_000_000:.1f}M"
    elif n >= 1000:
        return f"{n/1000:.0f}k"
    return str(n)


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--sizes",
        type=int,
        nargs="+",
        default=[
            16000,
            32000,
            64000,
            128000,
            200000,
            300000,
            400000,
            500000,
            600000,
            700000,
            800000,
            900000,
            1000000,
        ],
        help="System sizes to test (molecules)",
    )
    parser.add_argument(
        "--gpu", type=int, default=0, help="GPU device to use"
    )
    args = parser.parse_args()

    os.environ["CUDA_VISIBLE_DEVICES"] = str(args.gpu)

    print(f"\n{'='*70}")
    print(f"GPU Memory Scaling Probe - A100 80GB")
    print(f"Testing sizes: {[format_number(n) for n in args.sizes]}")
    print(f"{'='*70}\n")

    results = []
    for n in args.sizes:
        try:
            print(f"Testing N_mol={format_number(n):>6} ... ", end="", flush=True)
            mem, nat, box = probe(n)
            print(f"OK | atoms={format_number(nat):>8} | box={box:.1f}nm | GPU={mem:.0f} MiB")
            results.append((n, nat, box, mem, "OK"))

            # Stop if we've exceeded 75 GiB
            if mem > 75000:
                print(f"  -> Memory limit (~75 GiB) reached, stopping sweep")
                break
        except Exception as e:
            print(f"FAIL | {type(e).__name__}: {str(e)[:80]}")
            results.append((n, None, None, None, f"FAIL: {type(e).__name__}"))
            break

    print(f"\n{'='*70}")
    print(f"Summary:")
    print(f"{'='*70}")
    print(f"{'N_mol':>12} {'Atoms':>10} {'Box(nm)':>10} {'GPU(MiB)':>12} {'Status':>10}")
    print(f"{'-'*60}")
    for n, nat, box, mem, status in results:
        if nat:
            print(
                f"{format_number(n):>12} {format_number(nat):>10} {box:>10.1f} {mem:>12.0f} {status:>10}"
            )
        else:
            print(f"{format_number(n):>12} {'--':>10} {'--':>10} {'--':>12} {status:>10}")

    # Find max successful
    successful = [r for r in results if r[4] == "OK"]
    if successful:
        max_n, max_nat, max_box, max_mem, _ = successful[-1]
        print(f"\nMaximum tested: N_mol={format_number(max_n)} ({format_number(max_nat)} atoms) using {max_mem:.0f} MiB")
        if max_mem < 75000:
            print(f"Headroom to 75 GiB: ~{75000 - max_mem:.0f} MiB")
    print(f"{'='*70}\n")
