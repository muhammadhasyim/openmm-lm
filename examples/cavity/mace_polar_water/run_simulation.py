#!/usr/bin/env python3
"""MACE-POLAR-1 cavity MD of a water box (GPU bridge + PythonForce path).

Example (physical GPU 1, use ff-ml-dipole env python directly):

    CUDA_VISIBLE_DEVICES=1 \\
    OPENMMML_MACE_POLAR_MODEL=$PWD/third_party/models/MACE-POLAR-1-M.model \\
    python examples/cavity/mace_polar_water/run_simulation.py \\
        --num-molecules 100 --platform CUDA --steps 100
"""

from __future__ import annotations

import argparse
import math
import sys
import time
from pathlib import Path

import numpy as np

import openmm
from openmm import app, unit

from openmm.cavitymd.forcefields import CavityParams, build_system, evaluate_dipole

HARTREE_TO_CM = 219474.63
PHOTON_MASS_AMU = 1.0 / 1822.888
OH_BOND_NM = 0.09572
HOH_ANGLE_RAD = math.radians(104.52)


def box_size_nm_for_molecules(num_molecules: int, ref_molecules: int = 64, ref_box_nm: float = 2.0) -> float:
    """Scale cubic box edge at fixed number density."""
    return ref_box_nm * (num_molecules / ref_molecules) ** (1.0 / 3.0)


def make_water_topology(
    num_molecules: int,
    box_size_nm: float,
    seed: int = 42,
) -> tuple[app.Topology, np.ndarray]:
    """Build periodic HOH topology (no virtual sites) and positions in nm."""
    rng = np.random.RandomState(seed)
    topology = app.Topology()
    chain = topology.addChain()
    positions = []

    side = int(math.ceil(num_molecules ** (1.0 / 3.0)))
    spacing = box_size_nm / side
    half_angle = HOH_ANGLE_RAD / 2.0

    mol_count = 0
    for i in range(side):
        for j in range(side):
            for k in range(side):
                if mol_count >= num_molecules:
                    break
                cx = (i + 0.5) * spacing
                cy = (j + 0.5) * spacing
                cz = (k + 0.5) * spacing

                center = np.array([cx, cy, cz], dtype=np.float64)

                z_axis = rng.normal(size=3)
                z_axis /= np.linalg.norm(z_axis)
                x_axis = rng.normal(size=3)
                x_axis -= np.dot(x_axis, z_axis) * z_axis
                x_axis /= np.linalg.norm(x_axis)

                h1_dir = math.sin(half_angle) * x_axis + math.cos(half_angle) * z_axis
                h2_dir = math.sin(half_angle) * x_axis - math.cos(half_angle) * z_axis
                h1 = center + OH_BOND_NM * h1_dir
                h2 = center + OH_BOND_NM * h2_dir

                residue = topology.addResidue("HOH", chain)
                oxygen = topology.addAtom("O", app.Element.getBySymbol("O"), residue)
                hydrogen1 = topology.addAtom("H1", app.Element.getBySymbol("H"), residue)
                hydrogen2 = topology.addAtom("H2", app.Element.getBySymbol("H"), residue)
                topology.addBond(oxygen, hydrogen1)
                topology.addBond(oxygen, hydrogen2)

                positions.extend([center, h1, h2])
                mol_count += 1
            if mol_count >= num_molecules:
                break
        if mol_count >= num_molecules:
            break

    pos_nm = np.asarray(positions, dtype=np.float64)
    vectors = (
        openmm.Vec3(box_size_nm, 0, 0) * unit.nanometer,
        openmm.Vec3(0, box_size_nm, 0) * unit.nanometer,
        openmm.Vec3(0, 0, box_size_nm) * unit.nanometer,
    )
    topology.setPeriodicBoxVectors(vectors)
    return topology, pos_nm


def wavenumber_to_hartree(omega_cm: float) -> float:
    return omega_cm / HARTREE_TO_CM


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="MACE-POLAR-1 water cavity MD")
    parser.add_argument("--num-molecules", type=int, default=100)
    parser.add_argument("--box-size-nm", type=float, default=None)
    parser.add_argument("--omega-c-cm", type=float, default=3656.0, help="Cavity frequency cm^-1")
    parser.add_argument("--lambda-coupling", type=float, default=0.01)
    parser.add_argument("--temperature-K", type=float, default=300.0)
    parser.add_argument("--timestep-fs", type=float, default=1.0)
    parser.add_argument("--steps", type=int, default=100)
    parser.add_argument("--platform", default="CUDA", choices=("CUDA", "CPU", "Reference"))
    parser.add_argument("--precision", default="mixed", choices=("single", "mixed", "double"))
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--model-path", default=None, help="Override OPENMMML_MACE_POLAR_MODEL")
    parser.add_argument("--output-dir", type=Path, default=Path("runs/mace_polar_water"))
    parser.add_argument("--log-interval", type=int, default=10)
    parser.add_argument(
        "--use-cuda-bridge",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Use openmm_cuda_bridge for GPU-resident MACE (default: on for CUDA)",
    )
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)

    box_size_nm = args.box_size_nm
    if box_size_nm is None:
        box_size_nm = box_size_nm_for_molecules(args.num_molecules)

    print(f"Building {args.num_molecules} water molecules in {box_size_nm:.3f} nm box")
    topology, water_pos_nm = make_water_topology(
        args.num_molecules, box_size_nm, seed=args.seed
    )
    n_atoms = water_pos_nm.shape[0]

    omegac_au = wavenumber_to_hartree(args.omega_c_cm)
    cavity = CavityParams(
        omegac=omegac_au,
        lambda_coupling=args.lambda_coupling,
        photon_mass=PHOTON_MASS_AMU,
        include_dse=True,
    )

    use_cuda_bridge = args.use_cuda_bridge and args.platform == "CUDA"
    if use_cuda_bridge:
        import torch

        if not torch.cuda.is_available():
            raise RuntimeError(
                "CUDA platform requested with cuda bridge but torch.cuda.is_available() is False. "
                "Install CUDA PyTorch (see scripts/install_ff_ml_dipole.sh)."
            )

    print("Building mace-polar-1 system with cavity coupling ...")
    t0 = time.time()
    built = build_system(
        "mace-polar-1",
        topology=topology,
        cavity=cavity,
        model_path=args.model_path,
        charge=0,
        multiplicity=1,
        precision="single",
        use_cuda_bridge=use_cuda_bridge,
    )
    use_cuda_bridge = args.use_cuda_bridge and args.platform == "CUDA"
    print(
        f"  built in {time.time() - t0:.1f} s; backend={built.cavity_backend.value}; "
        f"cuda_bridge={use_cuda_bridge}"
    )

    photon_pos = np.zeros((1, 3), dtype=np.float64)
    all_pos_nm = np.vstack([water_pos_nm, photon_pos])
    positions = [
        openmm.Vec3(*row) * unit.nanometer for row in all_pos_nm
    ]

    dt = args.timestep_fs * unit.femtoseconds
    integrator = openmm.LangevinMiddleIntegrator(
        args.temperature_K * unit.kelvin,
        1.0 / unit.picosecond,
        dt,
    )

    platform = openmm.Platform.getPlatformByName(args.platform)
    properties = {}
    if args.platform == "CUDA":
        # Mixed precision triggers PTX 222 on this OpenMM CUDA build; default (double) is stable.
        if args.precision != "mixed":
            properties["Precision"] = args.precision
        properties["DeviceIndex"] = "0"

    simulation = app.Simulation(topology, built.system, integrator, platform, properties)
    if use_cuda_bridge:
        import torch

        if torch.cuda.is_available():
            torch.cuda.init()
            print("  PyTorch CUDA initialized after OpenMM Context")
        from openmmml.cuda_bridge import MACE_POLAR_BRIDGE_KEY, register_context

        register_context(MACE_POLAR_BRIDGE_KEY, simulation.context)
        bridge_key = built.metadata.get("cuda_bridge_key", MACE_POLAR_BRIDGE_KEY)
        print(f"  Registered CUDA bridge key={bridge_key}")

    simulation.context.setPositions(positions)
    simulation.context.setVelocitiesToTemperature(args.temperature_K * unit.kelvin)

    state0 = simulation.context.getState(getPositions=True, getEnergy=True)
    dipole0 = evaluate_dipole(built, state0)
    print(
        f"Initial: E={state0.getPotentialEnergy()} "
        f"|mu|={np.linalg.norm(dipole0.dipole_enm):.4f} e·nm "
        f"bec={dipole0.bec.shape}"
    )
    assert dipole0.bec.shape == (n_atoms, 3, 3)

    energies = []
    dipoles = []
    print(f"Running {args.steps} steps on {platform.getName()} ...")
    run_start = time.time()
    for step in range(1, args.steps + 1):
        simulation.step(1)
        if step % args.log_interval == 0 or step == args.steps:
            state = simulation.context.getState(getEnergy=True, getPositions=True)
            e_pot = state.getPotentialEnergy().value_in_unit(unit.kilojoules_per_mole)
            e_kin = state.getKineticEnergy().value_in_unit(unit.kilojoules_per_mole)
            resp = evaluate_dipole(built, state)
            energies.append(e_pot + e_kin)
            dipoles.append(resp.dipole_enm.copy())
            if not np.all(np.isfinite(resp.bec)):
                raise RuntimeError(f"Non-finite BEC at step {step}")
            print(
                f"  step {step:6d}  E_tot={e_pot + e_kin:12.3f} kJ/mol  "
                f"|mu|={np.linalg.norm(resp.dipole_enm):.4f} e·nm"
            )

    elapsed = time.time() - run_start
    print(f"Done: {args.steps} steps in {elapsed:.1f} s")

    out_npz = args.output_dir / "trajectory_summary.npz"
    np.savez(
        out_npz,
        energies_kjmol=np.asarray(energies),
        dipoles_enm=np.asarray(dipoles),
        num_molecules=args.num_molecules,
        box_size_nm=box_size_nm,
        omega_c_cm=args.omega_c_cm,
        lambda_coupling=args.lambda_coupling,
        steps=args.steps,
    )
    print(f"Saved summary to {out_npz}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
