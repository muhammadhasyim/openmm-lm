"""Shared runner for ML dipole cavity MD on a water box."""

from __future__ import annotations

import argparse
import sys
import time
from pathlib import Path

import numpy as np

import openmm
from openmm import app, unit

from openmm.cavitymd.forcefields import CavityParams, build_system, evaluate_dipole

from water_box import box_size_nm_for_molecules, make_water_topology

HARTREE_TO_CM = 219474.63
PHOTON_MASS_AMU = 1.0 / 1822.888


def wavenumber_to_hartree(omega_cm: float) -> float:
    return omega_cm / HARTREE_TO_CM


def parse_ml_water_args(description: str, registry_name: str) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=description)
    parser.add_argument("--num-molecules", type=int, default=100)
    parser.add_argument("--box-size-nm", type=float, default=None)
    parser.add_argument("--omega-c-cm", type=float, default=3656.0)
    parser.add_argument("--lambda-coupling", type=float, default=0.01)
    parser.add_argument("--temperature-K", type=float, default=300.0)
    parser.add_argument("--timestep-fs", type=float, default=1.0)
    parser.add_argument("--steps", type=int, default=100)
    parser.add_argument("--platform", default="CUDA", choices=("CUDA", "CPU", "Reference"))
    parser.add_argument("--precision", default="single", choices=("single", "mixed", "double"))
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--model-path", default=None)
    parser.add_argument("--output-dir", type=Path, default=None)
    parser.add_argument("--log-interval", type=int, default=10)
    parser.add_argument(
        "--use-cuda-bridge",
        action=argparse.BooleanOptionalAction,
        default=True,
    )
    parser.add_argument("--charge", type=int, default=0)
    parser.add_argument("--multiplicity", type=int, default=1)
    args = parser.parse_args()
    args.registry_name = registry_name
    return args


def run_ml_water_cavity_md(
    *,
    registry_name: str,
    default_output_dir: Path,
    bridge_key: str | None = None,
    build_kwargs: dict | None = None,
) -> int:
    args = parse_ml_water_args(f"{registry_name} water cavity MD", registry_name)
    if args.registry_name != registry_name:
        print(
            f"Warning: registry mismatch {args.registry_name} vs {registry_name}",
            file=sys.stderr,
        )

    output_dir = args.output_dir or default_output_dir
    output_dir.mkdir(parents=True, exist_ok=True)

    box_size_nm = args.box_size_nm
    if box_size_nm is None:
        box_size_nm = box_size_nm_for_molecules(args.num_molecules)

    print(f"Building {args.num_molecules} water molecules in {box_size_nm:.3f} nm box")
    topology, water_pos_nm = make_water_topology(
        args.num_molecules, box_size_nm, seed=args.seed
    )
    n_atoms = water_pos_nm.shape[0]

    cavity = CavityParams(
        omegac=wavenumber_to_hartree(args.omega_c_cm),
        lambda_coupling=args.lambda_coupling,
        photon_mass=PHOTON_MASS_AMU,
        include_dse=True,
    )

    use_cuda_bridge = args.use_cuda_bridge and args.platform == "CUDA"
    if use_cuda_bridge:
        import torch

        if not torch.cuda.is_available():
            raise RuntimeError(
                "CUDA platform with cuda bridge requires torch.cuda.is_available(). "
                "See scripts/install_ff_ml_dipole.sh."
            )

    build_kw = dict(build_kwargs or {})
    build_kw.update(
        topology=topology,
        cavity=cavity,
        model_path=args.model_path,
        charge=args.charge,
        multiplicity=args.multiplicity,
        use_cuda_bridge=use_cuda_bridge,
    )
    if registry_name == "mace-polar-1":
        build_kw["precision"] = "single"

    print(f"Building {registry_name} system with cavity coupling ...")
    t0 = time.time()
    built = build_system(registry_name, **build_kw)
    print(
        f"  built in {time.time() - t0:.1f} s; backend={built.cavity_backend.value}; "
        f"cuda_bridge={use_cuda_bridge}"
    )

    photon_pos = np.zeros((1, 3), dtype=np.float64)
    all_pos_nm = np.vstack([water_pos_nm, photon_pos])
    positions = [openmm.Vec3(*row) * unit.nanometer for row in all_pos_nm]

    integrator = openmm.LangevinMiddleIntegrator(
        args.temperature_K * unit.kelvin,
        1.0 / unit.picosecond,
        args.timestep_fs * unit.femtoseconds,
    )

    platform = openmm.Platform.getPlatformByName(args.platform)
    properties: dict[str, str] = {}
    if args.platform == "CUDA":
        if args.precision != "mixed":
            properties["Precision"] = args.precision
        properties["DeviceIndex"] = "0"

    simulation = app.Simulation(topology, built.system, integrator, platform, properties)
    if use_cuda_bridge and bridge_key is not None:
        import torch

        if torch.cuda.is_available():
            torch.cuda.init()
            print("  PyTorch CUDA initialized after OpenMM Context")
        from openmmml.cuda_bridge import register_context

        register_context(bridge_key, simulation.context)
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
    sim_ns = args.steps * args.timestep_fs * 1e-6
    ns_per_day = sim_ns / elapsed * 86400.0 if elapsed > 0 else 0.0
    print(f"Done: {args.steps} steps in {elapsed:.1f} s ({ns_per_day:.4f} ns/day)")

    out_npz = output_dir / "trajectory_summary.npz"
    np.savez(
        out_npz,
        energies_kjmol=np.asarray(energies),
        dipoles_enm=np.asarray(dipoles),
        num_molecules=args.num_molecules,
        box_size_nm=box_size_nm,
        omega_c_cm=args.omega_c_cm,
        lambda_coupling=args.lambda_coupling,
        steps=args.steps,
        registry_name=registry_name,
    )
    print(f"Saved summary to {out_npz}")
    return 0
