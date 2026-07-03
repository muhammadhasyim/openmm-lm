# Unified force fields for CavityMD

Every registered backend implements the same **dipole + position Jacobian** contract:

```python
from openmm.cavitymd.forcefields import build_system, CavityParams, evaluate_dipole

built = build_system(
    "mka",
    cavity=CavityParams(omegac=0.01, lambda_coupling=0.001),
    num_molecules=250,
    seed=42,
)
response = evaluate_dipole(built, simulation.context.getState(getPositions=True))
# response.dipole_enm  -> (3,) in e·nm
# response.bec         -> (n_atoms, 3, 3) with Z_{i,α,β} = ∂μ_α/∂r_{i,β}
```

## Registry (phase 1)

| Name | μ source | Z = ∂μ/∂r | Cavity execution |
|------|----------|-----------|------------------|
| `mka` | Σ qᵢ rᵢ | qᵢ δ | Native `CavityForce` |
| `dimer-xml` | partial charges | qᵢ δ | Native `CavityForce` |
| `tip4pew-flex` | real-atom charges | qᵢ δ | Native `CavityForce` |
| `amber-tip4pew-protein` | partial charges | qᵢ δ | Native `CavityForce` |
| `cace-les-bec` | CACE `polarization` | `CACE_bec` | `openmmml` PythonForce |
| `cace-les-bec-batch` | batched CACE | batched BEC | PythonForce (RPMD) |
| `aimnet2` | model dipole | autograd | PythonForce |
| `mace-polar-1` | polar MACE dipole | autograd / export | PythonForce |
| `mbpol-2023` | MBX `get_total_dipole` | site + induced BEC | CPU PythonForce |

Backends without both μ and Z are rejected at registration.

## Optional dependencies

```bash
pixi run -e ff-classical install-ff-classical   # openmmforcefields
pixi run -e ml install-ml                       # openmmml
pixi run -e ff-ml-dipole install-ff-ml          # cace, aimnet2, mace-torch
bash scripts/build_mbx.sh                       # MBPol(2023) via MBX
```

Environment variables for ML checkpoints:

- `OPENMMML_CACE_MODEL_PATH` / `OPENMMML_LES_BEC_WATER_MODEL`
- `OPENMMML_MACE_POLAR_MODEL`
- `MBX_HOME` / `OPENMMML_MBX_JSON` (MBPol)

## Tests

```bash
PYTHONPATH=wrappers/python pixi run -e test python -m pytest -v wrappers/python/tests/TestCavityMDForceFields.py
```
