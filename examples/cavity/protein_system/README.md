# Protein cavity simulation (3UTL)

Demonstrates cavity-coupled MD for a protein in explicit solvent (OH stretch coupling),
mirroring the water-system workflow.

## Dependencies

- `openmm` (built from this repo)
- `pdbfixer`
- `openmmforcefields` (AMBER ff14SB + TIP4P-Ew)

## Running

From the repository root:

```bash
pixi run -e test python examples/cavity/protein_system/run_simulation.py --test
```

Full run (downloads 3UTL, solvates, 100+900 ps production):

```bash
pixi run -e test python examples/cavity/protein_system/run_simulation.py
```

Use a local PDB:

```bash
pixi run -e test python examples/cavity/protein_system/run_simulation.py --pdb-path path/to/3UTL.pdb
```

## Output

Writes `protein_cavity_lambdaXXXX.npz` in the current working directory (same schema as
the water demo). Trajectories and movies are local-only and gitignored.

## Tests

Unit tests live in [`tests/protein_system/test_protein_cavity.py`](../../tests/protein_system/test_protein_cavity.py).
