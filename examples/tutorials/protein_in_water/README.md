# Protein in Water

Official OpenMM workshop notebook for solvating a protein and running a short NVT/NPT workflow, vendored from [`openmm/openmm_workshops`](https://github.com/openmm/openmm_workshops/blob/main/section_1/protein_in_water.ipynb) (Section 1).

## Contents

| File | Purpose |
|------|---------|
| [`protein_in_water.ipynb`](protein_in_water.ipynb) | Interactive tutorial (solvation, minimization, NVT/NPT, checkpoints, basic analysis) |
| [`villin.pdb`](villin.pdb) | Villin headpiece structure used by the notebook |
| [`images/`](images/) | Figures referenced by the notebook (`villin.png`, Colab/VMD screenshots) |

## What it covers

- Loading a PDB with `PDBFile`
- Force field selection (Amber14 + TIP3P-FB)
- Solvation and ions with `Modeller`
- Energy minimization
- NVT equilibration and NPT production (`MonteCarloBarostat`)
- Trajectory / state reporting
- Checkpointing and restarting long runs
- Light trajectory analysis and visualization hooks

## Prerequisites

Build OpenMM-LM from this repository:

```bash
pixi install
pixi run smoke
```

## Run

```bash
cd examples/tutorials/protein_in_water
```

Open `protein_in_water.ipynb` with a Jupyter frontend whose Python kernel can `import openmm` from this build.

On Google Colab you can still use the upstream badge in the notebook; Colab installs conda-forge OpenMM rather than this repository’s build.

## Provenance

Source: [openmm/openmm_workshops `section_1/protein_in_water.ipynb`](https://github.com/openmm/openmm_workshops/blob/main/section_1/protein_in_water.ipynb).

Local edits:

- Setup notes for OpenMM-LM / Pixi
- Load the bundled `villin.pdb` (with a download fallback for Colab)
- Guard conda-forge `mamba install` cells so they only run on Colab
- Vendor notebook figures under `images/`
