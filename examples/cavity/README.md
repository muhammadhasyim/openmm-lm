# Cavity-coupled MD examples

Short demos using `openmm.cavitymd` and core `CavityForce`. For the full C2F paper protocol,
see [`research/c2f/`](../../research/c2f/).

| Directory | Description |
|-----------|-------------|
| [`dimer_system/`](dimer_system/) | Two-component O-O / N-N dimer benchmark |
| [`water_system/`](water_system/) | Flexible TIP4P-Ew water, IR / Rabi splitting |
| [`protein_system/`](protein_system/) | 3UTL protein in solvent |
| [`cace_les_bec_water/`](cace_les_bec_water/) | CACE LES-BEC water (GPU bridge) |
| [`aimnet2_water/`](aimnet2_water/) | AIMNet2 water smoke test |
| [`mace_polar_water/`](mace_polar_water/) | MACE-POLAR water |
| [`mbpol_2023_water/`](mbpol_2023_water/) | MBPol(2023) via MBX (CPU) |
| [`c2f_protocol/`](c2f_protocol/README.md) | Redirect — moved to `research/c2f/` |

Tests: [`tests/dimer_system/`](../tests/dimer_system/), [`tests/protein_system/`](../tests/protein_system/).
