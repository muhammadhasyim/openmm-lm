# Cavity-coupled MD examples

Short demos using `openmm.cavitymd` and core `CavityForce`. For the full C2F paper protocol,
see [`research/c2f/`](../../research/c2f/).

| Directory | Description |
|-----------|-------------|
| [`dimer_system/`](dimer_system/) | Two-component O-O / N-N dimer benchmark |
| [`water_system/`](water_system/) | Flexible TIP4P-Ew water, IR / Rabi splitting |
| [`protein_system/`](protein_system/) | 3UTL protein in solvent |
| [`c2f_protocol/`](c2f_protocol/README.md) | Redirect — moved to `research/c2f/` |

Tests: [`tests/dimer_system/`](../tests/dimer_system/), [`tests/protein_system/`](../tests/protein_system/).
