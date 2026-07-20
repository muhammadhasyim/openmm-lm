# Tutorials

This directory contains interactive tutorials for OpenMM-LM.

## Protein in Water (upstream OpenMM workshop)

Reusable protein-in-solvent workflow (solvation, NVT/NPT, checkpoints):

| Path | Purpose |
|------|---------|
| [`protein_in_water/`](protein_in_water/) | Official OpenMM workshop notebook + `villin.pdb` |

See [`protein_in_water/README.md`](protein_in_water/README.md).

## mKA Cavity MD Tutorial

Step-by-step scripts and validation for cavity molecular dynamics with the modified Kob–Andersen (mKA) dimer model.

### Contents

| Path | Purpose |
|------|---------|
| [`01/01_nve_single_dimer.py`](01/01_nve_single_dimer.py) | Tutorial 01 — NVE single dimer, IR peak, finite-q demo |
| [`02/02_nvt_single_dimer.py`](02/02_nvt_single_dimer.py) | Tutorial 02 — Bussi NVT, kinetic T control, IR peak |
| [`03/03_nvt_two_dimers.py`](03/03_nvt_two_dimers.py) | Tutorial 03 — two dimers + LJ/Coulomb, polariton LP/UP |
| [`03/03_nvt_collective_scaling.py`](03/03_nvt_collective_scaling.py) | Tutorial 03 — collective coupling at fixed λ√N |
| [`tutorial_common.py`](tutorial_common.py) | Shared system builders and analysis helpers |
| [`run_tutorial_validation.py`](run_tutorial_validation.py) | Headless validation for all three tutorials |
| [`mka_cavity_md_tutorial.ipynb`](mka_cavity_md_tutorial.ipynb) | Interactive walkthrough (Sections 0–5) |

Each numbered folder also contains the matching Jupyter notebook (and any generated figures).

## Prerequisites

Build OpenMM from this repository with Python wrappers enabled:

```bash
pixi install
pixi run smoke   # verify import
```

## Run tutorials (01 → 03)

From the repository root:

```bash
python examples/tutorials/01/01_nve_single_dimer.py --platform Reference
python examples/tutorials/02/02_nvt_single_dimer.py --platform Reference
python examples/tutorials/03/03_nvt_two_dimers.py --platform Reference
```

Or validate all at once:

```bash
python examples/tutorials/run_tutorial_validation.py
python examples/tutorials/run_tutorial_validation.py --quick
```

## Physics acceptance criteria

| Tutorial | Checks |
|----------|--------|
| **01 NVE** | Single dominant IR peak near **1560 cm⁻¹**; finite-q shift suppresses photon displacement and potential-energy exchange vs `q = 0` |
| **02 NVT** | Mean molecular **T_kin ≈ 100 K** (3N DOF); same **~1560 cm⁻¹** peak |
| **03 Two dimers** | Resonant coupling shows **LP below** and **UP above** ω_c (λ = 0.03 by default for resolved splitting) |

## Finite-q displacement demo (Tutorial 01)

`CavityParticleDisplacer.displaceToEquilibrium()` sets `q_eq = -(λ/(m_ph·ω_c))·d_xy` before dynamics. With zero initial velocities:

- **No shift** (`q = 0`): photon drifts from equilibrium and the potential energy oscillates (molecule–cavity energy exchange).
- **With shift**: photon starts at equilibrium and stays there.

## Automated tests

```bash
pixi run -e test python -m pytest -v tests/tutorial/test_mka_tutorial_physics.py
```

## Physics notes

- **Finite-q displacement**: `displaceToEquilibrium()` matches `CavityForce` with `K = m_ph·ω_c²`.
- **NVT thermostat (02/03)**: Bussi on molecular indices only; Verlet integrator.
- **Temperature (02)**: molecular kinetic T from selected atoms with 3N translational DOF.
- **Spectrum**: dipole ACF + DCT (same method as the external tutorial notebooks).
- **Polaritons (03)**: increase λ or trajectory length if LP/UP are unresolved at λ = 0.01.
- **Platform**: scripts default to CPU/Reference (whichever is available); CUDA works when built.

## Troubleshooting

| Symptom | Likely cause |
|---------|----------------|
| `No registered Platform called "CPU"` | Use `--platform Reference` or rebuild with CPU platform |
| Peak ~1540 instead of ~1560 | Short trajectory; increase `--steps` |
| T_kin far from 100 K | Increase production length; check molecular-only T estimator |
| No LP/UP in Tutorial 03 | Increase `--lambda-coupling` (e.g. 0.03) or `--steps` |
