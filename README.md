[![Pixi CI](https://github.com/muhammadhasyim/openmm-lm/actions/workflows/pixi-ci.yml/badge.svg?branch=main)](https://github.com/muhammadhasyim/openmm-lm/actions/workflows/pixi-ci.yml?query=branch%3Amain)

## OpenMM-LM: A High Performance Molecular Dynamics Library with Light-Matter Interactions

### Introduction

**OpenMM-LM** is an independent molecular simulation toolkit built on the OpenMM foundation. It can be used either as a stand-alone application for running simulations, or as a library you call from your own code. It provides a combination of extreme flexibility (through custom forces and integrators), openness, and high performance (especially on recent GPUs).

This project adds light-matter interactions (particularly optical cavities), modules for nuclear quantum effects (ported from i-PI), and interfaces to state-of-the-art machine-learning interatomic potentials (e.g., FAIR Chemistry's UMA and AIMNet2).

OpenMM-LM is developed and maintained in [this repository](https://github.com/muhammadhasyim/openmm-lm). It is not distributed through upstream OpenMM channels (conda-forge, openmm.org downloads, etc.).

### Installation

OpenMM-LM is installed by building from source. The supported path is [Pixi](https://pixi.sh):

```bash
curl -fsSL https://pixi.sh/install.sh | sh   # if pixi not installed
pixi install                                  # build + install openmm
pixi run smoke                                # verify platforms
pixi run -e test test-py                      # Python wrapper tests
```

See [docs/BUILD_AND_REINSTALL.md](docs/BUILD_AND_REINSTALL.md) for environments (`test`, `ml`, `docs`), CUDA notes, and reinstall workflows.

### Getting Help

Use **this repository** for questions, bug reports, and feature requests about OpenMM-LM:

- [Issue Tracker](https://github.com/muhammadhasyim/openmm-lm/issues)
- [Discussion Forum](https://github.com/muhammadhasyim/openmm-lm/discussions)
- [Support guide](SUPPORT.md)

#### Documentation

| Topic | Location |
|--------|--------|
| Build, install, and Pixi environments | [docs/BUILD_AND_REINSTALL.md](docs/BUILD_AND_REINSTALL.md) |
| Architecture index (cavity, ML, RPMD, research) | [docs/openmm-lm/README.md](docs/openmm-lm/README.md) |
| RPMD + UMA fixes and regression context | [FIXES_SUMMARY.md](FIXES_SUMMARY.md) |
| Hybrid RPMD design | [plugins/rpmd/HYBRID_RPMD.md](plugins/rpmd/HYBRID_RPMD.md) |
| Ice Ih RPMD benchmarks and LAMMPS / i-PI parity | [tests/uma_ice_rpmd/README.md](tests/uma_ice_rpmd/README.md) |
| RPMD-focused tests | [tests/rpmd/README.md](tests/rpmd/README.md) |
| C2F protocol & aging campaigns | [research/c2f/](research/c2f/) |
| Examples | [examples/README.md](examples/README.md) |

For the shared OpenMM API that OpenMM-LM inherits, upstream reference docs remain useful:

- [User Manual](https://docs.openmm.org/latest/userguide/)
- [Python API Reference](https://docs.openmm.org/latest/api-python/)
- [C++ API Reference](https://docs.openmm.org/latest/api-c++/)
- [Developer Guide](https://docs.openmm.org/latest/developerguide/)

#### Contributing

See [CONTRIBUTING.md](CONTRIBUTING.md) for how to open issues and pull requests against this repository.

### License Information

OpenMM-LM is free and open-source software. There are several licenses which cover
different parts of the codebase, but most of the source is covered by the MIT
license or the GNU Lesser General Public License (LGPL). Portions copyright
© 2008-2025 Stanford University and the Authors. For more details, see
[Licenses.txt](docs-source/licenses/Licenses.txt).

### Lineage

OpenMM-LM began as a fork of [upstream OpenMM](https://github.com/openmm/openmm) and has since grown into an independent project with cavity-coupled MD, **machine-learning potentials** (bundled as **`openmmml`** in the Python install), **`PythonForce`** with **batched** evaluation for **RPMD**, and related **RPMD plugin** fixes (additive forces, hybrid classical–quantum RPMD, barostat coordination).

ML runtime dependencies (FairChem, PyTorch): `pixi install -e ml` or [requirements-ml.txt](requirements-ml.txt).

#### Submodule layout (FairChem)

Downstream **FairChem** work may point submodules at a personal fork; no change is required if you already use `https://github.com/muhammadhasyim/fairchem.git` (e.g. branch `les_branch`).
