# OpenMM-LM architecture index

This fork extends [upstream OpenMM](https://github.com/openmm/openmm) with cavity-coupled MD,
RPMD/ML integration, and research workflows. Code lives in the standard upstream layout plus
`research/` for paper-scale campaigns.

## Where things live

| Topic | Location |
|-------|----------|
| C++ cavity forces | [`openmmapi/include/openmm/CavityForce.h`](../openmmapi/include/openmm/CavityForce.h) |
| Platform layer | [`olla/`](../olla/), [`platforms/`](../platforms/) |
| Python cavity workflow API | [`wrappers/python/openmm/cavitymd/`](../wrappers/python/openmm/cavitymd/README.md) |
| Short cavity demos | [`examples/cavity/`](../examples/cavity/) |
| C2F protocol & aging campaigns | [`research/c2f/`](../research/c2f/) |
| ML potentials (`openmmml`) | [`third_party/openmm-ml/`](../third_party/openmm-ml/) (submodule) |
| ML examples | [`examples/ml/`](../examples/ml/) |
| Hybrid RPMD | [`plugins/rpmd/HYBRID_RPMD.md`](../plugins/rpmd/HYBRID_RPMD.md) |
| RPMD / UMA tests | [`tests/rpmd/`](../tests/rpmd/), [`tests/uma_ice_rpmd/`](../tests/uma_ice_rpmd/) |
| Build & Pixi environments | [BUILD_AND_REINSTALL.md](../BUILD_AND_REINSTALL.md) |
| Aging campaign handoff | [`research/c2f/aging_weak_lambda/TODO.md`](../research/c2f/aging_weak_lambda/TODO.md) |

## Design principles

1. **Core physics in-tree** — `CavityForce` and related APIs live in `openmmapi/` (not a separate plugin).
2. **Python workflow layer** — high-level cavity MD helpers are in `openmm.cavitymd`.
3. **Optional ML** — `openmmml` is an editable submodule under `third_party/openmm-ml/`.
4. **Research isolation** — long campaigns and generated outputs stay under `research/` and are gitignored.

See [ARCHITECTURE.md](ARCHITECTURE.md) for a diagram and module relationships.
