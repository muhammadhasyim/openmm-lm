# Third-party dependencies

Upstream [OpenMM](https://github.com/openmm/openmm) vendors C/C++ libraries under `libraries/` (compiled into the core build). This fork keeps **optional git submodules** here for reference tools, ML integration, and parity workflows.

## Submodules

| Path | Purpose |
|------|---------|
| `openmm-ml/` | Editable `openmmml` Python package (`pixi install -e ml`) |
| `openmmforcefields/` | Amber/CHARMM biomolecular XML (`pixi run -e ff-classical install-ff-classical`) |
| `cace/` | CACE source pin for LES-BEC cavity models |
| `aimnet2/` | AIMNet2 reference implementation |
| `mace/` | MACE / MACE-POLAR reference |
| `cav-hoomd/` | HOOMD reference parity and calibration tables |
| `i-pi/` | Ring-polymer MD parity (UMA ice RPMD tests) |
| `LES-BEC/` | CACE water model checkpoint for ML+cavity tests |

Initialize all:

```bash
git submodule update --init --recursive third_party/
```

Or selectively:

```bash
git submodule update --init third_party/openmm-ml
git submodule update --init third_party/cav-hoomd
```

Root-level paths (`openmm-ml/`, `cav-hoomd/`, etc.) are deprecated; use `third_party/` instead.
