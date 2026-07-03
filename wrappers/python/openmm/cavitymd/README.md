# openmm.cavitymd

Python helpers for cavity configurational-feedback (C2F) molecular dynamics on top of OpenMM `CavityForce`.

## Public API

Import from the package root:

```python
from openmm.cavitymd import (
    Units,
    CavityMDSimulation,
    configure_coupling,
    run_nvt_energy_calibration,
    validate_calibration_file,
    DualThermostat,
    DiffEqController,
    SimpleSetpointController,
    assign_force_groups,
)
```

Advanced helpers (adaptive timestepping, trackers, GPU setup wrappers) live in submodules:

- `openmm.cavitymd.adaptive`
- `openmm.cavitymd.simulation`
- `openmm.cavitymd.variants`
- `openmm.cavitymd.calibration`

## Coupling profiles

Use `configure_coupling(cavity_force, variant)` with a `CouplingVariant` subclass, or the legacy `setup_gpu_*` helpers in `simulation` (thin wrappers around `configure_coupling`).

## Tests

```bash
pixi run -e test test-py
pixi run -e test test-cavitymd-smoke   # research/c2f/run_c2f.py --smoke
```
