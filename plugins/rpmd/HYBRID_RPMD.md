# Hybrid RPMD in this fork

**Hybrid ring-polymer molecular dynamics (RPMD)** runs a mixed system: some particles are treated with full path-integral (P beads), others are **classical** (still replicated on the ring for bookkeeping) but should evolve under **centroid-averaged** forces from the quantum bath, not forces taken from a single bead.

## Design sketch

1. **Quantum particles:** Standard RPMD: each bead feels forces from the physical configuration where that bead sits; Langevin / thermostat acts on the ring.
2. **Classical particles in hybrid mode:** Positions are synchronized across beads; **velocities and forces** must respect the average over beads so the classical sector sees the effective force \(F_c = \frac{1}{P}\sum_k F^{(k)}\) appropriate to hybrid RPMD, not bead-0-only forces.

Implementation lives mainly in:

- `platforms/common/src/kernels/rpmd.cc` — `integrateStepHybrid`, `advanceVelocitiesHybrid`, `syncClassicalBeads`, and related kernels
- `platforms/common/src/CommonRpmdKernels.cpp` — host-side orchestration and batched vs non-batched force paths

When **batched** potentials (e.g. UMA via `PythonForce`) coexist with other forces, the RPMD path must **accumulate** context forces into the RPMD buffer (`+=`), not overwrite them; see the fix summary below.

## Further reading

| Topic | Document |
|--------|-----------|
| Bug fixes, regression context, file-level changes | [FIXES_SUMMARY.md](../../FIXES_SUMMARY.md) |
| Test suite, categories, when to run | [tests/rpmd/README.md](../../tests/rpmd/README.md) |
| Public API | `openmmapi/include/openmm/RPMDIntegrator.h` |

Upstream OpenMM documentation for the RPMD integrator and barostat applies unless this fork notes otherwise in the files above.
