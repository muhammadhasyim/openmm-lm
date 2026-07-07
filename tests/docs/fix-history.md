# Test Suite Fix History

Technical notes on fixes applied to the dimer and cavity simulation tests.

## Force Constants

HOOMD and OpenMM both use `E = 0.5 * k * (r - r0)²`; use the same k values. Do not double k.
- Correct: O-O k=0.73204, N-N k=1.4325 (a.u.) → ~1560, ~2325 cm⁻¹
- Wrong: 2× those values → frequencies √2 too high

## Cavity Coupling

CavityForce must receive `lambda_coupling` from t=0, not 0.0 with `setCouplingOnStep`. Otherwise coupling stays off during equilibration.

## Cavity on-switch sync (displacer vs coupling)

`CavityForce::setCouplingOnStep(N, λ)` and `CavityParticleDisplacer::setSwitchOnStep(N)` must fire on the **same** integrator step. Before the fix, coupling activated at kernel step `N-1` while the displacer waited until private counter `N`, leaving one step where full coupling acted on a photon still at `q≈0`. That injected an `O(λ ω_c |μ|)` impulse on the ultralight photon and blew up realistic-λ runs.

**Fix (Jul 2026):**
- `setCouplingOnStep` stores schedule entry `(N, λ)` directly (removed the `-1` offset).
- `CavityParticleDisplacerImpl::updateContextState` uses `context.getStepCount() == N` instead of a private step counter.

Regression: `tests/dimer_system/test_cavity_switch_sync.py`.

### Remaining parity / stability notes

- **Python OU cavity bath** (`cavitymd/thermostats.py`): per-step `getState`/`setVelocities` on all particles is a GPU sync hotspot and non-standard operator splitting vs cavHOOMD's in-integrator Langevin on the cavity DOF.
- **Reproducibility**: `apply_cavity_thermostat_step` uses unseeded `np.random.randn`; `run_simulation.py` calls `setVelocitiesToTemperature` without a seed.
- **Adaptive switch timing**: `switch_step = int(switch_time / dt_cap)` assumes steps run at the cap; VariableVerlet actual step sizes may shift the switch time slightly.
- **Photon PBC**: the cavity coordinate is a periodic particle at the box origin; fast photon motion can still stress `posCellOffsets` unwrapping on subsequent steps (separate from the one-step desync fix). Long-term parity may require a non-periodic cavity DOF as in cavHOOMD-blue.
- **`γ_c` convention**: paper uses `γ_c = 1/τ_c` with `τ_c = 1 ps`; confirm against SI if lifetimes disagree.

## LJ Parameters (Dimer)

Kob-Andersen model needs non-additive N-O cross-terms; Lorentz-Berthelot fails. Use explicit exceptions with sigma_NO, epsilon_NO from cav-hoomd.

## IR Spectrum (ML Water)

- Use unwrapped positions when computing dipoles: omit `enforcePeriodicBox` so molecules across box edges stay intact.
- ACF: subtract mean dipole, FFT-based method, ω² spectral weight.
- Real-time: `compute_ir_realtime.py` reads dipole file while simulation runs.
