# OpenMM weak-coupling aging — master TODO

Handoff document for reproducing **Figs 2–4** from  
*Non-Thermal Aging of Supercooled Liquids in Optical Cavities* (Hasyim, Damiani, Hoffmann; arXiv:2603.15693)  
using the OpenMM `c2f_protocol` pipeline.

**Working directory:** `research/c2f/aging_weak_lambda/`  
**Repo root:** `/media/extradrive/Trajectories/openmm` (adjust paths on other machines)

---

## 1. Environment & hardware

| Item | Value |
|------|--------|
| Python env | `pixi run --as-is -e test python …` from repo root |
| OpenMM | **cavity-md branch** build with `openmm.cavitymd` (not stock OpenMM) |
| GPU | CUDA strongly recommended; ~**316 MiB VRAM per trajectory** |
| Parallel jobs | **5** (one per λ) → ~**1.6 GiB** total on GPU |
| Reference node | RTX 4070 (12 GiB); fits 5 parallel jobs comfortably |
| Est. wall time | ~**52 min / replica round** × **500 rounds** ≈ **18 days** continuous (single node) |

Setup on a fresh machine:

```bash
cd /path/to/openmm
pixi install
pixi run -e test info          # verify OpenMM + CUDA platform
# Rebuild CUDA plugin if needed:
# CONDA_PREFIX=.pixi/envs/test bash scripts/rebuild_cuda_plugin.sh
```

Dependencies for MD + analysis (pixi `test` feature): `numpy`, `scipy`, `matplotlib`.

---

## 2. Physical system & shared parameters

Modified Kob–Andersen (mKA) dipole liquid; constants in `../run_c2f.py`.

| Parameter | Value | Notes |
|-----------|-------|--------|
| Molecules | 250 (200 A–A + 50 B–B) | 500 atoms + 1 cavity particle in OpenMM |
| Box | 40 Bohr cube (≈ 2.117 nm) | Constant density |
| Bath temperature | **100 K** | |
| Cavity frequency ω_c | **1560 cm⁻¹** | Resonant with A–A stretch |
| Bussi τ | **1 ps** | Thermostat time constant |
| MD timestep | **1 fs** (0.001 ps) | |
| LJ σ_AA | 6.2304 Bohr | Used for production \|k\| |
| Dipole self-energy (DSE) | **ON** in aging runs | `include_dipole_self_energy=True` |
| Finite-q photon shift | **OFF** in aging | `finite_q=False` |
| Paper FKT \|k\| (reference only) | 6.0 a.u. (Bohr⁻¹) | Diagnostic / HOOMD comparison |
| **Production FKT \|k\|** | **2π/σ_AA ≈ 1.0085 a.u.** | **≈ 19.057 nm⁻¹** in OpenMM |
| FKT sites | **atomic** (500 sites) | Not molecular COM |
| FKT wavevectors | 50 (Fibonacci sphere) | |
| FKT ref interval | 200 ps | New reference every 200 ps after turn-on |
| FKT output period | 1 ps | |
| FKT max refs / run | 13 | Covers 200 ps + 12×200 ps = 2600 ps window |
| Energy CSV interval | 1 ps | |
| Snapshot interval | 10 ps | For IR post-processing |
| ISF normalization (analysis) | φ = F / F(0) | Block-averaged \|φ\| for τ extraction |
| τ_s definition | φ(τ_s) = 0.1, linear interp | `min_lag_ps=10`, block window 10 ps |

Central config: `config.py` (imports `FKT_KMAG_AU` from `../run_c2f.py`).

Calibration reference (HOOMD, not used as production gate):  
`cav-hoomd/relaxation_times_vs_temperature.txt` → τ_s ≈ **105 ps** at 100 K.

---

## 3. Initial condition (prerequisite)

All aging replicas share one equilibrated structure; velocities are **resampled per replica**.

| File | Path |
|------|------|
| IC | `../equilibrium_output/eq10ns100K_lam0_final_state.npz` |

Must contain `positions_nm` (501 rows: 500 atoms + cavity; FKT uses first 500).  
**Copy this file to any new machine before launching the campaign.**

Regenerate if missing (example — adjust prefix as needed):

```bash
cd research/c2f
pixi run --as-is -e test python run_cavity_equilibrium.py \
  --temperature-K 100 \
  --runtime-ps 10000 \
  --lambda 0 \
  --with-dse \
  --finite-q \
  --output-prefix equilibrium_output/eq10ns100K_lam0 \
  --sample-interval-ps 10.0
# Use *_final_state.npz from the DSE-off stage if you run a two-stage protocol.
```

---

## 4. Production calculations (TO DO)

### 4.1 Main campaign — **N = 500 replicas × 5 λ = 2500 trajectories**

**Goal:** Weak-coupling step turn-on aging after 200 ps at λ = 0.

| Quantity | Value |
|----------|-------|
| λ (a.u.) | **0**, **0.01**, **0.016667**, **0.023333**, **0.03** |
| Replicas | **500** (indices **0–499**) |
| Seeds | `42 + replica` → seeds **42–541** |
| Switch-on time t_switch | **200 ps** |
| Total runtime | **2500 ps** (2.5 ns) |
| Schedule | `replica_round`: all 5 λ in parallel, then next replica |
| Jobs | **5** (must equal number of λ values) |

**Scripts**

| Role | Script |
|------|--------|
| Launch wrapper | `run_n500_campaign.sh` |
| Batch scheduler | `run_campaign.py` |
| Single trajectory | `run_single.py` → `../run_cavity_equilibrium.py` + `../fkt_tracker.py` |

**Launch (fresh start, overwrite incomplete runs):**

```bash
cd research/c2f/aging_weak_lambda
FRESH_START=1 nohup ./run_n500_campaign.sh >> campaign_n500_k2pi.log 2>&1 &
```

**Launch (resume — skip trajectories that already have complete CSV + FKT):**

```bash
nohup ./run_n500_campaign.sh >> campaign_n500_k2pi.log 2>&1 &
```

**Monitor:**

```bash
pixi run --as-is -e test python monitor_status.py
tail -f campaign_n500_k2pi.log
pgrep -af 'run_single.py'   # expect ≤ 5 workers
```

**Verify correct FKT k on finished runs:**

```bash
grep fkt_kmag lambda0/lam0_seed0042_meta.txt
# expect: fkt_kmag_nm_inv=19.05789556235437  (≈ 19.057)
grep fkt_sites lambda0/lam0_seed0042_meta.txt
# expect: fkt_sites=atomic
```

#### Output layout (per replica, per λ)

Directory: `lambda{tag}/` where tag is `0`, `0p01`, `0p016667`, `0p023333`, `0p03`.

Prefix: `lam{tag}_seed{seed:04d}` (e.g. `lam0_seed0042` for replica 0).

| File | Purpose |
|------|---------|
| `{prefix}_energies.csv` | Thermo + energy decomposition (1 ps) |
| `{prefix}_fkt_ref_{NNN}.txt` | F(k,t) autocorrelation refs (Re F, 1 ps lags) |
| `{prefix}_snapshots.npz` | Positions every 10 ps (IR) |
| `{prefix}_meta.txt` | Run metadata (λ, k, seeds, IC path) |
| `{prefix}_final_state.npz` | Final positions (optional archive) |

Campaign log (JSONL): `campaign_n500_log.jsonl` (one line per finished trajectory).

#### Multi-node / multi-GPU sharding

If running on **separate machines without a shared filesystem**, partition **replica index** ranges (each machine runs all 5 λ per replica):

```bash
# Machine A: replicas 0–124
FRESH_START=1 ./run_n500_campaign.sh --replica-start 0 --replica-end 124

# Machine B: replicas 125–249
FRESH_START=1 ./run_n500_campaign.sh --replica-start 125 --replica-end 249

# … etc.
```

Merge resulting `lambda*/` trees afterward (no filename collisions if replica ranges are disjoint).

With a **shared NFS**, use default resume (no `FRESH_START`) so `--replica-start/--replica-end` shards divide work safely.

**Important:** Run **only one** `run_campaign.py` instance per shard. Multiple accidental launches create duplicate workers.

---

### 4.2 Optional pre-flight checks (diagnostics)

Run once after env setup; not required for every replica.

| Check | Script | Pass criterion |
|-------|--------|----------------|
| Unit chain + F(0) round-trip | `diagnose_fkt_units.py` | `diagnose_fkt/diagnose_fkt_units.json` → `overall_pass: true` |
| OpenMM vs HOOMD-style F(0) | `fkt_parity.py` | rel diff ≪ 1 at production k |
| Equilibrium FKT benchmark (~600 ps) | `benchmark_fkt_equilibrium.py` | `benchmark_fkt/benchmark_atomic.json` → `pass: true` |
| S(k) + k-scan (hypothesis only) | `compute_sk_kscan.py` | `diagnose_fkt/compute_sk_kscan.json` |

```bash
pixi run --as-is -e test python diagnose_fkt_units.py
pixi run --as-is -e test python fkt_parity.py
pixi run --as-is -e test python benchmark_fkt_equilibrium.py --runtime-ps 600
```

---

## 5. Post-processing & figures (TO DO after ≥1 replica complete; full paper needs all 500)

Run from `aging_weak_lambda/`:

```bash
pixi run --as-is -e test python run_all_analysis.py
```

Or run panels individually:

| Paper panel | Script | Output figure | Inputs |
|-------------|--------|---------------|--------|
| **Master F(k,t)** | `build_master_fkt.py` | `master_fkt/lambda*/master_fkt_ref*.txt` | archived `*_fkt_ref_*.txt` |
| **Fig 2a** (top) IR spectra | `analyze_ir_from_dipole.py` | `figures/fig2a_ir_spectra.png` | `*_dipole.npz` (10 replicas), late window |
| **Fig 2a** (bottom) F(k,t) | `plot_isf_curves.py` | `figures/fig2a_fkt_lambda*.png` (one per λ) | master F(k,t) or live replica average |
| **Fig 2b** τ̃_s vs λ | `analyze_aging_relaxation.py` | `figures/fig2b_tau_tilde_vs_lambda.png` | master F(k,t), λ=0 baseline |
| **Fig 2c** τ̃_s vs t_w | `analyze_aging_relaxation.py` | `figures/fig2c_tau_tilde_vs_tw.png` | same |
| **Fig 3b** energy redistribution | `analyze_energy_redistribution.py` | `figures/fig3b_energy_redistribution_lam0.023333.png` | `*_energies.csv`, default λ=0.023333 |
| **Fig 3c** fictive temperatures | `analyze_fictive_temperatures.py` | `figures/fig3c_fictive_temperatures_lam0.023333.png` | `*_energies.csv`, default λ=0.023333 |
| **Fig 4a** material time | `analyze_material_time_aging.py` | `figures/fig4a_material_time.png` | CSV + FKT + `relaxation_times_vs_temperature.txt` |
| **Fig 4b** ISF collapse | `analyze_material_time_aging.py` | `figures/fig4b_isf_collapse.png` | same |
| **Fig 4c–d** TN overlays | `analyze_material_time_aging.py` | `figures/fig4c_*.png`, `fig4d_*.png` | same |
| Cavity energy diagnostics | `analyze_cavity_energies.py` | `figures/cavity_energies_vs_time.png` | `*_energies.csv` |

JSON summaries: `results/relaxation_summary.json`, `results/` (cavity energy CSV from `analyze_cavity_energies.py`).

**Analysis conventions**

- FKT/snapshot/dipole paths are resolved from archived `*_archive_full_rerun_*` dirs via `fkt_utils.build_replica_root_index()`.
- Build master files first: `build_master_fkt.py` → `master_fkt/lambda*/master_fkt_ref*.txt`.
- F(k,t) normalization: ref0-single (all t_w at a λ divided by ref0 F(k,0)).
- τ_s: `fkt_utils.find_relaxation_time(..., target_value=0.1)` via `extract_tau_s`.
- Fig 2a F(k,t): one panel per λ (`plot_isf_curves.py`, cav-hoomd `plot_individual_fkt_coupling.py` style).
- τ̃_s = τ_s(λ) / τ_s(λ=0) at matched t_w
- IR spectra: DCT + quantum correction (`analyze_ir_from_dipole.py`, from `process_dipole_autocorr.py`).
- Fig 3b/c default showcase λ = **0.023333** (highest complete λ; **λ=0.03 excluded** from analysis until N=1000 campaign completes — see `ANALYSIS_LAMBDAS` in `config.py`)

Provenance: `third_party/cavity_supercooled_archive/final_production_run/scripts/2026-01-29/`.

---

## 6. Status checklist (update as work proceeds)

Use `monitor_status.py` and filesystem counts; targets in **bold**.

| Task | Target | Status |
|------|--------|--------|
| IC `eq10ns100K_lam0_final_state.npz` | 1 file | ✅ exists on primary machine |
| Production aging trajectories | **2500** | 🔄 in progress (`campaign_n500_k2pi.log`, k = 2π/σ_AA) |
| Full replicas (5/5 λ complete) | **500** | 🔄 λ ∈ {0, 0.01, 0.016667, 0.023333}: **500/500**; λ=0.03: **307/1000** (see caveat below) |
| Fig 2a–c analysis | 3 figures | ✅ generated (`figure2_weak_coupling.{png,pdf}`) |
| Fig 3b–c analysis | 2 figures | ✅ generated at λ=0.023333, N=500 (`figure3bc_weak_coupling.{png,pdf}`); λ=0.03 excluded |
| Fig 4a–d analysis | 4 figures | ✅ generated for `ANALYSIS_LAMBDAS` (N=500, no λ=0.03; `figure4_weak_coupling.{png,pdf}`) |
| `run_all_analysis.py` end-to-end | pass | ⬜ |

**Stale data warning:** `pre_fkt_fix/` and partial files from earlier campaigns (k = 6 a.u. / 113.4 nm⁻¹) must **not** be mixed with k = 2π/σ_AA results. The k2pi relaunch uses `--no-skip` to overwrite. Confirm `fkt_kmag_nm_inv ≈ 19.057` in new `*_meta.txt` files.

### 6.1 Archive cross-check (2026-07-02) and bugs fixed in `analyze_material_time_aging.py`

Cross-checked `analyze_material_time_aging.py` / `wrappers/python/openmm/cavitymd/analysis.py`
against the actual paper-production pipeline in
`third_party/cavity_supercooled_archive/final_production_run/scripts/2026-01-29/`
(`material_time_correct.py`, `run_corrected_analysis.py`,
`MATERIAL_TIME_RECONSTRUCTION_ISSUE.md`). Findings:

- Our `ToolNarayanaswamy.reconstruct_material_time` already implements the same
  simultaneous regularized-least-squares MTTI reconstruction that the archive's own
  issue doc describes as the "corrected" fix for a naive-sequential-interpolation bug
  they had. No algorithmic port was needed there.
- Fig 4(b) previously fit a free amplitude `A` in `A·Φ(h)`; this deviated from the
  paper's Eq. 13 (`Φ_k(h) = e^{-h^β}`, no amplitude) and from the archive's own
  collapse plotting convention. Fixed to plot `Φ_k(h) = e^{-h^β}` directly (`A=1`).
- While regenerating panels, found and fixed three unrelated pre-existing bugs that
  were silently breaking this script (it could not have produced correct figures
  before this session):
  - `_load_cavitymd_analysis()` resolved the repo root via
    `Path(__file__).resolve().parents[4]` (one level too high); fixed to `parents[3]`.
  - `_load_Ts_timeseries()` looked for `{job_dir}/{prefix}_energies.csv` directly and
    silently found nothing (real CSVs live under archived/alternate subdirectories);
    fixed to use `fkt_utils.build_energy_csv_index` / `resolve_energy_csv`, same as
    `analyze_fictive_temperatures.py`. This was the root cause of Fig 4c/d being
    empty and the TN dashed overlay in Fig 4a being absent.
  - `paper_style.save_figure()` used `Path.with_suffix(".pdf")` on stems containing a
    literal decimal point (e.g. `fig3b_energy_redistribution_lam0.03`), which
    corrupted the filename to `..._lam0.pdf`. Fixed to append suffixes via string
    concatenation instead.
  - `paper_style.apply_paper_style()` now auto-detects the LaTeX backend: standard
    `latex`+`dvipng`, else `tectonic`+`ghostscript` (pixi global), else matplotlib
    `mathtext` (`cm` fontset). On this cluster, conda `texlive-core` is incomplete
    (missing format files); real Computer Modern rendering uses the tectonic path.

### 6.4 Energy equilibration bug — integrator mismatch (2026-07-02)

**Symptom:** Fig 3b for λ ∈ {0.01, 0.016667, 0.023333} showed persistent
post-switch offsets in ΔE_bond / ΔE_LJ+Coulomb (~5–7 kJ/mol, opposite signs) that
did not relax back to the pre-switch baseline by t ≈ 2000 ps, while λ=0 and HOOMD
reference runs relax cleanly at all coupling strengths.

**Root cause (confirmed by archive meta audit):** The `full_rerun_20260618` batch
used **different integrators per λ**:

| λ | `*_meta.txt` in `archive_full_rerun_20260618` | Integrator |
|---|-----------------------------------------------|------------|
| 0.01, 0.016667, 0.023333 | 500/500 replicas: **no** `adaptive=` field | Fixed Verlet, dt = 1 fs (no switch sub-step, no shock ramp) |
| 0.03 | 165/165 replicas: `adaptive=True`, `dt_max_ps=0.0015` | Max-metric adaptive + `would_cross_coupling_switch()` sub-step split |

The June adaptive fix (`force calibration + switch substep` in
`wrappers/python/openmm/cavitymd/adaptive.py`) was validated on λ=0.01/0.03 pilots
but was only applied in production for λ=0.03 reruns. Weak/intermediate λ data are
**not comparable** to λ=0.03 or to cav-hoomd until re-run with `--adaptive`.

**Fix applied in code (2026-07-02):**

1. `run_campaign.py` / `run_single.py`: **`--adaptive` is now the default**
   (use `--no-adaptive` only for diagnostics).
2. `run_cavity_equilibrium.py`: fixed-dt path now also splits Verlet steps at
   `coupling_start_ps` via `would_cross_coupling_switch()` (belt-and-suspenders).
3. `analyze_energy_redistribution.py`: **Total** = Δ(E_bond + E_nonbonded)
   (molecular PE only), matching cav-hoomd `plot_fictive_temperature_components.py`.
4. `validate_energy_equilibration.py`: quick late-time residual checker.

**Required re-run:** Archive and relaunch weak/intermediate λ with adaptive integrator:

```bash
cd research/c2f/aging_weak_lambda
bash archive_and_prepare_n1000.sh   # or per-λ archive
pixi run --as-is -e test python run_campaign.py \
  --lambdas 0.01 0.016667 0.023333 \
  --adaptive --no-resume --no-skip \
  --schedule replica_round --jobs 5
```

Or full SLURM resubmit: `bash slurm/submit_n1000_adaptive.sh full` (already passes
`--adaptive` for all λ).

**Validation commands:**

```bash
pixi run --as-is -e test python validate_energy_equilibration.py \
  --lambda 0.01 0.016667 0.023333 0.03
pixi run --as-is -e test python analyze_energy_redistribution.py --lambda 0.016667
```

Pass criterion after re-run: late-time (t ∈ [1800, 2000] ps)
|ΔE_bond|, |ΔE_nb|, |ΔE_bond + ΔE_nb| ≲ 1–2 kJ/mol (SEM), matching λ=0 control and
HOOMD reference behavior.

**Fig 3b note:** Regenerated plots with corrected Total definition show molecular
PE sum near zero even in stale fixed-dt data (components cancel partially); the
component-level offsets remain the diagnostic signature of the integrator bug.

### 6.5 Adaptive integrator override bug — VariableVerlet discarded setStepSize (2026-07-02)

**Symptom:** After making `--adaptive` the default (§6.4), new N=1000 production
runs (SLURM job 12196742) still showed sporadic `T_kin` blowups at random times
(not clustered at the coupling switch).

**Root cause:** `run_cavity_equilibrium.py` defaulted to `hybrid_safety_verlet=True`,
which builds `openmm.VariableVerletIntegrator`. OpenMM's `VariableVerletIntegrator::step()`
recomputes and overwrites `stepSize` every step from its internal Euler-vs-Verlet
error kernel — it **never uses** the cav-hoomd max-force `setStepSize()` from
`advance_to_time_step_on()`. The ported cav-hoomd policy was computed but discarded;
actual dt was governed by uncalibrated `errorTol = EPS_STAR_NM(5.0) × shock_ramp`.

Empirical confirmation: `research/c2f/verify_variableverlet_stepsize_override.py`
(requested 0.5 fs → VariableVerlet took 1.0 fs; plain Verlet respected 0.5 fs).

**Fix applied:**

1. `hybrid_safety_verlet` default → **`False`** in `adaptive.py` and
   `run_cavity_equilibrium.py` (plain `VerletIntegrator` + external max-force dt).
2. SLURM production capped to **`--runtime-ps 1500`** in `14_production_adaptive_n1000.sbatch`.
3. Pilot validation: `slurm/15_pilot_plain_verlet_adaptive.sbatch` (replica 42, all λ).

**Re-launch:**

```bash
scancel 12196742   # buggy hybrid campaign (if still running)
sbatch slurm/15_pilot_plain_verlet_adaptive.sbatch
sbatch slurm/14_production_adaptive_n1000.sbatch   # includes --runtime-ps 1500
```

Jobs submitted 2026-07-02: pilot **12276782**, production **12276783**.

### 6.6 Leapfrog dt-churn energy injection — plain Verlet unsafe under adaptive setStepSize (2026-07-02)

**Symptom:** After fixing the VariableVerlet override (§6.5), the plain-Verlet
adaptive pilot still blew up: λ=0.01 at t=156 ps (pre-switch), λ=0.03 at
t=242 ps (inside the 50 ps post-switch shock ramp).

**Root cause:** OpenMM's `VerletIntegrator` is **leapfrog** (half-step velocity
storage in `ReferenceVerletDynamics.cpp`). Our adaptive scheme calls
`setStepSize()` every ~1000 steps in steady state and **every step** during the
post-shock recovery window (`TAU_RAMP_PS = 50` ps). Each dt change reinterprets
the stored half-step velocity under the new dt, injecting spurious kinetic energy.
HOOMD's integrator is genuine velocity-Verlet (full-step state) and is immune.

Empirical confirmation: `research/c2f/verify_leapfrog_dt_churn_injection.py`
(toggle 0.5/1.0 fs every 100 steps on a stiff harmonic bond: leapfrog drift
+10 kJ/mol vs fixed-dt -1.6 kJ/mol; velocity-Verlet toggle drift +0.7 kJ/mol).

**Fix applied:**

1. `create_velocity_verlet_integrator()` — velocity-Verlet `CustomIntegrator`
   with full-step position/velocity state — added to `adaptive.py`.
2. `create_adaptive_integrator()` default now returns this integrator (not
   leapfrog `VerletIntegrator`).  Set `use_leapfrog_verlet=True` for regression
   tests of the old artifact only.
3. Matched non-zero RNG seeds on `CustomIntegrator` and `BussiThermostat`
   (required for CUDA Context construction).
4. Pilot re-validation: `run_pilot_local_gpu0.sh` → `pilot_velocity_verlet_1500ps`.

**Re-launch:**

```bash
scancel 12276783   # plain-leapfrog adaptive campaign (if still running)
bash run_pilot_local_gpu0.sh
sbatch slurm/14_production_adaptive_n1000.sbatch
```


- Extended `paper_style.py` with Adobe-safe LaTeX preamble (`amsmath`, `amsfonts`,
  `amssymb`; no `lmodern`), Type-1 font embedding (`pdf.fonttype=42`), and a
  tectonic+ghostscript fallback for matplotlib `text.usetex`.
- Updated `plot_isf_curves.py`: shared paper style, y-label
  `$\phi_k(t; t_{\mathrm{w}})$`, colorbar ticks every 400 ps, PDF+PNG via
  `save_figure()`.
- Applied `apply_paper_style()` to `analyze_ir_from_dipole.py`,
  `analyze_aging_relaxation.py`, and `analyze_cavity_energies.py`.
- Added `texlive-core` and `ghostscript` to `[feature.test.target.linux-64.dependencies]`
  in `pixi.toml` (runtime also requires `pixi global install texlive-core ghostscript tectonic`).

**Known caveat:** λ=0.03 is **excluded from figure analysis** (`ANALYSIS_LAMBDAS` in
`config.py`) until its N=1000 campaign completes. With only a partial ensemble,
measured `τ_s(t_w)` is too noisy for stable MTTI curves and Fig 4(a) showed
large unphysical excursions. Production continues for λ=0.03; re-enable in
analysis once the full ensemble is available.

### 6.2 Fig 4a pipeline fixes (2026-07-02, second pass)

Root-cause audit of measured-vs-TN disagreement in `fig4a_material_time.png`:

- **Unified T(E) inversion:** `EmpiricalTemperatureData.calculate_temperature_array`
  now uses the same fitted Rosenfeld–Tarazona closed-form inversion as the scalar
  path (was plain table interpolation). Validation script:
  `diagnose_ts_inversion.py` → `results/ts_inversion_validation.json`.
- **Standardized T_s source:** `_structural_Ts_from_csv` always infers T_s from
  `E_nonbonded_kjmol` via the fitted calibrator (no mixed use of CSV
  `T_s_fictive_K`, which is populated for only ~1/500 replicas in resolved
  archives). Tracked vs inferred delta on the one overlapping replica:
  mean ≈ +14 K (inferred hotter → faster TN).
- **Jensen-bias fix:** TN material time is integrated **per replica** then
  ensemble-averaged (`_load_h_tn_timeseries`), not `mean(T_s)` then integrate.
- **τ_s,eq(T) robustness:** parabolic-branch fit uses bounded `J > 0`; bootstrap
  spread at 100 K reported on Fig 4a (σ_τ ≈ 6 ps at τ ≈ 121 ps). Shaded bands
  on Eq. 12 and TN dashed curves use this relative uncertainty.
- **MTTI reconstruction:** solve grid scaled to ~8× constraint count (cap 200);
  monotonicity enforced via cumulative-increment `lsq_linear` solve; smoothness
  α scaled by constraint count. Sensitivity sweep:
  `diagnose_mtti_sensitivity.py` → `results/mtti_sensitivity.json`.

**Residual interpretation:** After fixes, TN dashed curves sit systematically
above measured MTTI solids (faster predicted aging). This reflects real tension
between energy-inferred T_s(t) + equilibrium τ(T) and measured τ_s(t_w) from
FKT — not archive/algorithm bugs. MTTI `h_end` is stable to ±0.03 across
α ∈ {0.5, 1, 2}, but λ ordering is not monotonic (λ=0.023333 late uptick gives
h_end ≈ 20 vs λ=0.01 ≈ 16); see `results/mtti_sensitivity.json`. λ=0.03 still
excluded (`ANALYSIS_LAMBDAS`).

---

## 7. Quick reference — one-liner commands

```bash
# Single smoke trajectory (replica 0, λ=0.01, short)
pixi run --as-is -e test python run_single.py --lambda 0.01 --replica 0 --smoke

# Dry-run campaign queue order
pixi run --as-is -e test python run_campaign.py --dry-run --replica-end 2 --jobs 5

# Full campaign + analysis (after all trajectories done)
./run_remaining_campaign.sh

# Progress snapshot
pixi run --as-is -e test python monitor_status.py
```

---

## 8. Related paths (do not rerun unless needed)

| Path | Notes |
|------|--------|
| `cav-hoomd/aging_weak_lambda/` | HOOMD reference data; **superseded** by OpenMM campaign for paper figures |
| `pre_fkt_fix/` | Archived outputs from pre–k-fix attempts |
| `pilot_n4/` | 4-replica pilot (if archived) |
| `diagnose_fkt/` | Unit audit, parity, k-scan, conclusion JSON |
| `../equilibrium_output/` | Long equilibrium runs (IC source) |
| `../run_c2f.py` | Force field + `FKT_KMAG_AU` definition |
| `../fkt_tracker.py` | In-situ F(k,t) tracker used by OpenMM MD |

---

## 9. Notes for downstream AI agents

1. **Do not change production \|k\| back to 6.0 a.u.** without explicit user approval; production uses **2π/σ_AA**.
2. **Do not use molecular COM** for FKT in production (`fkt_sites="atomic"` only).
3. Before starting a campaign on a new node, confirm **exactly one** `run_campaign.py` process and `JOBS=5`.
4. Copy **`eq10ns100K_lam0_final_state.npz`** and the git checkout of `config.py` / `run_c2f.py` together so k constants match.
5. After campaign completion, run `run_all_analysis.py` only when **all 500 replicas × 5 λ** pass `replica_complete()` (CSV ≥ 98% of 2500 ps + FKT ref 000 present).
6. Update the **Status checklist** (Section 6) when handing off again.
