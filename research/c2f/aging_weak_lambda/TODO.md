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
| **Fig 2a** (top) IR spectra | `analyze_ir_from_snapshots.py` | `figures/fig2a_ir_spectra.png` | `*_snapshots.npz`, replica 0 default |
| **Fig 2a** (bottom) ISF φ(t) | `plot_isf_curves.py` | `figures/fig2a_isf_vs_time.png` | `*_fkt_ref_*.txt`, block \|φ\| ensemble |
| **Fig 2b** τ̃_s vs λ | `analyze_aging_relaxation.py` | `figures/fig2b_tau_tilde_vs_lambda.png` | FKT files, λ=0 baseline |
| **Fig 2c** τ̃_s vs t_w | `analyze_aging_relaxation.py` | `figures/fig2c_tau_tilde_vs_tw.png` | same |
| **Fig 3b** energy redistribution | `analyze_energy_redistribution.py` | `figures/fig3b_energy_redistribution_lam0.03.png` | `*_energies.csv`, default λ=0.03 |
| **Fig 3c** fictive temperatures | `analyze_fictive_temperatures.py` | `figures/fig3c_fictive_temperatures_lam0.03.png` | `*_energies.csv`, default λ=0.03 |
| **Fig 4a** material time | `analyze_material_time_aging.py` | `figures/fig4a_material_time.png` | CSV + FKT + `relaxation_times_vs_temperature.txt` |
| **Fig 4b** ISF collapse | `analyze_material_time_aging.py` | `figures/fig4b_isf_collapse.png` | same |
| **Fig 4c–d** TN overlays | `analyze_material_time_aging.py` | `figures/fig4c_*.png`, `fig4d_*.png` | same |
| Cavity energy diagnostics | `analyze_cavity_energies.py` | `figures/cavity_energies_vs_time.png` | `*_energies.csv` |

JSON summaries: `results/relaxation_summary.json`, `results/` (cavity energy CSV from `analyze_cavity_energies.py`).

**Analysis conventions**

- τ_s: `fkt_utils.extract_tau_s(..., use_block_average=True, min_lag_ps=10.0)`
- ISF plots: `average_phi_over_replicas(..., block_window_ps=10.0)`
- τ̃_s = τ_s(λ) / τ_s(λ=0) at matched t_w
- Fig 3b/c default showcase λ = **0.03** (override with `--lambda`)

---

## 6. Status checklist (update as work proceeds)

Use `monitor_status.py` and filesystem counts; targets in **bold**.

| Task | Target | Status |
|------|--------|--------|
| IC `eq10ns100K_lam0_final_state.npz` | 1 file | ✅ exists on primary machine |
| Production aging trajectories | **2500** | 🔄 in progress (`campaign_n500_k2pi.log`, k = 2π/σ_AA) |
| Full replicas (5/5 λ complete) | **500** | ⬜ |
| Fig 2a–c analysis | 3 figures | ⬜ blocked on FKT ensemble |
| Fig 3b–c analysis | 2 figures | ⬜ blocked on CSV ensemble |
| Fig 4a–d analysis | 4 figures | ⬜ blocked on full campaign |
| `run_all_analysis.py` end-to-end | pass | ⬜ |

**Stale data warning:** `pre_fkt_fix/` and partial files from earlier campaigns (k = 6 a.u. / 113.4 nm⁻¹) must **not** be mixed with k = 2π/σ_AA results. The k2pi relaunch uses `--no-skip` to overwrite. Confirm `fkt_kmag_nm_inv ≈ 19.057` in new `*_meta.txt` files.

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
