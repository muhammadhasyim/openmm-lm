# OpenMM weak-coupling aging — deliverables

Regenerating Figs 2–4 from arXiv:2603.15693 on λ ∈ [0.01, 0.03] using **OpenMM**
(`run_cavity_equilibrium.py` + `fkt_tracker.py`).

## Simulation setup

| | |
|---|---|
| Engine | OpenMM c2f_protocol |
| λ (a.u.) | 0, 0.01, 0.016667, 0.023333, 0.03 |
| T | 100 K |
| ω_c | 1560 cm⁻¹ |
| System | 250 molecules, 500 atoms (+ cavity) |
| Replicas | **500 per λ** (seeds 42–541; paper-scale ensemble) |
| Total runs | **2500** (500 × 5 λ) |
| Schedule | All λ in parallel each replica round (`--schedule replica_round --jobs 5`) |
| GPU | ~316 MiB/job → ~1.6 GiB for 5 parallel (4070 12 GiB) |
| Est. wall time | ~52 min/round × 500 ≈ **18 days** |
| Protocol | Step turn-on 200 ps, 2.5 ns total |
| Outputs | `*_energies.csv`, `*_fkt_ref_*.txt`, `*_snapshots.npz` |

Production data live in per-replica `*_archive_full_rerun_*` subdirectories under
`lambda*/`. Analysis scripts resolve archived paths automatically via `fkt_utils.py`.

HOOMD partial data in `cav-hoomd/aging_weak_lambda/` is superseded.

## Figure 2 pipeline (individual panels)

Algorithms ported from
`third_party/cavity_supercooled_archive/final_production_run/scripts/2026-01-29/`
(`process_fskt_only.py`, `plot_fkt_analysis.py`, `plot_individual_fkt_coupling.py`,
`process_dipole_autocorr.py`).

| Panel | Script | Output |
|-------|--------|--------|
| Master F(k,t) | `build_master_fkt.py` | `master_fkt/lambda*/master_fkt_ref*.txt` |
| Fig 2a IR | `analyze_ir_from_dipole.py` | `figures/fig2a_ir_spectra.{png,pdf}` |
| Fig 2a F(k,t) | `plot_isf_curves.py` | `figures/fig2a_fkt_lambda*.{png,pdf}` (one per λ) |
| Fig 2b/c | `analyze_aging_relaxation.py` | `figures/fig2b_*.{png,pdf}`, `fig2c_*.{png,pdf}` |

F(k,t) normalization: all waiting times at a given λ share ref0-single normalization
(first non-zero F(k,0) from `master_fkt_ref000.txt`). Relaxation times use
`find_relaxation_time()` at F(k,t)=0.1; τ̃_s = τ(λ,t_w)/τ(λ=0,t_w).

Run Figure 2 only (background):

```bash
cd research/c2f/aging_weak_lambda
chmod +x run_figure2_nohup.sh
nohup ./run_figure2_nohup.sh >> figure2_pipeline.log 2>&1 &
echo $! > figure2_pipeline.pid
tail -f figure2_pipeline.log
```

Run full analysis: `python run_all_analysis.py`

## Other figures

| Panel | Script | Output |
|-------|--------|--------|
| Fig 3b | `analyze_energy_redistribution.py` | `figures/fig3b_energy_redistribution_lam0.016667.{png,pdf}` |
| Fig 3c | `analyze_fictive_temperatures.py` | `figures/fig3c_fictive_temperatures_lam0.016667.{png,pdf}` |
| Fig 3(b,c) composite | `make_figure3bc.py` | `figures/figure3bc_weak_coupling.{png,pdf}` |
| Fig 4a–d | `analyze_material_time_aging.py` | `figures/fig4{a,b,c,d}_*.png` |
| Fig 4 composite | `make_figure4.py` | `figures/figure4_weak_coupling.{png,pdf}` |
| DSE/bilinear | `analyze_cavity_energies.py` | `figures/cavity_energies_vs_time.{png,pdf}` |

Fig 3(b,c) and Fig 4 use N=500 replicas for λ ∈ {0, 0.01, 0.016667, 0.023333}.
**λ=0.03 is excluded from analysis** until its N=1000 campaign completes
(`ANALYSIS_LAMBDAS` / `FIG3_SHOWCASE_LAMBDA=0.016667` in `config.py`).

**Publication styling** is centralized in `paper_style.py` (ported from
`third_party/cav-hoomd/plotting/plot_fictive_temperature.py`). All Fig 2–4 scripts
call `apply_paper_style()` before plotting. LaTeX rendering uses, in order:
(1) standard `latex`+`dvipng` if fully functional; (2) **`tectonic`+`ghostscript`**
via `pixi global install texlive-core ghostscript tectonic` (required on this cluster
because conda `texlive-core` lacks format files and `dvipng` is unavailable); (3)
matplotlib `mathtext` with the `cm` fontset as last resort. Fig 2a F(k,t) panels use
y-label `$\phi_k(t; t_{\mathrm{w}})$` (archive/paper convention), viridis colormap,
dashed grid, and colorbar ticks every 400 ps. The Fig 4(b) ISF collapse plots
`Φ_k(h) = e^{-h^β}` (paper Eq. 13, no free amplitude), cross-checked against
`third_party/cavity_supercooled_archive/final_production_run/scripts/2026-01-29/`
(see `TODO.md` §6.1 for the full cross-check and bugs found/fixed).
