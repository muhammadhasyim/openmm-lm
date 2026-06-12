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

HOOMD partial data in `cav-hoomd/aging_weak_lambda/` is superseded.

## Figures

| Panel | Script |
|-------|--------|
| Fig 2a IR | `analyze_ir_from_snapshots.py` |
| Fig 2a ISF | `plot_isf_curves.py` |
| Fig 2b/c | `analyze_aging_relaxation.py` |
| Fig 3b | `analyze_energy_redistribution.py` |
| Fig 3c | `analyze_fictive_temperatures.py` |
| Fig 4a–d | `analyze_material_time_aging.py` |
| DSE/bilinear | `analyze_cavity_energies.py` |

Run all: `python run_all_analysis.py`
