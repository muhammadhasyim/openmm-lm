# OpenMM weak-coupling aging campaign

Paper-matched non-thermal aging (Figs 2–4) using OpenMM `run_cavity_equilibrium.py`.

## Parameters

- λ: 0, 0.01, 0.016667, 0.023333, 0.03 a.u.
- **500 replicas per λ** (seeds 42–541), 2.5 ns, step turn-on at 200 ps
- F(k,t): |k| = 113.4 nm⁻¹, refs every 200 ps, output every 1 ps, **F(0) at each t_w**
- **2500 trajectories total** (500 × 5 λ)

## Scheduling (N=500)

Each **replica round** runs **all 5 λ in parallel** (`--jobs 5`), then advances to the next replica.
~316 MiB GPU per job → ~1.6 GiB for 5 jobs (fits easily on 12 GiB 4070).

```bash
cd research/c2f/aging_weak_lambda

# Dry-run first 10 queued runs (order check)
pixi run --as-is -e test python run_campaign.py --dry-run --replica-end 1 --jobs 5 | head

# Fresh production (archive pilot first if needed)
chmod +x archive_pilot_and_launch_n500.sh run_n500_campaign.sh
./archive_pilot_and_launch_n500.sh

# Or manual launch
FRESH_START=1 nohup ./run_n500_campaign.sh >> campaign_n500.log 2>&1 &

# Monitor
pixi run --as-is -e test python monitor_status.py
tail -f campaign_n500.log
```

## Wall time (estimate)

~52 min per replica round × 500 rounds ≈ **18 days** continuous on RTX 4070.

## Analysis

```bash
pixi run --as-is -e test python run_all_analysis.py
```

IC: `../equilibrium_output/eq10ns100K_lam0_final_state.npz` (generate with `run_cavity_equilibrium.py` if missing).

See [DELIVERABLES.md](DELIVERABLES.md) for figure mapping.
