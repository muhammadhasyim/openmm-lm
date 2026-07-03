# Research workflows

Paper-scale campaigns and long-running studies live here—not under `examples/`, which
is reserved for short, reproducible demos.

## Layout

| Path | Purpose |
|------|---------|
| [`c2f/`](c2f/) | C2F (cavity configurational feedback) protocol, figure reproduction, aging campaigns |

## Campaign outputs

Generated outputs (trajectories, CSVs, `lambda*/`, SLURM logs, equilibrium checkpoints)
are **not** committed to git. Store them on scratch or project storage and rely on
[`.gitignore`](../.gitignore) patterns under `research/c2f/`.

## SLURM path migration (one-time)

C2F code moved from `research/c2f/` to `research/c2f/`. Before
submitting new jobs:

1. Pull this layout change and cancel or finish jobs using the old path.
2. Resubmit from the repo root using scripts under `research/c2f/aging_weak_lambda/slurm/`.
3. Update any personal wrappers that hardcode `research/c2f`.

Pixi tasks (`test-cavitymd-smoke`, `figure5`, `calibrate-fictive`) already point at
`research/c2f/`.
