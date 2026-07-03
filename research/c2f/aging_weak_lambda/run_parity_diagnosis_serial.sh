#!/usr/bin/env bash
# Serial OpenMM vs cav-hoomd adaptive parity diagnosis (one GPU job at a time).
set -euo pipefail

REPO=/scratch/mh7373/openmm
export OPENMM_PLUGIN_DIR="${REPO}/.pixi/envs/test/lib/plugins"
export PYTHONUNBUFFERED=1
PY="${REPO}/.pixi/envs/test/bin/python"
CAMPAIGN="${REPO}/research/c2f/aging_weak_lambda"
C2F="${REPO}/research/c2f"
OUT="${CAMPAIGN}/diagnose_fkt/parity_runs"

cp "${REPO}/wrappers/python/openmm/cavitymd/adaptive.py" \
   "${REPO}/.pixi/envs/test/lib/python3.13/site-packages/openmm/cavitymd/"

cd "${CAMPAIGN}"
mkdir -p "${OUT}"

echo "=== Unit preflight ==="
cd "${REPO}"
"${PY}" -m pytest research/c2f/tests/test_adaptive_equilibrium.py \
  -k "pre_switch or coupling_epsilon or effective_force or parity or default_parity or effective_dt_max" \
  -q --maxfail=3

echo "=== Phase 1: pre-switch determinism (250 ps, serial) ==="
cd "${CAMPAIGN}"
"${PY}" diagnose_adaptive_parity.py \
  --phases pre_switch \
  --seed 42 \
  --pre-switch-runtime-ps 250 \
  --output-dir "${OUT}" \
  --platform CUDA

echo "=== dt logs: t=90-120 (lam=0.03) and t=740-760 (lam=0.01) ==="
cd "${C2F}"
IC="${C2F}/equilibrium_output/eq10ns100K_lam0_final_state.npz"
DT_OUT="${OUT}/dt_logs"
mkdir -p "${DT_OUT}"
"${PY}" diagnose_adaptive_switch.py --seed 42 --lambda 0.03 \
  --t-start-ps 90 --t-end-ps 120 --sample-interval-ps 0.5 \
  --initial-state "${IC}" --output "${DT_OUT}/lam003_t90_120.csv" --platform CUDA
"${PY}" diagnose_adaptive_switch.py --seed 42 --lambda 0.01 \
  --t-start-ps 740 --t-end-ps 760 --sample-interval-ps 0.5 \
  --initial-state "${IC}" --output "${DT_OUT}/lam001_t740_760.csv" --platform CUDA

echo "=== Phase 3: OpenMM dt reference + archived HOOMD logs ==="
cd "${CAMPAIGN}"
"${PY}" diagnose_hoomd_adaptive_reference.py --output-dir "${OUT}/phase3"

echo "=== Phases 2+4: ablation + knob sweep (800 ps quick) ==="
"${PY}" diagnose_adaptive_parity.py \
  --phases ablation knobs \
  --seed 42 \
  --quick \
  --output-dir "${OUT}" \
  --platform CUDA

echo "=== Phase 5 gate: seed-42 all-lambda 2500 ps CUDA pytest ==="
cd "${REPO}"
"${PY}" -m pytest research/c2f/tests/test_adaptive_equilibrium.py \
  -k "test_adaptive_seed42_all_lambdas_2500ps" \
  --maxfail=1 -q

echo "=== Done ==="
