#!/usr/bin/env bash
# Continue parity diagnosis from dt-logs onward (after Phase 1 pre-switch sweep).
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

echo "=== Waiting for GPU (no concurrent pre_switch run) ==="
while pgrep -f "diagnose_adaptive_parity.py --phases pre_switch" >/dev/null 2>&1; do
  sleep 30
done
echo "=== GPU free — continuing from dt logs ==="

cd "${C2F}"
IC="${C2F}/equilibrium_output/eq10ns100K_lam0_final_state.npz"
DT_OUT="${OUT}/dt_logs"
mkdir -p "${DT_OUT}"

if [[ ! -f "${DT_OUT}/lam003_t90_120.csv" ]]; then
  "${PY}" diagnose_adaptive_switch.py --seed 42 --lambda 0.03 \
    --t-start-ps 90 --t-end-ps 120 --sample-interval-ps 0.5 \
    --initial-state "${IC}" --output "${DT_OUT}/lam003_t90_120.csv" --platform CUDA
fi
if [[ ! -f "${DT_OUT}/lam001_t740_760.csv" ]]; then
  "${PY}" diagnose_adaptive_switch.py --seed 42 --lambda 0.01 \
    --t-start-ps 740 --t-end-ps 760 --sample-interval-ps 0.5 \
    --initial-state "${IC}" --output "${DT_OUT}/lam001_t740_760.csv" --platform CUDA
fi

echo "=== Phase 3: OpenMM dt reference + archived HOOMD logs ==="
cd "${CAMPAIGN}"
"${PY}" diagnose_hoomd_adaptive_reference.py --output-dir "${OUT}/phase3"

echo "=== Phases 2+4: ablation + knob sweep (quick) ==="
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

echo "=== Done (phases 2-5) ==="
