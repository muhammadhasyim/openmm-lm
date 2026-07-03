#!/usr/bin/env bash
# Saturate GPU 0 with λ=0.03 and GPU 1 with λ=0.023333 (N=250, ~844 MiB/job).
set -euo pipefail

ROOT="/scratch/mh7373/openmm"
CAMPAIGN="${ROOT}/research/c2f/aging_weak_lambda"
PY="${ROOT}/.pixi/envs/test/bin/python"
PLUGIN_DIR="${ROOT}/.pixi/envs/test/lib/plugins"
CAVDIR="${ROOT}/.pixi/envs/test/lib/python3.13/site-packages/openmm/cavitymd"
LOG_DIR="${CAMPAIGN}/logs/dual_gpu_fkt_resubmit"

export PYTHONUNBUFFERED=1
export OPENMM_PLUGIN_DIR="${PLUGIN_DIR}"

mkdir -p "${LOG_DIR}"
cp "${ROOT}/wrappers/python/openmm/cavitymd/"*.py "${CAVDIR}/"

cd "${CAMPAIGN}"

# Default 20/GPU after disk-quota incident at 92/GPU (844 MiB/job -> ~16.9 GiB/GPU).
MAX_JOBS_PER_GPU="${MAX_JOBS_PER_GPU:-20}"

echo "=== Dual-GPU FKT resubmit $(date -Is) ===" | tee -a "${LOG_DIR}/orchestrator.log"
echo "GPU0 -> lambda=0.03 | GPU1 -> lambda=0.023333 | ${MAX_JOBS_PER_GPU} concurrent/GPU" | tee -a "${LOG_DIR}/orchestrator.log"

exec "${PY}" run_dual_gpu_saturated.py \
  --replica-start "${REPLICA_START:-0}" \
  --replica-end "${REPLICA_END:-999}" \
  --max-jobs-gpu0 "${MAX_JOBS_PER_GPU}" \
  --max-jobs-gpu1 "${MAX_JOBS_PER_GPU}" \
  --no-resume \
  --adaptive \
  "$@" \
  2>&1 | tee -a "${LOG_DIR}/orchestrator.log"
