#!/usr/bin/env bash
# Local GPU pilot: replica 42, four lambdas, 1500 ps, velocity-Verlet adaptive.
set -euo pipefail

REPO=/scratch/mh7373/openmm
CAMPAIGN="${REPO}/research/c2f/aging_weak_lambda"
PILOT="${CAMPAIGN}/pilot_velocity_verlet_1500ps"
LOG="${CAMPAIGN}/slurm/logs/pilot_velocity_verlet_gpu0.log"
PY="${REPO}/.pixi/envs/test/bin/python"
SITE="${REPO}/.pixi/envs/test/lib/python3.13/site-packages/openmm/cavitymd"

export CUDA_VISIBLE_DEVICES=0
export OPENMM_PLUGIN_DIR="${REPO}/.pixi/envs/test/lib/plugins"
export PYTHONUNBUFFERED=1

# Ensure runtime picks up workspace cavitymd (velocity-Verlet integrator fix)
cp "${REPO}/wrappers/python/openmm/cavitymd/adaptive.py" "${SITE}/"
cp "${REPO}/wrappers/python/openmm/cavitymd/__init__.py" "${SITE}/"
cp "${REPO}/wrappers/python/openmm/cavitymd/thermostats.py" "${SITE}/"

cd "${CAMPAIGN}"
: > "${LOG}"
echo "=== pilot start $(date -Is) ===" | tee -a "${LOG}"

pids=()
for lam in 0.01 0.016667 0.023333 0.03; do
  echo "launch lam=${lam}" | tee -a "${LOG}"
  "${PY}" run_single.py \
    --lambda "${lam}" \
    --replica 42 \
    --runtime-ps 1500 \
    --switch-time-ps 200 \
    --adaptive \
    --no-resume \
    --no-fkt \
    --no-dipole \
    --platform CUDA \
    --campaign-dir "${PILOT}" >> "${LOG}" 2>&1 &
  pids+=("$!")
done

fail=0
for pid in "${pids[@]}"; do
  if ! wait "${pid}"; then
    fail=1
  fi
done

echo "=== validate $(date -Is) ===" | tee -a "${LOG}"
if "${PY}" validate_pilot_blowups.py \
  --lambda 0.01 0.016667 0.023333 0.03 \
  --replica 42 \
  --runtime-ps 1500 \
  --campaign-dir "${PILOT}" | tee -a "${LOG}"; then
  echo "=== PILOT PASS $(date -Is) ===" | tee -a "${LOG}"
else
  echo "=== PILOT FAIL $(date -Is) ===" | tee -a "${LOG}"
  exit 1
fi

exit "${fail}"
