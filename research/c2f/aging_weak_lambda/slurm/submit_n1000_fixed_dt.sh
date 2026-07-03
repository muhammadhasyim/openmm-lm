#!/usr/bin/env bash
# Archive existing outputs, rebuild CUDA plugin, submit N=1000 fixed-dt campaign.
set -euo pipefail

REPO_ROOT=/scratch/mh7373/openmm
CAMPAIGN_DIR="${REPO_ROOT}/research/c2f/aging_weak_lambda"
SLURM_DIR="${CAMPAIGN_DIR}/slurm"
LOG_DIR="${SLURM_DIR}/logs"
PY="${REPO_ROOT}/.pixi/envs/test/bin/python"
SBATCH="${SLURM_DIR}/16_production_fixed_dt_n1000.sbatch"
DT_PS="${DT_PS:-0.0001}"

export PATH="/scratch/mh7373/.pixi/bin:${PATH}"
export OPENMM_PLUGIN_DIR="${REPO_ROOT}/.pixi/envs/test/lib/plugins"

mkdir -p "${LOG_DIR}"
cd "${CAMPAIGN_DIR}"

MODE="${1:-full}"

echo "=== Cancel stale cavity aging jobs ==="
for pattern in cavity-aging cavity-lam003-resubmit cavity-n1000-adapt cavity-n1000-fix01; do
  STALE=$(squeue -u "${USER}" -h -o "%i %j" 2>/dev/null | awk -v p="${pattern}" '$2 ~ p {print $1}' || true)
  if [[ -n "${STALE}" ]]; then
    echo "Cancelling: ${STALE}"
    scancel ${STALE}
  fi
done

echo "=== Archive all existing campaign outputs ==="
"${PY}" archive_campaign_outputs.py --full-rerun

echo "=== Install cluster NVRTC into pixi env (GPU node; login node has no /usr/local/cuda) ==="
FIX_SBATCH="${SLURM_DIR}/00_rebuild_cuda_plugin.sbatch"
FIX_JOB=$(sbatch --parsable "${FIX_SBATCH}")
echo "Submitted fix-cuda job ${FIX_JOB}; waiting for completion..."
while true; do
  STATE=$(sacct -j "${FIX_JOB}" --format=State -n -P 2>/dev/null | head -1 || true)
  case "${STATE}" in
    COMPLETED) break ;;
    FAILED|CANCELLED|TIMEOUT|NODE_FAIL)
      echo "ERROR: fix-cuda job ${FIX_JOB} ended with state ${STATE}" >&2
      echo "Check ${LOG_DIR}/fix_cuda_${FIX_JOB}.out and .err" >&2
      exit 1
      ;;
    "") sleep 5 ;;
    *) sleep 10 ;;
  esac
done
echo "fix-cuda job ${FIX_JOB} completed OK"
cd "${CAMPAIGN_DIR}"

if [[ "${MODE}" == "pilot" ]]; then
  echo "=== Submit pilot: array=0, STACK=1 (1 replica, 5 lambda, jobs=5), dt=${DT_PS} ps ==="
  JOB_ID=$(STACK=1 DT_PS="${DT_PS}" sbatch --parsable --array=0 "${SBATCH}")
  echo "Pilot job ${JOB_ID}"
  echo "Monitor: tail -f ${LOG_DIR}/n1000_fix01fs_${JOB_ID}_0.out"
  exit 0
fi

if [[ "${MODE}" != "full" ]]; then
  echo "Usage: $0 [pilot|full]" >&2
  exit 1
fi

echo "=== Submit full N=1000 campaign: array=0-249%4, STACK=4, dt=${DT_PS} ps (0.1 fs) ==="
JOB_ID=$(DT_PS="${DT_PS}" sbatch --parsable --array=0-249%4 "${SBATCH}")
echo "Submitted production job ${JOB_ID}"
echo "Monitor: squeue -u \$USER"
echo "Status:  ${PY} monitor_status.py"
