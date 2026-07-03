#!/usr/bin/env bash
# Relaunch fixed adaptive validation (λ=0.01 + 0.03, seed 42, 2500 ps) via nohup.
set -euo pipefail

REPO=/scratch/mh7373/openmm
CAMPAIGN_DIR="${REPO}/research/c2f/aging_weak_lambda"
LOG_DIR="${CAMPAIGN_DIR}/slurm/logs"
PY="${REPO}/.pixi/envs/test/bin/python"
CAVDIR="${REPO}/.pixi/envs/test/lib/python3.13/site-packages/openmm/cavitymd"

export OPENMM_PLUGIN_DIR="${REPO}/.pixi/envs/test/lib/plugins"
export PYTHONUNBUFFERED=1

mkdir -p "${LOG_DIR}" "${CAMPAIGN_DIR}/md_validation_rerun"
cp "${REPO}/wrappers/python/openmm/cavitymd/adaptive.py" "${CAVDIR}/"

cd "${CAMPAIGN_DIR}"

for pattern in "run_single.py --lambda 0.01.*md_validation_rerun" \
               "run_single.py --lambda 0.03.*md_validation_rerun"; do
  pids=$(pgrep -f "${pattern}" || true)
  if [[ -n "${pids}" ]]; then
    echo "Already running (${pattern}): ${pids}"
  fi
done

launch() {
  local gpu="$1"
  local lam="$2"
  local tag="${3}"
  if pgrep -f "run_single.py --lambda ${lam}.*md_validation_rerun" >/dev/null 2>&1; then
    echo "Skip λ=${lam} (already running)"
    return
  fi
  local resume_flag="--no-resume"
  local csv="${CAMPAIGN_DIR}/md_validation_rerun/lambda${tag}/lam${tag}_seed0042_energies.csv"
  if [[ -f "${csv}" ]]; then
    resume_flag=""
    echo "Resuming λ=${lam} from checkpoint"
  fi
  CUDA_VISIBLE_DEVICES="${gpu}" nohup "${PY}" run_single.py \
    --lambda "${lam}" --replica 0 --adaptive ${resume_flag} \
    --campaign-dir md_validation_rerun --platform CUDA \
    > "${LOG_DIR}/rerun_lam${tag}_rep0.log" 2>&1 &
  echo "Started λ=${lam} on GPU ${gpu} PID=$!"
}

launch 0 0.01 0p01
launch 1 0.03 0p03

echo ""
echo "Monitor:"
echo "  tail -f ${LOG_DIR}/rerun_lam0p01_rep0.log"
echo "  tail -f ${LOG_DIR}/rerun_lam0p03_rep0.log"
echo "  nvidia-smi"
echo ""
echo "After both finish (~2-3 h):"
echo "  ${PY} validate_replica_stability.py --campaign-dir md_validation_rerun --replica-start 0 --replica-end 0 --lambdas 0.01 0.03"
