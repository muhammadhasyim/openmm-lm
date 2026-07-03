#!/bin/bash
#SBATCH --job-name=cavity-aging
#SBATCH --account=torch_pr_283_chemistry
#SBATCH --partition=a100_chemistry
#SBATCH --qos=gpu48
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=6
#SBATCH --mem=32G
#SBATCH --time=08:00:00
#SBATCH --array=0-499%30
#SBATCH --output=/scratch/mh7373/openmm/research/c2f/aging_weak_lambda/slurm/logs/prod_%A_%a.out
#SBATCH --error=/scratch/mh7373/openmm/research/c2f/aging_weak_lambda/slurm/logs/prod_%A_%a.err

set -euo pipefail

REPO_ROOT=/scratch/mh7373/openmm
CAMPAIGN_DIR="${REPO_ROOT}/research/c2f/aging_weak_lambda"
IC_FILE="${REPO_ROOT}/research/c2f/equilibrium_output/eq10ns100K_lam0_final_state.npz"
REPLICA="${SLURM_ARRAY_TASK_ID}"

export PATH="/scratch/mh7373/.pixi/bin:${PATH}"
export PYTHONUNBUFFERED=1
export OPENMM_PLUGIN_DIR="${REPO_ROOT}/.pixi/envs/test/lib/plugins"
PY="${REPO_ROOT}/.pixi/envs/test/bin/python"

cd "${REPO_ROOT}"
if [[ ! -x "${PY}" ]]; then
  echo "ERROR: missing ${PY}; run pixi install -e test on login node first"
  exit 1
fi
if [[ ! -d "${OPENMM_PLUGIN_DIR}" ]]; then
  echo "ERROR: missing OpenMM plugins at ${OPENMM_PLUGIN_DIR}"
  exit 1
fi
cd "${CAMPAIGN_DIR}"

if [[ ! -f "${IC_FILE}" ]]; then
  echo "ERROR: missing IC ${IC_FILE}; run 00_equilibrate_ic.sbatch first"
  exit 1
fi

echo "=== replica ${REPLICA}: 5 lambda in parallel (jobs=3, checkpoint resume) ==="
"${PY}" run_campaign.py \
  --schedule replica_round \
  --jobs 3 \
  --replica-start "${REPLICA}" \
  --replica-end "${REPLICA}"

echo "=== replica ${REPLICA} complete ==="
