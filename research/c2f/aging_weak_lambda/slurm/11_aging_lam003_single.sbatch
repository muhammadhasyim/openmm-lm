#!/bin/bash
#SBATCH --job-name=cavity-lam003
#SBATCH --account=torch_pr_283_chemistry
#SBATCH --partition=a100_chemistry
#SBATCH --qos=gpu48
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=6
#SBATCH --mem=32G
#SBATCH --time=04:00:00
#SBATCH --output=/scratch/mh7373/openmm/research/c2f/aging_weak_lambda/slurm/logs/lam003_%j.out
#SBATCH --error=/scratch/mh7373/openmm/research/c2f/aging_weak_lambda/slurm/logs/lam003_%j.err

set -euo pipefail

REPO_ROOT=/scratch/mh7373/openmm
CAMPAIGN_DIR="${REPO_ROOT}/research/c2f/aging_weak_lambda"
C2F_ROOT="${REPO_ROOT}/research/c2f"
IC_FILE="${C2F_ROOT}/equilibrium_output/eq5ns100K_lam0_final_state.npz"

# Protocol: lam=0 for 100 ps, then lam=0.03 at N=250 (g=λ√N≈0.474), aging 2.5 ns.
# λ values in run_single are reference values at N=250; scaling is automatic via g=λ√N.
SWITCH_PS=100
RUNTIME_PS=2600
LAMBDA_N250=0.03
REPLICA=0

export PATH="/scratch/mh7373/.pixi/bin:${PATH}"
export OPENMM_PLUGIN_DIR="${REPO_ROOT}/.pixi/envs/test/lib/plugins"
export PYTHONUNBUFFERED=1
PY="${REPO_ROOT}/.pixi/envs/test/bin/python"

cd "${REPO_ROOT}"
test -x "${PY}"
test -f "${IC_FILE}"

cd "${CAMPAIGN_DIR}"
echo "=== Aging production: switch=${SWITCH_PS} ps, runtime=${RUNTIME_PS} ps, lambda=${LAMBDA_N250} @ N=250 ==="
echo "=== (g=λ√N held fixed when --num-molecules ≠ 250) ==="
echo "=== IC: ${IC_FILE} ==="

"${PY}" run_single.py \
  --lambda "${LAMBDA_N250}" \
  --replica "${REPLICA}" \
  --runtime-ps "${RUNTIME_PS}" \
  --switch-time-ps "${SWITCH_PS}" \
  --initial-state "${IC_FILE}" \
  --platform CUDA

echo "=== Production complete ==="
