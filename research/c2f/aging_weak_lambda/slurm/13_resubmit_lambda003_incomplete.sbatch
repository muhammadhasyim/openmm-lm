#!/bin/bash
#SBATCH --job-name=cavity-lam003-resubmit
#SBATCH --account=torch_pr_283_chemistry
#SBATCH --partition=a100_chemistry
#SBATCH --qos=gpu48
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=6
#SBATCH --mem=32G
#SBATCH --time=08:00:00
#SBATCH --output=/scratch/mh7373/openmm/research/c2f/aging_weak_lambda/slurm/logs/lam003_resubmit_%A_%a.out
#SBATCH --error=/scratch/mh7373/openmm/research/c2f/aging_weak_lambda/slurm/logs/lam003_resubmit_%A_%a.err

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

CAVDIR="${REPO_ROOT}/.pixi/envs/test/lib/python3.13/site-packages/openmm/cavitymd"
cp "${REPO_ROOT}/wrappers/python/openmm/cavitymd/adaptive.py" "${CAVDIR}/"
cp "${REPO_ROOT}/wrappers/python/openmm/cavitymd/__init__.py" "${CAVDIR}/"

echo "=== CUDA preflight ==="
"${PY}" - <<'PY'
import sys
import openmm
from openmm import unit

platforms = [openmm.Platform.getPlatform(i).getName() for i in range(openmm.Platform.getNumPlatforms())]
print("Platforms:", platforms)
if "CUDA" not in platforms:
    sys.exit("CUDA platform not available (check OPENMM_PLUGIN_DIR)")
system = openmm.System()
system.addParticle(1.0)
integrator = openmm.VerletIntegrator(0.001 * unit.picoseconds)
ctx = openmm.Context(system, integrator, openmm.Platform.getPlatformByName("CUDA"))
del ctx
print("CUDA context OK")
PY

cd "${CAMPAIGN_DIR}"

if [[ ! -f "${IC_FILE}" ]]; then
  echo "ERROR: missing IC ${IC_FILE}; run 00_equilibrate_ic.sbatch first"
  exit 1
fi

echo "=== replica ${REPLICA}: lambda=0.03 resubmit (max-metric adaptive) ==="
"${PY}" run_campaign.py \
  --lambdas 0.03 \
  --schedule replica_round \
  --jobs 1 \
  --replica-start "${REPLICA}" \
  --replica-end "${REPLICA}" \
  --no-skip \
  --adaptive \
  --no-resume

echo "=== replica ${REPLICA} lambda=0.03 resubmit complete ==="
