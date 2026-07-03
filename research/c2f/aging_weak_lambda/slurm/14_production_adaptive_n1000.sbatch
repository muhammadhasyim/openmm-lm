#!/bin/bash
#SBATCH --job-name=cavity-n1000-adapt
#SBATCH --account=torch_pr_283_chemistry
#SBATCH --partition=a100_chemistry
#SBATCH --qos=gpu48
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=10
#SBATCH --mem=48G
#SBATCH --time=10:00:00
#SBATCH --array=0-249%4
#SBATCH --output=/scratch/mh7373/openmm/research/c2f/aging_weak_lambda/slurm/logs/n1000_adapt_%A_%a.out
#SBATCH --error=/scratch/mh7373/openmm/research/c2f/aging_weak_lambda/slurm/logs/n1000_adapt_%A_%a.err

set -euo pipefail

REPO_ROOT=/scratch/mh7373/openmm
CAMPAIGN_DIR="${REPO_ROOT}/research/c2f/aging_weak_lambda"
IC_FILE="${REPO_ROOT}/research/c2f/equilibrium_output/eq10ns100K_lam0_final_state.npz"

# Replicas per array task (override for pilot: export STACK=1)
STACK="${STACK:-4}"
REPLICA_START=$(( SLURM_ARRAY_TASK_ID * STACK ))
REPLICA_END=$(( REPLICA_START + STACK - 1 ))

export PATH="/scratch/mh7373/.pixi/bin:${PATH}"
export PYTHONUNBUFFERED=1
export OPENMM_PLUGIN_DIR="${REPO_ROOT}/.pixi/envs/test/lib/plugins"
# Prefer cluster NVRTC (13.0.x) over pixi conda-forge NVRTC (13.3.x); mismatched PTX
# triggers CUDA_ERROR_UNSUPPORTED_PTX_VERSION (222) on A100 nodes.
if [[ -d /usr/local/cuda/lib64 ]]; then
  export LD_LIBRARY_PATH="/usr/local/cuda/lib64:${LD_LIBRARY_PATH:-}"
fi
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
REPO_ADAPT="${REPO_ROOT}/wrappers/python/openmm/cavitymd/adaptive.py"
cp "${REPO_ADAPT}" "${CAVDIR}/"
cp "${REPO_ROOT}/wrappers/python/openmm/cavitymd/__init__.py" "${CAVDIR}/"
cp "${REPO_ROOT}/wrappers/python/openmm/cavitymd/thermostats.py" "${CAVDIR}/"

HASH_REPO=$(sha256sum "${REPO_ADAPT}" | awk '{print $1}')
HASH_DEPLOY=$(sha256sum "${CAVDIR}/adaptive.py" | awk '{print $1}')
if [[ "${HASH_REPO}" != "${HASH_DEPLOY}" ]]; then
  echo "ERROR: adaptive.py hash mismatch after deploy copy"
  exit 1
fi
echo "adaptive.py sha256=${HASH_REPO}"

echo "=== adaptive integrator unit preflight ==="
cd "${REPO_ROOT}"
"${PY}" -m pytest research/c2f/tests/test_adaptive_equilibrium.py \
  -k "pre_switch_guard or coupling_epsilon or effective_force_update or epsilon_shock or create_adaptive_state or parity_config or default_parity" \
  --maxfail=1 -q

echo "=== cav-hoomd timestep policy parity (fast, no GPU MD) ==="
"${PY}" -m pytest research/c2f/tests/test_timestep_parity.py --maxfail=1 -q

if [[ "${STACK}" == "1" ]]; then
  echo "=== replica-0 note: 2500 ps all-lambda gate replaced by test_timestep_parity.py ==="
fi

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

echo "=== array ${SLURM_ARRAY_TASK_ID}: replicas ${REPLICA_START}-${REPLICA_END} (STACK=${STACK}) ==="
echo "=== max-metric adaptive, jobs=5, all lambda including 0.0 ==="

WALL_START=$(date +%s)
JOBS="${JOBS:-5}"
if [[ "${STACK}" == "1" ]]; then
  JOBS=1
fi
"${PY}" run_campaign.py \
  --schedule replica_round \
  --jobs "${JOBS}" \
  --replica-start "${REPLICA_START}" \
  --replica-end "${REPLICA_END}" \
  --runtime-ps 1500 \
  --adaptive \
  --no-resume \
  --no-skip \
  --log campaign_n1000_log.jsonl

WALL_END=$(date +%s)
WALL_ELAPSED=$(( WALL_END - WALL_START ))
echo "=== Wall time: ${WALL_ELAPSED}s for replicas ${REPLICA_START}-${REPLICA_END} ==="
echo "=== array ${SLURM_ARRAY_TASK_ID} complete ==="
