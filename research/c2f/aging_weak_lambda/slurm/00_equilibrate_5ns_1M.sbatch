#!/bin/bash
#SBATCH --job-name=cavity-eq5ns-1M
#SBATCH --account=torch_pr_283_chemistry
#SBATCH --partition=a100_chemistry
#SBATCH --qos=gpu48
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=128G
#SBATCH --time=2-00:00:00
#SBATCH --output=/scratch/mh7373/openmm/research/c2f/aging_weak_lambda/slurm/logs/eq5ns_1M_%j.out
#SBATCH --error=/scratch/mh7373/openmm/research/c2f/aging_weak_lambda/slurm/logs/eq5ns_1M_%j.err

set -euo pipefail

REPO_ROOT=/scratch/mh7373/openmm
C2F_ROOT="${REPO_ROOT}/research/c2f"
IC_PREFIX="${C2F_ROOT}/equilibrium_output/eq5ns100K_N1M_lam0"
NUM_MOL=1000000
EQUIL_PS=5000

export PATH="/scratch/mh7373/.pixi/bin:${PATH}"
export OPENMM_PLUGIN_DIR="${REPO_ROOT}/.pixi/envs/test/lib/plugins"
export PYTHONUNBUFFERED=1
cd "${REPO_ROOT}"

echo "=== CUDA preflight ==="
pixi run --as-is -e test python - <<'PY'
import openmm
from openmm import unit
assert "CUDA" in [openmm.Platform.getPlatform(i).getName() for i in range(openmm.Platform.getNumPlatforms())]
s = openmm.System(); s.addParticle(1.0)
openmm.Context(s, openmm.VerletIntegrator(0.001*unit.picoseconds), openmm.Platform.getPlatformByName("CUDA"))
print("CUDA OK")
PY

echo "=== 5 ns lambda=0 equilibration at 100 K, N=${NUM_MOL} ==="
pixi run --as-is -e test python "${C2F_ROOT}/run_cavity_equilibrium.py" \
  --temperature-K 100 \
  --runtime-ps "${EQUIL_PS}" \
  --lambda 0 \
  --num-molecules "${NUM_MOL}" \
  --with-dse \
  --no-finite-q \
  --sample-interval-ps 10 \
  --platform CUDA \
  --output-prefix "${IC_PREFIX}"

IC_FILE="${IC_PREFIX}_final_state.npz"
test -f "${IC_FILE}"
echo "IC ready: ${IC_FILE}"
ls -la "${IC_FILE}" "${IC_PREFIX}_meta.txt"
