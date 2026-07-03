#!/bin/bash
#SBATCH --job-name=cavity-fix-cuda
#SBATCH --account=torch_pr_283_chemistry
#SBATCH --partition=a100_chemistry
#SBATCH --qos=gpu48
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=4
#SBATCH --mem=16G
#SBATCH --time=00:30:00
#SBATCH --output=/scratch/mh7373/openmm/research/c2f/aging_weak_lambda/slurm/logs/fix_cuda_%j.out
#SBATCH --error=/scratch/mh7373/openmm/research/c2f/aging_weak_lambda/slurm/logs/fix_cuda_%j.err

set -euo pipefail

REPO_ROOT=/scratch/mh7373/openmm
export PATH="/scratch/mh7373/.pixi/bin:${PATH}"
export OPENMM_PLUGIN_DIR="${REPO_ROOT}/.pixi/envs/test/lib/plugins"
PY="${REPO_ROOT}/.pixi/envs/test/bin/python"

echo "=== Node: $(hostname) ==="
echo "=== Driver / CUDA (nvidia-smi) ==="
nvidia-smi || true
echo "=== System CUDA toolkit candidates ==="
for d in /usr/local/cuda /usr/local/cuda-12 /usr/local/cuda-12.0 /usr/local/cuda-12.4; do
  if [[ -d "${d}" ]]; then
    echo "  ${d}:"
    ls -la "${d}/lib64/libnvrtc"* 2>/dev/null || true
    "${d}/bin/nvcc" --version 2>/dev/null | tail -1 || true
  fi
done

cd "${REPO_ROOT}"
echo "=== Install cluster NVRTC into pixi env (PTX 13.0 for A100 driver) ==="
CONDA_PREFIX="${REPO_ROOT}/.pixi/envs/test" bash scripts/install_cluster_nvrtc.sh

echo "=== Plugin NVRTC linkage ==="
ldd "${REPO_ROOT}/.pixi/envs/test/lib/plugins/libOpenMMCUDA.so" | grep -i nvrtc || true
if [[ -d /usr/local/cuda/lib64 ]]; then
  export LD_LIBRARY_PATH="/usr/local/cuda/lib64:${LD_LIBRARY_PATH:-}"
fi

echo "=== CUDA smoke test ==="
"${PY}" - <<'PY'
import openmm
from openmm import unit

platforms = [openmm.Platform.getPlatform(i).getName() for i in range(openmm.Platform.getNumPlatforms())]
print("Platforms:", platforms)
system = openmm.System()
system.addParticle(1.0)
integrator = openmm.VerletIntegrator(0.001 * unit.picoseconds)
ctx = openmm.Context(system, integrator, openmm.Platform.getPlatformByName("CUDA"))
del ctx
print("CUDA context OK")
PY

echo "=== fix-cuda complete ==="
