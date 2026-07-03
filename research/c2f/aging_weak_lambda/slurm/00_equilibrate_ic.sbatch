#!/bin/bash
#SBATCH --job-name=cavity-ic-eq
#SBATCH --account=torch_pr_283_chemistry
#SBATCH --partition=a100_chemistry
#SBATCH --qos=gpu48
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=6
#SBATCH --mem=32G
#SBATCH --time=24:00:00
#SBATCH --output=/scratch/mh7373/openmm/research/c2f/aging_weak_lambda/slurm/logs/ic_%j.out
#SBATCH --error=/scratch/mh7373/openmm/research/c2f/aging_weak_lambda/slurm/logs/ic_%j.err

set -euo pipefail

REPO_ROOT=/scratch/mh7373/openmm
C2F_ROOT="${REPO_ROOT}/research/c2f"
CAMPAIGN_DIR="${C2F_ROOT}/aging_weak_lambda"
IC_PREFIX="${C2F_ROOT}/equilibrium_output/eq10ns100K_lam0"

export PATH="/scratch/mh7373/.pixi/bin:${PATH}"
cd "${REPO_ROOT}"

echo "=== pixi environment ==="
UV_LOCK_TIMEOUT=7200 pixi install -e default --frozen
UV_LOCK_TIMEOUT=7200 pixi install -e test --frozen

echo "=== CUDA preflight ==="
pixi run --as-is -e test python - <<'PY'
import sys
try:
    import openmm
    from openmm import unit
except ImportError as exc:
    sys.exit(f"OpenMM import failed: {exc}")
platforms = [openmm.Platform.getPlatform(i).getName() for i in range(openmm.Platform.getNumPlatforms())]
print("Platforms:", platforms)
if "CUDA" not in platforms:
    sys.exit("CUDA platform not available")
system = openmm.System()
system.addParticle(1.0)
integrator = openmm.VerletIntegrator(0.001 * unit.picoseconds)
ctx = openmm.Context(system, integrator, openmm.Platform.getPlatformByName("CUDA"))
del ctx
print("CUDA context OK")
PY

echo "=== 10 ns lambda=0 equilibration (IC) ==="
pixi run --as-is -e test python "${C2F_ROOT}/run_cavity_equilibrium.py" \
  --temperature-K 100 \
  --runtime-ps 10000 \
  --lambda 0 \
  --with-dse \
  --no-finite-q \
  --sample-interval-ps 10 \
  --platform CUDA \
  --output-prefix "${IC_PREFIX}"

IC_FILE="${IC_PREFIX}_final_state.npz"
if [[ ! -f "${IC_FILE}" ]]; then
  echo "ERROR: missing IC file ${IC_FILE}"
  exit 1
fi
echo "IC ready: ${IC_FILE}"
ls -la "${IC_FILE}" "${IC_PREFIX}_meta.txt"
