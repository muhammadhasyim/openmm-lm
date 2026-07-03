#!/bin/bash
#SBATCH --job-name=cavity-analyze
#SBATCH --account=torch_pr_283_chemistry
#SBATCH --partition=a100_chemistry
#SBATCH --qos=gpu48
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=4
#SBATCH --mem=16G
#SBATCH --time=04:00:00
#SBATCH --output=/scratch/mh7373/openmm/research/c2f/aging_weak_lambda/slurm/logs/analysis_%j.out
#SBATCH --error=/scratch/mh7373/openmm/research/c2f/aging_weak_lambda/slurm/logs/analysis_%j.err

set -euo pipefail

REPO_ROOT=/scratch/mh7373/openmm
CAMPAIGN_DIR="${REPO_ROOT}/research/c2f/aging_weak_lambda"

export PATH="/scratch/mh7373/.pixi/bin:${PATH}"
cd "${REPO_ROOT}"
UV_LOCK_TIMEOUT=7200 pixi install -e default --frozen
UV_LOCK_TIMEOUT=7200 pixi install -e test --frozen
cd "${CAMPAIGN_DIR}"

echo "=== campaign completion check ==="
pixi run --as-is -e test python monitor_status.py

echo "=== stability spot-check (first 10 replicas) ==="
pixi run --as-is -e test python validate_replica_stability.py \
  --replica-start 0 --replica-end 9 || echo "WARN: stability check failed (campaign may still be running)"

echo "=== full analysis pipeline ==="
pixi run --as-is -e test python run_all_analysis.py

echo "=== FKT k / sites verification (replica 0, lambda=0) ==="
META="${CAMPAIGN_DIR}/lambda0/lam0_seed0042_meta.txt"
if [[ -f "${META}" ]]; then
  grep fkt_kmag "${META}" || true
  grep fkt_sites "${META}" || true
else
  echo "WARN: ${META} not found yet"
fi

echo "=== figures ==="
ls -la "${CAMPAIGN_DIR}/figures/"
