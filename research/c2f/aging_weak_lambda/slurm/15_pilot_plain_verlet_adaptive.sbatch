#!/bin/bash
#SBATCH --job-name=cavity-pilot-fix
#SBATCH --account=torch_pr_283_chemistry
#SBATCH --partition=a100_chemistry
#SBATCH --qos=gpu48
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=32G
#SBATCH --time=08:00:00
#SBATCH --output=/scratch/mh7373/openmm/research/c2f/aging_weak_lambda/slurm/logs/pilot_fix_%j.out
#SBATCH --error=/scratch/mh7373/openmm/research/c2f/aging_weak_lambda/slurm/logs/pilot_fix_%j.err

set -euo pipefail

REPO_ROOT=/scratch/mh7373/openmm
CAMPAIGN_DIR="${REPO_ROOT}/research/c2f/aging_weak_lambda"
export PATH="/scratch/mh7373/.pixi/bin:${PATH}"
export PYTHONUNBUFFERED=1
export OPENMM_PLUGIN_DIR="${REPO_ROOT}/.pixi/envs/test/lib/plugins"
PY="${REPO_ROOT}/.pixi/envs/test/bin/python"

CAVDIR="${REPO_ROOT}/.pixi/envs/test/lib/python3.13/site-packages/openmm/cavitymd"
cp "${REPO_ROOT}/wrappers/python/openmm/cavitymd/adaptive.py" "${CAVDIR}/"
cp "${REPO_ROOT}/wrappers/python/openmm/cavitymd/__init__.py" "${CAVDIR}/"
cp "${REPO_ROOT}/wrappers/python/openmm/cavitymd/thermostats.py" "${CAVDIR}/"

cd "${CAMPAIGN_DIR}"
echo "=== pilot: replica 42, all non-zero lambda, 1500 ps, plain Verlet adaptive ==="
"${PY}" run_campaign.py \
  --lambdas 0.01 0.016667 0.023333 0.03 \
  --replica-start 42 \
  --replica-end 42 \
  --runtime-ps 1500 \
  --adaptive \
  --no-resume \
  --no-skip \
  --jobs 4 \
  --log campaign_pilot_fix_log.jsonl

echo "=== pilot complete ==="
