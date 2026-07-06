#!/bin/bash
#SBATCH --job-name=cavity-lam003-verify
#SBATCH --account=torch_pr_283_chemistry
#SBATCH --partition=a100_chemistry
#SBATCH --qos=gpu48
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=4
#SBATCH --mem=16G
#SBATCH --time=01:00:00
#SBATCH --output=/scratch/mh7373/openmm/research/c2f/aging_weak_lambda/slurm/logs/lam003_verify_%j.out
#SBATCH --error=/scratch/mh7373/openmm/research/c2f/aging_weak_lambda/slurm/logs/lam003_verify_%j.err

set -euo pipefail

REPO_ROOT=/scratch/mh7373/openmm
CAMPAIGN_DIR="${REPO_ROOT}/research/c2f/aging_weak_lambda"
PY="${REPO_ROOT}/.pixi/envs/test/bin/python"

cd "${CAMPAIGN_DIR}"

echo "=== Verify lambda0p03 rerun (N=1000 campaign) ==="
"${PY}" verify_lambda003_rerun.py \
  --output results/lambda003_rerun_verification.json

echo "=== Replot fig2a ==="
"${PY}" plot_isf_curves.py

echo "=== Done ==="
ls -la figures/fig2a_isf_vs_time.png
