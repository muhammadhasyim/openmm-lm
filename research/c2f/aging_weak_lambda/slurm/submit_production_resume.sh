#!/usr/bin/env bash
# Resubmit production only (IC already complete). Archives stale partials first.
set -euo pipefail

REPO_ROOT=/scratch/mh7373/openmm
CAMPAIGN_DIR="${REPO_ROOT}/research/c2f/aging_weak_lambda"
SLURM_DIR="${CAMPAIGN_DIR}/slurm"
LOG_DIR="${SLURM_DIR}/logs"

mkdir -p "${LOG_DIR}"

echo "=== Archive partial outputs from prior TIMEOUT runs (no checkpoint) ==="
bash "${CAMPAIGN_DIR}/archive_partial_before_resubmit.sh"

echo ""
echo "=== Cancel prior production array if still active ==="
scancel --name=cavity-aging 2>/dev/null || true

echo ""
echo "=== Submit production array (8 h wall, jobs=3, checkpoint resume) ==="
JOB_PROD=$(sbatch --parsable "${SLURM_DIR}/10_production_array.sbatch")
echo "Submitted production array: ${JOB_PROD}"

JOB_ANALYSIS=$(sbatch --parsable --dependency=afterok:"${JOB_PROD}" "${SLURM_DIR}/20_analysis.sbatch")
echo "Submitted analysis job: ${JOB_ANALYSIS}"

cat <<EOF

Production resume chain:
  production  ${JOB_PROD}
  analysis    ${JOB_ANALYSIS}  (afterok:${JOB_PROD})

Monitor: squeue -u \$USER
Logs:    ${LOG_DIR}/
EOF
