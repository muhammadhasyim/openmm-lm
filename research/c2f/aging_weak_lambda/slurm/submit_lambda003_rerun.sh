#!/usr/bin/env bash
# Archive lambda0p03 outputs and submit λ=0.03-only production rerun array.
set -euo pipefail

REPO_ROOT=/scratch/mh7373/openmm
CAMPAIGN_DIR="${REPO_ROOT}/research/c2f/aging_weak_lambda"
SLURM_DIR="${CAMPAIGN_DIR}/slurm"
LOG_DIR="${SLURM_DIR}/logs"

mkdir -p "${LOG_DIR}"

echo "=== Archive all lambda0p03 outputs for intentional rerun ==="
bash "${CAMPAIGN_DIR}/archive_lambda003_for_rerun.sh"

echo ""
echo "=== Cancel prior lambda0.03 rerun array if still active ==="
scancel --name=cavity-lam003-rerun 2>/dev/null || true

echo ""
echo "=== Submit lambda=0.03-only production array ==="
JOB_RERUN=$(sbatch --parsable "${SLURM_DIR}/12_rerun_lambda003_array.sbatch")
echo "Submitted lambda0.03 rerun array: ${JOB_RERUN}"

JOB_VERIFY=$(sbatch --parsable --dependency=afterok:"${JOB_RERUN}" "${SLURM_DIR}/13_verify_lambda003_rerun.sbatch")
echo "Submitted post-rerun verify + fig2a replot: ${JOB_VERIFY}"

cat <<EOF

Lambda 0.03 rerun submitted:
  array job  ${JOB_RERUN}
  verify job ${JOB_VERIFY}  (afterok:${JOB_RERUN})
  scope      lambda0p03/ replicas 0-499 only
  manifest   ${CAMPAIGN_DIR}/results/lambda003_archive_manifest.json

Monitor: squeue -u \$USER -n cavity-lam003-rerun
Logs:    ${LOG_DIR}/lam003_rerun_${JOB_RERUN}_*.out

Verification + fig2a replot run automatically after array completes (job ${JOB_VERIFY}).
EOF
