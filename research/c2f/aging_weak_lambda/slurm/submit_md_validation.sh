#!/usr/bin/env bash
# Submit production-faithful MD validation before N=1000 adaptive campaign.
set -euo pipefail

REPO_ROOT=/scratch/mh7373/openmm
CAMPAIGN_DIR="${REPO_ROOT}/research/c2f/aging_weak_lambda"
SLURM_DIR="${CAMPAIGN_DIR}/slurm"
LOG_DIR="${SLURM_DIR}/logs"
SBATCH="${SLURM_DIR}/15_md_validation.sbatch"

export PATH="/scratch/mh7373/.pixi/bin:${PATH}"

mkdir -p "${LOG_DIR}"
cd "${CAMPAIGN_DIR}"

MODE="${1:-both}"

echo "=== Cancel stale MD validation jobs ==="
for pattern in cavity-md-val; do
  STALE=$(squeue -u "${USER}" -h -o "%i %j" 2>/dev/null | awk -v p="${pattern}" '$2 ~ p {print $1}' || true)
  if [[ -n "${STALE}" ]]; then
    echo "Cancelling: ${STALE}"
    scancel ${STALE}
  fi
done

if [[ "${MODE}" == "primary" ]]; then
  echo "=== Submit primary validation: array=0 (rep0, all lambda, jobs=5) ==="
  JOB_ID=$(sbatch --parsable --array=0 "${SBATCH}")
elif [[ "${MODE}" == "both" ]]; then
  echo "=== Submit primary + spot-check: array=0-1 ==="
  JOB_ID=$(sbatch --parsable --array=0-1 "${SBATCH}")
else
  echo "Usage: $0 [primary|both]" >&2
  exit 1
fi

echo "Submitted MD validation job ${JOB_ID}"
echo "Monitor:"
echo "  tail -f ${LOG_DIR}/md_validation_${JOB_ID}_0.out"
echo "  tail -f ${LOG_DIR}/md_validation_${JOB_ID}_1.out"
echo ""
echo "After completion, run:"
echo "  bash ${SLURM_DIR}/check_md_validation.sh"
