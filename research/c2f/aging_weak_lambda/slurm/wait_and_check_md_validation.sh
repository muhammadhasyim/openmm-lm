#!/usr/bin/env bash
# Wait for MD validation SLURM job, then run post-run gate.
set -euo pipefail

JOB_ID="${1:?usage: wait_and_check_md_validation.sh JOB_ID}"
SLURM_DIR="/scratch/mh7373/openmm/research/c2f/aging_weak_lambda/slurm"
LOG="${SLURM_DIR}/logs/wait_md_validation_${JOB_ID}.log"

exec > >(tee -a "${LOG}") 2>&1
echo "Waiting for SLURM job ${JOB_ID} to finish..."
while squeue -h -j "${JOB_ID}" 2>/dev/null | grep -q .; do
  sleep 120
done
echo "Job ${JOB_ID} finished at $(date -u +%Y-%m-%dT%H:%M:%SZ)"
bash "${SLURM_DIR}/check_md_validation.sh"
