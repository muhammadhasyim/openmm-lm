#!/usr/bin/env bash
# Archive incomplete lambda=0.03 rerun outputs and resubmit only those replicas.
set -euo pipefail

REPO_ROOT=/scratch/mh7373/openmm
CAMPAIGN_DIR="${REPO_ROOT}/research/c2f/aging_weak_lambda"
PY="${REPO_ROOT}/.pixi/envs/test/bin/python"
SBATCH="${CAMPAIGN_DIR}/slurm/13_resubmit_lambda003_incomplete.sbatch"

export OPENMM_PLUGIN_DIR="${REPO_ROOT}/.pixi/envs/test/lib/plugins"

cd "${CAMPAIGN_DIR}"

echo "=== Identifying incomplete lambda=0.03 replicas ==="
INCOMPLETE=$("${PY}" - <<PY
from pathlib import Path
import sys

repo = Path("${REPO_ROOT}")
sys.path.insert(0, str(repo / "research/c2f/aging_weak_lambda"))
sys.path.insert(0, str(repo / "research/c2f"))
from config import RUNTIME_PS, N_REPLICAS, job_dir_path
from fkt_utils import replica_complete

job_dir = job_dir_path(0.03)
incomplete = [
    r for r in range(N_REPLICAS) if not replica_complete(job_dir, 0.03, r, RUNTIME_PS)
]
print(",".join(str(r) for r in incomplete))
print(f"# count={len(incomplete)}", file=sys.stderr)
PY
)

if [[ -z "${INCOMPLETE}" ]]; then
  echo "All replicas complete — nothing to resubmit."
  exit 0
fi

COUNT=$(echo "${INCOMPLETE}" | tr ',' '\n' | grep -c . || true)
echo "Incomplete replicas: ${COUNT}"

echo "=== Archiving poisoned / blown-up outputs ==="
"${PY}" archive_poisoned_lambda003.py --lambda 0.03

echo "=== Archiving partial outputs (no checkpoint) ==="
bash "${CAMPAIGN_DIR}/archive_partial_before_resubmit.sh"

echo "=== Submitting array job for incomplete replicas ==="
JOB_ID=$(sbatch --parsable --array="${INCOMPLETE}%30" "${SBATCH}")
echo "Submitted job ${JOB_ID} (array=${INCOMPLETE}%30)"
