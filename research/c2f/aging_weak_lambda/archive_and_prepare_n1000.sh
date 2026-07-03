#!/usr/bin/env bash
# Archive all existing campaign outputs before N=1000 adaptive rerun.
set -euo pipefail

REPO_ROOT=/scratch/mh7373/openmm
CAMPAIGN_DIR="${REPO_ROOT}/research/c2f/aging_weak_lambda"
PY="${REPO_ROOT}/.pixi/envs/test/bin/python"

cd "${CAMPAIGN_DIR}"
echo "=== Archiving all existing outputs (full rerun) ==="
"${PY}" archive_campaign_outputs.py --full-rerun "$@"
