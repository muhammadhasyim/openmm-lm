#!/usr/bin/env bash
# Archive incomplete trajectories (no checkpoint) before resubmitting production.
# Preserves partial CSV/FKT from the 2 h TIMEOUT batch under *_archive_no_checkpoint_*.
set -euo pipefail

REPO_ROOT=/scratch/mh7373/openmm
CAMPAIGN_DIR="${REPO_ROOT}/research/c2f/aging_weak_lambda"
PY="${REPO_ROOT}/.pixi/envs/test/bin/python"

cd "${CAMPAIGN_DIR}"
"${PY}" - <<PY
from pathlib import Path
import sys

c2f = Path("${REPO_ROOT}/research/c2f")
sys.path.insert(0, str(c2f))
from checkpoint_utils import archive_stale_partial_outputs, checkpoint_path
from config import LAMBDAS, N_REPLICAS, RUNTIME_PS, job_dir_path, run_prefix

archived = 0
for lam in LAMBDAS:
    job_dir = job_dir_path(lam)
    for rep in range(N_REPLICAS):
        prefix = job_dir / run_prefix(lam, rep)
        if checkpoint_path(prefix).exists():
            continue
        out = archive_stale_partial_outputs(
            prefix, runtime_ps=RUNTIME_PS, reason="pre_resubmit_timeout"
        )
        if out is not None:
            archived += 1
            print(f"archived {out}")
print(f"Done: {archived} trajectories archived")
PY
