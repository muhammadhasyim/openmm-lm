#!/usr/bin/env bash
# Archive all lambda0p03 replica outputs before intentional full rerun.
set -euo pipefail

REPO_ROOT=/scratch/mh7373/openmm
CAMPAIGN_DIR="${REPO_ROOT}/research/c2f/aging_weak_lambda"
PY="${REPO_ROOT}/.pixi/envs/test/bin/python"
LAM003=0.03
MANIFEST="${CAMPAIGN_DIR}/results/lambda003_archive_manifest.json"

mkdir -p "${CAMPAIGN_DIR}/results"

cd "${CAMPAIGN_DIR}"
"${PY}" - <<PY
import json
import sys
from datetime import datetime, timezone
from pathlib import Path

c2f = Path("${REPO_ROOT}/research/c2f")
sys.path.insert(0, str(c2f))
sys.path.insert(0, str(Path("${CAMPAIGN_DIR}")))

from checkpoint_utils import archive_replica_outputs
from config import N_REPLICAS, RUNTIME_PS, job_dir_path, run_prefix

lam = ${LAM003}
job_dir = job_dir_path(lam)
manifest_path = Path("${MANIFEST}")
entries: dict[str, str | None] = {}
archived = 0
skipped = 0

for rep in range(N_REPLICAS):
    prefix = job_dir / run_prefix(lam, rep)
    out = archive_replica_outputs(
        prefix,
        reason="lambda003_rerun",
        runtime_ps=RUNTIME_PS,
        lambda_coupling=lam,
        replica=rep,
    )
    if out is None:
        skipped += 1
        entries[str(rep)] = None
    else:
        archived += 1
        entries[str(rep)] = str(out)
        print(f"archived rep={rep} -> {out.name}")

payload = {
    "timestamp_utc": datetime.now(timezone.utc).isoformat(),
    "lambda": lam,
    "job_dir": job_dir.name,
    "reason": "lambda003_rerun",
    "replicas": entries,
    "archived_count": archived,
    "skipped_count": skipped,
}
manifest_path.write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")
print(f"Done: archived={archived} skipped={skipped} manifest={manifest_path}")
PY
