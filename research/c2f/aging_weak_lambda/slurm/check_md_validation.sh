#!/usr/bin/env bash
# Post-run gate: validate md_validation/ outputs and update conclusion JSON.
set -euo pipefail

REPO_ROOT=/scratch/mh7373/openmm
CAMPAIGN_DIR="${REPO_ROOT}/research/c2f/aging_weak_lambda"
PY="${REPO_ROOT}/.pixi/envs/test/bin/python"
export OPENMM_PLUGIN_DIR="${REPO_ROOT}/.pixi/envs/test/lib/plugins"
VALIDATION_LOG="${CAMPAIGN_DIR}/md_validation/validation_log.jsonl"
REPO_ADAPT="${REPO_ROOT}/wrappers/python/openmm/cavitymd/adaptive.py"

cd "${CAMPAIGN_DIR}"

echo "=== Check validation_log.jsonl return codes ==="
if [[ ! -f "${VALIDATION_LOG}" ]]; then
  echo "ERROR: missing ${VALIDATION_LOG}" >&2
  exit 1
fi
"${PY}" - <<PY
import hashlib
import json
import sys
from pathlib import Path

log_path = Path("${VALIDATION_LOG}")
repo_adapt = Path("${REPO_ADAPT}")
expected_sha = hashlib.sha256(repo_adapt.read_bytes()).hexdigest()
failed = 0
for line in log_path.read_text(encoding="utf-8").splitlines():
    if not line.strip():
        continue
    rec = json.loads(line)
    rc = rec.get("returncode")
    lam = rec.get("lambda")
    rep = rec.get("replica")
    label = f"lam={lam} rep={rep}"
    if rc != 0:
        failed += 1
        print(f"FAIL {label}: returncode={rc} (SIGBUS if -7)")
    else:
        print(f"OK   {label}: returncode=0")
    prefix = rec.get("prefix") or rec.get("output_prefix")
    if prefix:
        meta = Path(f"{prefix}_meta.txt")
        if meta.exists():
            text = meta.read_text(encoding="utf-8")
            for line_meta in text.splitlines():
                if line_meta.startswith("adaptive_module_sha256="):
                    got = line_meta.split("=", 1)[1].strip()
                    if got != expected_sha:
                        failed += 1
                        print(f"FAIL {label}: adaptive_module_sha256 mismatch")
                    break
if failed:
    print(f"\n{failed} log/meta checks failed", file=sys.stderr)
    sys.exit(1)
print("\nAll validation_log return codes and sha256 checks passed")
PY

echo "=== Primary gate: replica 0, all lambda ==="
"${PY}" validate_replica_stability.py \
  --campaign-dir md_validation \
  --replica-start 0 \
  --replica-end 0

echo "=== Spot-check: replica 1, lambda 0.01 + 0.03 ==="
"${PY}" validate_replica_stability.py \
  --campaign-dir md_validation \
  --lambdas 0.01 0.03 \
  --replica-start 1 \
  --replica-end 1

"${PY}" - <<'PY'
import json
from datetime import datetime, timezone
from pathlib import Path

campaign = Path("/scratch/mh7373/openmm/research/c2f/aging_weak_lambda")
conclusion_path = campaign / "diagnose_fkt" / "parity_diagnosis_conclusion.json"
marker_path = campaign / "md_validation" / "validation_passed.json"
passed_at = datetime.now(timezone.utc).isoformat().replace("+00:00", "Z")

data = json.loads(conclusion_path.read_text(encoding="utf-8"))
md_val = data.setdefault("md_validation", {})
md_val["md_validation_passed"] = True
md_val["passed_at"] = passed_at
md_val["status"] = "passed"
data["md_validation_passed"] = True
data["md_validation_passed_at"] = passed_at
data["production_gate"] = (
    "MD validation passed; safe to run: bash slurm/submit_n1000_adaptive.sh full"
)
conclusion_path.write_text(json.dumps(data, indent=2) + "\n", encoding="utf-8")

marker = {
    "passed": True,
    "passed_at": passed_at,
    "note": "All MD validation stability checks passed",
}
marker_path.parent.mkdir(parents=True, exist_ok=True)
marker_path.write_text(json.dumps(marker, indent=2) + "\n", encoding="utf-8")
print(f"Updated {conclusion_path}")
print(f"Wrote {marker_path}")
PY

echo "=== MD validation gate PASSED ==="
echo "Next: bash slurm/submit_n1000_adaptive.sh full"
