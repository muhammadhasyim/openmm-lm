#!/usr/bin/env bash
# Paper-scale campaign: 1000 replicas × 5 λ = 5000 trajectories.
# Schedule: all couplings in parallel, then advance replica index (see --schedule replica_round).
set -euo pipefail
cd "$(dirname "$0")"
PY=python3
if command -v pixi >/dev/null 2>&1; then
  PY="pixi run --as-is -e test python"
fi

JOBS="${JOBS:-5}"
EXTRA=(--adaptive --no-resume)
if [[ "${FRESH_START:-0}" == "1" ]]; then
  EXTRA+=(--no-skip)
fi

echo "Starting N=1000 adaptive aging campaign (jobs=${JOBS}, schedule=replica_round)"
exec $PY run_campaign.py \
  --schedule replica_round \
  --jobs "$JOBS" \
  --replica-start 0 \
  --replica-end 999 \
  --log campaign_n1000_log.jsonl \
  "${EXTRA[@]}" \
  "$@"
