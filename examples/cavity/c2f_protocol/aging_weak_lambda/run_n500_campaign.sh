#!/usr/bin/env bash
# Paper-scale campaign: 500 replicas × 5 λ = 2500 trajectories.
# Schedule: all couplings in parallel, then advance replica index (see --schedule replica_round).
set -euo pipefail
cd "$(dirname "$0")"
PY=python3
if command -v pixi >/dev/null 2>&1; then
  PY="pixi run --as-is -e test python"
fi

JOBS="${JOBS:-5}"
EXTRA=()
if [[ "${FRESH_START:-0}" == "1" ]]; then
  EXTRA+=(--no-skip)
fi

echo "Starting N=500 aging campaign (jobs=${JOBS}, schedule=replica_round)"
exec $PY run_campaign.py \
  --schedule replica_round \
  --jobs "$JOBS" \
  --replica-start 0 \
  --replica-end 499 \
  "${EXTRA[@]}" \
  "$@"
