#!/usr/bin/env bash
# Run full campaign then analysis (resume-safe via run_campaign.py skip-complete).
set -euo pipefail
cd "$(dirname "$0")"
PY=python3
if command -v pixi >/dev/null 2>&1; then
  PY="pixi run --as-is -e test python"
fi

JOBS="${JOBS:-4}"
$PY run_campaign.py --jobs "$JOBS" "$@"
echo "Running analysis..."
$PY run_all_analysis.py
