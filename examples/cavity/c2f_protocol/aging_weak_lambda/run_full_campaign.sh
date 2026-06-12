#!/usr/bin/env/bash
set -euo pipefail
cd "$(dirname "$0")"
PY=python3
if command -v pixi >/dev/null 2>&1; then
  PY="pixi run --as-is -e test python"
fi

$PY run_campaign.py --jobs 2 "$@"
echo "Running analysis..."
$PY run_all_analysis.py
