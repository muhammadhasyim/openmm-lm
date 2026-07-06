#!/usr/bin/env bash
# Long-running fictive temperature calibration (pixi test env).
#
# Usage:
#   bash research/c2f/run_fictive_calibration.sh
#   bash research/c2f/run_fictive_calibration.sh --quick
#   bash research/c2f/run_fictive_calibration.sh --resume

set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../../.." && pwd)"
cd "$ROOT"

SCRIPT_DIR="research/c2f"
OUTPUT_DIR="${SCRIPT_DIR}/calibration_output"
mkdir -p "${OUTPUT_DIR}"

echo "=== Fictive temperature energy calibration ==="
echo "Working directory: $ROOT"
echo "Output directory: ${OUTPUT_DIR}"
echo ""

# Line-buffered stdout so nohup logs update during long MD runs.
export PYTHONUNBUFFERED=1

pixi run -e test python "${SCRIPT_DIR}/run_fictive_calibration.py" \
    --output "${OUTPUT_DIR}/potential_energy_components_vs_temperature.txt" \
    --slim-output "${OUTPUT_DIR}/calibration_data.txt" \
    --timeseries-dir "${OUTPUT_DIR}/timeseries" \
    "$@"

echo ""
echo "Calibration files written under ${OUTPUT_DIR}/"
