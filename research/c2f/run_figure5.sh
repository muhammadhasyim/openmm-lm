#!/usr/bin/env bash
# Reproduce Figure 5b (C2F cooling) end-to-end via pixi test env.
#
# Usage:
#   pixi run -e test figure5
#   pixi run -e test figure5 -- --quick
#   bash research/c2f/run_figure5.sh --quick

set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../../.." && pwd)"
cd "$ROOT"

SCRIPT_DIR="research/c2f"
OUTPUT_DIR="${SCRIPT_DIR}/fig5_output"

echo "=== C2F Figure 5 replication (pixi test env) ==="
echo "Working directory: $ROOT"
echo ""

pixi run -e test python "${SCRIPT_DIR}/reproduce_figure5.py" \
    --output-dir "${OUTPUT_DIR}" \
    "$@"

echo ""
echo "=== Plotting Figure 5b ==="
pixi run -e test python "${SCRIPT_DIR}/plot_figure5.py" \
    --input "${OUTPUT_DIR}/fig5_averaged.csv" \
    --meta "${OUTPUT_DIR}/fig5_meta.txt" \
    --output "${OUTPUT_DIR}/Figure5b_reproduced"

echo ""
echo "Done. Outputs in ${OUTPUT_DIR}/"
