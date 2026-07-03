#!/usr/bin/env bash
# Finish Figure 2 pipeline (IR only) after master FKT + relaxation/F(k,t) steps complete.
set -euo pipefail

ROOT="/scratch/mh7373/openmm"
CAMPAIGN="${ROOT}/research/c2f/aging_weak_lambda"
LOG="${CAMPAIGN}/figure2_pipeline.log"

cd "${ROOT}"

echo "=== Figure 2 finish started $(date -Is) ===" >> "${LOG}"

pixi run --as-is -e test python "${CAMPAIGN}/analyze_ir_from_dipole.py" \
  >> "${LOG}" 2>&1

echo "=== Figure 2 pipeline finished $(date -Is) ===" >> "${LOG}"
