#!/usr/bin/env bash
# Regenerate Figure 2 panels (weak coupling) from archived OpenMM aging data.
set -euo pipefail

ROOT="/scratch/mh7373/openmm"
CAMPAIGN="${ROOT}/research/c2f/aging_weak_lambda"
LOG="${CAMPAIGN}/figure2_pipeline.log"
PIDFILE="${CAMPAIGN}/figure2_pipeline.pid"

cd "${ROOT}"

echo "=== Figure 2 pipeline started $(date -Is) ===" >> "${LOG}"

pixi run --as-is -e test python "${CAMPAIGN}/build_master_fkt.py" \
  >> "${LOG}" 2>&1

pixi run --as-is -e test python "${CAMPAIGN}/analyze_aging_relaxation.py" \
  >> "${LOG}" 2>&1

pixi run --as-is -e test python "${CAMPAIGN}/plot_isf_curves.py" \
  >> "${LOG}" 2>&1

pixi run --as-is -e test python "${CAMPAIGN}/analyze_ir_from_dipole.py" \
  >> "${LOG}" 2>&1

echo "=== Figure 2 pipeline finished $(date -Is) ===" >> "${LOG}"
