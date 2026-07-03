#!/usr/bin/env bash
# Clone LES-BEC and point OPENMMML_LES_BEC_WATER_MODEL at the water BEC checkpoint.
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
LES_DIR="${ROOT}/third_party/LES-BEC"
CHECKPOINT="${LES_DIR}/water/fit/fit_version_4/best_model.pth"
URL="https://github.com/BingqingCheng/LES-BEC.git"

if [[ ! -d "${LES_DIR}/.git" ]]; then
  echo "Cloning LES-BEC into ${LES_DIR} ..."
  git clone --depth 1 "${URL}" "${LES_DIR}"
fi

if [[ ! -f "${CHECKPOINT}" ]]; then
  echo "ERROR: LES-BEC checkpoint not found at ${CHECKPOINT}" >&2
  exit 1
fi

echo "LES-BEC checkpoint: ${CHECKPOINT}"
echo ""
echo "Set before running CACE cavity MD:"
echo "  export OPENMMML_LES_BEC_WATER_MODEL=${CHECKPOINT}"
