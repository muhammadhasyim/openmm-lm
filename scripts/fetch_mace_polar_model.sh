#!/usr/bin/env bash
# Download MACE-POLAR-1-M checkpoint for cavity MD (OpenMM-ML path).
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
MODEL_DIR="${ROOT}/third_party/models"
MODEL_PATH="${MODEL_DIR}/MACE-POLAR-1-M.model"
URL="https://github.com/ACEsuit/mace-foundations/releases/download/mace_polar_1/MACE-POLAR-1-M.model"

mkdir -p "${MODEL_DIR}"

if [[ -f "${MODEL_PATH}" ]]; then
  echo "Model already present: ${MODEL_PATH}"
else
  echo "Downloading MACE-POLAR-1-M.model ..."
  curl -fsSL -o "${MODEL_PATH}" "${URL}"
  echo "Saved to ${MODEL_PATH}"
fi

echo ""
echo "Set before running cavity MD:"
echo "  export OPENMMML_MACE_POLAR_MODEL=${MODEL_PATH}"
