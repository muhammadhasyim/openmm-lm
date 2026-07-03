#!/usr/bin/env bash
# Initialize git submodules under third_party/ required by optional Pixi environments.
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "${ROOT}"

DEFAULT_SUBMODULES=(
  third_party/openmm-ml
)

FF_CLASSICAL_SUBMODULES=(
  third_party/openmmforcefields
)

FF_ML_DIPOLE_SUBMODULES=(
  third_party/openmm-ml
  third_party/cace
  third_party/aimnet2
  third_party/mace
  third_party/LES-BEC
)

if [[ "$#" -gt 0 ]]; then
  SUBMODULES=("$@")
else
  SUBMODULES=("${DEFAULT_SUBMODULES[@]}")
fi

for path in "${SUBMODULES[@]}"; do
  if [[ ! -d "${path}" ]] || [[ ! -f "${path}/setup.py" && ! -f "${path}/pyproject.toml" && ! -f "${path}/README.md" ]]; then
    echo "Initializing submodule: ${path}"
    git submodule update --init "${path}"
  fi
  if [[ ! -f "${path}/setup.py" && ! -f "${path}/pyproject.toml" && ! -f "${path}/README.md" ]]; then
    echo "ERROR: ${path} is not a usable checkout after submodule init" >&2
    exit 1
  fi
done

echo "Submodules ready: ${SUBMODULES[*]}"
