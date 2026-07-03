#!/usr/bin/env bash
# Run MACE-POLAR water cavity MD on physical GPU 1 (ff-ml-dipole env, no pixi rebuild).
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../../.." && pwd)"
cd "${ROOT}"

ENV_BIN="${ROOT}/.pixi/envs/ff-ml-dipole/bin"
MODEL="${OPENMMML_MACE_POLAR_MODEL:-${ROOT}/third_party/models/MACE-POLAR-1-M.model}"

if [[ ! -x "${ENV_BIN}/python" ]]; then
  echo "ERROR: ff-ml-dipole env not found at ${ENV_BIN}" >&2
  exit 1
fi

if [[ ! -f "${MODEL}" ]]; then
  echo "Model missing; fetching ..."
  bash scripts/fetch_mace_polar_model.sh
fi

export OPENMMML_MACE_POLAR_MODEL="${MODEL}"
export OPENMM_PLUGIN_DIR="${ENV_BIN}/../lib/plugins"
export OPENMM_LIB_DIR="${ENV_BIN}/../lib"
export OPENMM_CUDA_LIB_DIR="${OPENMM_PLUGIN_DIR}"
export CUDA_HOME="${CUDA_HOME:-/usr/local/cuda}"
export CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-1}"
export LD_LIBRARY_PATH="${ENV_BIN}/../lib:${ENV_BIN}/../lib/plugins:${CUDA_HOME}/lib64${LD_LIBRARY_PATH:+:${LD_LIBRARY_PATH}}"

exec env -i \
  HOME="${HOME}" \
  PATH="${ENV_BIN}:/usr/bin:/bin" \
  PYTHONUNBUFFERED=1 \
  OPENMM_PLUGIN_DIR="${OPENMM_PLUGIN_DIR}" \
  OPENMM_LIB_DIR="${OPENMM_LIB_DIR}" \
  OPENMM_CUDA_LIB_DIR="${OPENMM_CUDA_LIB_DIR}" \
  OPENMMML_MACE_POLAR_MODEL="${OPENMMML_MACE_POLAR_MODEL}" \
  CUDA_HOME="${CUDA_HOME}" \
  CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES}" \
  LD_LIBRARY_PATH="${LD_LIBRARY_PATH}" \
  "${ENV_BIN}/python" examples/cavity/mace_polar_water/run_simulation.py "$@"
