#!/usr/bin/env bash
# Run CACE/LES-BEC water cavity MD on physical GPU 1.
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../../.." && pwd)"
cd "${ROOT}"

ENV_BIN="${ROOT}/.pixi/envs/ff-ml-dipole/bin"
CHECKPOINT="${OPENMMML_LES_BEC_WATER_MODEL:-${ROOT}/third_party/LES-BEC/water/fit/fit_version_4/best_model.pth}"

if [[ ! -x "${ENV_BIN}/python" ]]; then
  echo "ERROR: ff-ml-dipole env not found at ${ENV_BIN}" >&2
  exit 1
fi

if [[ ! -f "${CHECKPOINT}" ]]; then
  echo "Checkpoint missing; fetching LES-BEC ..."
  bash scripts/fetch_les_bec_model.sh
  CHECKPOINT="${ROOT}/third_party/LES-BEC/water/fit/fit_version_4/best_model.pth"
fi

export OPENMMML_LES_BEC_WATER_MODEL="${CHECKPOINT}"
export OPENMM_PLUGIN_DIR="${ENV_BIN}/../lib/plugins"
export OPENMM_LIB_DIR="${ENV_BIN}/../lib"
export OPENMM_CUDA_LIB_DIR="${OPENMM_PLUGIN_DIR}"
export CUDA_HOME="${CUDA_HOME:-/usr/local/cuda}"
export CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-1}"
export LD_LIBRARY_PATH="${ENV_BIN}/../lib:${ENV_BIN}/../lib/plugins:${CUDA_HOME}/lib64${LD_LIBRARY_PATH:+:${LD_LIBRARY_PATH}}"
export OPENMM_DIR="${ROOT}"

exec env -i \
  HOME="${HOME}" \
  PATH="${ENV_BIN}:/usr/bin:/bin" \
  PYTHONUNBUFFERED=1 \
  OPENMM_DIR="${OPENMM_DIR}" \
  OPENMM_PLUGIN_DIR="${OPENMM_PLUGIN_DIR}" \
  OPENMM_LIB_DIR="${OPENMM_LIB_DIR}" \
  OPENMM_CUDA_LIB_DIR="${OPENMM_CUDA_LIB_DIR}" \
  OPENMMML_LES_BEC_WATER_MODEL="${OPENMMML_LES_BEC_WATER_MODEL}" \
  CUDA_HOME="${CUDA_HOME}" \
  CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES}" \
  LD_LIBRARY_PATH="${LD_LIBRARY_PATH}" \
  "${ENV_BIN}/python" examples/cavity/cace_les_bec_water/run_simulation.py "$@"
