#!/usr/bin/env bash
# Run AIMNet2 water cavity MD on physical GPU 1.
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../../.." && pwd)"
cd "${ROOT}"

ENV_BIN="${ROOT}/.pixi/envs/ff-ml-dipole/bin"

if [[ ! -x "${ENV_BIN}/python" ]]; then
  echo "ERROR: ff-ml-dipole env not found at ${ENV_BIN}" >&2
  exit 1
fi

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
  CUDA_HOME="${CUDA_HOME}" \
  CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES}" \
  LD_LIBRARY_PATH="${LD_LIBRARY_PATH}" \
  "${ENV_BIN}/python" examples/cavity/aimnet2_water/run_simulation.py "$@"
