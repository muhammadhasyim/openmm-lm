#!/usr/bin/env bash
# MBPol(2023) water cavity MD smoke test (CPU / Reference platform).
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../../.." && pwd)"
cd "${ROOT}/examples/cavity/mbpol_2023_water"

if [[ -z "${MBX_HOME:-}" ]]; then
  export MBX_HOME="${ROOT}/third_party/MBX"
fi
export LD_LIBRARY_PATH="${MBX_HOME}/lib:${ROOT}/third_party/fftw/install/lib:${LD_LIBRARY_PATH:-}"

PYTHON="${ROOT}/.pixi/envs/ff-ml-dipole/bin/python"
if [[ ! -x "${PYTHON}" ]]; then
  PYTHON="$(command -v python3)"
fi

export PYTHONPATH="${ROOT}/wrappers/python:${ROOT}/third_party/openmm-ml:${PYTHONPATH:-}"

"${PYTHON}" run_simulation.py \
  --num-molecules 100 \
  --steps 100 \
  --platform CPU \
  --no-use-cuda-bridge \
  --output-dir "${ROOT}/runs/mbpol_2023_water_smoke"
