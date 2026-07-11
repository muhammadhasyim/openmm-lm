#!/usr/bin/env bash
# Long NVT equilibration: 1 µs at lambda=0 (no cavity coupling), Bussi + cavity bath.
set -euo pipefail
ROOT="$(cd "$(dirname "$0")/../../.." && pwd)"
export LD_LIBRARY_PATH="${ROOT}/build:${ROOT}/build/plugins/amoeba:${ROOT}/build/plugins/cpupme:${ROOT}/build/plugins/drude:${ROOT}/build/plugins/rpmd${LD_LIBRARY_PATH:+:$LD_LIBRARY_PATH}"
export OPENMM_PLUGIN_DIR="${ROOT}/build"
export PYTHONPATH="${ROOT}/build/python/build/lib.linux-x86_64-cpython-311:${ROOT}/wrappers/python${PYTHONPATH:+:$PYTHONPATH}"
cd "$(dirname "$0")"
exec python3 -u run_energy_recovery_test.py \
  --lambda 0.0 \
  --equil-time 1000000 \
  --switch-time 0 \
  --post-switch-time 0 \
  --cavity-friction 1.0 \
  --sample-interval 1000 \
  --output energy_equil_1us_lambda0.npz \
  "$@"
