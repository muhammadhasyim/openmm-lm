#!/usr/bin/env bash
# Run equilibrated non-thermal aging test (4 ns equil + switch + 2.5 ns post-switch).
set -euo pipefail
ROOT="$(cd "$(dirname "$0")/../../.." && pwd)"
export LD_LIBRARY_PATH="${ROOT}/build:${ROOT}/build/plugins/amoeba:${ROOT}/build/plugins/cpupme:${ROOT}/build/plugins/drude:${ROOT}/build/plugins/rpmd${LD_LIBRARY_PATH:+:$LD_LIBRARY_PATH}"
export OPENMM_PLUGIN_DIR="${ROOT}/build"
export PYTHONPATH="${ROOT}/build/python/build/lib.linux-x86_64-cpython-311:${ROOT}/wrappers/python${PYTHONPATH:+:$PYTHONPATH}"
cd "$(dirname "$0")"
exec python3 -u run_energy_recovery_test.py \
  --equil-time 4000 \
  --switch-time 200 \
  --post-switch-time 2500 \
  --cavity-friction 1.0 \
  --sample-interval 10 \
  "$@"
