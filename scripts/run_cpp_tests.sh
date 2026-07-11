#!/usr/bin/env bash
# Build OpenMM with C++ tests enabled and run CTest (Reference/CPU by default).
#
# The pixi package build uses -DBUILD_TESTING=OFF, so installed prefixes do not
# contain Test* binaries. This script configures a separate build directory.
#
# Env overrides:
#   OPENMM_CPP_TEST_DIR   build directory (default: <repo>/build-cpp-tests)
#   OPENMM_CPP_TEST_REGEX ctest -R pattern (default: Cavity)
#   CMAKE_BUILD_PARALLEL_LEVEL  parallel build jobs
set -euo pipefail

ROOT="$(cd "$(dirname "$0")/.." && pwd)"
BUILD_DIR="${OPENMM_CPP_TEST_DIR:-$ROOT/build-cpp-tests}"
TEST_REGEX="${OPENMM_CPP_TEST_REGEX:-Cavity}"
JOBS="${CMAKE_BUILD_PARALLEL_LEVEL:-$(nproc 2>/dev/null || sysctl -n hw.ncpu 2>/dev/null || echo 2)}"
PREFIX="${CONDA_PREFIX:-${PREFIX:-}}"

mkdir -p "$BUILD_DIR"
cd "$BUILD_DIR"

EXTRA=()
# Default to CPU/Reference for CI speed and reliability. Opt in to CUDA with
# OPENMM_CPP_TEST_CUDA=1 when a working CUDAToolkit is available.
CUDA_LIB=OFF
if [[ "${OPENMM_CPP_TEST_CUDA:-0}" == "1" ]]; then
  if [[ -n "$PREFIX" ]]; then
    EXTRA+=("-DCMAKE_PREFIX_PATH=$PREFIX" "-DCUDAToolkit_ROOT=$PREFIX")
  fi
  CUDA_LIB=ON
elif [[ -n "$PREFIX" ]]; then
  EXTRA+=("-DCMAKE_PREFIX_PATH=$PREFIX")
fi

# Reconfigure only when needed (missing cache or generator change).
if [[ ! -f CMakeCache.txt ]]; then
  cmake "$ROOT" \
    -DCMAKE_BUILD_TYPE=Release \
    -DBUILD_TESTING=ON \
    -DOPENMM_BUILD_PYTHON_WRAPPERS=OFF \
    -DOPENMM_BUILD_SHARED_LIB=ON \
    -DOPENMM_BUILD_CPU_LIB=ON \
    -DOPENMM_BUILD_CUDA_LIB="$CUDA_LIB" \
    -DOPENMM_BUILD_OPENCL_LIB=OFF \
    -DOPENMM_BUILD_AMOEBA_PLUGIN=ON \
    -DOPENMM_BUILD_RPMD_PLUGIN=ON \
    -DOPENMM_BUILD_DRUDE_PLUGIN=ON \
    -DOPENMM_BUILD_PME_PLUGIN=ON \
    -DOPENMM_BUILD_EXAMPLES=OFF \
    -DOPENMM_BUILD_REFERENCE_TESTS=ON \
    -DOPENMM_BUILD_C_AND_FORTRAN_WRAPPERS=OFF \
    "${EXTRA[@]}"
fi

cmake --build . --parallel "$JOBS"

# Prefer the repo runner; fall back to ctest if Python helper is unavailable.
if [[ -f "$ROOT/devtools/run-ctest.py" ]]; then
  python "$ROOT/devtools/run-ctest.py" \
    --parallel 2 \
    --timeout 600 \
    --job-duration 900 \
    --attempts 2 \
    -R "$TEST_REGEX"
else
  ctest --output-on-failure -j 2 -R "$TEST_REGEX"
fi
