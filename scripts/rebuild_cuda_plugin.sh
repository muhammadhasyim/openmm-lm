#!/usr/bin/env bash
# Rebuild libOpenMMCUDA against a system CUDA toolkit's libnvrtc (e.g. /usr/local/cuda).
# Pixi's isolated build links conda-forge NVRTC 13.x; PTX from it can exceed driver
# support (CUDA_ERROR_UNSUPPORTED_PTX_VERSION / 222) when nvidia-smi reports CUDA 13.0.
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
BUILD="${ROOT}/build-cuda-nvrtc-fix"
CONDA="${CONDA_PREFIX:?CONDA_PREFIX is not set; run via pixi run}"
CUDA_NVRTC_ROOT="${OPENMM_NVRTC_ROOT:-}"

_nvrtc_present() {
    local root="$1"
    [[ -f "${root}/lib64/libnvrtc.so" || -f "${root}/lib64/libnvrtc.so.12" || -f "${root}/lib64/libnvrtc.so.13" ]]
}

if [[ -z "${CUDA_NVRTC_ROOT}" ]]; then
    for candidate in /usr/local/cuda /usr/local/cuda-13 /usr/local/cuda-12.4 /usr/local/cuda-12.0 /usr/local/cuda-12; do
        if _nvrtc_present "${candidate}"; then
            CUDA_NVRTC_ROOT="${candidate}"
            break
        fi
    done
fi

if [[ -z "${CUDA_NVRTC_ROOT}" ]]; then
    echo "No system CUDA toolkit found. Set OPENMM_NVRTC_ROOT to a CUDA install with libnvrtc." >&2
    exit 1
fi

echo "Rebuilding OpenMMCUDA with NVRTC from ${CUDA_NVRTC_ROOT}"
echo "Install prefix: ${CONDA}"

rm -rf "${BUILD}"

cmake -S "${ROOT}" -B "${BUILD}" -G Ninja \
    -DCMAKE_BUILD_TYPE=Release \
    -DCMAKE_INSTALL_PREFIX="${CONDA}" \
    -DCUDAToolkit_ROOT="${CONDA}" \
    -DOPENMM_NVRTC_ROOT="${CUDA_NVRTC_ROOT}" \
    -DBUILD_TESTING=OFF \
    -DOPENMM_BUILD_PYTHON_WRAPPERS=OFF \
    -DOPENMM_BUILD_SHARED_LIB=ON \
    -DOPENMM_BUILD_CPU_LIB=ON \
    -DOPENMM_BUILD_CUDA_LIB=ON \
    -DOPENMM_BUILD_OPENCL_LIB=ON \
    -DOPENMM_BUILD_AMOEBA_PLUGIN=ON \
    -DOPENMM_BUILD_RPMD_PLUGIN=ON \
    -DOPENMM_BUILD_DRUDE_PLUGIN=ON \
    -DOPENMM_BUILD_PME_PLUGIN=ON \
    -DOPENMM_BUILD_EXAMPLES=OFF \
    -DOPENMM_BUILD_REFERENCE_TESTS=OFF \
    -DOPENMM_BUILD_C_AND_FORTRAN_WRAPPERS=OFF

cmake --build "${BUILD}" --target OpenMMCUDA -j "$(nproc)"
install -m 0644 "${BUILD}/libOpenMMCUDA.so" "${CONDA}/lib/plugins/libOpenMMCUDA.so"
echo "Installed ${CONDA}/lib/plugins/libOpenMMCUDA.so"
