#!/usr/bin/env bash
# Install cluster /usr/local/cuda NVRTC into the active Pixi/conda prefix.
# Pixi links OpenMMCUDA against conda-forge NVRTC 13.3.x; NYU A100 nodes ship
# NVRTC 13.0.x. Runtime JIT then fails with CUDA_ERROR_UNSUPPORTED_PTX_VERSION.
set -euo pipefail

CONDA="${CONDA_PREFIX:?CONDA_PREFIX is not set}"
CUDA_ROOT="${OPENMM_NVRTC_ROOT:-/usr/local/cuda}"
CLUSTER_LIB="${CUDA_ROOT}/lib64"
TARGET_LIB="${CONDA}/targets/x86_64-linux/lib"
STAMP="${CONDA}/lib/.cluster_nvrtc_13.0_installed"

_nvrtc_present() {
    local root="$1"
    [[ -f "${root}/lib64/libnvrtc.so" || -f "${root}/lib64/libnvrtc.so.13" || -f "${root}/lib64/libnvrtc.so.13.0.88" ]]
}

if ! _nvrtc_present "${CUDA_ROOT}"; then
    echo "No cluster NVRTC under ${CUDA_ROOT}; set OPENMM_NVRTC_ROOT." >&2
    exit 1
fi

NVRTC_SRC="${CLUSTER_LIB}/libnvrtc.so.13.0.88"
BUILTINS_SRC="${CLUSTER_LIB}/libnvrtc-builtins.so.13.0.88"
if [[ ! -f "${NVRTC_SRC}" ]]; then
    NVRTC_SRC="${CLUSTER_LIB}/libnvrtc.so.13"
fi
if [[ ! -f "${BUILTINS_SRC}" ]]; then
    BUILTINS_SRC="${CLUSTER_LIB}/libnvrtc-builtins.so.13.0.88"
fi
if [[ ! -f "${NVRTC_SRC}" || ! -f "${BUILTINS_SRC}" ]]; then
    echo "Missing cluster NVRTC libraries under ${CLUSTER_LIB}" >&2
    exit 1
fi

mkdir -p "${TARGET_LIB}" "${CONDA}/lib"

if [[ -f "${STAMP}" ]]; then
    echo "Cluster NVRTC already installed ($(cat "${STAMP}"))"
    exit 0
fi

BACKUP="${CONDA}/lib/.nvrtc_backup_pixi_13.3"
mkdir -p "${BACKUP}"
for name in libnvrtc.so.13.3.33 libnvrtc-builtins.so.13.3.33; do
    if [[ -f "${TARGET_LIB}/${name}" && ! -f "${BACKUP}/${name}" ]]; then
        cp -a "${TARGET_LIB}/${name}" "${BACKUP}/${name}"
    fi
done

cp -f "${NVRTC_SRC}" "${TARGET_LIB}/libnvrtc.so.13.0.88"
cp -f "${BUILTINS_SRC}" "${TARGET_LIB}/libnvrtc-builtins.so.13.0.88"
ln -sfn libnvrtc.so.13.0.88 "${TARGET_LIB}/libnvrtc.so.13.3.33"
ln -sfn libnvrtc-builtins.so.13.0.88 "${TARGET_LIB}/libnvrtc-builtins.so.13.3.33"
ln -sfn ../targets/x86_64-linux/lib/libnvrtc.so.13.3.33 "${CONDA}/lib/libnvrtc.so.13"
ln -sfn ../targets/x86_64-linux/lib/libnvrtc.so.13.3.33 "${CONDA}/lib/libnvrtc.so.13.3.33"
ln -sfn ../targets/x86_64-linux/lib/libnvrtc-builtins.so.13.3.33 "${CONDA}/lib/libnvrtc-builtins.so.13.3"
ln -sfn ../targets/x86_64-linux/lib/libnvrtc-builtins.so.13.3.33 "${CONDA}/lib/libnvrtc-builtins.so.13.3.33"

date -Iseconds > "${STAMP}"
echo "Installed cluster NVRTC from ${CUDA_ROOT} into ${CONDA}"
echo "Backup of pixi NVRTC: ${BACKUP}"
