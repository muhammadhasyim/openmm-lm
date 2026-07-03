#!/usr/bin/env bash
# Install ML dipole backend libraries for CACE, AIMNet2, and MACE-POLAR.
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "${ROOT}"

bash scripts/init_submodules.sh third_party/openmm-ml

# Install CUDA PyTorch if the Pixi-provided build is CPU-only.
if ! python -c "import torch; assert torch.cuda.is_available()" 2>/dev/null; then
  echo "Installing CUDA-enabled PyTorch (cu124) ..."
  pip uninstall -y torch 2>/dev/null || true
  pip install torch --index-url https://download.pytorch.org/whl/cu124
fi

pip install -e third_party/openmm-ml
# MACE-POLAR-1 requires mace@main + graph_electrostatics (not PyPI mace-torch alone).
pip install "mace-torch @ git+https://github.com/ACEsuit/mace.git@main"
pip install "git+https://github.com/WillBaldwin0/graph_electrostatics.git"
# NNPOps is not on PyPI; install via conda-forge into CONDA_PREFIX if pip fails.
if ! pip install nnpops 2>/dev/null; then
  echo "NOTE: pip nnpops unavailable; install with: CONDA_PREFIX=\$CONDA_PREFIX conda install -c conda-forge nnpops -y"
fi
# Optional backends (skip if submodules are not initialized).
bash scripts/clone_ml_submodules.sh 2>/dev/null || true
pip install --no-build-isolation --no-deps -e third_party/cace 2>/dev/null || true
pip install --no-build-isolation --no-deps -e third_party/aimnet2 2>/dev/null || true
pip install matscipy 2>/dev/null || true

echo ""
echo "LES-BEC checkpoint (CACE water): run bash scripts/fetch_les_bec_model.sh"

# CUDA tensor bridge (links against OPENMM_LIB_DIR / OPENMM_CUDA_LIB_DIR).
OPENMM_LIB_DIR="${OPENMM_LIB_DIR:-${CONDA_PREFIX:-}/lib}"
OPENMM_CUDA_LIB_DIR="${OPENMM_CUDA_LIB_DIR:-${OPENMM_LIB_DIR}/plugins}"
export OPENMM_DIR="${OPENMM_DIR:-${ROOT}}"
export OPENMM_LIB_DIR OPENMM_CUDA_LIB_DIR CUDA_HOME="${CUDA_HOME:-/usr/local/cuda}"
pip install --no-build-isolation -e python/openmm_cuda_bridge

if [[ -x "${ROOT}/scripts/build_mbx.sh" ]]; then
  echo ""
  echo "MBPol/MBX (optional): run bash scripts/build_mbx.sh"
  echo "  export MBX_HOME=\${ROOT}/third_party/MBX"
  echo "  export LD_LIBRARY_PATH=\${MBX_HOME}/lib:\${LD_LIBRARY_PATH:-}"
fi

echo "ML dipole FF dependencies installed (openmmml + polar MACE + cuda bridge)."
