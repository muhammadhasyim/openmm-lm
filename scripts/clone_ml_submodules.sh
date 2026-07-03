#!/usr/bin/env bash
# Clone ML dipole third_party deps when git submodule init is unavailable.
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "${ROOT}/third_party"

clone_if_missing() {
  local dir="$1"
  local url="$2"
  if [[ ! -d "${dir}/.git" ]]; then
    echo "Cloning ${dir} from ${url} ..."
    git clone --depth 1 "${url}" "${dir}"
  else
    echo "${dir} already present"
  fi
}

clone_if_missing cace "https://github.com/BingqingCheng/cace.git"
clone_if_missing aimnet2 "https://github.com/isayevlab/AIMNet2.git"
clone_if_missing LES-BEC "https://github.com/BingqingCheng/LES-BEC.git"

bash "${ROOT}/scripts/fetch_les_bec_model.sh"
