#!/usr/bin/env bash
# Install classical force field dependencies (openmmforcefields from submodule).
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "${ROOT}"

bash scripts/init_submodules.sh third_party/openmmforcefields

if [[ -f third_party/openmmforcefields/setup.py || -f third_party/openmmforcefields/pyproject.toml ]]; then
  pip install -e third_party/openmmforcefields
else
  pip install openmmforcefields
fi

echo "Classical FF dependencies installed (openmmforcefields)."
