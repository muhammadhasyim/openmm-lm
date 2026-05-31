#!/bin/bash
# DEPRECATED: Use Pixi instead.
#
#   pixi install
#   pixi run smoke
#   pixi install -e ml   # optional ML deps
#
# See docs/BUILD_AND_REINSTALL.md

echo "ERROR: scripts/install_openmm_fairchem_base.sh is deprecated." >&2
echo "Use: pixi install && pixi run smoke" >&2
echo "See: docs/BUILD_AND_REINSTALL.md" >&2
exit 1
