#!/usr/bin/env bash
# Install merged OpenMM calibration as the default fictive-temperature reference.
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../../.." && pwd)"
C2F="$ROOT/research/c2f"
SLIM="$C2F/calibration_output/calibration_data.txt"
REF="$C2F/reference_potential_energy_vs_T.txt"

if [[ ! -f "$SLIM" ]]; then
  echo "ERROR: merged calibration not found: $SLIM" >&2
  echo "Run merge_fictive_calibration.py after the shard sweep completes." >&2
  exit 1
fi

cp "$SLIM" "$REF"
echo "Installed $SLIM → $REF"
echo "run_c2f.REFERENCE_CALIBRATION_FILE will prefer calibration_output/calibration_data.txt when present."
