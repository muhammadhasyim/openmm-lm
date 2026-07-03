#!/usr/bin/env bash
# Parallel fictive-temperature calibration on ga043 (2× A100, 8 workers).
set -euo pipefail

export PYTHONUNBUFFERED=1

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../../.." && pwd)"
C2F="$ROOT/research/c2f"
OUT="$C2F/calibration_output"
PYTHON="$ROOT/.pixi/envs/test/bin/python"
MANIFEST="${MANIFEST:-$OUT/plan.json}"
N_SHARDS="${N_SHARDS:-8}"
PLATFORM="${OPENMM_PLATFORM:-CUDA}"

if [[ ! -x "$PYTHON" ]]; then
  echo "ERROR: test env python not found at $PYTHON" >&2
  echo "Run: cd $ROOT && pixi install -e test --frozen" >&2
  exit 1
fi

# Optional stats deps (pyblock/pymbar) when not in the env.
if ! "$PYTHON" -c "import pyblock" 2>/dev/null; then
  echo "Installing pyblock/pymbar into test env..."
  "$PYTHON" -m ensurepip --upgrade >/dev/null 2>&1 || true
  "$PYTHON" -m pip install -q pyblock pymbar
fi

mkdir -p "$OUT/logs" "$OUT/timeseries"

if [[ ! -f "$MANIFEST" ]]; then
  echo "Generating manifest $MANIFEST"
  "$PYTHON" "$C2F/plan_fictive_calibration.py" --output "$MANIFEST"
fi

echo "Launching $N_SHARDS calibration shards (platform=$PLATFORM)"
pids=()
for shard in $(seq 0 $((N_SHARDS - 1))); do
  if (( shard < N_SHARDS / 2 )); then
    gpu=0
  else
    gpu=1
  fi
  log="$OUT/logs/shard_${shard}.log"
  echo "  shard $shard → GPU $gpu (log: $log)"
  (
    export CUDA_VISIBLE_DEVICES="$gpu"
    exec "$PYTHON" "$C2F/run_fictive_calibration.py" \
      --manifest "$MANIFEST" \
      --shard-id "$shard" \
      --n-shards "$N_SHARDS" \
      --platform "$PLATFORM" \
      --output "$OUT/shard_${shard}.txt" \
      --slim-output "$OUT/shard_${shard}_slim.txt" \
      --timeseries-dir "$OUT/timeseries" \
      --n-eff-report "$OUT/shard_${shard}_n_eff.json" \
      --resume
  ) >"$log" 2>&1 &
  pids+=($!)
done

fail=0
for pid in "${pids[@]}"; do
  if ! wait "$pid"; then
    fail=1
  fi
done

if (( fail != 0 )); then
  echo "ERROR: one or more shards failed; inspect $OUT/logs/" >&2
  exit 1
fi

echo "Merging shard outputs..."
"$PYTHON" "$C2F/merge_fictive_calibration.py" \
  --shard-glob "$OUT/shard_[0-9].txt" \
  --full-output "$OUT/potential_energy_components_vs_temperature.txt" \
  --slim-output "$OUT/calibration_data.txt" \
  --n-eff-report "$OUT/n_eff_report.json"

echo "Validating merged calibration..."
"$PYTHON" - <<'PY' "$OUT/calibration_data.txt" "$OUT/potential_energy_components_vs_temperature.txt"
import sys
from pathlib import Path
from openmm.cavitymd.calibration import validate_calibration_file, crosscheck_calibration_against_reference
slim = Path(sys.argv[1])
full = Path(sys.argv[2])
ref = slim.parent.parent / "reference_potential_energy_vs_T.txt"
ok = validate_calibration_file(full, slim)
if ref.exists():
    crosscheck_calibration_against_reference(slim, ref)
sys.exit(0 if ok else 1)
PY

echo "Calibration sweep complete: $OUT/calibration_data.txt"
