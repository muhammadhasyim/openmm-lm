#!/usr/bin/env bash
# Resume incomplete fictive-calibration shards on 2 GPUs (2 shards per GPU).
set -euo pipefail

export PYTHONUNBUFFERED=1

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../../.." && pwd)"
C2F="$ROOT/research/c2f"
OUT="$C2F/calibration_output"
PYTHON="$ROOT/.pixi/envs/test/bin/python"
export OPENMM_PLUGIN_DIR="$ROOT/.pixi/envs/test/lib/plugins"
MANIFEST="${MANIFEST:-$OUT/plan.json}"
N_SHARDS="${N_SHARDS:-8}"
PLATFORM="${OPENMM_PLATFORM:-CUDA}"

if [[ ! -x "$PYTHON" ]]; then
  echo "ERROR: test env python not found at $PYTHON" >&2
  exit 1
fi

mkdir -p "$OUT/logs" "$OUT/timeseries"

run_shard() {
  local shard="$1"
  local gpu="$2"
  local log="$OUT/logs/shard_${shard}.log"
  {
    echo "=== resume start $(date -Is) shard=$shard gpu=$gpu ==="
    CUDA_VISIBLE_DEVICES="$gpu" "$PYTHON" "$C2F/run_fictive_calibration.py" \
      --manifest "$MANIFEST" \
      --shard-id "$shard" \
      --n-shards "$N_SHARDS" \
      --platform "$PLATFORM" \
      --output "$OUT/shard_${shard}.txt" \
      --slim-output "$OUT/shard_${shard}_slim.txt" \
      --timeseries-dir "$OUT/timeseries" \
      --n-eff-report "$OUT/shard_${shard}_n_eff.json" \
      --resume
    echo "=== shard $shard finished $(date -Is) exit=$? ==="
  } >>"$log" 2>&1
}

echo "Resuming shards 0-3 on 2 GPUs (2 shards per GPU)"
echo "  GPU 0: shards 0, 2"
echo "  GPU 1: shards 1, 3"

pids=()
run_shard 0 0 & pids+=("$!")
run_shard 2 0 & pids+=("$!")
run_shard 1 1 & pids+=("$!")
run_shard 3 1 & pids+=("$!")

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

"$PYTHON" "$C2F/write_calibration_provenance.py" \
  --output "$OUT/provenance.json"

echo "Calibration resume complete: $OUT/calibration_data.txt"
