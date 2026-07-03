#!/usr/bin/env bash
# Resume incomplete fictive-calibration shards on a single GPU (max 3 concurrent).
set -euo pipefail

export PYTHONUNBUFFERED=1

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../../.." && pwd)"
C2F="$ROOT/research/c2f"
OUT="$C2F/calibration_output"
PYTHON="$ROOT/.pixi/envs/test/bin/python"
MANIFEST="${MANIFEST:-$OUT/plan.json}"
N_SHARDS="${N_SHARDS:-8}"
MAX_JOBS="${MAX_JOBS:-3}"
PLATFORM="${OPENMM_PLATFORM:-CUDA}"
GPU="${CUDA_VISIBLE_DEVICES:-0}"
export OPENMM_PLUGIN_DIR="$ROOT/.pixi/envs/test/lib/plugins"

if [[ ! -x "$PYTHON" ]]; then
  echo "ERROR: test env python not found at $PYTHON" >&2
  exit 1
fi

mkdir -p "$OUT/logs" "$OUT/timeseries"

wait_for_gpu_slots() {
  local max_busy="$1"
  while true; do
    local busy
    busy="$(ps -u "$(whoami)" -o cmd= 2>/dev/null | rg -c "run_single\.py|run_cavity_equilibrium\.py" || true)"
    if (( busy < max_busy )); then
      echo "GPU slots available (busy=$busy, limit=$max_busy)"
      return 0
    fi
    echo "Waiting for GPU slots: $busy non-calibration OpenMM jobs running (limit $max_busy)..."
    sleep 60
  done
}

run_shard() {
  local shard="$1"
  local log="$OUT/logs/shard_${shard}.log"
  {
    echo "=== resume start $(date -Is) shard=$shard gpu=$GPU ==="
    CUDA_VISIBLE_DEVICES="$GPU" "$PYTHON" "$C2F/run_fictive_calibration.py" \
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

echo "Resuming incomplete shards 0-3 (max $MAX_JOBS concurrent, GPU $GPU)"
wait_for_gpu_slots "$MAX_JOBS"
pids=()
for shard in 0 1 2 3; do
  while ((${#pids[@]} >= MAX_JOBS)); do
    if ! wait -n -p finished_pid; then
      echo "ERROR: shard worker $finished_pid failed" >&2
      exit 1
    fi
    new_pids=()
    for pid in "${pids[@]}"; do
      [[ "$pid" != "$finished_pid" ]] && new_pids+=("$pid")
    done
    pids=("${new_pids[@]}")
  done
  echo "Launching shard $shard"
  run_shard "$shard" &
  pids+=("$!")
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

"$PYTHON" "$C2F/write_calibration_provenance.py" \
  --output "$OUT/provenance.json"

echo "Calibration resume complete: $OUT/calibration_data.txt"
