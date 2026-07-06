#!/usr/bin/env bash
# Finish incomplete fictive-temperature calibration with 2 concurrent jobs per A100.
set -euo pipefail

export PYTHONUNBUFFERED=1

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../../.." && pwd)"
C2F="$ROOT/research/c2f"
OUT="$C2F/calibration_output"
PARTIAL="$OUT/partial"
PYTHON="$ROOT/.pixi/envs/test/bin/python"
MANIFEST="${MANIFEST:-$OUT/plan.json}"
N_SHARDS="${N_SHARDS:-8}"
PLATFORM="${OPENMM_PLATFORM:-CUDA}"
JOBS_PER_GPU="${JOBS_PER_GPU:-2}"
export OPENMM_PLUGIN_DIR="$ROOT/.pixi/envs/test/lib/plugins"

if [[ ! -x "$PYTHON" ]]; then
  echo "ERROR: test env python not found at $PYTHON" >&2
  exit 1
fi

cd "$C2F"
mkdir -p "$OUT/logs" "$OUT/timeseries" "$PARTIAL"

mapfile -t MISSING < <(
  "$PYTHON" - <<'PY'
import json
from pathlib import Path

plan = json.loads(Path("calibration_output/plan.json").read_text())
entries = plan["entries"]
ts_dir = Path("calibration_output/timeseries")
done = {
    float(p.stem.replace("calibration_T", "").replace("K", ""))
    for p in ts_dir.glob("*.csv")
}

for i, entry in enumerate(entries):
    t = entry["temperature_K"]
    if any(abs(t - d) < 0.5 for d in done):
        continue
    prod_ns = entry["prod_ps"] / 1000.0
    print(f"{t:.6f},{i % 8},{prod_ns:.6f}")
PY
)

if ((${#MISSING[@]} == 0)); then
  echo "All calibration temperatures complete."
  exit 0
fi

echo "Missing ${#MISSING[@]} temperatures:"
for item in "${MISSING[@]}"; do
  IFS=',' read -r temp shard prod_ns <<<"$item"
  printf "  T=%8.1f K  shard=%s  prod=%.2f ns\n" "$temp" "$shard" "$prod_ns"
done

run_temperature() {
  local temp="$1"
  local shard="$2"
  local gpu="$3"
  local tag
  tag="$(printf 'T%06.1f' "$temp")"
  local work="$PARTIAL/$tag"
  mkdir -p "$work"

  {
    echo "=== start $(date -Is) T=$temp shard=$shard gpu=$gpu ==="
    printf '{"temperature_K": %s, "shard_id": %s}\n' "$temp" "$shard" >"$work/meta.json"
    CUDA_VISIBLE_DEVICES="$gpu" "$PYTHON" "$C2F/run_fictive_calibration.py" \
      --manifest "$MANIFEST" \
      --shard-id "$shard" \
      --n-shards "$N_SHARDS" \
      --only-temperatures "$temp" \
      --platform "$PLATFORM" \
      --output "$work/run.txt" \
      --slim-output "$work/run_slim.txt" \
      --timeseries-dir "$OUT/timeseries" \
      --n-eff-report "$work/n_eff.json"
    echo "=== done $(date -Is) T=$temp exit=$? ==="
  } >>"$OUT/logs/partial_${tag}.log" 2>&1
}

wait_for_gpu_slot() {
  local gpu="$1"
  if ((gpu == 0)); then
    while ((${#GPU0_PIDS[@]} >= JOBS_PER_GPU)); do
      if ! wait "${GPU0_PIDS[0]}"; then
        echo "ERROR: calibration worker failed on GPU 0" >&2
        exit 1
      fi
      GPU0_PIDS=("${GPU0_PIDS[@]:1}")
    done
  else
    while ((${#GPU1_PIDS[@]} >= JOBS_PER_GPU)); do
      if ! wait "${GPU1_PIDS[0]}"; then
        echo "ERROR: calibration worker failed on GPU 1" >&2
        exit 1
      fi
      GPU1_PIDS=("${GPU1_PIDS[@]:1}")
    done
  fi
}

GPU0_PIDS=()
GPU1_PIDS=()

launch_on_gpu() {
  local gpu="$1"
  local temp="$2"
  local shard="$3"
  wait_for_gpu_slot "$gpu"
  run_temperature "$temp" "$shard" "$gpu" &
  local pid=$!
  if ((gpu == 0)); then
    GPU0_PIDS+=("$pid")
  else
    GPU1_PIDS+=("$pid")
  fi
  echo "Launched T=$temp on GPU $gpu (pid=$pid, active_gpu0=${#GPU0_PIDS[@]}, active_gpu1=${#GPU1_PIDS[@]})"
}

mapfile -t SORTED < <(printf '%s\n' "${MISSING[@]}" | sort -t, -k3 -nr)

gpu_toggle=0
for item in "${SORTED[@]}"; do
  IFS=',' read -r temp shard _prod <<<"$item"
  launch_on_gpu "$gpu_toggle" "$temp" "$shard"
  gpu_toggle=$((1 - gpu_toggle))
done

fail=0
for pid in "${GPU0_PIDS[@]}" "${GPU1_PIDS[@]}"; do
  if ! wait "$pid"; then
    fail=1
  fi
done

if ((fail != 0)); then
  echo "ERROR: one or more temperature jobs failed; inspect $OUT/logs/partial_*.log" >&2
  exit 1
fi

echo "Assembling shard outputs from partial runs..."
"$PYTHON" "$C2F/assemble_calibration_partials.py" --shards 0 2

echo "Merging all shard outputs..."
"$PYTHON" "$C2F/merge_fictive_calibration.py" \
  --shard-glob "$OUT/shard_[0-9].txt" \
  --full-output "$OUT/potential_energy_components_vs_temperature.txt" \
  --slim-output "$OUT/calibration_data.txt" \
  --n-eff-report "$OUT/n_eff_report.json"

"$PYTHON" "$C2F/write_calibration_provenance.py" \
  --output "$OUT/provenance.json"

echo "Remaining calibration complete: $OUT/calibration_data.txt"
