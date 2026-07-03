#!/usr/bin/env bash
# Paper-aligned C²F cooling at λ=0.03 (weak-coupling pool).
# 500 replicas (paper ensemble), 2 GPUs, sequential pairs.
set -euo pipefail

REPO=/scratch/mh7373/openmm
C2F_DIR="${REPO}/research/c2f"
CAMPAIGN_DIR="${C2F_DIR}/c2f_campaign/lam0p03"
LOG_DIR="${CAMPAIGN_DIR}/logs"
PY="${REPO}/.pixi/envs/test/bin/python"
IC="${C2F_DIR}/equilibrium_output/eq10ns100K_lam0_final_state.npz"
CAL="${C2F_DIR}/reference_potential_energy_vs_T.txt"

export OPENMM_PLUGIN_DIR="${REPO}/.pixi/envs/test/lib/plugins"
export PYTHONUNBUFFERED=1

mkdir -p "${LOG_DIR}"
cd "${C2F_DIR}"

if [[ ! -f "${IC}" ]]; then
  echo "ERROR: missing equilibrium IC ${IC}" >&2
  exit 1
fi

N_REPLICAS=500
BASE_SEED=42
LAM=0.03

echo "=== C²F paper protocol λ=${LAM}: ${N_REPLICAS} replicas, equilibrium IC ==="
echo "  IC: ${IC}"
echo "  coupling start: 10 ps | feedback: every-step DiffEq | finite-q: false"
echo "Started: $(date -Iseconds)"

for ((rep = 0; rep < N_REPLICAS; rep += 2)); do
  pids=()
  for offset in 0 1; do
    r=$((rep + offset))
    if ((r >= N_REPLICAS)); then
      break
    fi
    seed=$((BASE_SEED + r))
    gpu=${offset}
    prefix="${CAMPAIGN_DIR}/c2f_seed$(printf '%04d' "${seed}")"
    log="${LOG_DIR}/replica_${r}.log"

    if [[ -f "${prefix}_final_state.npz" ]]; then
      echo "Skip replica ${r} (complete): ${prefix}"
      continue
    fi

    echo "Launch replica ${r} seed=${seed} GPU=${gpu} -> ${prefix}"
    CUDA_VISIBLE_DEVICES="${gpu}" "${PY}" reproduce_c2f_campaign.py \
      --lambda "${LAM}" \
      --output-dir "${CAMPAIGN_DIR}" \
      --output-prefix c2f \
      --initial-state "${IC}" \
      --calibration-file "${CAL}" \
      --replica-start "${r}" \
      --replica-end "${r}" \
      --platform CUDA \
      > "${log}" 2>&1 &
    pids+=($!)
  done

  if ((${#pids[@]} > 0)); then
    echo "Waiting for pair replicas ${rep}-$((rep + ${#pids[@]} - 1)) (PIDs: ${pids[*]})"
    for pid in "${pids[@]}"; do
      if ! wait "${pid}"; then
        echo "ERROR: replica failed (PID ${pid})" >&2
        exit 1
      fi
    done
    echo "Pair complete at $(date -Iseconds)"
  fi
done

completed=$(find "${CAMPAIGN_DIR}" -maxdepth 1 -name 'c2f_seed*_final_state.npz' | wc -l)
echo "=== Campaign finished: ${completed}/${N_REPLICAS} final states ==="
echo "Finished: $(date -Iseconds)"

# Ensemble average over all completed trajectories
"${PY}" reproduce_c2f_campaign.py \
  --lambda "${LAM}" \
  --output-dir "${CAMPAIGN_DIR}" \
  --output-prefix c2f \
  --initial-state "${IC}" \
  --calibration-file "${CAL}" \
  --skip-simulation
