#!/usr/bin/env bash
# N=10k local aging campaign: 10 ns equil + 500 adaptive production replicas
# on two A100 GPUs, coexisting with other OpenMM jobs (e.g. calibration).
set -euo pipefail

ROOT="/scratch/mh7373/openmm"
C2F="${ROOT}/research/c2f"
CAMPAIGN="${C2F}/aging_weak_lambda"
PY="${ROOT}/.pixi/envs/test/bin/python"
PLUGIN_DIR="${ROOT}/.pixi/envs/test/lib/plugins"
CAVDIR="${ROOT}/.pixi/envs/test/lib/python3.13/site-packages/openmm/cavitymd"

export PYTHONUNBUFFERED=1
export OPENMM_PLUGIN_DIR="${PLUGIN_DIR}"

LOG_DIR="${CAMPAIGN}/N10k/logs"
mkdir -p "${LOG_DIR}" "${CAMPAIGN}/N10k"

if [[ ! -x "${PY}" ]]; then
  echo "ERROR: missing ${PY}; run: cd ${ROOT} && pixi install -e test --frozen" >&2
  exit 1
fi

# Sync cavitymd Python modules into pixi env (adaptive + public API exports).
cp "${ROOT}/wrappers/python/openmm/cavitymd/"*.py "${CAVDIR}/"

echo "=== CUDA preflight ==="
"${PY}" - <<'PY'
import openmm
from openmm import unit

names = [openmm.Platform.getPlatform(i).getName() for i in range(openmm.Platform.getNumPlatforms())]
print("Platforms:", names)
if "CUDA" not in names:
    raise SystemExit("CUDA platform not available")
system = openmm.System()
system.addParticle(1.0)
ctx = openmm.Context(
    system,
    openmm.VerletIntegrator(0.001 * unit.picoseconds),
    openmm.Platform.getPlatformByName("CUDA"),
)
del ctx
print("CUDA context OK")
PY

echo "=== Probe N=10k GPU memory (with calibration jobs running) ==="
JOB_MEM_MIB="$("${PY}" "${C2F}/probe_gpu_scaling.py" --sizes 10000 --gpu 0 2>&1 \
  | awk -F'GPU=' '/GPU=/{gsub(/ MiB.*/,"",$2); print $2; exit}' \
  || true)"
if [[ -z "${JOB_MEM_MIB}" || "${JOB_MEM_MIB}" == "nan" ]]; then
  echo "WARN: probe failed; defaulting JOB_MEM_MIB=14000"
  JOB_MEM_MIB=14000
fi
echo "N=10k job memory estimate: ${JOB_MEM_MIB} MiB"

read -r CAL0 CAL1 <<<"$("${PY}" - <<'PY'
import subprocess

def gpu_used_mib(gpu_id: int) -> float:
    out = subprocess.check_output(
        [
            "nvidia-smi",
            "--query-gpu=index,memory.used",
            "--format=csv,noheader,nounits",
        ],
        text=True,
    )
    for line in out.strip().splitlines():
        idx, used = [x.strip() for x in line.split(",")]
        if int(idx) == gpu_id:
            return float(used)
    return 0.0

print(gpu_used_mib(0), gpu_used_mib(1))
PY
)"
GPU_TOTAL_MIB=81920
HEADROOM_MIB=4000
PER_GPU_JOBS="$("${PY}" - <<PY
job = float("${JOB_MEM_MIB}")
cal0 = float("${CAL0}")
cal1 = float("${CAL1}")
total = ${GPU_TOTAL_MIB}
head = ${HEADROOM_MIB}
j0 = max(1, int((total - cal0 - head) // job))
j1 = max(1, int((total - cal1 - head) // job))
print(j0, j1, j0 + j1)
PY
)"
read -r JOBS_GPU0 JOBS_GPU1 MAX_JOBS <<<"${PER_GPU_JOBS}"
MAX_JOBS="${MAX_JOBS:-4}"
if (( MAX_JOBS > 6 )); then
  MAX_JOBS=6
fi
if (( MAX_JOBS < 2 )); then
  MAX_JOBS=2
fi
echo "GPU0 used=${CAL0} MiB -> up to ${JOBS_GPU0} concurrent N=10k jobs"
echo "GPU1 used=${CAL1} MiB -> up to ${JOBS_GPU1} concurrent N=10k jobs"
echo "MAX_JOBS (total concurrent prod replicas)=${MAX_JOBS}"

IC_PREFIX="${C2F}/equilibrium_output/eq10ns100K_N10k_lam0"
IC_FILE="${IC_PREFIX}_final_state.npz"
EQUIL_LOG="${LOG_DIR}/equil.log"
ORCH_LOG="${LOG_DIR}/orchestrator.log"

launch_equil() {
  if [[ -f "${IC_FILE}" ]]; then
    echo "IC already exists: ${IC_FILE}"
    return 0
  fi
  if pgrep -u "$(whoami)" -f "eq10ns100K_N10k_lam0" >/dev/null 2>&1; then
    echo "Equil already running (eq10ns100K_N10k_lam0)"
    return 0
  fi
  echo "=== Launching 10 ns N=10k equil on GPU 0 ==="
  (
    export CUDA_VISIBLE_DEVICES=0
    exec "${PY}" "${C2F}/run_cavity_equilibrium.py" \
      --temperature-K 100 \
      --runtime-ps 10000 \
      --lambda 0 \
      --num-molecules 10000 \
      --with-dse \
      --no-finite-q \
      --sample-interval-ps 10 \
      --platform CUDA \
      --output-prefix "${IC_PREFIX}"
  ) >>"${EQUIL_LOG}" 2>&1 &
  echo "Equil PID=$! log=${EQUIL_LOG}"
}

wait_for_ic() {
  echo "=== Waiting for IC: ${IC_FILE} ==="
  while [[ ! -f "${IC_FILE}" ]]; do
    if ! pgrep -u "$(whoami)" -f "eq10ns100K_N10k_lam0" >/dev/null 2>&1; then
      if [[ -f "${EQUIL_LOG}" ]] && tail -5 "${EQUIL_LOG}" | grep -q "Simulation complete"; then
        :
      elif [[ ! -f "${IC_FILE}" ]]; then
        echo "ERROR: equil process exited without IC; see ${EQUIL_LOG}" >&2
        exit 1
      fi
    fi
    sleep 60
  done
  echo "IC ready: ${IC_FILE}"
}

run_replica() {
  local replica="$1"
  local gpu="$2"
  local log="${LOG_DIR}/rep_${replica}.log"
  local rc=0
  {
    echo "=== replica ${replica} GPU ${gpu} start $(date -Is) ==="
    CUDA_VISIBLE_DEVICES="${gpu}" "${PY}" "${CAMPAIGN}/run_single.py" \
      --lambda 0.03 \
      --replica "${replica}" \
      --runtime-ps 2200 \
      --switch-time-ps 200 \
      --num-molecules 10000 \
      --initial-state "${IC_FILE}" \
      --campaign-dir "${CAMPAIGN}/N10k" \
      --platform CUDA \
      --adaptive \
      --ir-windows 150 50 \
      --ir-windows 2150 50 \
      || {
        echo "Adaptive failed for replica ${replica}; retrying fixed dt"
        CUDA_VISIBLE_DEVICES="${gpu}" "${PY}" "${CAMPAIGN}/run_single.py" \
          --lambda 0.03 \
          --replica "${replica}" \
          --runtime-ps 2200 \
          --switch-time-ps 200 \
          --num-molecules 10000 \
          --initial-state "${IC_FILE}" \
          --campaign-dir "${CAMPAIGN}/N10k" \
          --platform CUDA \
          --ir-windows 150 50 \
          --ir-windows 2150 50
      }
    rc=$?
    echo "=== replica ${replica} finished $(date -Is) exit=${rc} ==="
  } >>"${log}" 2>&1
  echo "{\"replica\":${replica},\"gpu\":${gpu},\"returncode\":${rc},\"finished\":\"$(date -Is)\"}" \
    >>"${CAMPAIGN}/N10k/campaign_log.jsonl"
  return "${rc}"
}

run_production_pool() {
  echo "=== Production: replicas ${N10K_REPLICA_START:-0}-${N10K_REPLICA_END:-499}, MAX_JOBS=${MAX_JOBS} ==="
  local -a pids=()
  local -a replica_gpu=()
  local replica
  local gpu_idx=0
  local fail=0

  for replica in $(seq "${REPLICA_START:-0}" "${REPLICA_END:-499}"); do
    local prefix="${CAMPAIGN}/N10k/lambda0p03/lam0p03_seed$(printf '%04d' $((42 + replica)))"
    if [[ -f "${prefix}_final_state.npz" ]]; then
      echo "Skip complete replica ${replica}"
      continue
    fi
    while ((${#pids[@]} >= MAX_JOBS)); do
      local new_pids=()
      for pid in "${pids[@]}"; do
        if kill -0 "${pid}" 2>/dev/null; then
          new_pids+=("${pid}")
        fi
      done
      pids=("${new_pids[@]}")
      if ((${#pids[@]} >= MAX_JOBS)); then
        sleep 30
      fi
    done
    if (( gpu_idx % 2 == 0 )); then
      gpu=0
    else
      gpu=1
    fi
    gpu_idx=$((gpu_idx + 1))
    run_replica "${replica}" "${gpu}" &
    pids+=("$!")
    replica_gpu+=("${replica}:${gpu}")
    echo "Started replica ${replica} on GPU ${gpu} (running=${#pids[@]})"
  done

  for pid in "${pids[@]}"; do
    if ! wait "${pid}"; then
      fail=1
    fi
  done
  return "${fail}"
}

MODE="${1:-all}"
REPLICA_START="${REPLICA_START:-0}"
REPLICA_END="${REPLICA_END:-499}"

case "${MODE}" in
  equil)
    launch_equil
    ;;
  prod)
    if [[ ! -f "${IC_FILE}" ]]; then
      echo "ERROR: missing IC ${IC_FILE}; run: $0 equil" >&2
      exit 1
    fi
    run_production_pool
    ;;
  all)
    launch_equil
    wait_for_ic
    run_production_pool
    ;;
  wait-ic)
    wait_for_ic
    ;;
  *)
    echo "Usage: $0 [all|equil|wait-ic|prod]" >&2
    exit 1
    ;;
esac

echo "Done ($(date -Is)). Logs: ${LOG_DIR}/"
