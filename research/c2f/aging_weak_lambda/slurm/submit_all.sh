#!/usr/bin/env bash
# Submit IC equilibration -> production array -> analysis dependency chain.
set -euo pipefail

REPO_ROOT=/scratch/mh7373/openmm
CAMPAIGN_DIR="${REPO_ROOT}/research/c2f/aging_weak_lambda"
SLURM_DIR="${CAMPAIGN_DIR}/slurm"
RESULTS_DIR="${CAMPAIGN_DIR}/results"
LOG_DIR="${SLURM_DIR}/logs"

mkdir -p "${LOG_DIR}" "${RESULTS_DIR}"

cd "${REPO_ROOT}"
export PATH="/scratch/mh7373/.pixi/bin:${PATH}"

PROVENANCE="${RESULTS_DIR}/provenance.json"
python3 - <<'PY'
import hashlib
import json
import subprocess
from datetime import datetime, timezone
from pathlib import Path

repo = Path("/scratch/mh7373/openmm")
campaign = repo / "research/c2f/aging_weak_lambda"


def sh(cmd: list[str]) -> str:
    return subprocess.check_output(cmd, cwd=repo, text=True).strip()


git_commit = sh(["git", "rev-parse", "HEAD"])
git_branch = sh(["git", "rev-parse", "--abbrev-ref", "HEAD"])
pixi_lock = repo / "pixi.lock"
lock_sha = hashlib.sha256(pixi_lock.read_bytes()).hexdigest() if pixi_lock.exists() else None

payload = {
    "created_utc": datetime.now(timezone.utc).isoformat(),
    "hostname": sh(["hostname"]),
    "repo_root": str(repo),
    "git_commit": git_commit,
    "git_branch": git_branch,
    "pixi_lock_sha256": lock_sha,
    "slurm_account": "torch_pr_283_chemistry",
    "slurm_partition": "a100_chemistry",
    "slurm_qos": "gpu48",
    "campaign": {
        "n_replicas": 1000,
        "lambdas": [0.0, 0.01, 0.016667, 0.023333, 0.03],
        "base_seed": 42,
        "seed_formula": "42 + replica",
        "runtime_ps": 2500.0,
        "switch_time_ps": 200.0,
        "integrator": "max_metric_adaptive",
        "dt_max_ps": 0.001,
        "fkt_kmag_nm_inv": 19.05789556235437,
        "fkt_sites": "atomic",
        "ir_subset_replicas": 10,
        "ir_windows_ps": [(150.0, 50.0), (2450.0, 50.0)],
        "dipole_interval_ps": 0.001,
    },
    "ic_command": (
        "run_cavity_equilibrium.py --temperature-K 100 --runtime-ps 10000 "
        "--lambda 0 --with-dse --no-finite-q --sample-interval-ps 10 "
        "--platform CUDA --output-prefix equilibrium_output/eq10ns100K_lam0"
    ),
    "jobs": {
        "ic": "slurm/00_equilibrate_ic.sbatch",
        "production": "slurm/14_production_adaptive_n1000.sbatch --array=0-249%40",
        "analysis": "slurm/20_analysis.sbatch",
    },
}
out = campaign / "results" / "provenance.json"
out.write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")
print(f"Wrote {out}")
PY

echo "Provenance: ${PROVENANCE}"

JOB_IC=$(sbatch --parsable "${SLURM_DIR}/00_equilibrate_ic.sbatch")
echo "Submitted IC job: ${JOB_IC}"

JOB_PROD=$(sbatch --parsable --dependency=afterok:"${JOB_IC}" "${SLURM_DIR}/14_production_adaptive_n1000.sbatch")
echo "Submitted production array: ${JOB_PROD}"

JOB_ANALYSIS=$(sbatch --parsable --dependency=afterok:"${JOB_PROD}" "${SLURM_DIR}/20_analysis.sbatch")
echo "Submitted analysis job: ${JOB_ANALYSIS}"

cat <<EOF

Dependency chain:
  IC          ${JOB_IC}
  production  ${JOB_PROD}  (afterok:${JOB_IC})
  analysis    ${JOB_ANALYSIS}  (afterok:${JOB_PROD})

Monitor: squeue -u \$USER
Logs:    ${LOG_DIR}/
EOF
