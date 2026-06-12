"""Configuration for OpenMM weak-coupling non-thermal aging replication."""

from __future__ import annotations

import sys
from pathlib import Path

CAMPAIGN_DIR = Path(__file__).resolve().parent
C2F_ROOT = CAMPAIGN_DIR.parent
if str(C2F_ROOT) not in sys.path:
    sys.path.insert(0, str(C2F_ROOT))

from run_c2f import FKT_KMAG_AU, FKT_KMAG_PAPER_AU  # noqa: E402

REPO_ROOT = CAMPAIGN_DIR.parent.parent.parent.parent

INITIAL_STATE = C2F_ROOT / "equilibrium_output" / "eq10ns100K_lam0_final_state.npz"
REFERENCE_CALIBRATION = C2F_ROOT / "run_c2f.py"  # REFERENCE_CALIBRATION_FILE in run_c2f

SWITCH_TIME_PS = 200.0
RUNTIME_PS = 2500.0
TEMPERATURE_K = 100.0
FREQUENCY_CM1 = 1560.0

BOHR_TO_NM = 0.0529177
FKT_KMAG_NM_INV = FKT_KMAG_AU / BOHR_TO_NM
FKT_NUM_WAVEVECTORS = 50
FKT_REF_INTERVAL_PS = 200.0
FKT_MAX_REFS = 13
FKT_OUTPUT_PERIOD_PS = 1.0

CSV_INTERVAL_PS = 1.0
SNAPSHOT_INTERVAL_PS = 10.0

# Paper-scale ensemble: 500 independent velocity resamples from common IC per λ.
N_REPLICAS = 500
REPLICA_START = 0
REPLICA_END = 499
BASE_SEED = 42

# Pilot (4-replica) data lives under pilot_n4/ after archive; production uses dirs below.
LAMBDAS: list[float] = [0.0, 0.01, 0.016667, 0.023333, 0.03]

# One OpenMM job per λ per replica round (~316 MiB each; 5 jobs ≈ 1.6 GiB on RTX 4070).
DEFAULT_JOBS = len(LAMBDAS)
CAMPAIGN_LOG = CAMPAIGN_DIR / "campaign_n500_log.jsonl"

POTENTIAL_ENERGY_VS_T = REPO_ROOT / "cav-hoomd" / "potential_energy_vs_T.txt"
RELAXATION_TIMES_VS_T = REPO_ROOT / "cav-hoomd" / "relaxation_times_vs_temperature.txt"

FIGURES_DIR = CAMPAIGN_DIR / "figures"
RESULTS_DIR = CAMPAIGN_DIR / "results"


def lambda_tag(lam: float) -> str:
    if lam == 0.0:
        return "0"
    return f"{lam:g}".replace(".", "p")


def job_dir_name(lam: float) -> str:
    return f"lambda{lambda_tag(lam)}"


def job_dir_path(lam: float) -> Path:
    return CAMPAIGN_DIR / job_dir_name(lam)


def replica_seed(replica: int) -> int:
    return BASE_SEED + replica


def run_prefix(lam: float, replica: int) -> str:
    return f"lam{lambda_tag(lam)}_seed{replica_seed(replica):04d}"
