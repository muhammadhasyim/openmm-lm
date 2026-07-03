"""Configuration for OpenMM weak-coupling non-thermal aging replication."""

from __future__ import annotations

import math
import sys
from pathlib import Path

CAMPAIGN_DIR = Path(__file__).resolve().parent
C2F_ROOT = CAMPAIGN_DIR.parent
if str(C2F_ROOT) not in sys.path:
    sys.path.insert(0, str(C2F_ROOT))

from run_c2f import FKT_KMAG_AU, FKT_KMAG_PAPER_AU, NUM_MOL  # noqa: E402

REPO_ROOT = CAMPAIGN_DIR.parent.parent.parent

# Collective coupling g = λ√N (intensive coupling strength). Campaign λ values below
# are defined at REFERENCE_NUM_MOL; scale with scale_lambda() when N changes.
REFERENCE_NUM_MOL = NUM_MOL  # 250


def collective_coupling_g(lambda_coupling: float, num_molecules: int) -> float:
    """Return g = λ√N for the given per-molecule coupling."""
    return lambda_coupling * math.sqrt(num_molecules)


def lambda_for_constant_g(g: float, num_molecules: int) -> float:
    """Return λ = g/√N holding collective coupling fixed."""
    if g == 0.0:
        return 0.0
    return g / math.sqrt(num_molecules)


def scale_lambda(
    lambda_ref: float,
    num_molecules: int,
    reference_num_mol: int = REFERENCE_NUM_MOL,
) -> float:
    """Scale λ from reference N so g = λ√N stays constant."""
    if lambda_ref == 0.0:
        return 0.0
    g = collective_coupling_g(lambda_ref, reference_num_mol)
    return lambda_for_constant_g(g, num_molecules)


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

# Fine dipole sampling for Fig 2a IR (replicas 0-9 only; Nyquist ~16,700 cm^-1 at 1 fs).
IR_SUBSET_REPLICAS = 10
DIPOLE_INTERVAL_PS = 0.001
IR_WINDOWS: list[tuple[float, float]] = [
    (150.0, 50.0),   # baseline pre turn-on at 200 ps
    (2450.0, 50.0),  # late aged window
]

# Paper-scale ensemble: 1000 independent velocity resamples from common IC per λ.
N_REPLICAS = 1000
REPLICA_START = 0
REPLICA_END = 999
BASE_SEED = 42

# Per-molecule λ at N=250 (reference). Use scale_lambda(lam, N) when N ≠ 250.
LAMBDAS: list[float] = [0.0, 0.01, 0.016667, 0.023333, 0.03]
# Analysis/figures: exclude λ=0.03 until the N=1000 campaign completes (partial
# ensemble gives noisy τ_s and unstable MTTI curves).
ANALYSIS_LAMBDAS: list[float] = [0.0, 0.01, 0.016667, 0.023333]
FIG3_SHOWCASE_LAMBDA = 0.016667
LAMBDA_G_AT_N250: dict[float, float] = {
    lam: collective_coupling_g(lam, REFERENCE_NUM_MOL) for lam in LAMBDAS
}

# One OpenMM job per λ per replica round (~316 MiB each; 5 jobs ≈ 1.6 GiB on RTX 4070).
DEFAULT_JOBS = len(LAMBDAS)
CAMPAIGN_LOG = CAMPAIGN_DIR / "campaign_n1000_log.jsonl"

# N=10k local aging campaign (g-scaled λ equivalent to λ=0.03 @ N=250).
N10K_CAMPAIGN_DIR = CAMPAIGN_DIR / "N10k"
N10K_NUM_MOL = 10_000
N10K_IC = C2F_ROOT / "equilibrium_output" / "eq10ns100K_N10k_lam0_final_state.npz"
N10K_LAMBDA_REF = 0.03
N10K_LAMBDA_EFF = scale_lambda(N10K_LAMBDA_REF, N10K_NUM_MOL)
N10K_SWITCH_PS = 200.0
N10K_RUNTIME_PS = 2200.0  # 200 ps idle + 2 ns aging
N10K_REPLICA_START = 0
N10K_REPLICA_END = 499
N10K_REPLICAS = 500
N10K_IR_WINDOWS: list[tuple[float, float]] = [
    (150.0, 50.0),
    (2150.0, 50.0),
]
N10K_CAMPAIGN_LOG = N10K_CAMPAIGN_DIR / "campaign_log.jsonl"

_CAV_HOOMD_CANDIDATES = (
    Path("/scratch/mh7373/projects/cav-hoomd"),
    REPO_ROOT / "third_party/cav-hoomd",
)


def _resolve_cav_hoomd_file(name: str) -> Path:
    """Return first non-empty cav-hoomd calibration table on disk."""
    for base in _CAV_HOOMD_CANDIDATES:
        path = base / name
        if path.is_file() and path.stat().st_size > 0:
            return path
    return _CAV_HOOMD_CANDIDATES[-1] / name


POTENTIAL_ENERGY_VS_T = _resolve_cav_hoomd_file("potential_energy_vs_T.txt")
RELAXATION_TIMES_VS_T = _resolve_cav_hoomd_file("relaxation_times_vs_temperature.txt")

FIGURES_DIR = CAMPAIGN_DIR / "figures"
RESULTS_DIR = CAMPAIGN_DIR / "results"
MASTER_FKT_DIR = CAMPAIGN_DIR / "master_fkt"


def lambda_tag(lam: float) -> str:
    if lam == 0.0:
        return "0"
    return f"{lam:g}".replace(".", "p")


def job_dir_name(lam: float) -> str:
    return f"lambda{lambda_tag(lam)}"


def job_dir_path(lam: float, campaign_root: Path | None = None) -> Path:
    root = CAMPAIGN_DIR if campaign_root is None else campaign_root
    return root / job_dir_name(lam)


def replica_seed(replica: int) -> int:
    return BASE_SEED + replica


def run_prefix(lam: float, replica: int) -> str:
    return f"lam{lambda_tag(lam)}_seed{replica_seed(replica):04d}"
