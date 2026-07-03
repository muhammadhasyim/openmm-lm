#!/usr/bin/env python3
"""CACE/LES-BEC water cavity MD (GPU bridge + PythonForce)."""

from __future__ import annotations

import sys
from pathlib import Path

_COMMON = Path(__file__).resolve().parents[1] / "common"
if str(_COMMON) not in sys.path:
    sys.path.insert(0, str(_COMMON))

from ml_cavity_runner import run_ml_water_cavity_md


def main() -> int:
    from openmmml.cuda_bridge import CACE_BRIDGE_KEY

    return run_ml_water_cavity_md(
        registry_name="cace-les-bec",
        default_output_dir=Path("runs/cace_les_bec_water"),
        bridge_key=CACE_BRIDGE_KEY,
    )


if __name__ == "__main__":
    sys.exit(main())
