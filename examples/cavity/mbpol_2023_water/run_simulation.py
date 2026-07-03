#!/usr/bin/env python3
"""MBPol(2023) / MBX water cavity MD (CPU PythonForce)."""

from __future__ import annotations

import sys
from pathlib import Path

_COMMON = Path(__file__).resolve().parents[1] / "common"
if str(_COMMON) not in sys.path:
    sys.path.insert(0, str(_COMMON))

from ml_cavity_runner import run_ml_water_cavity_md


def main() -> int:
    return run_ml_water_cavity_md(
        registry_name="mbpol-2023",
        default_output_dir=Path("runs/mbpol_2023_water"),
        bridge_key=None,
        build_kwargs={"use_cuda_bridge": False},
    )


if __name__ == "__main__":
    sys.exit(main())
