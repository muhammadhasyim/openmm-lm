#!/usr/bin/env python3
"""Backward-compatible entry point — see 03_nvt_collective_scaling.py """

from __future__ import annotations

from pathlib import Path
import runpy

runpy.run_path(str(Path(__file__).resolve().parent / "03_nvt_collective_scaling.py"), run_name="__main__")
