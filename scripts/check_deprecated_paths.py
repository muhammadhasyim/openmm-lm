#!/usr/bin/env python3
"""Fail if deprecated root-level submodule directories exist."""

from __future__ import annotations

import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
DEPRECATED = ("openmm-ml", "cav-hoomd", "i-pi", "LES-BEC")
found = [name for name in DEPRECATED if (REPO_ROOT / name).exists()]
if found:
    print(
        "Deprecated submodule paths at repo root (use third_party/ instead): "
        + ", ".join(found),
        file=sys.stderr,
    )
    sys.exit(1)
print("OK: no deprecated root-level submodule paths")
