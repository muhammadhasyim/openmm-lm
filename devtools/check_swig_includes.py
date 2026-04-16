#!/usr/bin/env python3
"""
Verify that every #include "..." in wrappers/python/src/swig_doxygen/OpenMM.i
inside the %{ ... %} block resolves to an existing file under the OpenMM source tree.

Exit code 1 if any include is missing (prevents CI failures from phantom headers).
"""
from __future__ import annotations

import re
import sys
from pathlib import Path


def _repo_root() -> Path:
    return Path(__file__).resolve().parent.parent


def _parse_quoted_includes(openmm_i: Path) -> list[str]:
    text = openmm_i.read_text(encoding="utf-8")
    m = re.search(r"%\{(.*?)%\}", text, re.DOTALL)
    if not m:
        print("ERROR: no %{ ... %} block in OpenMM.i", file=sys.stderr)
        sys.exit(1)
    block = m.group(1)
    return re.findall(r'#include\s+"([^"]+)"', block)


def _search_roots(repo: Path) -> list[Path]:
    return [
        repo / "openmmapi/include",
        repo / "olla/include",
        repo / "serialization/include",
        repo / "plugins/amoeba/openmmapi/include",
        repo / "plugins/rpmd/openmmapi/include",
        repo / "plugins/drude/openmmapi/include",
        repo / "include",
    ]


def _resolve_include(repo: Path, rel: str, roots: list[Path]) -> Path | None:
    rel_path = Path(rel)
    for root in roots:
        candidate = root / rel_path
        if candidate.is_file():
            return candidate
    return None


def main() -> int:
    repo = _repo_root()
    openmm_i = repo / "wrappers/python/src/swig_doxygen/OpenMM.i"
    if not openmm_i.is_file():
        print(f"ERROR: {openmm_i} not found", file=sys.stderr)
        return 1

    includes = _parse_quoted_includes(openmm_i)
    roots = _search_roots(repo)
    missing: list[str] = []
    for inc in includes:
        if _resolve_include(repo, inc, roots) is None:
            missing.append(inc)

    if missing:
        print("SWIG OpenMM.i includes missing from source tree:", file=sys.stderr)
        for m in missing:
            print(f"  {m}", file=sys.stderr)
        print("\nSearched under:", file=sys.stderr)
        for r in roots:
            print(f"  {r}", file=sys.stderr)
        return 1

    print(f"OK: {len(includes)} includes in OpenMM.i resolve under {repo}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
