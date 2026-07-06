"""Matplotlib backend helper for headless plotting scripts."""

from __future__ import annotations


def use_agg_backend() -> None:
    import matplotlib

    matplotlib.use("Agg")
