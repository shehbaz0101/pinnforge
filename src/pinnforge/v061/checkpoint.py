"""Resumable v0.6.1 case log."""

from __future__ import annotations

import os

from pinnforge.v041.checkpoint import CaseLog, json_ready, run_cases, single_thread

__all__ = ["CaseLog", "default_workers", "json_ready", "run_cases", "single_thread"]


def default_workers() -> int:
    raw = os.environ.get("PINNFORGE_V061_WORKERS")
    if raw:
        return max(1, int(raw))
    return max(1, int(os.cpu_count() or 1))


def feature_key(kind: str, density: str, noise: float, seed: int, instance_id: int) -> str:
    return f"{kind}|{density}|{noise}|{seed}|{instance_id}"
