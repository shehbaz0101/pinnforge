"""Resumable per-instance library extraction for v0.5.1."""

from __future__ import annotations

import os
import traceback
from typing import Any

import numpy as np

from pinnforge.v04.sensors import read_sensors
from pinnforge.v041.checkpoint import CaseLog, json_ready, run_cases, single_thread
from pinnforge.v05.checkpoint import replay_noise
from pinnforge.v05.library import terms_for
from pinnforge.v05.wsindy import assemble_weak_system
from pinnforge.v051.masks import mask_for_lattice

__all__ = [
    "CaseLog",
    "default_workers",
    "feature_key",
    "json_ready",
    "replay_noise",
    "run_cases",
    "single_thread",
]


def default_workers() -> int:
    raw = os.environ.get("PINNFORGE_V051_WORKERS")
    if raw:
        return max(1, int(raw))
    return max(1, int(os.cpu_count() or 1))


def feature_key(problem: str, lattice: str, noise: float, seed: int, instance_id: int) -> str:
    return f"feat|{problem}|{lattice}|{noise}|{seed}|{instance_id}"


_PROBLEM_CACHE: dict[str, tuple[np.ndarray, np.ndarray, dict[str, Any]]] = {}


def case_worker(job: dict[str, Any]) -> dict[str, Any]:
    single_thread()
    try:
        return _run_case(job)
    except Exception:
        raise RuntimeError(f"case {job.get('key')} crashed:\n{traceback.format_exc()}") from None


def _run_case(job: dict[str, Any]) -> dict[str, Any]:
    from pathlib import Path

    from pinnforge.reference.spectral2d.dataset import load_protocol

    protocol = load_protocol(Path(job["protocol"]))
    root = Path(job["root"])
    problem = str(job["problem"])
    fields, nu, manifest = _load_problem(root, problem)
    instance_id = int(job["instance_id"])
    observed = replay_noise(fields, instance_id, float(job["noise"]), int(job["seed"]), protocol)
    mask = mask_for_lattice(protocol, int(job["n_side"]), n=int(manifest["n"]))
    used = read_sensors(observed, mask)
    design, target = assemble_weak_system(
        used,
        problem=problem,
        protocol=protocol,
        mask=mask,
        length=float(manifest["length"]),
    )
    return {
        "terms": list(terms_for(problem)),
        "design": design.tolist(),
        "target": target.tolist(),
        "n_rows": int(design.shape[0]),
        "lattice": str(job["lattice"]),
        "n_side": int(job["n_side"]),
        "family": manifest["forcing"][instance_id] if problem == "ns2d" else "all",
        "nu": float(nu[instance_id]),
    }


def _load_problem(root: Any, problem: str) -> tuple[np.ndarray, np.ndarray, dict[str, Any]]:
    import json
    from pathlib import Path

    root = Path(root)
    key = f"{root.resolve()}|{problem}"
    cached = _PROBLEM_CACHE.get(key)
    if cached is not None:
        return cached
    manifest = json.loads((root / f"{problem}_manifest.json").read_text(encoding="utf-8"))
    fields = np.load(root / manifest["files"]["fields"])
    viscosities = np.load(root / manifest["files"]["nu"])
    packed = (fields, viscosities, manifest)
    _PROBLEM_CACHE[key] = packed
    return packed
