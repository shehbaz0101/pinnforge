"""Resumable per-instance feature extraction for v0.5. Does not change the protocol.

Each library matrix is a JSONL row keyed by problem, noise, seed, method, and
instance. Restarts skip completed keys. A process pool runs pending cases.
Noise is replayed from ``noise_seed + 100 * seed`` in instance-id order.
"""

from __future__ import annotations

import os
import traceback
from typing import Any

import numpy as np

from pinnforge.v04.sensors import add_noise, mask_from_protocol, read_sensors
from pinnforge.v041.checkpoint import CaseLog, json_ready, run_cases, single_thread
from pinnforge.v05.library import terms_for
from pinnforge.v05.wsindy import assemble_weak_system

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
    raw = os.environ.get("PINNFORGE_V05_WORKERS")
    if raw:
        return max(1, int(raw))
    return max(1, int(os.cpu_count() or 1))


def feature_key(problem: str, noise: float, seed: int, method: str, instance_id: int) -> str:
    return f"feat|{problem}|{noise}|{seed}|{method}|{instance_id}"


def replay_noise(
    fields: np.ndarray,
    instance_id: int,
    noise: float,
    seed: int,
    protocol: dict[str, Any],
) -> np.ndarray:
    """Replay the per-seed ``add_noise`` stream in instance-id order."""

    target = int(instance_id)
    values = np.asarray(fields, dtype=np.float64)
    if float(noise) == 0.0:
        return values[target].copy()
    rng = np.random.default_rng(int(protocol["noise"]["noise_seed"]) + 100 * int(seed))
    drawn = None
    for item in range(values.shape[0]):
        noisy = add_noise(values[item], float(noise), rng)
        if item == target:
            drawn = noisy
    if drawn is None:
        raise KeyError(f"instance {target} is not in the noise stream")
    return drawn


_PROBLEM_CACHE: dict[str, tuple[np.ndarray, np.ndarray, dict[str, Any]]] = {}
_MODEL_CACHE: dict[str, Any] = {}


def case_worker(job: dict[str, Any]) -> dict[str, Any]:
    single_thread()
    try:
        return _run_case(job)
    except Exception:
        raise RuntimeError(f"case {job.get('key')} crashed:\n{traceback.format_exc()}") from None


def _run_case(job: dict[str, Any]) -> dict[str, Any]:
    from pathlib import Path

    from pinnforge.reference.spectral2d.dataset import load_protocol
    from pinnforge.v04.train import relative_l2

    protocol = load_protocol(Path(job["protocol"]))
    root = Path(job["root"])
    problem = str(job["problem"])
    fields, _nu, manifest = _load_problem(root, problem)
    instance_id = int(job["instance_id"])
    seed = int(job["seed"])
    noise = float(job["noise"])
    method = str(job["method"])
    observed = replay_noise(fields, instance_id, noise, seed, protocol)
    mask = mask_from_protocol(protocol, n=int(manifest["n"]))
    length = float(manifest["length"])
    used = observed
    fno_rel = None
    if method == "wsindy_raw":
        used = read_sensors(observed, mask)
    elif method == "wsindy_fno":
        from pinnforge.v05.denoise import predict_denoised

        model, payload = _load_model(root, problem, seed)
        used = predict_denoised(observed, mask, model=model, payload=payload)
        fno_rel = relative_l2(used, fields[instance_id])
    else:
        raise ValueError(method)
    design, target = assemble_weak_system(used, problem=problem, protocol=protocol, mask=mask, length=length)
    payload = {
        "terms": list(terms_for(problem)),
        "design": design.tolist(),
        "target": target.tolist(),
        "n_rows": int(design.shape[0]),
        "method": method,
        "family": manifest["forcing"][instance_id] if problem == "ns2d" else "all",
        "nu": float(_nu[instance_id]),
        "fno_rel_l2": fno_rel,
    }
    return payload


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
    nu = np.load(root / manifest["files"]["nu"])
    packed = (fields, nu, manifest)
    _PROBLEM_CACHE[key] = packed
    return packed


def _load_model(root: Any, problem: str, seed: int) -> Any:
    from pathlib import Path

    from pinnforge.v05.denoise import load_denoiser

    path = Path(root) / "train" / problem / f"seed_{seed}" / "checkpoint.pt"
    key = str(path.resolve())
    cached = _MODEL_CACHE.get(key)
    if cached is not None:
        return cached
    loaded = load_denoiser(path)
    _MODEL_CACHE[key] = loaded
    return loaded
