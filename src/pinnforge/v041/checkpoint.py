"""Resumable per-case scoring for v0.4.1. Does not change the frozen protocol.

Each atomic inverse or solver call is a JSONL row keyed by problem, split
order, noise, method, instance, and seed. Restarts skip completed keys.
A process pool runs pending cases; noise is replayed from the serial RNG
so estimates match a one-process run. Wall-clock seconds may differ.
"""

from __future__ import annotations

import json
import math
import multiprocessing as mp
import os
import traceback
from collections.abc import Callable, Sequence
from concurrent.futures import ProcessPoolExecutor, as_completed
from pathlib import Path
from typing import Any

import numpy as np

from pinnforge.v04.sensors import add_noise


def default_workers() -> int:
    raw = os.environ.get("PINNFORGE_V041_WORKERS")
    if raw:
        return max(1, int(raw))
    return max(1, int(os.cpu_count() or 1))


def json_ready(value: Any) -> Any:
    if isinstance(value, dict):
        return {str(key): json_ready(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [json_ready(item) for item in value]
    if isinstance(value, np.generic):
        return value.item()
    if isinstance(value, np.ndarray):
        return value.tolist()
    return value


def fno_key(problem: str, noise: float, instance_id: int, loss: str, seed: int) -> str:
    return f"fno|{problem}|{noise}|{instance_id}|{loss}|{seed}"


def solver_key(problem: str, noise: float, instance_id: int, initial: str) -> str:
    return f"solver|{problem}|{noise}|{instance_id}|{initial}"


def classical_key(problem: str, noise: float, instance_id: int, method: str) -> str:
    return f"classical|{problem}|{noise}|{instance_id}|{method}"


def warm_key(problem: str, noise: float, instance_id: int, loss: str, seed: int, n_local: int) -> str:
    return f"warm|{problem}|{noise}|{instance_id}|{loss}|{seed}|n{n_local}"


def replay_noise(
    fields: np.ndarray,
    instance_ids: Sequence[int],
    instance_id: int,
    noise: float,
    protocol: dict[str, Any],
) -> np.ndarray:
    """Replay the serial ``add_noise`` stream used by ``_score_split``.

    The parent loop seeds one generator with ``noise_seed + int(1000 * noise)``
    and then calls ``add_noise`` once per instance in ``instance_ids`` order.
    Workers reconstruct that same draw without sharing state.
    """

    target = int(instance_id)
    if float(noise) == 0.0:
        return np.asarray(fields[target], dtype=np.float64).copy()
    rng = np.random.default_rng(int(protocol["noise"]["noise_seed"]) + int(1000 * noise))
    drawn = None
    for item in instance_ids:
        noisy = add_noise(fields[int(item)], float(noise), rng)
        if int(item) == target:
            drawn = noisy
    if drawn is None:
        raise KeyError(f"instance {target} is not in the noise stream")
    return drawn


class CaseLog:
    """Append-only JSONL of finished cases. Stale protocol hashes are ignored."""

    def __init__(self, path: Path, protocol_sha256: str) -> None:
        self.path = Path(path)
        self.protocol_sha256 = protocol_sha256
        self.done: dict[str, dict[str, Any]] = {}
        self._load()

    def _load(self) -> None:
        if not self.path.is_file():
            return
        for line in self.path.read_text(encoding="utf-8").splitlines():
            if not line.strip():
                continue
            try:
                item = json.loads(line)
            except json.JSONDecodeError:
                continue
            if item.get("protocol_sha256") != self.protocol_sha256:
                continue
            key = item.get("key")
            payload = item.get("payload")
            if isinstance(key, str) and isinstance(payload, dict):
                self.done[key] = payload

    def get(self, key: str) -> dict[str, Any] | None:
        return self.done.get(key)

    def require(self, key: str) -> dict[str, Any]:
        payload = self.get(key)
        if payload is None:
            raise KeyError(key)
        return payload

    def put(self, key: str, payload: dict[str, Any]) -> dict[str, Any]:
        ready = json_ready(payload)
        if not isinstance(ready, dict):
            raise TypeError("case payload must be a mapping")
        self.done[key] = ready
        self.path.parent.mkdir(parents=True, exist_ok=True)
        record = {"protocol_sha256": self.protocol_sha256, "key": key, "payload": ready}
        with self.path.open("a", encoding="utf-8") as handle:
            handle.write(json.dumps(record) + "\n")
            handle.flush()
            os.fsync(handle.fileno())
        return ready

    def __contains__(self, key: object) -> bool:
        return key in self.done

    def __len__(self) -> int:
        return len(self.done)


def single_thread() -> None:
    for name in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS", "NUMEXPR_NUM_THREADS"):
        os.environ[name] = "1"
    try:
        import torch

        torch.set_num_threads(1)
        torch.set_num_interop_threads(1)
    except Exception:
        pass


def run_cases(
    jobs: Sequence[dict[str, Any]],
    *,
    log: CaseLog,
    workers: int,
    worker: Callable[[dict[str, Any]], dict[str, Any]] | None = None,
) -> dict[str, dict[str, Any]]:
    """Run pending jobs and return every requested key, including cached rows."""

    worker_fn = case_worker if worker is None else worker
    pending = [job for job in jobs if job["key"] not in log]
    print(
        f"cases total={len(jobs)} cached={len(jobs) - len(pending)} pending={len(pending)} workers={workers}",
        flush=True,
    )
    if pending and int(workers) <= 1:
        for job in pending:
            log.put(job["key"], worker_fn(job))
            print(f"done {job['key']}", flush=True)
    elif pending:
        single_thread()
        ctx = mp.get_context("spawn")
        with ProcessPoolExecutor(
            max_workers=int(workers),
            mp_context=ctx,
            initializer=single_thread,
        ) as pool:
            futures = {pool.submit(worker_fn, job): job for job in pending}
            finished = 0
            for future in as_completed(futures):
                job = futures[future]
                try:
                    payload = future.result()
                except Exception as exc:
                    raise RuntimeError(f"case {job['key']} failed: {exc}") from exc
                log.put(job["key"], payload)
                finished += 1
                print(f"done {finished}/{len(pending)} {job['key']}", flush=True)
    return {job["key"]: log.require(job["key"]) for job in jobs}


_PROBLEM_CACHE: dict[str, tuple[np.ndarray, np.ndarray, dict[str, Any]]] = {}
_MODEL_CACHE: dict[str, Any] = {}


def _load_problem(root: Path, problem: str) -> tuple[np.ndarray, np.ndarray, dict[str, Any]]:
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


def case_worker(job: dict[str, Any]) -> dict[str, Any]:
    """One inverse or solver evaluation. Import-heavy work stays in the worker."""

    single_thread()
    try:
        return _run_case(job)
    except Exception:
        raise RuntimeError(f"case {job.get('key')} crashed:\n{traceback.format_exc()}") from None


def _run_case(job: dict[str, Any]) -> dict[str, Any]:
    from pinnforge.reference.spectral2d.dataset import load_protocol
    from pinnforge.v04.baselines import closed_form_ls, weak_form
    from pinnforge.v04.inverse import fno_inverse, solver_fit
    from pinnforge.v04.sensors import mask_from_protocol
    from pinnforge.v041.inverse import bracket_search

    protocol = load_protocol(Path(job["protocol"]))
    root = Path(job["root"])
    problem = str(job["problem"])
    fields, _nu, manifest = _load_problem(root, problem)
    instance_id = int(job["instance_id"])
    field = replay_noise(fields, job["instance_ids"], instance_id, float(job["noise"]), protocol)
    mask = mask_from_protocol(protocol)
    forcing = manifest["forcing"][instance_id]
    kind = job["kind"]
    if kind == "fno":
        guess = fno_inverse(
            field,
            mask,
            model_path=_model_path(root, problem, str(job["loss"]), int(job["seed"])),
            protocol=protocol,
            problem=problem,
        )
        return {
            "nu_hat": float(guess["nu_hat"]),
            "seconds": float(guess["seconds"]),
            "loss": float(guess["loss"]),
        }
    if kind == "solver":
        fitted = solver_fit(
            field,
            mask,
            problem=problem,
            protocol=protocol,
            forcing=forcing,
            initial=str(job["initial"]),
        )
        return {"nu_hat": float(fitted["nu_hat"]), "seconds": float(fitted["seconds"])}
    if kind == "classical":
        method = str(job["method"])
        dt = float(protocol["time"]["save_dt"])
        length = float(manifest["length"])
        if method == "closed_form_ls":
            hat = closed_form_ls(field, mask, problem=problem, dt=dt, forcing=forcing, length=length)
        elif method == "weak_form":
            hat = weak_form(field, mask, problem=problem, dt=dt, forcing=forcing, length=length)
        else:
            raise ValueError(method)
        return {"nu_hat": float(hat), "seconds": 0.0}
    if kind == "warm":
        warm = bracket_search(
            field,
            mask,
            first_guess=float(job["first_guess"]),
            q_log=float(job["q_log"]),
            n_local=int(job["n_local"]),
            problem=problem,
            protocol=protocol,
            forcing=forcing,
        )
        return {
            "nu_hat": float(warm["nu_hat"]),
            "seconds": float(warm["seconds"]),
            "bracket_log_half_width": float(warm["bracket_log_half_width"]),
            "bracket_low": float(warm["bracket_low"]),
            "bracket_high": float(warm["bracket_high"]),
            "nu_guess": float(job["first_guess"]),
            "guess_seconds": float(job["guess_seconds"]),
        }
    raise ValueError(kind)


def _model_path(root: Path, problem: str, loss: str, seed: int) -> Path:
    return root / "train" / problem / loss / f"seed_{seed}" / "checkpoint.pt"


def finite_or_none(value: Any) -> float | None:
    if value is None:
        return None
    number = float(value)
    return number if math.isfinite(number) else None
