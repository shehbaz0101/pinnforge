"""Pilot trajectories for periodic Burgers, split by problem instance.

Generation draws instance ``0 .. N_total-1`` from the seeded Fourier
family, integrates each with the spectral solver, and only then writes
files. The train / validation / test assignment is a Fisher–Yates
permutation of those instance ids, using PCG64 and
``SeedSequence([split_seed, 0x53504C54])``. It is computed before any
trajectory is integrated and before any window is cut. This package does
not cut windows. Every resolution of one instance id belongs to that
instance's split; the pilot stores a single resolution per instance.

Stored fields are the raw solver output in ``float64``. Normalization
statistics are the population mean and standard deviation (``ddof = 0``)
of those values on the training split only. They are recorded in the
manifest and are not applied to the arrays.

``manifest.json`` records the solver config, its canonical JSON SHA-256,
the SHA-256 of the solver sources, the seeds, the split lists, and the
SHA-256 of each ``.npz`` file.
"""

from __future__ import annotations

import hashlib
import json
import math
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np

from pinnforge.reference.numerical.initial import (
    CANONICAL_N,
    IC_SALT,
    N_MODES,
    NU_MAX,
    NU_MIN,
    draw_initial_condition,
)
from pinnforge.reference.numerical.solver import SolverConfig, solve_batch

FORMAT = "pinnforge.burgers_pilot.v1"
SPLIT_SALT = 0x53504C54
_SOURCES = ("solver.py", "checks.py", "initial.py", "dataset.py")


@dataclass(frozen=True)
class PilotConfig:
    """Sizes and discrete settings for one pilot generation.

    Counts are problem instances, not windows. ``n``, ``dt``, and
    ``save_dt`` are the label resolution. They are not a proof of
    accuracy; that is the convergence study.
    """

    n_train: int = 512
    n_val: int = 128
    n_test: int = 128
    n: int = 256
    dt: float = 1e-3
    t_final: float = 1.0
    save_dt: float = 0.01
    master_seed: int = 20260926
    split_seed: int = 20260926
    n_modes: int = N_MODES
    batch_size: int = 8
    nu_min: float = NU_MIN
    nu_max: float = NU_MAX

    def total(self) -> int:
        return self.n_train + self.n_val + self.n_test

    def to_dict(self) -> dict[str, Any]:
        return {
            "n_train": self.n_train,
            "n_val": self.n_val,
            "n_test": self.n_test,
            "n": self.n,
            "dt": self.dt,
            "t_final": self.t_final,
            "save_dt": self.save_dt,
            "master_seed": self.master_seed,
            "split_seed": self.split_seed,
            "n_modes": self.n_modes,
            "batch_size": self.batch_size,
            "nu_min": self.nu_min,
            "nu_max": self.nu_max,
            "ic_salt": IC_SALT,
            "split_salt": SPLIT_SALT,
            "canonical_n": CANONICAL_N,
            "bit_generator": "PCG64",
            "windowing": "none",
            "split_unit": "problem_instance",
        }


@dataclass
class RunningStats:
    """Population mean and second moment, updated in batches."""

    count: int = 0
    mean: float = 0.0
    m2: float = 0.0

    def update(self, values: np.ndarray) -> None:
        flat = np.asarray(values, dtype=np.float64).ravel()
        if flat.size == 0:
            return
        batch_count = int(flat.size)
        batch_mean = float(np.mean(flat))
        batch_m2 = float(np.sum((flat - batch_mean) ** 2))
        total = self.count + batch_count
        delta = batch_mean - self.mean
        self.mean += delta * batch_count / total
        self.m2 += batch_m2 + delta * delta * self.count * batch_count / total
        self.count = total

    def population_std(self) -> float:
        if self.count < 1:
            raise ValueError("normalization needs at least one training value")
        return math.sqrt(self.m2 / self.count)


def assign_splits(n_train: int, n_val: int, n_test: int, split_seed: int) -> dict[str, list[int]]:
    """Permute instance ids, then cut train, validation, and test in that order.

    The permutation is Fisher–Yates driven by ``Generator.integers``. It
    does not use ``Generator.shuffle``, so the split does not follow
    NumPy's shuffle implementation.
    """

    _require_count(n_train, label="n_train")
    _require_count(n_val, label="n_val")
    _require_count(n_test, label="n_test")
    _require_count(split_seed, label="split_seed", allow_zero=True)
    total = n_train + n_val + n_test
    rng = np.random.Generator(np.random.PCG64(np.random.SeedSequence([split_seed, SPLIT_SALT])))
    order = np.arange(total, dtype=np.int64)
    for index in range(total - 1, 0, -1):
        swap = int(rng.integers(0, index + 1))
        order[index], order[swap] = order[swap], order[index]
    train = [int(value) for value in order[:n_train]]
    val = [int(value) for value in order[n_train : n_train + n_val]]
    test = [int(value) for value in order[n_train + n_val :]]
    return {"train": train, "val": val, "test": test}


def estimate_storage(config: PilotConfig) -> dict[str, float | int]:
    """Uncompressed ``float64`` field bytes, excluding JSON and coordinates."""

    _validate_config(config)
    n_times = _frame_count(config.t_final, config.save_dt)
    per_trajectory = n_times * config.n * 8
    total = config.total() * per_trajectory
    return {
        "n_instances": config.total(),
        "n_times": n_times,
        "n": config.n,
        "bytes_per_trajectory": per_trajectory,
        "field_bytes": total,
        "field_mebibytes": total / (1024 * 1024),
    }


def generate_pilot(output: Path, config: PilotConfig) -> dict[str, Any]:
    """Write a pilot directory and return the manifest.

    ``output`` is created and must not already exist. Each instance is
    ``{split}/instance_{id:06d}.npz``.
    """

    _validate_config(config)
    destination = Path(output)
    if destination.exists():
        raise FileExistsError(f"pilot output already exists: {destination}")
    splits = assign_splits(config.n_train, config.n_val, config.n_test, config.split_seed)
    split_of = {instance_id: name for name, ids in splits.items() for instance_id in ids}
    destination.mkdir(parents=True)
    for name in ("train", "val", "test"):
        (destination / name).mkdir()
    stats = RunningStats()
    records: list[dict[str, Any]] = []
    solver_config: SolverConfig | None = None
    batch = config.batch_size
    for start in range(0, config.total(), batch):
        ids = list(range(start, min(start + batch, config.total())))
        drawn = [
            draw_initial_condition(
                config.master_seed,
                instance_id,
                config.n,
                n_modes=config.n_modes,
                nu_min=config.nu_min,
                nu_max=config.nu_max,
            )
            for instance_id in ids
        ]
        fields = np.stack([item.u0 for item in drawn], axis=0)
        viscosities = np.asarray([item.nu for item in drawn], dtype=np.float64)
        stacked, times, nodes, solver_config = solve_batch(
            fields,
            viscosities,
            dt=config.dt,
            t_final=config.t_final,
            save_dt=config.save_dt,
        )
        print(f"integrated instances {ids[0]}..{ids[-1]}", flush=True)
        for local, item in enumerate(drawn):
            split = split_of[item.instance_id]
            relative = Path(split) / f"instance_{item.instance_id:06d}.npz"
            path = destination / relative
            payload = {
                "u": stacked[local],
                "x": nodes,
                "t": times,
                "nu": np.float64(item.nu),
                "instance_id": np.int64(item.instance_id),
                "master_seed": np.int64(item.master_seed),
                "coefficients_a": item.a,
                "coefficients_b": item.b,
                "amplitude_scale": np.float64(item.scale),
            }
            np.savez_compressed(path, **payload)
            digest = hashlib.sha256(path.read_bytes()).hexdigest()
            if split == "train":
                stats.update(stacked[local])
            records.append(
                {
                    "instance_id": item.instance_id,
                    "split": split,
                    "path": relative.as_posix(),
                    "sha256": digest,
                    "field_sha256": _field_sha256(stacked[local]),
                    "nu": item.nu,
                    "amplitude_scale": item.scale,
                    "n": config.n,
                    "n_times": int(times.shape[0]),
                }
            )
    if solver_config is None:
        raise RuntimeError("pilot generation produced no trajectories")
    manifest = {
        "format": FORMAT,
        "pilot": config.to_dict(),
        "solver": solver_config.to_dict(),
        "solver_config_sha256": sha256_json(solver_config.to_dict()),
        "pilot_config_sha256": sha256_json(config.to_dict()),
        "code_sha256": source_hashes(),
        "numpy": np.__version__,
        "splits": {name: sorted(ids) for name, ids in splits.items()},
        "split_order_note": (
            "splits lists are sorted for display. Assignment used Fisher-Yates "
            "on instance ids before integration. No windows were extracted."
        ),
        "normalization": {
            "fit_on": "train",
            "applied_to_files": False,
            "u_mean": stats.mean,
            "u_std": stats.population_std(),
            "n_values": stats.count,
            "definition": "population mean and std (ddof=0) of raw training u(x, t)",
        },
        "instances": records,
    }
    manifest_path = destination / "manifest.json"
    manifest_path.write_text(
        json.dumps(manifest, indent=2, sort_keys=True, allow_nan=False) + "\n",
        encoding="utf-8",
    )
    return manifest


def _field_sha256(field: np.ndarray) -> str:
    """SHA-256 of the contiguous float64 field, independent of the zip wrapper."""

    array = np.ascontiguousarray(field, dtype=np.float64)
    header = f"{array.shape[0]},{array.shape[1]}:float64:".encode("ascii")
    return hashlib.sha256(header + array.tobytes()).hexdigest()


def canonical_json(payload: dict[str, Any]) -> str:
    """Stable JSON: sorted keys, no extra whitespace, finite numbers only."""

    return json.dumps(payload, sort_keys=True, separators=(",", ":"), allow_nan=False)


def sha256_json(payload: dict[str, Any]) -> str:
    return hashlib.sha256(canonical_json(payload).encode("utf-8")).hexdigest()


def source_hashes() -> dict[str, str]:
    directory = Path(__file__).resolve().parent
    return {name: hashlib.sha256((directory / name).read_bytes()).hexdigest() for name in _SOURCES}


def _frame_count(t_final: float, save_dt: float) -> int:
    ratio = t_final / save_dt
    count = int(round(ratio))
    if count < 1 or abs(count * save_dt - t_final) > 1e-8 * max(1.0, abs(t_final)):
        raise ValueError("t_final / save_dt must be an integer >= 1")
    return count + 1


def _validate_config(config: PilotConfig) -> None:
    _require_count(config.n_train, label="n_train")
    _require_count(config.n_val, label="n_val")
    _require_count(config.n_test, label="n_test")
    _require_count(config.batch_size, label="batch_size")
    _require_count(config.master_seed, label="master_seed", allow_zero=True)
    _require_count(config.split_seed, label="split_seed", allow_zero=True)
    if isinstance(config.n, bool) or not isinstance(config.n, int) or config.n < 4 or config.n % 2:
        raise ValueError("n must be an even integer >= 4")
    if config.n // 2 <= config.n_modes:
        raise ValueError("n/2 must be greater than n_modes")
    for label, value in (
        ("dt", config.dt),
        ("t_final", config.t_final),
        ("save_dt", config.save_dt),
        ("nu_min", config.nu_min),
        ("nu_max", config.nu_max),
    ):
        if isinstance(value, bool) or not isinstance(value, (int, float)) or not math.isfinite(value):
            raise ValueError(f"{label} must be finite")
    if config.dt <= 0 or config.t_final <= 0 or config.save_dt <= 0:
        raise ValueError("dt, t_final, and save_dt must be > 0")
    if config.nu_min <= 0 or config.nu_max < config.nu_min:
        raise ValueError("viscosity bounds must satisfy 0 < nu_min <= nu_max")


def _require_count(value: int, *, label: str, allow_zero: bool = False) -> None:
    if isinstance(value, bool) or not isinstance(value, (int, np.integer)):
        raise ValueError(f"{label} must be an integer")
    if int(value) < 0 or (int(value) == 0 and not allow_zero):
        raise ValueError(f"{label} must be >= {0 if allow_zero else 1}")
