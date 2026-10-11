"""Operator first guesses and timed warm-start fits.

Torch is imported here. One instance at a time, the Stage D sparse
rollout is evaluated for every viscosity on a grid in a single batch, and
the grid argmin is the first guess. Ties keep the smaller viscosity. The
solver refinement and the cold-start baseline then run in the same worker
process, so their wall clocks share one machine and one load.
"""

from __future__ import annotations

import math
import os
import time
from collections.abc import Mapping, Sequence
from pathlib import Path
from typing import Any

import numpy as np

from pinnforge.operator.fno import pack_inputs
from pinnforge.operator.stage_d import initial_window_from_mask, read_mask, target_slots
from pinnforge.operator.windows import FieldNorm, WindowSpec

WINDOW = WindowSpec(input_frames=8, output_frames=8, stride=8)


def first_guess_grid(kind: str, n_points: int, nu_min: float, nu_max: float) -> np.ndarray:
    """``linear`` is the Stage D grid when ``n_points`` is 191. ``geometric`` is log-spaced."""

    count = int(n_points)
    if count < 3:
        raise ValueError("a first-guess grid needs at least three points")
    low = float(nu_min)
    high = float(nu_max)
    if not 0.0 < low < high:
        raise ValueError("grid bounds must satisfy 0 < nu_min < nu_max")
    if kind == "linear":
        grid = np.linspace(low, high, count)
    elif kind == "geometric":
        grid = np.geomspace(low, high, count)
        grid[0] = low
        grid[-1] = high
    else:
        raise ValueError("grid kind must be linear or geometric")
    return np.asarray(grid, dtype=np.float64)


def sensor_curve(
    model: Any,
    norm: FieldNorm,
    window: np.ndarray,
    targets: np.ndarray,
    slots: Sequence[tuple[int, int, int, int]],
    grid: np.ndarray,
    n_sensors: int,
) -> np.ndarray:
    """Sensor mean square of the rollout at every grid viscosity, one batch.

    ``window`` is the physical eight-frame window ``(8, n_space)``.
    ``targets`` is ``(n_targets, length, n_sensors)``.
    """

    import torch

    frames = np.asarray(window, dtype=np.float64)
    observed = np.asarray(targets, dtype=np.float64)
    nodes = np.asarray(grid, dtype=np.float64)
    if frames.ndim != 2 or observed.ndim != 3 or nodes.ndim != 1:
        raise ValueError("sensor curve inputs have the wrong rank")
    n_steps = max(step for _, step, _, _ in slots) + 1
    stride = int(frames.shape[-1]) // int(n_sensors)
    batch = int(nodes.size)
    current = torch.as_tensor(norm.normalize_u(frames), dtype=torch.float32).unsqueeze(0).expand(batch, -1, -1)
    channel = torch.as_tensor(norm.normalize_nu(nodes), dtype=torch.float32)
    stored: list[np.ndarray] = []
    model.eval()
    with torch.no_grad():
        for _ in range(n_steps):
            predicted = model(pack_inputs(current.contiguous(), channel))
            sensors = predicted[:, :, ::stride].detach().cpu().numpy().astype(np.float64)
            stored.append(norm.denormalize_u(sensors))
            current = predicted
    total = np.zeros(batch, dtype=np.float64)
    count = 0
    for target_index, step, local, length in slots:
        piece = stored[step][:, local : local + length, :]
        if piece.shape[1:] != observed[target_index].shape:
            raise ValueError("predicted sensors do not match the mask")
        total += np.sum((piece - observed[target_index][None]) ** 2, axis=(1, 2))
        count += int(piece.shape[1] * piece.shape[2])
    return total / count


def operator_first_guess(
    model: Any,
    norm: FieldNorm,
    field: np.ndarray,
    spec: Any,
    grid: np.ndarray,
) -> dict[str, Any]:
    """Grid argmin of the Stage D sparse objective for one stored field.

    Only the mask is read from ``field``. The wall clock covers the window
    construction and every rollout.
    """

    started = time.perf_counter()
    mask = read_mask(field, spec)
    window = initial_window_from_mask(mask, int(np.asarray(field).shape[1]))
    raw_slots = target_slots(spec, WINDOW)
    targets = np.stack([mask[series] for series, _, _, _ in raw_slots], axis=0)
    slots = tuple((index, step, local, length) for index, (_, step, local, length) in enumerate(raw_slots))
    curve = sensor_curve(model, norm, window, targets, slots, grid, int(spec.n_sensors))
    if not np.isfinite(curve).all():
        raise ValueError("operator sensor curve is not finite")
    index = int(np.argmin(curve))
    return {
        "nu_hat": float(grid[index]),
        "grid_index": index,
        "curve_min": float(curve[index]),
        "seconds": float(time.perf_counter() - started),
    }


def load_models(checkpoints: Mapping[str, str]) -> dict[str, Any]:
    """Load each named checkpoint once. Keys are ``arm/seed`` labels."""

    from pinnforge.operator.checkpoint import load_fno_checkpoint

    loaded = {}
    for key, path in checkpoints.items():
        item = load_fno_checkpoint(Path(path))
        if item.spec != WINDOW:
            raise ValueError(f"{key} window does not match the Stage D window")
        loaded[key] = item
    return loaded


def single_thread() -> None:
    os.environ["OMP_NUM_THREADS"] = "1"
    os.environ["OPENBLAS_NUM_THREADS"] = "1"
    os.environ["MKL_NUM_THREADS"] = "1"
    import torch

    torch.set_num_threads(1)


def log_std(values: Sequence[float]) -> float:
    """Sample standard deviation of log values (``ddof = 1``)."""

    logs = np.log(np.asarray(list(values), dtype=np.float64))
    if logs.size < 2:
        return math.nan
    return float(np.std(logs, ddof=1))
