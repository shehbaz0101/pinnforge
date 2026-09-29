"""Roll out a trained FNO on the Stage D viscosity grid.

Torch is imported here. The sparse window is built in
:mod:`pinnforge.operator.stage_d` from the sensor mask. This module does
not choose ``λ`` and does not read the test split when ``split`` is
``val``.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

import numpy as np

from pinnforge.operator.fno import pack_inputs
from pinnforge.operator.inverse import (
    PREREGISTERED_OBSERVATION,
    observed_residual_terms,
    training_baseline_nu,
)
from pinnforge.operator.stage_d import (
    RUN_FORMAT,
    load_stage_d_protocol,
    protocol_sha256,
    sensor_field,
    viscosity_grid,
    window_from_protocol,
)
from pinnforge.operator.windows import FieldNorm, field_norm_from_manifest, load_pilot_manifest
from pinnforge.reference.numerical.stress import frame_diagnostics

_NORM_ABS = 1e-12


def evaluate_curves(
    model: Any,
    norm: FieldNorm,
    initial_windows: np.ndarray,
    targets: np.ndarray,
    slots: tuple[tuple[int, int, int, int], ...],
    grid: np.ndarray,
    baseline_nu: float,
    oracle_windows: np.ndarray,
    *,
    batch_size: int,
    n_sensors: int,
) -> dict[str, np.ndarray]:
    """Sensor mean squares on the viscosity grid and at the training mean.

    ``initial_windows`` is ``(batch, 8, n_space)`` in physical units.
    ``targets`` is ``(batch, n_targets, length, n_sensors)``. ``slots``
    indexes those targets inside the autoregressive blocks. The oracle
    windows are the true fields for the same targets, one step each.
    """

    from pinnforge.ml_import import require_torch

    require_torch()
    if not isinstance(norm, FieldNorm):
        raise TypeError("norm must be a FieldNorm")
    windows = np.asarray(initial_windows, dtype=np.float64)
    observed = np.asarray(targets, dtype=np.float64)
    oracle = np.asarray(oracle_windows, dtype=np.float64)
    nodes = np.asarray(grid, dtype=np.float64)
    if windows.ndim != 3 or observed.ndim != 4 or oracle.ndim != 4:
        raise ValueError("curve tensors have the wrong rank")
    if windows.shape[0] != observed.shape[0] or windows.shape[0] != oracle.shape[0]:
        raise ValueError("curve batch sizes do not match")
    if oracle.shape[1] != observed.shape[1] or oracle.shape[2] != windows.shape[1]:
        raise ValueError("oracle windows do not match the target bursts")
    if not np.isfinite(windows).all() or not np.isfinite(observed).all() or not np.isfinite(oracle).all():
        raise ValueError("curve inputs must be finite")
    n_steps = max(step for _, step, _, _ in slots) + 1
    model.eval()
    sensor = np.empty((windows.shape[0], nodes.size), dtype=np.float64)
    oracle_sensor = np.empty_like(sensor)
    for index, nu in enumerate(nodes):
        blocks = _rollout_sensors(model, norm, windows, float(nu), n_steps, n_sensors, batch_size)
        sensor[:, index] = _burst_mse(blocks, observed, slots, rollout=True)
        one_step = _onestep_sensors(model, norm, oracle, float(nu), n_sensors, batch_size)
        oracle_sensor[:, index] = _burst_mse(one_step, observed, slots, rollout=False)
        if index % 20 == 0 or index + 1 == nodes.size:
            print(f"grid {index + 1}/{nodes.size} nu={float(nu):.6f}", flush=True)
    anchor_blocks = _rollout_sensors(model, norm, windows, float(baseline_nu), n_steps, n_sensors, batch_size)
    anchor = _burst_mse(anchor_blocks, observed, slots, rollout=True)
    oracle_anchor_blocks = _onestep_sensors(model, norm, oracle, float(baseline_nu), n_sensors, batch_size)
    oracle_anchor = _burst_mse(oracle_anchor_blocks, observed, slots, rollout=False)
    if np.any(anchor <= 0.0) or np.any(oracle_anchor <= 0.0):
        raise ValueError("baseline sensor mismatch must be > 0 so lambda can be normalized")
    return {
        "sensor_mse": sensor,
        "sensor_mse_at_baseline": anchor,
        "oracle_sensor_mse": oracle_sensor,
        "oracle_sensor_mse_at_baseline": oracle_anchor,
    }


def run_operator_inverse(
    *,
    protocol_path: Path,
    pilot_dir: Path,
    manifest_path: Path,
    checkpoint: Path,
    arm: str,
    split: str,
) -> dict[str, Any]:
    """Curve file for one checkpoint. ``λ`` is not applied here."""

    from pinnforge.operator.checkpoint import load_fno_checkpoint
    from pinnforge.operator.data import load_split_trajectories
    from pinnforge.operator.stage_d import prepare_oracle_windows, prepare_sparse_batch

    if split not in ("val", "test"):
        raise ValueError(f"unknown split {split!r}")
    if arm not in ("data_only", "hybrid_1e-2"):
        raise ValueError(f"unknown arm {arm!r}")
    source = Path(protocol_path)
    protocol = load_stage_d_protocol(source)
    loaded = load_fno_checkpoint(Path(checkpoint))
    _require_checkpoint(loaded, protocol, arm)
    manifest = load_pilot_manifest(Path(manifest_path))
    norm = field_norm_from_manifest(manifest)
    _require_same_norm(loaded.norm, norm)
    trajectories = load_split_trajectories(
        Path(pilot_dir),
        Path(manifest_path),
        (split,),
        check_field_hash=True,
    )
    window = window_from_protocol(protocol)
    prepared = prepare_sparse_batch(trajectories, PREREGISTERED_OBSERVATION, window)
    oracle = prepare_oracle_windows(trajectories, PREREGISTERED_OBSERVATION, window)
    grid = viscosity_grid(protocol)
    baseline = training_baseline_nu(Path(manifest_path))
    curves = evaluate_curves(
        loaded.model,
        loaded.norm,
        prepared["windows"],
        prepared["targets"],
        prepared["slots"],
        grid,
        baseline,
        oracle,
        batch_size=int(protocol["inference_batch_size"]),
        n_sensors=int(PREREGISTERED_OBSERVATION.n_sensors),
    )
    instances = []
    for index, item in enumerate(trajectories):
        mask = prepared["masks"][index]
        advection, diffusion = observed_residual_terms(
            sensor_field(mask, PREREGISTERED_OBSERVATION, int(item.u.shape[0])),
            PREREGISTERED_OBSERVATION,
        )
        alias = frame_diagnostics(
            item.u,
            frame_dt=PREREGISTERED_OBSERVATION.frame_dt,
            n_sensors=int(PREREGISTERED_OBSERVATION.n_sensors or 0),
        )["alias_rel_l2"]
        instances.append(
            {
                "instance_id": item.instance_id,
                "nu": item.nu,
                "alias_rel_l2": alias,
                "sensor_mse": curves["sensor_mse"][index],
                "sensor_mse_at_baseline": curves["sensor_mse_at_baseline"][index],
                "residual_mse": _residual_grid(advection, diffusion, grid),
                "residual_mse_at_baseline": _residual_grid(advection, diffusion, np.asarray([baseline]))[0],
                "oracle_sensor_mse": curves["oracle_sensor_mse"][index],
                "oracle_sensor_mse_at_baseline": curves["oracle_sensor_mse_at_baseline"][index],
                "advection": advection,
                "diffusion": diffusion,
            }
        )
    return {
        "format": RUN_FORMAT,
        "split": split,
        "arm": arm,
        "seed": loaded.seed,
        "selected_epoch": loaded.epoch,
        "val_relative_l2": loaded.val_relative_l2,
        "loss_mode": loaded.loss.mode,
        "residual_weight": loaded.loss.residual_weight,
        "checkpoint": Path(checkpoint).as_posix(),
        "baseline_nu": baseline,
        "grid": grid,
        "protocol_sha256": protocol_sha256(source),
        "test_used_for_curves": split == "test",
        "reads_unmasked_field_for_sparse_objective": False,
        "instances": instances,
    }


def _rollout_sensors(
    model: Any,
    norm: FieldNorm,
    windows: np.ndarray,
    nu: float,
    n_steps: int,
    n_sensors: int,
    batch_size: int,
) -> list[np.ndarray]:
    import torch

    stored: list[list[np.ndarray]] = [[] for _ in range(n_steps)]
    stride = int(windows.shape[-1]) // n_sensors
    for start in range(0, int(windows.shape[0]), batch_size):
        stop = min(start + batch_size, int(windows.shape[0]))
        current = torch.as_tensor(norm.normalize_u(windows[start:stop]), dtype=torch.float32)
        channel = torch.as_tensor(
            norm.normalize_nu(np.full(stop - start, nu, dtype=np.float64)),
            dtype=torch.float32,
        )
        with torch.no_grad():
            for step in range(n_steps):
                predicted = model(pack_inputs(current, channel))
                physical = norm.denormalize_u(predicted.detach().cpu().numpy())
                stored[step].append(np.ascontiguousarray(physical[:, :, ::stride]))
                current = predicted
    return [np.concatenate(chunks, axis=0) for chunks in stored]


def _onestep_sensors(
    model: Any,
    norm: FieldNorm,
    oracle_windows: np.ndarray,
    nu: float,
    n_sensors: int,
    batch_size: int,
) -> np.ndarray:
    import torch

    batch, n_slot, frames, n_space = oracle_windows.shape
    flat = np.reshape(oracle_windows, (batch * n_slot, frames, n_space))
    stride = n_space // n_sensors
    pieces: list[np.ndarray] = []
    for start in range(0, flat.shape[0], batch_size):
        stop = min(start + batch_size, flat.shape[0])
        current = torch.as_tensor(norm.normalize_u(flat[start:stop]), dtype=torch.float32)
        channel = torch.as_tensor(
            norm.normalize_nu(np.full(stop - start, nu, dtype=np.float64)),
            dtype=torch.float32,
        )
        with torch.no_grad():
            predicted = model(pack_inputs(current, channel)).detach().cpu().numpy()
        physical = norm.denormalize_u(predicted)
        pieces.append(np.ascontiguousarray(physical[:, :, ::stride]))
    stacked = np.concatenate(pieces, axis=0)
    return np.reshape(stacked, (batch, n_slot, frames, n_sensors))


def _burst_mse(
    blocks: list[np.ndarray] | np.ndarray,
    targets: np.ndarray,
    slots: tuple[tuple[int, int, int, int], ...],
    *,
    rollout: bool,
) -> np.ndarray:
    total = None
    count = 0
    for target_index, step, local, length in slots:
        if rollout:
            predicted = blocks[step][:, local : local + length, :]  # type: ignore[index]
        else:
            predicted = blocks[:, target_index, local : local + length, :]  # type: ignore[index]
        observed = targets[:, target_index]
        if predicted.shape != observed.shape:
            raise ValueError("predicted sensors do not match the mask")
        square = np.sum((predicted - observed) ** 2, axis=(1, 2))
        total = square if total is None else total + square
        count += int(predicted.shape[1] * predicted.shape[2])
    if total is None or count < 1:
        raise ValueError("sensor objective has no nodes")
    return total / count


def _residual_grid(advection: np.ndarray, diffusion: np.ndarray, grid: np.ndarray) -> np.ndarray:
    residual = advection.reshape(1, -1) - grid.reshape(-1, 1) * diffusion.reshape(1, -1)
    return np.mean(residual * residual, axis=1)


def _require_checkpoint(loaded: Any, protocol: dict[str, Any], arm: str) -> None:
    seeds = [int(seed) for seed in protocol["arms"][arm]["seeds"]]
    if int(loaded.seed) not in seeds:
        raise ValueError(f"checkpoint seed {loaded.seed} is not in the frozen arm")
    window = window_from_protocol(protocol)
    if loaded.spec != window:
        raise ValueError("checkpoint window does not match the stage D protocol")
    if arm == "data_only":
        if loaded.loss.mode != "data":
            raise ValueError("data_only checkpoint must use the data loss")
        return
    if loaded.loss.mode != "hybrid" or abs(float(loaded.loss.residual_weight) - 0.01) > 1e-12:
        raise ValueError("hybrid checkpoint must use residual weight 1e-2")


def _require_same_norm(left: FieldNorm, right: FieldNorm) -> None:
    for label, first, second in (
        ("u_mean", left.u_mean, right.u_mean),
        ("u_std", left.u_std, right.u_std),
        ("nu_mean", left.nu_mean, right.nu_mean),
        ("nu_std", left.nu_std, right.nu_std),
    ):
        if abs(first - second) > _NORM_ABS:
            raise ValueError(f"checkpoint {label} does not match the manifest")
