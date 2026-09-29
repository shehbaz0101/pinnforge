"""Autoregressive rollout on saved Burgers frames.

The first input window is the true field. Each predicted block is then
fed back as the next input. This requires the window stride to equal the
output length, so the handoff is exactly one predicted block. The
one-step window metric is a different score: later windows there still
see the true field.

The open-loop persistence baseline holds the last frame of that true
initial window for the whole predicted horizon. It is not trained.

Neither score chooses the checkpoint.
"""

from __future__ import annotations

import math
from collections.abc import Callable, Mapping, Sequence

import numpy as np

from pinnforge.operator.windows import WindowSpec, window_starts
from pinnforge.reference.numerical.checks import relative_l2

PredictBatch = Callable[[np.ndarray, np.ndarray], np.ndarray]


def rollout_trajectories(
    fields: Sequence[np.ndarray],
    instance_ids: Sequence[int],
    nu: Sequence[float],
    spec: WindowSpec,
    predict_batch: PredictBatch,
    *,
    batch_size: int,
) -> dict[str, np.ndarray]:
    """Roll every trajectory forward from its true initial window.

    ``fields[i]`` has shape ``(n_times, n_space)`` in physical units.
    ``predict_batch`` takes physical windows ``(batch, input_frames, n_space)``
    and physical viscosities ``(batch,)`` and returns the next physical
    block ``(batch, output_frames, n_space)``.
    """

    if not isinstance(spec, WindowSpec):
        raise TypeError("spec must be a WindowSpec")
    if spec.stride != spec.output_frames:
        raise ValueError("autoregressive rollout requires stride == output_frames")
    if isinstance(batch_size, bool) or not isinstance(batch_size, int) or batch_size < 1:
        raise ValueError("batch_size must be an integer >= 1")
    if len(fields) < 1 or len(fields) != len(instance_ids) or len(fields) != len(nu):
        raise ValueError("fields, instance_ids, and nu must be the same non-empty length")
    order = sorted(range(len(fields)), key=lambda index: int(instance_ids[index]))
    ordered_fields = [np.asarray(fields[index], dtype=np.float64) for index in order]
    ordered_ids = np.asarray([int(instance_ids[index]) for index in order], dtype=np.int64)
    ordered_nu = np.asarray([float(nu[index]) for index in order], dtype=np.float64)
    if len(set(int(value) for value in ordered_ids)) != len(ordered_ids):
        raise ValueError("rollout instance ids must be unique")
    if not np.isfinite(ordered_nu).all() or np.any(ordered_nu <= 0.0):
        raise ValueError("rollout viscosities must be finite and > 0")
    n_times, n_space = _shared_shape(ordered_fields)
    starts = window_starts(n_times, spec)
    if not starts:
        raise ValueError("trajectory is shorter than one rollout window")
    current = np.stack([field[0 : spec.input_frames] for field in ordered_fields], axis=0)
    step_scores = np.empty((len(ordered_fields), len(starts)), dtype=np.float64)
    predicted_blocks: list[np.ndarray] = []
    truth_blocks: list[np.ndarray] = []
    for step, start in enumerate(starts):
        truth = np.stack(
            [field[start + spec.input_frames : start + spec.span()] for field in ordered_fields],
            axis=0,
        )
        predicted = _predict_in_batches(predict_batch, current, ordered_nu, batch_size)
        if predicted.shape != truth.shape or not np.isfinite(predicted).all():
            raise ValueError("rollout prediction must be finite and match the truth block")
        for index in range(predicted.shape[0]):
            step_scores[index, step] = relative_l2(predicted[index], truth[index])
        predicted_blocks.append(predicted)
        truth_blocks.append(truth)
        current = predicted
    predicted_horizon = np.concatenate(predicted_blocks, axis=1)
    truth_horizon = np.concatenate(truth_blocks, axis=1)
    instance_scores = np.empty(len(ordered_fields), dtype=np.float64)
    persistence_scores = np.empty(len(ordered_fields), dtype=np.float64)
    for index, field in enumerate(ordered_fields):
        instance_scores[index] = relative_l2(predicted_horizon[index], truth_horizon[index])
        hold = field[spec.input_frames - 1]
        repeated = np.repeat(hold[None, :], truth_horizon.shape[1], axis=0)
        persistence_scores[index] = relative_l2(repeated, truth_horizon[index])
    return {
        "instance_ids": ordered_ids,
        "nu": ordered_nu,
        "instance_relative_l2": instance_scores,
        "persistence_instance_relative_l2": persistence_scores,
        "step_relative_l2": step_scores,
        "first_prediction": predicted_blocks[0],
        "n_space": np.asarray(n_space),
    }


def summarize_rollout(
    rollout: Mapping[str, np.ndarray],
    threshold: float,
    *,
    worst_count: int = 5,
) -> dict[str, dict[str, object]]:
    """Reduce a rollout onto full test, ``hard_ood``, and the complement."""

    if isinstance(threshold, bool) or not isinstance(threshold, (int, float)):
        raise ValueError("threshold_nu must be a finite number > 0")
    if not math.isfinite(float(threshold)) or float(threshold) <= 0.0:
        raise ValueError("threshold_nu must be a finite number > 0")
    nu = np.asarray(rollout["nu"], dtype=np.float64)
    masks = {
        "full_test": np.ones(nu.shape[0], dtype=bool),
        "hard_ood": nu <= float(threshold),
        "complement": nu > float(threshold),
    }
    if not np.any(masks["hard_ood"]) or not np.any(masks["complement"]):
        raise ValueError("rollout slices must both be non-empty")
    summary: dict[str, dict[str, object]] = {}
    instance_scores = np.asarray(rollout["instance_relative_l2"], dtype=np.float64)
    persistence = np.asarray(rollout["persistence_instance_relative_l2"], dtype=np.float64)
    steps = np.asarray(rollout["step_relative_l2"], dtype=np.float64)
    ids = np.asarray(rollout["instance_ids"], dtype=np.int64)
    for name, mask in masks.items():
        chosen_scores = instance_scores[mask]
        chosen_steps = steps[mask]
        summary[name] = {
            "n_instances": int(np.count_nonzero(mask)),
            "n_windows": int(np.count_nonzero(mask) * steps.shape[1]),
            "n_steps": int(steps.shape[1]),
            "mean_instance_relative_l2": float(np.mean(chosen_scores)),
            "median_instance_relative_l2": float(np.median(chosen_scores)),
            "persistence_mean_instance_relative_l2": float(np.mean(persistence[mask])),
            "per_step_mean_relative_l2": [float(value) for value in np.mean(chosen_steps, axis=0)],
            "worst": _worst_instances(
                ids[mask],
                nu[mask],
                chosen_scores,
                persistence[mask],
                worst_count,
            ),
        }
    return summary


def _predict_in_batches(
    predict_batch: PredictBatch,
    windows: np.ndarray,
    nu: np.ndarray,
    batch_size: int,
) -> np.ndarray:
    blocks: list[np.ndarray] = []
    for start in range(0, int(nu.shape[0]), batch_size):
        stop = min(start + batch_size, int(nu.shape[0]))
        predicted = np.asarray(predict_batch(windows[start:stop], nu[start:stop]), dtype=np.float64)
        if predicted.shape[0] != stop - start:
            raise ValueError("rollout predict_batch returned the wrong batch count")
        blocks.append(predicted)
    return np.concatenate(blocks, axis=0)


def _shared_shape(fields: Sequence[np.ndarray]) -> tuple[int, int]:
    first = fields[0]
    if first.ndim != 2 or first.shape[0] < 1 or first.shape[1] < 1:
        raise ValueError("each field must have shape (n_times, n_space)")
    if not np.isfinite(first).all():
        raise ValueError("rollout fields must be finite")
    for field in fields[1:]:
        if field.shape != first.shape or not np.isfinite(field).all():
            raise ValueError("rollout fields must share a finite shape")
    return int(first.shape[0]), int(first.shape[1])


def _worst_instances(
    instance_ids: np.ndarray,
    nu: np.ndarray,
    scores: np.ndarray,
    persistence: np.ndarray,
    worst_count: int,
) -> list[dict[str, object]]:
    rows = [
        {
            "instance_id": int(instance_ids[index]),
            "nu": float(nu[index]),
            "relative_l2": float(scores[index]),
            "persistence_relative_l2": float(persistence[index]),
        }
        for index in range(instance_ids.shape[0])
    ]
    rows.sort(key=lambda row: (-float(row["relative_l2"]), int(row["instance_id"])))
    return rows[:worst_count]
