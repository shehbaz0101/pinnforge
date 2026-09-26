"""Relative L2 on Burgers windows, in the stored field units.

Training loss is mean squared error in normalized space and is not
computed here. These scores denormalize first. Relative L2 matches
:func:`pinnforge.reference.numerical.checks.relative_l2`: the Euclidean
norm over the flattened window, with no extra grid weight. The same nodes
are used for the prediction and the target, so a ``dx`` factor would cancel.
"""

from __future__ import annotations

import math

import numpy as np

from pinnforge.operator.windows import WindowDataset
from pinnforge.reference.numerical.checks import relative_l2


def per_window_relative_l2(candidate: np.ndarray, reference: np.ndarray) -> np.ndarray:
    """Relative L2 of each window.

    Both arrays have shape ``(n_windows, frames, n_space)`` and are in
    physical units. A window whose reference norm is zero raises, because
    the ratio is undefined.
    """

    got = np.asarray(candidate, dtype=np.float64)
    truth = np.asarray(reference, dtype=np.float64)
    if got.shape != truth.shape or got.ndim != 3:
        raise ValueError(f"window arrays must share shape (n, frames, n_space); got {got.shape} vs {truth.shape}")
    residual = (got - truth).reshape(got.shape[0], -1)
    target = truth.reshape(truth.shape[0], -1)
    numerator = np.linalg.norm(residual, axis=1)
    denominator = np.linalg.norm(target, axis=1)
    if np.any(denominator == 0.0) or not np.isfinite(denominator).all():
        raise ValueError("each reference window must have a finite non-zero norm")
    scores = numerator / denominator
    if not np.isfinite(scores).all():
        raise ValueError("relative L2 is not finite")
    return scores


def mean_relative_l2(candidate: np.ndarray, reference: np.ndarray) -> float:
    """Mean of the per-window relative L2 scores."""

    scores = per_window_relative_l2(candidate, reference)
    value = float(np.mean(scores))
    if not math.isfinite(value):
        raise ValueError("mean relative L2 is not finite")
    return value


def median_relative_l2(candidate: np.ndarray, reference: np.ndarray) -> float:
    """Median of the per-window relative L2 scores."""

    scores = per_window_relative_l2(candidate, reference)
    value = float(np.median(scores))
    if not math.isfinite(value):
        raise ValueError("median relative L2 is not finite")
    return value


def pooled_relative_l2(candidate: np.ndarray, reference: np.ndarray) -> float:
    """One relative L2 over every window stacked together."""

    return relative_l2(candidate, reference)


def physical_targets(dataset: WindowDataset) -> np.ndarray:
    """Denormalized target windows."""

    return dataset.norm.denormalize_u(dataset.targets)


def physical_inputs(dataset: WindowDataset) -> np.ndarray:
    """Denormalized input windows."""

    return dataset.norm.denormalize_u(dataset.inputs)


def persistence_prediction(dataset: WindowDataset) -> np.ndarray:
    """Repeat the last input frame across the target horizon.

    This is a data baseline, not a trained model. The repeated frame is
    the last frame of the input window in physical units.
    """

    inputs = physical_inputs(dataset)
    last = inputs[:, -1:, :]
    return np.repeat(last, dataset.spec.output_frames, axis=1)


def normalized_mse(candidate: np.ndarray, reference: np.ndarray) -> float:
    """Mean squared error. Arrays are already in the space being scored."""

    got = np.asarray(candidate, dtype=np.float64)
    truth = np.asarray(reference, dtype=np.float64)
    if got.shape != truth.shape:
        raise ValueError(f"shape mismatch: {got.shape} vs {truth.shape}")
    value = float(np.mean((got - truth) ** 2))
    if not math.isfinite(value):
        raise ValueError("mean squared error is not finite")
    return value
