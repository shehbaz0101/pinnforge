"""FNO log-bracket, WSINDy clip, and split-conformal intervals."""

from __future__ import annotations

import math

import numpy as np


def conformal_quantile(scores: np.ndarray, alpha: float) -> float:
    values = np.sort(np.asarray(scores, dtype=np.float64).reshape(-1))
    n = int(values.size)
    if n < 1:
        return float("inf")
    rank = int(math.ceil((n + 1) * (1.0 - float(alpha))))
    if rank >= n:
        return float(values[-1])
    return float(values[max(0, rank - 1)])


def interval(hat: float, q_log: float) -> tuple[float, float]:
    if hat <= 0.0 or not math.isfinite(hat) or not math.isfinite(q_log):
        return (float("nan"), float("nan"))
    return (float(hat) * math.exp(-float(q_log)), float(hat) * math.exp(float(q_log)))


def clip_to_bracket(value: float, low: float, high: float) -> float:
    if not math.isfinite(value):
        return float(low) if math.isfinite(low) else float("nan")
    return float(min(max(value, low), high))


def covered(truth: float, low: float, high: float) -> bool:
    return bool(math.isfinite(truth) and math.isfinite(low) and math.isfinite(high) and low <= truth <= high)
