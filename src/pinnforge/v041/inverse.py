"""FNO-located conformal bracket and a short local solver search."""

from __future__ import annotations

import math
import time
from typing import Any

import numpy as np

from pinnforge.v04.inverse import solve_forward
from pinnforge.v04.sensors import SensorMask, lift_sensors, read_sensors, target_frames


def bracket_bounds(nu_fno: float, q_log: float, nu_min: float, nu_max: float) -> tuple[float, float]:
    guess = float(nu_fno)
    width = float(q_log)
    if guess <= 0.0 or not math.isfinite(guess):
        return (float(nu_min), float(nu_max))
    if not math.isfinite(width) or width < 0.0:
        return (float(nu_min), float(nu_max))
    low = max(float(nu_min), guess * math.exp(-width))
    high = min(float(nu_max), guess * math.exp(width))
    if high <= low:
        high = min(float(nu_max), low * 1.01)
    return (low, high)


def bracket_search(
    field: np.ndarray,
    mask: SensorMask,
    *,
    first_guess: float,
    q_log: float,
    n_local: int,
    problem: str,
    protocol: dict[str, Any],
    forcing: str,
) -> dict[str, Any]:
    block = protocol["methods"]["warm_start"]["bracket"]
    time_spec = protocol["time"]
    length = float(protocol["domain"]["length"])
    sensors = read_sensors(field, mask)
    u0 = lift_sensors(sensors[:1], mask)[0]
    if problem == "ns2d" and u0.ndim == 2:
        u0 = u0[None]
    targets = target_frames(mask)
    low, high = bracket_bounds(first_guess, q_log, float(block["nu_min"]), float(block["nu_max"]))
    grid = np.linspace(low, high, int(n_local))
    start = time.perf_counter()
    losses: dict[float, float] = {}
    for nu in grid:
        pred = solve_forward(
            u0,
            float(nu),
            problem=problem,
            dt=float(time_spec["dt"]),
            t_final=float(time_spec["t_final"]),
            save_dt=float(time_spec["save_dt"]),
            forcing=forcing,
            length=length,
        )
        err = (read_sensors(pred, mask)[targets] - sensors[targets]).ravel()
        losses[float(nu)] = float(np.mean(err**2))
    best_nu = min(losses, key=lambda nu: (losses[nu], nu))
    log_half = 0.5 * math.log(high / low) if high > 0 and low > 0 else math.inf
    return {
        "nu_hat": float(best_nu),
        "loss": float(losses[best_nu]),
        "seconds": time.perf_counter() - start,
        "n_evaluations": int(n_local),
        "bracket_low": low,
        "bracket_high": high,
        "bracket_log_half_width": log_half,
        "q_log": float(q_log),
        "nu_guess": float(first_guess),
    }
