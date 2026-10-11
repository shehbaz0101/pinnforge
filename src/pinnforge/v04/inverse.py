"""FNO-conditioned and solver-in-the-loop viscosity fits for v0.4."""

from __future__ import annotations

import math
import time
from typing import Any, Callable

import numpy as np

from pinnforge.operator.solver_fit import SolverSearch
from pinnforge.operator.warmstart import RefineConfig, refine_viscosity
from pinnforge.reference.spectral2d.burgers import solve_burgers2d
from pinnforge.reference.spectral2d.navier_stokes import solve_ns2d
from pinnforge.v04.sensors import SensorMask, lift_sensors, read_sensors, target_frames
from pinnforge.v04.train import load_model, predict_window

GOLDEN = (math.sqrt(5.0) - 1.0) / 2.0


def geometric_grid(nu_min: float, nu_max: float, n_points: int) -> np.ndarray:
    return np.geomspace(nu_min, nu_max, n_points)


def fno_inverse(
    field: np.ndarray,
    mask: SensorMask,
    *,
    model_path: Any,
    protocol: dict[str, Any],
    problem: str,
) -> dict[str, Any]:
    model, config = load_model(model_path)
    spec = protocol["methods"]["fno_inverse"]
    grid = geometric_grid(float(spec["nu_min"]), float(spec["nu_max"]), int(spec["n_points"]))
    input_frames = int(protocol["fno"]["input_frames"])
    sensors = read_sensors(field, mask)
    lifted = lift_sensors(sensors, mask)
    opening = lifted[:input_frames]
    targets = target_frames(mask)
    observed = sensors
    start = time.perf_counter()
    best_nu = float(grid[0])
    best_loss = math.inf
    for nu in grid:
        pred = _rollout_fno(model, config, opening, float(nu), field.shape[0], protocol)
        loss = _sensor_loss(pred, observed, mask, targets)
        if loss < best_loss - 1e-18 or (abs(loss - best_loss) <= 1e-18 and float(nu) < best_nu):
            best_loss = loss
            best_nu = float(nu)
    return {
        "nu_hat": best_nu,
        "loss": best_loss,
        "seconds": time.perf_counter() - start,
        "n_grid": int(grid.size),
    }


def _rollout_fno(model, config, opening, nu, n_times, protocol) -> np.ndarray:
    input_frames = int(protocol["fno"]["input_frames"])
    output_frames = int(protocol["fno"]["output_frames"])
    frames = [opening[i] for i in range(input_frames)]
    while len(frames) < n_times:
        window = np.stack(frames[-input_frames :], axis=0)
        pred = predict_window(model, window, nu, config)
        for item in pred:
            frames.append(item)
            if len(frames) >= n_times:
                break
        if output_frames <= 0:
            break
    return np.stack(frames[:n_times], axis=0)


def _sensor_loss(pred: np.ndarray, observed: np.ndarray, mask: SensorMask, targets: list[int]) -> float:
    pred_s = read_sensors(pred, mask)
    err = pred_s[targets] - observed[targets]
    return float(np.mean(err**2))


def solve_forward(
    u0: np.ndarray,
    nu: float,
    *,
    problem: str,
    dt: float,
    t_final: float,
    save_dt: float,
    forcing: str,
    length: float,
) -> np.ndarray:
    if problem == "burgers2d":
        return solve_burgers2d(u0, nu, dt=dt, t_final=t_final, save_dt=save_dt, length=length).u
    omega = solve_ns2d(
        u0[0],
        nu,
        dt=dt,
        t_final=t_final,
        save_dt=save_dt,
        forcing="kolmogorov" if forcing == "kolmogorov" else "decaying",
        length=length,
    ).omega
    return omega[:, None]


def solver_fit(
    field: np.ndarray,
    mask: SensorMask,
    *,
    problem: str,
    protocol: dict[str, Any],
    forcing: str,
    initial: str,
    n: int | None = None,
    dt: float | None = None,
) -> dict[str, Any]:
    search_block = protocol["methods"]["solver_sensor_ic"]["search"]
    time_spec = protocol["time"]
    length = float(protocol["domain"]["length"])
    solver_n = int(protocol["domain"]["n"] if n is None else n)
    solver_dt = float(time_spec["dt"] if dt is None else dt)
    t_final = float(time_spec["t_final"])
    save_dt = float(time_spec["save_dt"])
    sensors = read_sensors(field, mask)
    if initial == "oracle":
        u0 = field[0]
    else:
        u0 = lift_sensors(sensors[:1], mask)[0]
    if problem == "ns2d" and u0.ndim == 2:
        u0 = u0[None]
    targets = target_frames(mask)
    search = SolverSearch(
        nu_min=float(search_block["nu_min"]),
        nu_max=float(search_block["nu_max"]),
        n_coarse=int(search_block["n_coarse"]),
        refine_width=float(search_block["refine_width"]),
        max_refine=int(search_block["max_refine"]),
        n=solver_n,
        dt=solver_dt,
        save_dt=save_dt,
    )

    def residual(nu: float) -> np.ndarray:
        pred = solve_forward(
            u0,
            nu,
            problem=problem,
            dt=solver_dt,
            t_final=t_final,
            save_dt=save_dt,
            forcing=forcing,
            length=length,
        )
        return (read_sensors(pred, mask)[targets] - sensors[targets]).ravel()

    start = time.perf_counter()
    nu_hat, loss, queries = _golden_search(residual, search)
    return {
        "nu_hat": nu_hat,
        "loss": loss,
        "seconds": time.perf_counter() - start,
        "n_evaluations": len(queries),
        "queries": queries,
        "initial": initial,
    }


def warm_refine(
    field: np.ndarray,
    mask: SensorMask,
    *,
    first_guess: float,
    problem: str,
    protocol: dict[str, Any],
    forcing: str,
    max_evaluations: int,
) -> dict[str, Any]:
    refine_block = protocol["methods"]["warm_start"]["refine"]
    time_spec = protocol["time"]
    length = float(protocol["domain"]["length"])
    sensors = read_sensors(field, mask)
    u0 = lift_sensors(sensors[:1], mask)[0]
    if problem == "ns2d" and u0.ndim == 2:
        u0 = u0[None]
    targets = target_frames(mask)

    def residual(nu: float) -> np.ndarray:
        pred = solve_forward(
            u0,
            nu,
            problem=problem,
            dt=float(time_spec["dt"]),
            t_final=float(time_spec["t_final"]),
            save_dt=float(time_spec["save_dt"]),
            forcing=forcing,
            length=length,
        )
        return (read_sensors(pred, mask)[targets] - sensors[targets]).ravel()

    config = RefineConfig(
        max_evaluations=int(max_evaluations),
        probe_step=float(refine_block["probe_step"]),
        trust_radius=float(refine_block["trust_radius"]),
        step_tolerance=float(refine_block["step_tolerance"]),
        nu_min=float(refine_block["nu_min"]),
        nu_max=float(refine_block["nu_max"]),
        return_mode=str(refine_block["return_mode"]),
    )
    start = time.perf_counter()
    result = refine_viscosity(residual, float(first_guess), config)
    payload = result.to_dict()
    payload["seconds"] = time.perf_counter() - start
    return payload


def _golden_search(residual: Callable[[float], np.ndarray], search: SolverSearch) -> tuple[float, float, list[float]]:
    queries: list[float] = []
    losses: dict[float, float] = {}

    def loss_at(nu: float) -> float:
        key = round(float(nu), 12)
        if key not in losses:
            queries.append(float(nu))
            losses[key] = float(np.mean(residual(float(nu)) ** 2))
        return losses[key]

    coarse = np.linspace(search.nu_min, search.nu_max, search.n_coarse)
    values = [loss_at(float(nu)) for nu in coarse]
    best_i = int(np.argmin(values))
    left = float(coarse[max(0, best_i - 1)])
    right = float(coarse[min(len(coarse) - 1, best_i + 1)])
    for _ in range(search.max_refine):
        if right - left <= search.refine_width:
            break
        span = GOLDEN * (right - left)
        c = right - span
        d = left + span
        if loss_at(c) <= loss_at(d):
            right = d
        else:
            left = c
    best_nu = min(losses, key=lambda nu: (losses[nu], nu))
    return float(best_nu), float(losses[best_nu]), queries
