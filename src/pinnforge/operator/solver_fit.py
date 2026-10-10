"""Fit viscosity by matching the spectral Burgers solver at the sensors.

The forward model is :func:`pinnforge.reference.numerical.solver.solve`,
the same ETDRK4 scheme that built the pilot. The objective is the mean
square error on the sensor nodes of every burst except the opening one.
That is the same target set the Stage D operator matches. The opening
burst is the initial condition, not a target.

Two initial conditions are pre-specified.

``sensor_lift`` reads the first frame of the opening burst at the
sensors and extends it with :func:`pinnforge.operator.stage_d.lift_sensors`.
That is a trigonometric interpolant. Modes above the sensor Nyquist are
zero. It is not the stored initial condition, and it is not a claim that
the interpolant solves Burgers.

``oracle_ic`` reads the full stored field at that same frame. It is a
ceiling. It is not a sparse-sensor estimator.

The search is a uniform coarse grid on ``[nu_min, nu_max]``, then
golden-section refinement in the neighboring coarse bracket until the
bracket is at most ``refine_width`` or ``max_refine`` new evaluations
have been spent. The returned viscosity is the evaluated point with the
smallest loss. An exact tie keeps the smaller viscosity. The search does
not see the true viscosity.
"""

from __future__ import annotations

import math
import time
from dataclasses import dataclass
from typing import Any, Callable

import numpy as np

from pinnforge.operator.inverse import ObservationSpec
from pinnforge.operator.stage_d import lift_sensors

_GOLDEN = (math.sqrt(5.0) - 1.0) / 2.0
_QUERY_DECIMALS = 12


@dataclass(frozen=True, slots=True)
class SolverSearch:
    """Bounded scalar search. Losses are queried, not interpolated."""

    nu_min: float = 0.005
    nu_max: float = 0.1
    n_coarse: int = 7
    refine_width: float = 5e-4
    max_refine: int = 10
    n: int = 1024
    dt: float = 2.5e-4
    save_dt: float = 0.01

    def __post_init__(self) -> None:
        low = _finite(self.nu_min, "nu_min")
        high = _finite(self.nu_max, "nu_max")
        if low <= 0.0 or high <= low:
            raise ValueError("viscosity bounds must satisfy 0 < nu_min < nu_max")
        object.__setattr__(self, "nu_min", low)
        object.__setattr__(self, "nu_max", high)
        if isinstance(self.n_coarse, bool) or not isinstance(self.n_coarse, (int, np.integer)):
            raise ValueError("n_coarse must be an integer >= 3")
        if int(self.n_coarse) < 3:
            raise ValueError("n_coarse must be an integer >= 3")
        object.__setattr__(self, "n_coarse", int(self.n_coarse))
        width = _finite(self.refine_width, "refine_width")
        if width <= 0.0:
            raise ValueError("refine_width must be > 0")
        object.__setattr__(self, "refine_width", width)
        if isinstance(self.max_refine, bool) or not isinstance(self.max_refine, (int, np.integer)):
            raise ValueError("max_refine must be an integer >= 0")
        if int(self.max_refine) < 0:
            raise ValueError("max_refine must be an integer >= 0")
        object.__setattr__(self, "max_refine", int(self.max_refine))
        for label, value in (("n", self.n),):
            if isinstance(value, bool) or not isinstance(value, (int, np.integer)) or int(value) < 4 or int(value) % 2:
                raise ValueError(f"{label} must be an even integer >= 4")
            object.__setattr__(self, label, int(value))
        for label, value in (("dt", self.dt), ("save_dt", self.save_dt)):
            number = _finite(value, label)
            if number <= 0.0:
                raise ValueError(f"{label} must be > 0")
            object.__setattr__(self, label, number)

    def to_dict(self) -> dict[str, Any]:
        return {
            "nu_min": self.nu_min,
            "nu_max": self.nu_max,
            "n_coarse": self.n_coarse,
            "refine_width": self.refine_width,
            "max_refine": self.max_refine,
            "n": self.n,
            "dt": self.dt,
            "save_dt": self.save_dt,
            "optimizer": "coarse_grid_plus_golden_section",
            "tie_break": "smallest evaluated nu",
            "returned_value": "evaluated point, not an interpolated vertex",
        }


@dataclass(frozen=True, slots=True)
class SearchResult:
    """One scalar minimization."""

    nu_hat: float
    objective: float
    n_evaluations: int
    queries: tuple[tuple[float, float], ...]


@dataclass(frozen=True, slots=True)
class SolverProblem:
    """Sensor targets and one initial condition, ready to integrate."""

    initial_condition: str
    u0: np.ndarray
    frame_index: tuple[int, ...]
    observed: np.ndarray
    t_final: float
    stride: int


def opening_frame(spec: ObservationSpec) -> int:
    """First frame of the first burst. The solve starts there."""

    if not isinstance(spec, ObservationSpec):
        raise TypeError("spec must be an ObservationSpec")
    return int(spec.series[0][0])


def target_frames(spec: ObservationSpec) -> tuple[int, ...]:
    """Observed frames after the opening burst. Empty means inapplicable."""

    if not isinstance(spec, ObservationSpec):
        raise TypeError("spec must be an ObservationSpec")
    frames = [int(index) for series in spec.series[1:] for index in series]
    return tuple(frames)


def build_solver_problem(
    field: np.ndarray,
    spec: ObservationSpec,
    search: SolverSearch,
    *,
    initial_condition: str,
) -> SolverProblem:
    """Slice the mask. Off-target samples are not copied into the problem."""

    if initial_condition not in ("sensor_lift", "oracle_ic"):
        raise ValueError("initial_condition must be sensor_lift or oracle_ic")
    if not isinstance(spec, ObservationSpec):
        raise TypeError("spec must be an ObservationSpec")
    if spec.n_sensors is None:
        raise ValueError("solver fit requires a finite sensor count")
    frames = target_frames(spec)
    if len(frames) < 1:
        raise ValueError("solver fit has no target burst")
    if opening_frame(spec) != 0:
        raise ValueError("solver fit starts at frame 0 of the stored trajectory")
    values = np.asarray(field, dtype=np.float64)
    if values.ndim != 2 or not np.isfinite(values).all():
        raise ValueError("field must be a finite array of shape (n_times, n_space)")
    n_sensors = spec.check_grid(int(values.shape[0]), int(values.shape[1]))
    if int(values.shape[1]) != int(search.n):
        raise ValueError("field grid does not match the solver search grid")
    if max(frames) >= int(values.shape[0]):
        raise ValueError("a target frame is outside the stored field")
    stride = int(values.shape[1]) // n_sensors
    observed = np.ascontiguousarray(values[list(frames)][:, ::stride])
    start = np.ascontiguousarray(values[0])
    if initial_condition == "sensor_lift":
        u0 = lift_sensors(start[::stride], int(search.n))
    else:
        u0 = start.copy()
    t_final = float(max(frames)) * float(spec.frame_dt)
    if abs(float(spec.frame_dt) - float(search.save_dt)) > 0.0:
        raise ValueError("solver save_dt must equal the observation frame_dt")
    _require_divisible(t_final, search.dt, "t_final / dt")
    _require_divisible(t_final, search.save_dt, "t_final / save_dt")
    return SolverProblem(
        initial_condition=initial_condition,
        u0=np.asarray(u0, dtype=np.float64),
        frame_index=frames,
        observed=observed,
        t_final=t_final,
        stride=stride,
    )


def sensor_mismatch(problem: SolverProblem, nu: float, search: SolverSearch) -> float:
    """Mean square error of one forward solve against the sensor targets."""

    from pinnforge.reference.numerical.solver import solve

    if not isinstance(problem, SolverProblem) or not isinstance(search, SolverSearch):
        raise TypeError("problem and search must use the solver-fit dataclasses")
    viscosity = _finite(nu, "nu")
    if viscosity <= 0.0:
        raise ValueError("nu must be > 0")
    trajectory = solve(
        problem.u0,
        viscosity,
        dt=search.dt,
        t_final=problem.t_final,
        save_dt=search.save_dt,
    )
    predicted = np.asarray(trajectory.u[list(problem.frame_index)][:, :: problem.stride], dtype=np.float64)
    if predicted.shape != problem.observed.shape:
        raise ValueError("solver sensor slice does not match the observations")
    residual = predicted - problem.observed
    if not np.isfinite(residual).all():
        raise ValueError("solver sensor mismatch is not finite")
    return float(np.mean(residual * residual))


def minimize_scalar(objective: Callable[[float], float], search: SolverSearch) -> SearchResult:
    """Coarse grid, then golden section. ``objective`` is a loss."""

    if not isinstance(search, SolverSearch):
        raise TypeError("search must be a SolverSearch")
    cache: dict[float, float] = {}
    queries: list[tuple[float, float]] = []

    def evaluate(nu: float) -> float:
        key = round(float(nu), _QUERY_DECIMALS)
        if key < search.nu_min or key > search.nu_max:
            key = min(search.nu_max, max(search.nu_min, key))
        cached = cache.get(key)
        if cached is not None:
            return cached
        loss = float(objective(key))
        if not math.isfinite(loss):
            raise ValueError("objective is not finite")
        cache[key] = loss
        queries.append((key, loss))
        return loss

    coarse = np.linspace(search.nu_min, search.nu_max, search.n_coarse)
    coarse_losses = [evaluate(float(node)) for node in coarse]
    best = min(coarse_losses)
    index = min(i for i, loss in enumerate(coarse_losses) if loss == best)
    left = float(coarse[max(0, index - 1)])
    right = float(coarse[min(search.n_coarse - 1, index + 1)])
    refinements = 0
    if right - left > search.refine_width and search.max_refine > 0:
        interior_c = right - _GOLDEN * (right - left)
        interior_d = left + _GOLDEN * (right - left)
        loss_c = evaluate(interior_c)
        loss_d = evaluate(interior_d)
        while right - left > search.refine_width and refinements < search.max_refine:
            refinements += 1
            if loss_c < loss_d or (loss_c == loss_d and interior_c <= interior_d):
                right = interior_d
                interior_d = interior_c
                loss_d = loss_c
                interior_c = right - _GOLDEN * (right - left)
                loss_c = evaluate(interior_c)
            else:
                left = interior_c
                interior_c = interior_d
                loss_c = loss_d
                interior_d = left + _GOLDEN * (right - left)
                loss_d = evaluate(interior_d)
    winner = min(queries, key=lambda item: (item[1], item[0]))
    return SearchResult(
        nu_hat=float(winner[0]),
        objective=float(winner[1]),
        n_evaluations=len(queries),
        queries=tuple(queries),
    )


def fit_solver_problem(
    problem: SolverProblem,
    search: SolverSearch,
    *,
    truth: float | None = None,
) -> dict[str, Any]:
    """Minimize the sensor mismatch. ``truth`` is an optional diagnostic solve."""

    started = time.perf_counter()
    found = minimize_scalar(lambda nu: sensor_mismatch(problem, nu, search), search)
    diagnostic = None
    if truth is not None:
        diagnostic = sensor_mismatch(problem, float(truth), search)
    elapsed = time.perf_counter() - started
    return {
        "initial_condition": problem.initial_condition,
        "nu_hat": found.nu_hat,
        "objective": found.objective,
        "objective_at_truth": diagnostic,
        "n_evaluations": found.n_evaluations,
        "seconds": float(elapsed),
    }


def _finite(value: object, label: str) -> float:
    if isinstance(value, bool) or not isinstance(value, (int, float, np.floating)):
        raise ValueError(f"{label} must be a finite number")
    number = float(value)
    if not math.isfinite(number):
        raise ValueError(f"{label} must be a finite number")
    return number


def _require_divisible(span: float, step: float, label: str) -> None:
    ratio = span / step
    nearest = round(ratio)
    if nearest < 1 or abs(nearest * step - span) > 1e-8 * max(1.0, abs(span)):
        raise ValueError(f"{label} must be an integer >= 1")
