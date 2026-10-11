"""Warm-started solver-in-the-loop viscosity fit with calibrated intervals.

A Fourier neural operator estimate initializes a short, bounded local fit
of the same sensor-lift problem that :mod:`pinnforge.operator.solver_fit`
solves from a cold coarse grid. The forward model, the initial condition,
the target bursts, and the sensor mean square are unchanged. Only the
search differs.

The local fit is Gauss-Newton in ``s = log ν`` on the vector of sensor
residuals. The first Jacobian is a one-sided finite difference with step
``probe_step``. Later Jacobians are secants between the lowest-loss point
and its nearest evaluated neighbour. Every iterate is clipped to a trust
box of half-width ``trust_radius`` around the first guess, intersected
with the pilot viscosity range. The fit stops when the next step is at
most ``step_tolerance`` or after ``max_evaluations`` forward solves. The
returned value is the final Gauss-Newton prediction, or the best evaluated
point when ``return_mode`` is ``best_evaluated``.

Intervals are split-conformal on validation residuals of ``log ν``,
optionally normalized by a per-instance scale and computed separately for
the instances predicted to lie in ``hard_ood`` (``ν̂ <= threshold``) and
the rest. The abstain flag compares a sensor-residual statistic with a
threshold fit on validation. This module does not import torch.
"""

from __future__ import annotations

import math
import time
from collections.abc import Callable, Mapping, Sequence
from dataclasses import dataclass
from typing import Any

import numpy as np

from pinnforge.operator.solver_fit import SolverProblem, SolverSearch

RETURN_MODES = ("prediction", "best_evaluated")
SCALE_NAMES = ("none", "gauss_newton", "ensemble_spread")
GROUP_NAMES = ("predicted_hard_ood", "predicted_complement")


@dataclass(frozen=True, slots=True)
class RefineConfig:
    """Bounded local search around a first guess. Lengths are in log ν."""

    max_evaluations: int = 3
    probe_step: float = 0.02
    trust_radius: float = math.log(2.0)
    step_tolerance: float = 1e-3
    nu_min: float = 0.005
    nu_max: float = 0.1
    return_mode: str = "prediction"

    def __post_init__(self) -> None:
        if isinstance(self.max_evaluations, bool) or not isinstance(self.max_evaluations, (int, np.integer)):
            raise ValueError("max_evaluations must be an integer >= 2")
        if int(self.max_evaluations) < 2:
            raise ValueError("max_evaluations must be an integer >= 2")
        object.__setattr__(self, "max_evaluations", int(self.max_evaluations))
        for label in ("probe_step", "trust_radius", "step_tolerance", "nu_min", "nu_max"):
            value = getattr(self, label)
            if isinstance(value, bool) or not isinstance(value, (int, float, np.floating)):
                raise ValueError(f"{label} must be a finite number")
            number = float(value)
            if not math.isfinite(number) or number <= 0.0:
                raise ValueError(f"{label} must be finite and > 0")
            object.__setattr__(self, label, number)
        if self.nu_max <= self.nu_min:
            raise ValueError("nu_max must exceed nu_min")
        if self.probe_step >= self.trust_radius:
            raise ValueError("probe_step must be smaller than trust_radius")
        if self.return_mode not in RETURN_MODES:
            raise ValueError(f"return_mode must be one of {RETURN_MODES}")

    def to_dict(self) -> dict[str, Any]:
        return {
            "max_evaluations": self.max_evaluations,
            "probe_step": self.probe_step,
            "trust_radius": self.trust_radius,
            "step_tolerance": self.step_tolerance,
            "nu_min": self.nu_min,
            "nu_max": self.nu_max,
            "return_mode": self.return_mode,
        }

    @classmethod
    def from_mapping(cls, raw: Mapping[str, Any]) -> RefineConfig:
        return cls(
            max_evaluations=int(raw["max_evaluations"]),
            probe_step=float(raw["probe_step"]),
            trust_radius=float(raw["trust_radius"]),
            step_tolerance=float(raw["step_tolerance"]),
            nu_min=float(raw["nu_min"]),
            nu_max=float(raw["nu_max"]),
            return_mode=str(raw["return_mode"]),
        )


@dataclass(frozen=True, slots=True)
class RefineStep:
    """The estimate the loop would return if it stopped after ``n_evaluations`` solves."""

    n_evaluations: int
    nu_hat: float
    seconds: float
    loss_hat: float
    log_standard_error: float
    relative_residual: float
    at_trust_bound: bool
    stopped_on_tolerance: bool

    def to_dict(self) -> dict[str, Any]:
        return {
            "n_evaluations": self.n_evaluations,
            "nu_hat": self.nu_hat,
            "seconds": self.seconds,
            "loss_hat": self.loss_hat,
            "log_standard_error": self.log_standard_error,
            "relative_residual": self.relative_residual,
            "at_trust_bound": self.at_trust_bound,
            "stopped_on_tolerance": self.stopped_on_tolerance,
        }


@dataclass(frozen=True, slots=True)
class RefineResult:
    """One warm-started fit. ``trace`` holds every intermediate stop."""

    nu_guess: float
    nu_hat: float
    n_evaluations: int
    queries: tuple[tuple[float, float], ...]
    loss_at_guess: float
    loss_hat: float
    log_standard_error: float
    relative_residual: float
    at_trust_bound: bool
    stopped_on_tolerance: bool
    trace: tuple[RefineStep, ...] = ()

    def to_dict(self) -> dict[str, Any]:
        return {
            "nu_guess": self.nu_guess,
            "nu_hat": self.nu_hat,
            "n_evaluations": self.n_evaluations,
            "queries": [[nu, loss] for nu, loss in self.queries],
            "loss_at_guess": self.loss_at_guess,
            "loss_hat": self.loss_hat,
            "log_standard_error": self.log_standard_error,
            "relative_residual": self.relative_residual,
            "at_trust_bound": self.at_trust_bound,
            "stopped_on_tolerance": self.stopped_on_tolerance,
            "trace": [step.to_dict() for step in self.trace],
        }


def refine_viscosity(
    residual: Callable[[float], np.ndarray],
    nu_guess: float,
    config: RefineConfig,
    *,
    observed_rms: float = 1.0,
) -> RefineResult:
    """Gauss-Newton in log ν from ``nu_guess``. ``residual`` returns a vector.

    ``observed_rms`` normalizes the residual statistic used by the abstain
    flag. It does not change the iterates.
    """

    if not isinstance(config, RefineConfig):
        raise TypeError("config must be a RefineConfig")
    guess = float(nu_guess)
    if not math.isfinite(guess) or guess <= 0.0:
        raise ValueError("nu_guess must be finite and > 0")
    scale = float(observed_rms)
    if not math.isfinite(scale) or scale <= 0.0:
        raise ValueError("observed_rms must be finite and > 0")
    low = math.log(config.nu_min)
    high = math.log(config.nu_max)
    start = min(high, max(low, math.log(guess)))
    box_low = max(low, start - config.trust_radius)
    box_high = min(high, start + config.trust_radius)
    points: list[tuple[float, np.ndarray, float]] = []
    trace: list[RefineStep] = []
    started = time.perf_counter()

    def evaluate(s: float) -> None:
        vector = np.asarray(residual(math.exp(s)), dtype=np.float64).ravel()
        if vector.size < 2 or not np.isfinite(vector).all():
            raise ValueError("residual must be a finite vector with at least two entries")
        points.append((s, vector, float(np.mean(vector * vector))))

    evaluate(start)
    probe = start + config.probe_step
    if probe > box_high:
        probe = start - config.probe_step
    evaluate(probe)
    while True:
        best_s, best_r, _ = _best(points)
        jacobian = _secant(points, best_s, best_r)
        curvature = float(np.dot(jacobian, jacobian))
        step = 0.0 if curvature <= 0.0 else -float(np.dot(jacobian, best_r)) / curvature
        proposal = min(box_high, max(box_low, best_s + step))
        stopped = abs(proposal - best_s) <= config.step_tolerance
        final = proposal if config.return_mode == "prediction" else best_s
        linear = best_r + jacobian * (final - best_s)
        loss_hat = float(np.mean(linear * linear))
        sigma2 = float(np.dot(linear, linear)) / max(1, int(linear.size) - 1)
        standard_error = math.sqrt(sigma2 / curvature) if curvature > 0.0 else math.inf
        edge = 1e-12
        trace.append(
            RefineStep(
                n_evaluations=len(points),
                nu_hat=float(math.exp(final)),
                seconds=float(time.perf_counter() - started),
                loss_hat=loss_hat,
                log_standard_error=float(standard_error),
                relative_residual=float(math.sqrt(loss_hat) / scale),
                at_trust_bound=bool(final <= box_low + edge or final >= box_high - edge),
                stopped_on_tolerance=bool(stopped),
            )
        )
        if stopped or len(points) >= config.max_evaluations:
            break
        evaluate(proposal)
    last = trace[-1]
    return RefineResult(
        nu_guess=guess,
        nu_hat=last.nu_hat,
        n_evaluations=last.n_evaluations,
        queries=tuple((float(math.exp(s)), float(loss)) for s, _, loss in points),
        loss_at_guess=float(points[0][2]),
        loss_hat=last.loss_hat,
        log_standard_error=last.log_standard_error,
        relative_residual=last.relative_residual,
        at_trust_bound=last.at_trust_bound,
        stopped_on_tolerance=last.stopped_on_tolerance,
        trace=tuple(trace),
    )


def step_at_cap(result: Mapping[str, Any], cap: int) -> dict[str, Any]:
    """The trace entry a run with ``max_evaluations = cap`` would have returned.

    The loop does not read ``max_evaluations`` before it stops, so a run
    with a larger cap passes through the same states.
    """

    limit = int(cap)
    steps = [step for step in result["trace"] if int(step["n_evaluations"]) <= limit]
    if not steps:
        raise ValueError("cap is below the first trace entry")
    return dict(steps[-1])


def sensor_residual(problem: SolverProblem, nu: float, search: SolverSearch) -> np.ndarray:
    """Solver sensors minus observations on every target frame, flattened.

    The mean square of this vector is
    :func:`pinnforge.operator.solver_fit.sensor_mismatch`.
    """

    from pinnforge.reference.numerical.solver import solve

    if not isinstance(problem, SolverProblem) or not isinstance(search, SolverSearch):
        raise TypeError("problem and search must use the solver-fit dataclasses")
    viscosity = float(nu)
    if not math.isfinite(viscosity) or viscosity <= 0.0:
        raise ValueError("nu must be finite and > 0")
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
        raise ValueError("solver sensor residual is not finite")
    return residual.ravel()


def fit_warm_start(
    problem: SolverProblem,
    search: SolverSearch,
    nu_guess: float,
    config: RefineConfig,
) -> dict[str, Any]:
    """Refine ``nu_guess`` on the solver problem. Wall clock covers every solve."""

    started = time.perf_counter()
    observed_rms = float(np.sqrt(np.mean(problem.observed * problem.observed)))
    result = refine_viscosity(
        lambda nu: sensor_residual(problem, nu, search),
        nu_guess,
        config,
        observed_rms=observed_rms,
    )
    payload = result.to_dict()
    payload["seconds"] = float(time.perf_counter() - started)
    return payload


def conformal_quantile(scores: Sequence[float], alpha: float) -> float:
    """Split-conformal quantile: the ``ceil((n + 1)(1 - alpha))``-th smallest score.

    Returns ``inf`` when ``n`` is too small for that rank.
    """

    values = np.sort(np.asarray(list(scores), dtype=np.float64))
    level = float(alpha)
    if not 0.0 < level < 1.0:
        raise ValueError("alpha must be in (0, 1)")
    if values.size and (not np.isfinite(values).all() or np.any(values < 0.0)):
        raise ValueError("conformal scores must be finite and >= 0")
    rank = math.ceil((values.size + 1) * (1.0 - level))
    if rank > values.size:
        return math.inf
    return float(values[rank - 1])


def conformal_score(nu: float, nu_hat: float, scale: float) -> float:
    """|log(ν / ν̂)| divided by a positive scale."""

    truth = float(nu)
    estimate = float(nu_hat)
    spread = float(scale)
    if truth <= 0.0 or estimate <= 0.0 or not math.isfinite(spread) or spread <= 0.0:
        raise ValueError("conformal score needs positive nu, nu_hat, and scale")
    return abs(math.log(truth / estimate)) / spread


def instance_scale(kind: str, *, log_standard_error: float, ensemble_log_std: float, floor: float) -> float:
    """Per-instance interval scale. ``floor`` keeps it positive."""

    if kind not in SCALE_NAMES:
        raise ValueError(f"scale must be one of {SCALE_NAMES}")
    bottom = float(floor)
    if not math.isfinite(bottom) or bottom <= 0.0:
        raise ValueError("scale floor must be finite and > 0")
    if kind == "none":
        return 1.0
    raw = float(log_standard_error) if kind == "gauss_newton" else float(ensemble_log_std)
    if not math.isfinite(raw):
        return math.inf
    return math.sqrt(raw * raw + bottom * bottom)


def predicted_group(nu_hat: float, threshold: float) -> str:
    """Calibration group from the estimate. The true ν is not read."""

    return GROUP_NAMES[0] if float(nu_hat) <= float(threshold) else GROUP_NAMES[1]


def calibrate_groups(
    rows: Sequence[Mapping[str, Any]],
    alpha: float,
    *,
    min_group: int,
) -> dict[str, Any]:
    """Quantile per predicted group, with a pooled fallback for a small group.

    Each row has ``score`` and ``group``. A group with fewer than
    ``min_group`` rows uses the pooled quantile.
    """

    pooled = [float(row["score"]) for row in rows]
    pooled_q = conformal_quantile(pooled, alpha)
    groups: dict[str, Any] = {}
    for name in GROUP_NAMES:
        values = [float(row["score"]) for row in rows if row["group"] == name]
        if len(values) >= int(min_group):
            groups[name] = {"n": len(values), "quantile": conformal_quantile(values, alpha), "pooled_fallback": False}
        else:
            groups[name] = {"n": len(values), "quantile": pooled_q, "pooled_fallback": True}
    return {"alpha": float(alpha), "pooled": {"n": len(pooled), "quantile": pooled_q}, "groups": groups}


def interval(nu_hat: float, scale: float, quantile: float) -> tuple[float, float]:
    """Multiplicative interval ``ν̂ exp(± q s)``."""

    estimate = float(nu_hat)
    half = float(scale) * float(quantile)
    if estimate <= 0.0 or not math.isfinite(half) or half < 0.0:
        return (0.0, math.inf)
    return (estimate * math.exp(-half), estimate * math.exp(half))


def abstain_threshold(values: Sequence[float], rate: float) -> float:
    """Validation quantile above which roughly ``rate`` of instances abstain.

    The threshold is the ``ceil(n (1 - rate))``-th smallest value, so at
    most ``floor(n rate)`` validation instances lie strictly above it.
    """

    data = np.sort(np.asarray(list(values), dtype=np.float64))
    share = float(rate)
    if data.size < 1 or not np.isfinite(data).all():
        raise ValueError("abstain threshold needs finite values")
    if not 0.0 <= share < 1.0:
        raise ValueError("abstain rate must be in [0, 1)")
    rank = max(1, math.ceil(data.size * (1.0 - share)))
    return float(data[rank - 1])


def mean_std(values: Sequence[float]) -> tuple[float, float]:
    """Arithmetic mean and sample standard deviation (``ddof = 1``)."""

    array = np.asarray(list(values), dtype=np.float64)
    if array.size < 2 or not np.isfinite(array).all():
        raise ValueError("mean_std needs at least two finite values")
    return float(np.mean(array)), float(np.std(array, ddof=1))


def _best(points: Sequence[tuple[float, np.ndarray, float]]) -> tuple[float, np.ndarray, float]:
    return min(points, key=lambda item: (item[2], item[0]))


def _secant(points: Sequence[tuple[float, np.ndarray, float]], best_s: float, best_r: np.ndarray) -> np.ndarray:
    others = [item for item in points if item[0] != best_s]
    if not others:
        raise ValueError("secant needs two distinct evaluated points")
    partner = min(others, key=lambda item: (abs(item[0] - best_s), item[0]))
    return (best_r - partner[1]) / (best_s - partner[0])
