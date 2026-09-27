"""Recover a scalar viscosity from sparse Burgers observations.

The forward operator is not retrained. Each instance has one unknown
``ν`` in ``u_t + u u_x = ν u_xx``. On a preregistered set of sensors the
Stage 4 central residual is linear in that scalar,

    R = a - ν b,    a = u_t + u u_x,    b = u_xx,

so the minimizer of the mean of ``R²`` is the normal equation

    ν̂ = (a · b) / (b · b).

``a`` and ``b`` are read off :func:`pinnforge.operator.residual.central_burgers_residual`
by evaluating it at two positive probe viscosities. The stencil, the
spectral derivative, and the saved-frame spacing stay the ones from
Stage 4. No Fourier layer is fit, and no test viscosity enters the
estimator. The training split is used only for the mean-``ν`` baseline,
which is read from the pilot manifest.

:data:`PREREGISTERED_OBSERVATION` is the pattern scored on the held-out
instances. :data:`ABLATION_OBSERVATION` keeps the same sensors and
replaces the short bursts with the Stage 3 stride as the observation
clock. :func:`dense_reference_observation` uses every stored sample. It
is a stencil ceiling, not a sparse sensor pattern.
"""

from __future__ import annotations

import json
import math
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np

from pinnforge.operator.data import load_split_trajectories
from pinnforge.operator.residual import central_burgers_residual
from pinnforge.operator.windows import (
    SPLIT_NAMES,
    Trajectory,
    load_pilot_manifest,
    viscosity_train_stats,
)

# Locked before the Stage 5 test scoring. 32 equispaced sensors on the
# N = 256 pilot grid have Nyquist mode 16, the band the Stage 3 Fourier
# layer keeps. Each burst is five consecutive saved frames so the Stage 4
# central stencil at Δt = 0.01 has three residual nodes. Burst starts are
# Stage 3 window starts spaced by 24 frames.
PREREGISTERED_N_SENSORS = 32
PREREGISTERED_BURST_STARTS = (0, 24, 48, 72)
PREREGISTERED_BURST_LENGTH = 5
PREREGISTERED_FRAME_DT = 0.01
# An estimate at or below this relative error is not, by itself, a failure.
# Nonpositive ν̂ is always a failure. Fixed with the observation pattern.
FAILURE_RELATIVE_ERROR = 0.5
_WORST_COUNT = 5
_PROBE_NU = (1.0, 2.0)


def _burst(start: int, length: int) -> tuple[int, ...]:
    return tuple(range(start, start + length))


@dataclass(frozen=True, slots=True)
class ObservationSpec:
    """Sparse sensors on one saved trajectory.

    ``series`` is one or more strictly increasing frame-index tuples.
    Inside a series the gap is constant, and that gap times ``frame_dt``
    is the ``Δt`` of the central difference. Series do not share indexes,
    so a frame is not counted twice in the normal equation. ``n_sensors``
    is the number of equispaced periodic nodes. ``None`` means every grid
    point. The sensor count must be even and at least 4 when it is set,
    because that is the spectral grid.
    """

    name: str
    n_sensors: int | None
    series: tuple[tuple[int, ...], ...]
    frame_dt: float = PREREGISTERED_FRAME_DT

    def __post_init__(self) -> None:
        if not isinstance(self.name, str) or self.name == "":
            raise ValueError("observation name must be a non-empty string")
        if self.n_sensors is not None:
            _require_even_count(self.n_sensors, "n_sensors")
        frame_dt = _require_finite(self.frame_dt, "frame_dt")
        if frame_dt <= 0.0:
            raise ValueError("frame_dt must be > 0")
        object.__setattr__(self, "frame_dt", frame_dt)
        if len(self.series) < 1:
            raise ValueError("observation series must be non-empty")
        seen: set[int] = set()
        normalized: list[tuple[int, ...]] = []
        for series in self.series:
            indexes = _validate_series(series)
            overlap = seen.intersection(indexes)
            if overlap:
                raise ValueError(f"observation series share frame indexes {sorted(overlap)[:5]}")
            seen.update(indexes)
            normalized.append(indexes)
        object.__setattr__(self, "series", tuple(normalized))

    def series_dt(self, indexes: tuple[int, ...]) -> float:
        """Central-difference step for one series, in the same units as time."""

        return (indexes[1] - indexes[0]) * self.frame_dt

    def maximum_index(self) -> int:
        return max(index for series in self.series for index in series)

    def n_observed_frames(self) -> int:
        return sum(len(series) for series in self.series)

    def n_residual_times(self) -> int:
        return sum(len(series) - 2 for series in self.series)

    def check_grid(self, n_times: int, n_space: int) -> int:
        """Raise if ``series`` or the sensor count does not fit this grid.

        Returns the number of spatial sensors actually read.
        """

        _require_positive_int(n_times, "n_times")
        _require_even_count(n_space, "n_space")
        if self.maximum_index() >= n_times:
            raise ValueError(
                f"observation index {self.maximum_index()} does not fit n_times={n_times}"
            )
        if self.n_sensors is None:
            return n_space
        if n_space % self.n_sensors != 0:
            raise ValueError(f"n_space {n_space} is not divisible by n_sensors {self.n_sensors}")
        return self.n_sensors

    def to_dict(self) -> dict[str, Any]:
        return {
            "name": self.name,
            "n_sensors": self.n_sensors,
            "frame_dt": self.frame_dt,
            "n_series": len(self.series),
            "n_observed_frames": self.n_observed_frames(),
            "n_residual_times": self.n_residual_times(),
            "series_dt": [self.series_dt(series) for series in self.series],
            "series": [list(series) for series in self.series],
        }


def dense_reference_observation(
    n_times: int,
    frame_dt: float = PREREGISTERED_FRAME_DT,
) -> ObservationSpec:
    """Every saved frame and every grid point. Not a sparse pattern."""

    _require_positive_int(n_times, "n_times")
    if n_times < 3:
        raise ValueError("dense reference needs at least 3 frames")
    return ObservationSpec(
        name="dense_reference",
        n_sensors=None,
        series=(tuple(range(n_times)),),
        frame_dt=frame_dt,
    )


@dataclass(frozen=True, slots=True)
class InstanceRecovery:
    """One instance under one observation pattern."""

    instance_id: int
    nu: float
    nu_hat: float
    abs_error: float
    rel_error: float
    baseline_abs_error: float
    baseline_rel_error: float
    mean_abs_residual: float
    mean_abs_residual_at_truth: float
    mean_square_residual: float
    mean_square_residual_at_truth: float
    failure: bool

    def to_dict(self) -> dict[str, Any]:
        return {
            "instance_id": self.instance_id,
            "nu": self.nu,
            "nu_hat": self.nu_hat,
            "abs_error": self.abs_error,
            "rel_error": self.rel_error,
            "baseline_abs_error": self.baseline_abs_error,
            "baseline_rel_error": self.baseline_rel_error,
            "mean_abs_residual": self.mean_abs_residual,
            "mean_abs_residual_at_truth": self.mean_abs_residual_at_truth,
            "mean_square_residual": self.mean_square_residual,
            "mean_square_residual_at_truth": self.mean_square_residual_at_truth,
            "failure": self.failure,
        }


@dataclass(frozen=True, slots=True)
class RecoveryScore:
    """Aggregate viscosity error on one split and one pattern."""

    pattern: str
    n_instances: int
    mean_abs_error: float
    median_abs_error: float
    max_abs_error: float
    mean_rel_error: float
    median_rel_error: float
    max_rel_error: float
    correlation: float | None
    n_failures: int
    n_nonpositive: int
    n_worse_than_baseline: int
    baseline_nu: float
    baseline_mean_abs_error: float
    baseline_mean_rel_error: float
    mean_abs_residual: float
    mean_abs_residual_at_truth: float
    mean_square_residual: float
    mean_square_residual_at_truth: float
    instances: tuple[InstanceRecovery, ...]

    def worst(self, count: int = _WORST_COUNT) -> tuple[InstanceRecovery, ...]:
        """Largest relative errors, then smallest instance id on a tie."""

        ordered = sorted(self.instances, key=lambda row: (-row.rel_error, row.instance_id))
        return tuple(ordered[:count])

    def to_dict(self, *, include_instances: bool) -> dict[str, Any]:
        payload: dict[str, Any] = {
            "pattern": self.pattern,
            "n_instances": self.n_instances,
            "mean_abs_error": self.mean_abs_error,
            "median_abs_error": self.median_abs_error,
            "max_abs_error": self.max_abs_error,
            "mean_rel_error": self.mean_rel_error,
            "median_rel_error": self.median_rel_error,
            "max_rel_error": self.max_rel_error,
            "correlation": self.correlation,
            "n_failures": self.n_failures,
            "n_nonpositive": self.n_nonpositive,
            "n_worse_than_baseline": self.n_worse_than_baseline,
            "baseline_nu": self.baseline_nu,
            "baseline_mean_abs_error": self.baseline_mean_abs_error,
            "baseline_mean_rel_error": self.baseline_mean_rel_error,
            "mean_abs_residual": self.mean_abs_residual,
            "mean_abs_residual_at_truth": self.mean_abs_residual_at_truth,
            "mean_square_residual": self.mean_square_residual,
            "mean_square_residual_at_truth": self.mean_square_residual_at_truth,
            "failure_rule": (
                f"nu_hat <= 0 or rel_error > {FAILURE_RELATIVE_ERROR}"
            ),
            "worse_than_baseline_ids": [
                row.instance_id for row in self.instances if row.abs_error > row.baseline_abs_error
            ],
            "worst": [row.to_dict() for row in self.worst()],
        }
        if include_instances:
            payload["instances"] = [row.to_dict() for row in self.instances]
        return payload


def least_squares_viscosity(field: np.ndarray, spec: ObservationSpec) -> float:
    """Viscosity that minimizes the observed residual mean square.

    ``field`` has shape ``(n_times, n_space)`` and is the raw trajectory.
    The returned scalar is the critical point of ``mean(R²)``. It is not
    projected onto ``ν > 0``. A field with no ``u_xx`` energy on the
    sensors raises, because the normal equation is undefined.
    """

    _require_spec(spec)
    parts = _observation_terms(field, spec)
    nu_hat = _solve_normal(parts)
    if not math.isfinite(nu_hat):
        raise ValueError("recovered viscosity is not finite")
    return nu_hat


def score_trajectories(
    trajectories: list[Trajectory] | tuple[Trajectory, ...],
    spec: ObservationSpec,
    baseline_nu: float,
) -> RecoveryScore:
    """Score one pattern. ``baseline_nu`` is the training-split mean.

    Instance order in the result follows ``instance_id``. The true
    viscosity is used only in the error columns. It is not an input to
    :func:`least_squares_viscosity`.
    """

    _require_spec(spec)
    baseline = _require_finite(baseline_nu, "baseline_nu")
    if baseline <= 0.0:
        raise ValueError("baseline_nu must be > 0")
    if len(trajectories) < 1:
        raise ValueError("at least one trajectory is required")
    rows: list[InstanceRecovery] = []
    seen: set[int] = set()
    for item in trajectories:
        if not isinstance(item, Trajectory):
            raise TypeError("trajectories must be Trajectory values")
        if item.instance_id in seen:
            raise ValueError(f"instance {item.instance_id} is listed more than once")
        seen.add(item.instance_id)
        rows.append(_score_one(item, spec, baseline))
    rows.sort(key=lambda row: row.instance_id)
    return _aggregate(spec.name, baseline, tuple(rows))


def training_baseline_nu(manifest_path: Path) -> float:
    """Mean viscosity of the training instances recorded in the manifest.

    This does not open trajectory files and does not read validation or
    test viscosities into the average. Those records are still parsed,
    because the manifest reader checks that the split lists match.
    """

    manifest = load_pilot_manifest(Path(manifest_path))
    mean, _std = viscosity_train_stats(manifest)
    return mean


def recover_split(
    pilot_dir: Path,
    manifest_path: Path,
    spec: ObservationSpec,
    *,
    split: str = "test",
    check_field_hash: bool = True,
) -> RecoveryScore:
    """Recover ``ν`` on one split. Other splits' files are not opened."""

    if split not in SPLIT_NAMES:
        raise ValueError(f"unknown split {split!r}")
    baseline = training_baseline_nu(manifest_path)
    trajectories = load_split_trajectories(
        pilot_dir,
        manifest_path,
        (split,),
        check_field_hash=check_field_hash,
    )
    return score_trajectories(trajectories, spec, baseline)


def load_recovery_split(
    pilot_dir: Path,
    manifest_path: Path,
    *,
    split: str = "test",
    check_field_hash: bool = True,
) -> tuple[list[Trajectory], float]:
    """Trajectories for ``split`` and the manifest training-mean viscosity."""

    if split not in SPLIT_NAMES:
        raise ValueError(f"unknown split {split!r}")
    baseline = training_baseline_nu(manifest_path)
    trajectories = load_split_trajectories(
        pilot_dir,
        manifest_path,
        (split,),
        check_field_hash=check_field_hash,
    )
    return trajectories, baseline


def adam_viscosity(
    field: np.ndarray,
    spec: ObservationSpec,
    *,
    init: float,
    steps: int,
    lr: float,
) -> tuple[float, float, float]:
    """Adam on the same residual mean square, starting at ``init``.

    Returns ``(nu, objective_at_init, objective_at_final)``. The field is
    data. Only the scalar viscosity is a parameter. The minimizer of this
    quadratic is :func:`least_squares_viscosity`. The reported recovery
    uses that closed form. This path is the check that a bad
    initialization moves toward it. Torch is imported here.
    """

    from pinnforge.ml_import import require_torch
    from pinnforge.operator.loss import _central

    torch = require_torch()
    _require_spec(spec)
    start = _require_finite(init, "init")
    if start <= 0.0:
        raise ValueError("init must be > 0")
    _require_positive_int(steps, "steps")
    step_size = _require_finite(lr, "lr")
    if step_size <= 0.0:
        raise ValueError("lr must be > 0")
    batches = _torch_batches(field, spec, torch)
    viscosity = torch.nn.Parameter(torch.tensor(start, dtype=torch.float64))
    optimizer = torch.optim.Adam([viscosity], lr=step_size)
    initial_objective = float(_adam_objective(batches, viscosity, _central).detach())
    for _ in range(steps):
        optimizer.zero_grad()
        objective = _adam_objective(batches, viscosity, _central)
        objective.backward()
        optimizer.step()
    final = float(viscosity.detach())
    final_objective = float(_adam_objective(batches, viscosity, _central).detach())
    if not math.isfinite(final) or not math.isfinite(final_objective):
        raise ValueError("Adam viscosity fit is not finite")
    return final, initial_objective, final_objective


def write_inverse_json(payload: dict[str, Any], path: Path) -> Path:
    """Write a recovery record. Parent directories are created."""

    destination = Path(path)
    destination.parent.mkdir(parents=True, exist_ok=True)
    destination.write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")
    return destination


def _score_one(item: Trajectory, spec: ObservationSpec, baseline: float) -> InstanceRecovery:
    terms = _observation_terms(item.u, spec)
    nu_hat = _solve_normal(terms)
    if not math.isfinite(nu_hat):
        raise ValueError(f"instance {item.instance_id} recovered viscosity is not finite")
    abs_error = abs(nu_hat - item.nu)
    rel_error = abs_error / item.nu
    baseline_abs = abs(baseline - item.nu)
    residual, square = _residual_moments(terms, nu_hat)
    truth_residual, truth_square = _residual_moments(terms, item.nu)
    failure = nu_hat <= 0.0 or rel_error > FAILURE_RELATIVE_ERROR
    return InstanceRecovery(
        instance_id=item.instance_id,
        nu=item.nu,
        nu_hat=nu_hat,
        abs_error=abs_error,
        rel_error=rel_error,
        baseline_abs_error=baseline_abs,
        baseline_rel_error=baseline_abs / item.nu,
        mean_abs_residual=residual,
        mean_abs_residual_at_truth=truth_residual,
        mean_square_residual=square,
        mean_square_residual_at_truth=truth_square,
        failure=failure,
    )


def _aggregate(pattern: str, baseline: float, rows: tuple[InstanceRecovery, ...]) -> RecoveryScore:
    nu = np.array([row.nu for row in rows], dtype=np.float64)
    hat = np.array([row.nu_hat for row in rows], dtype=np.float64)
    abs_error = np.array([row.abs_error for row in rows], dtype=np.float64)
    rel_error = np.array([row.rel_error for row in rows], dtype=np.float64)
    baseline_abs = np.array([row.baseline_abs_error for row in rows], dtype=np.float64)
    baseline_rel = np.array([row.baseline_rel_error for row in rows], dtype=np.float64)
    return RecoveryScore(
        pattern=pattern,
        n_instances=len(rows),
        mean_abs_error=float(np.mean(abs_error)),
        median_abs_error=float(np.median(abs_error)),
        max_abs_error=float(np.max(abs_error)),
        mean_rel_error=float(np.mean(rel_error)),
        median_rel_error=float(np.median(rel_error)),
        max_rel_error=float(np.max(rel_error)),
        correlation=_pearson(hat, nu),
        n_failures=sum(1 for row in rows if row.failure),
        n_nonpositive=sum(1 for row in rows if row.nu_hat <= 0.0),
        n_worse_than_baseline=sum(1 for row in rows if row.abs_error > row.baseline_abs_error),
        baseline_nu=baseline,
        baseline_mean_abs_error=float(np.mean(baseline_abs)),
        baseline_mean_rel_error=float(np.mean(baseline_rel)),
        mean_abs_residual=float(np.mean([row.mean_abs_residual for row in rows])),
        mean_abs_residual_at_truth=float(np.mean([row.mean_abs_residual_at_truth for row in rows])),
        mean_square_residual=float(np.mean([row.mean_square_residual for row in rows])),
        mean_square_residual_at_truth=float(np.mean([row.mean_square_residual_at_truth for row in rows])),
        instances=rows,
    )


def _observation_terms(field: np.ndarray, spec: ObservationSpec) -> tuple[np.ndarray, np.ndarray]:
    values = np.asarray(field, dtype=np.float64)
    if values.ndim != 2:
        raise ValueError("field must have shape (n_times, n_space)")
    if not np.isfinite(values).all():
        raise ValueError("field must be finite")
    n_sensors = spec.check_grid(int(values.shape[0]), int(values.shape[1]))
    observed = _sensors(values, n_sensors)
    advection: list[np.ndarray] = []
    diffusion: list[np.ndarray] = []
    for indexes in spec.series:
        frames = np.ascontiguousarray(observed[list(indexes)])
        adv, diff = _split_residual(frames, spec.series_dt(indexes))
        advection.append(adv.ravel())
        diffusion.append(diff.ravel())
    return np.concatenate(advection), np.concatenate(diffusion)


def _sensors(field: np.ndarray, n_sensors: int) -> np.ndarray:
    if n_sensors == field.shape[1]:
        return field
    stride = field.shape[1] // n_sensors
    return np.ascontiguousarray(field[:, ::stride])


def _split_residual(frames: np.ndarray, dt: float) -> tuple[np.ndarray, np.ndarray]:
    """Return ``a = u_t + u u_x`` and ``b = u_xx`` from the Stage 4 stencil.

    ``R(ν) = a - ν b``. Two probe viscosities identify ``a`` and ``b``
    without a second derivative implementation in this module.
    """

    batch = frames[None, :, :]
    first, second = _PROBE_NU
    residual_first = central_burgers_residual(batch, np.array([first]), dt=dt)[0]
    residual_second = central_burgers_residual(batch, np.array([second]), dt=dt)[0]
    # R(1) - R(2) = (a - b) - (a - 2b) = b.
    diffusion = residual_first - residual_second
    advection = residual_first + first * diffusion
    return advection, diffusion


def _solve_normal(parts: tuple[np.ndarray, np.ndarray]) -> float:
    advection, diffusion = parts
    denom = float(np.dot(diffusion, diffusion))
    if not math.isfinite(denom) or denom <= 0.0:
        raise ValueError("u_xx vanishes on the observed nodes, so nu is not identifiable")
    return float(np.dot(advection, diffusion) / denom)


def _residual_moments(parts: tuple[np.ndarray, np.ndarray], nu: float) -> tuple[float, float]:
    advection, diffusion = parts
    residual = advection - float(nu) * diffusion
    if not np.isfinite(residual).all():
        raise ValueError("observed residual is not finite")
    return float(np.mean(np.abs(residual))), float(np.mean(residual * residual))


def _torch_batches(field: np.ndarray, spec: ObservationSpec, torch: Any) -> list[tuple[Any, float]]:
    values = np.asarray(field, dtype=np.float64)
    if values.ndim != 2 or not np.isfinite(values).all():
        raise ValueError("field must be a finite array of shape (n_times, n_space)")
    n_sensors = spec.check_grid(int(values.shape[0]), int(values.shape[1]))
    observed = _sensors(values, n_sensors)
    batches: list[tuple[Any, float]] = []
    for indexes in spec.series:
        frames = np.ascontiguousarray(observed[list(indexes)])
        tensor = torch.tensor(frames, dtype=torch.float64).unsqueeze(0)
        batches.append((tensor, spec.series_dt(indexes)))
    return batches


def _adam_objective(batches: list[tuple[Any, float]], viscosity: Any, central: Any) -> Any:
    total = None
    count = 0
    for frames, dt in batches:
        residual = central(
            frames,
            viscosity.view(1),
            dt=dt,
            space="physical",
            u_mean=0.0,
            u_std=1.0,
        )
        square = residual.square().sum()
        total = square if total is None else total + square
        count += int(residual.numel())
    if total is None or count < 1:
        raise ValueError("Adam residual has no nodes")
    return total / count


def _pearson(left: np.ndarray, right: np.ndarray) -> float | None:
    if left.size < 2:
        return None
    left_centered = left - float(np.mean(left))
    right_centered = right - float(np.mean(right))
    denom = math.sqrt(float(np.dot(left_centered, left_centered) * np.dot(right_centered, right_centered)))
    if denom == 0.0:
        return None
    value = float(np.dot(left_centered, right_centered) / denom)
    if not math.isfinite(value):
        return None
    return value


def _validate_series(series: tuple[int, ...] | list[int]) -> tuple[int, ...]:
    if len(series) < 3:
        raise ValueError("each observation series needs at least 3 frames")
    indexes: list[int] = []
    for item in series:
        indexes.append(_require_nonnegative_int(item, "frame index"))
    gaps = [indexes[i + 1] - indexes[i] for i in range(len(indexes) - 1)]
    if gaps[0] < 1 or any(gap != gaps[0] for gap in gaps):
        raise ValueError("each observation series must be strictly increasing with a constant gap")
    return tuple(indexes)


def _require_spec(spec: ObservationSpec) -> None:
    if not isinstance(spec, ObservationSpec):
        raise TypeError("spec must be an ObservationSpec")


def _require_even_count(value: object, label: str) -> int:
    number = _require_positive_int(value, label)
    if number < 4 or number % 2 != 0:
        raise ValueError(f"{label} must be an even integer >= 4")
    return number


def _require_positive_int(value: object, label: str) -> int:
    if isinstance(value, bool) or not isinstance(value, (int, np.integer)):
        raise ValueError(f"{label} must be an integer >= 1")
    number = int(value)
    if number < 1:
        raise ValueError(f"{label} must be an integer >= 1")
    return number


def _require_nonnegative_int(value: object, label: str) -> int:
    if isinstance(value, bool) or not isinstance(value, (int, np.integer)):
        raise ValueError(f"{label} must be a non-negative integer")
    number = int(value)
    if number < 0:
        raise ValueError(f"{label} must be a non-negative integer")
    return number


def _require_finite(value: object, label: str) -> float:
    if isinstance(value, bool) or not isinstance(value, (int, float, np.floating)):
        raise ValueError(f"{label} must be a finite number")
    number = float(value)
    if not math.isfinite(number):
        raise ValueError(f"{label} must be a finite number")
    return number


PREREGISTERED_OBSERVATION = ObservationSpec(
    name="sensors32_bursts",
    n_sensors=PREREGISTERED_N_SENSORS,
    series=tuple(_burst(start, PREREGISTERED_BURST_LENGTH) for start in PREREGISTERED_BURST_STARTS),
    frame_dt=PREREGISTERED_FRAME_DT,
)

ABLATION_OBSERVATION = ObservationSpec(
    name="sensors32_stride8",
    n_sensors=PREREGISTERED_N_SENSORS,
    series=(tuple(range(0, 97, 8)),),
    frame_dt=PREREGISTERED_FRAME_DT,
)
