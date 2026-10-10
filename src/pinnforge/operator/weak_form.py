"""Weak-form viscosity estimate on sparse Burgers sensors.

The PDE is known: ``u_t + (u² / 2)_x = ν u_xx``. Viscosity is the only
unknown. Each test function is separable, ``φ(x, t) = ψ(x) η(t)``, with
``ψ`` and ``η`` the compact polynomial bumps used by WSINDy
(Messenger and Bortz, arXiv:2007.02848): ``(1 - ξ²)_+^p``. Derivatives
fall on ``φ``. The field is never differentiated and never interpolated.

On a burst, ``η`` vanishes at the first and last frame, so integration
by parts in time has no endpoint term. Space is periodic, so integration
by parts in ``x`` has no boundary term even when a bump does not decay
inside one period. The integrals are the trapezoidal rule on the sensor
nodes only: spacing ``dx = 2 / n_sensors`` in space, and half-weights on
the first and last frame in time. That time rule is a 5-node quadrature
when the burst has five frames. It is not an interpolant of ``u``.

Rows from every burst are stacked and ``ν`` is the normal equation of
``a - ν b``. A row whose diffusion projection is below the numerical
floor, relative to the largest row, is dropped. Nothing here is chosen
on a test score.
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Any

import numpy as np

from pinnforge.operator.inverse import ObservationSpec
from pinnforge.reference.numerical.solver import LENGTH, grid

_DEGREE_MIN = 2


@dataclass(frozen=True, slots=True)
class WeakFormConfig:
    """Frozen test-function and quadrature choices.

    ``radius_in_spacings`` is the bump radius in units of the sensor
    spacing. ``center_stride`` subsamples sensor nodes as bump centers.
    ``projection_floor`` drops a row only when its diffusion projection
    is numerically negligible next to the largest row.
    """

    degree: int = 4
    radius_in_spacings: int = 8
    center_stride: int = 4
    projection_floor: float = 1e-12

    def __post_init__(self) -> None:
        if isinstance(self.degree, bool) or not isinstance(self.degree, (int, np.integer)):
            raise ValueError("degree must be an integer")
        if int(self.degree) < _DEGREE_MIN:
            raise ValueError(f"degree must be >= {_DEGREE_MIN}")
        object.__setattr__(self, "degree", int(self.degree))
        for label, value in (
            ("radius_in_spacings", self.radius_in_spacings),
            ("center_stride", self.center_stride),
        ):
            if isinstance(value, bool) or not isinstance(value, (int, np.integer)) or int(value) < 1:
                raise ValueError(f"{label} must be an integer >= 1")
            object.__setattr__(self, label, int(value))
        floor = float(self.projection_floor)
        if not math.isfinite(floor) or floor < 0.0:
            raise ValueError("projection_floor must be finite and >= 0")
        object.__setattr__(self, "projection_floor", floor)

    def to_dict(self) -> dict[str, Any]:
        return {
            "degree": self.degree,
            "radius_in_spacings": self.radius_in_spacings,
            "center_stride": self.center_stride,
            "projection_floor": self.projection_floor,
            "family": "(1 - xi^2)_+^degree",
            "interpolation_of_u": "none",
            "quadrature": "periodic trapezoid in x, trapezoid in t",
        }


def bump_derivatives(xi: np.ndarray, degree: int) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Value and ``ξ`` derivatives of ``(1 - ξ²)_+^degree``.

    Outside ``|ξ| < 1`` every returned array is zero, including at the
    endpoints, where the bump of degree >= 2 is flat.
    """

    if isinstance(degree, bool) or not isinstance(degree, (int, np.integer)) or int(degree) < _DEGREE_MIN:
        raise ValueError(f"degree must be an integer >= {_DEGREE_MIN}")
    order = int(degree)
    points = np.asarray(xi, dtype=np.float64)
    value = np.zeros(points.shape, dtype=np.float64)
    first = np.zeros(points.shape, dtype=np.float64)
    second = np.zeros(points.shape, dtype=np.float64)
    inside = np.abs(points) < 1.0
    if not np.any(inside):
        return value, first, second
    local = points[inside]
    base = 1.0 - local * local
    value[inside] = base**order
    first[inside] = -2.0 * order * local * base ** (order - 1)
    second[inside] = -2.0 * order * base ** (order - 2) * (1.0 - (2.0 * order - 1.0) * local * local)
    return value, first, second


def sensor_nodes(n_space: int, n_sensors: int) -> np.ndarray:
    """Equispaced periodic nodes, the same stride the residual mask uses."""

    if isinstance(n_space, bool) or not isinstance(n_space, (int, np.integer)):
        raise ValueError("n_space must be an even integer")
    width = int(n_space)
    count = int(n_sensors)
    if count < 4 or count % 2 != 0 or width % count != 0:
        raise ValueError("n_sensors must be an even divisor of n_space and >= 4")
    stride = width // count
    return np.asarray(grid(width)[::stride], dtype=np.float64)


def weak_form_terms(
    samples: np.ndarray,
    nodes: np.ndarray,
    times: np.ndarray,
    config: WeakFormConfig,
) -> tuple[np.ndarray, np.ndarray]:
    """One burst. ``samples`` is ``(n_times, n_sensors)`` at ``nodes``.

    Returns ``a, b`` with ``a - ν b ≈ 0``. ``a`` holds the integrated
    time derivative and nonlinear flux. ``b`` holds the integrated
    diffusion test. Neither vector depends on ``ν``.
    """

    if not isinstance(config, WeakFormConfig):
        raise TypeError("config must be a WeakFormConfig")
    values = np.asarray(samples, dtype=np.float64)
    coordinates = np.asarray(nodes, dtype=np.float64)
    clock = np.asarray(times, dtype=np.float64)
    if values.ndim != 2:
        raise ValueError("samples must have shape (n_times, n_sensors)")
    if coordinates.shape != (values.shape[1],) or clock.shape != (values.shape[0],):
        raise ValueError("nodes and times must match the sample axes")
    if values.shape[0] < 3:
        raise ValueError("a weak-form burst needs at least 3 frames")
    if not np.isfinite(values).all() or not np.isfinite(coordinates).all() or not np.isfinite(clock).all():
        raise ValueError("weak-form samples must be finite")
    weights = _time_weights(clock)
    eta, eta_t = _temporal_bump(clock, config.degree)
    spacing = LENGTH / int(values.shape[1])
    radius = float(config.radius_in_spacings) * spacing
    centers = coordinates[:: config.center_stride]
    if centers.size < 1:
        raise ValueError("weak-form centers are empty")
    advection: list[float] = []
    diffusion: list[float] = []
    for center in centers:
        psi, psi_x, psi_xx = _spatial_bump(coordinates, float(center), radius, config.degree)
        weight = weights[:, None] * spacing
        time_term = float(np.sum(values * psi[None, :] * eta_t[:, None] * weight))
        flux_term = float(np.sum((values * values) * 0.5 * psi_x[None, :] * eta[:, None] * weight))
        diffusion_term = float(np.sum(values * psi_xx[None, :] * eta[:, None] * weight))
        advection.append(-time_term - flux_term)
        diffusion.append(diffusion_term)
    return np.asarray(advection, dtype=np.float64), np.asarray(diffusion, dtype=np.float64)


def weak_form_viscosity(
    field: np.ndarray,
    spec: ObservationSpec,
    config: WeakFormConfig | None = None,
) -> float:
    """Estimate ``ν`` from the sensor bursts in ``spec``.

    Columns that are not sensors, and frames that are not in a burst, are
    not read. The returned scalar is not projected onto ``ν > 0``.
    """

    if not isinstance(spec, ObservationSpec):
        raise TypeError("spec must be an ObservationSpec")
    if spec.n_sensors is None:
        raise ValueError("weak form requires a finite sensor count")
    settings = config if config is not None else WeakFormConfig()
    if not isinstance(settings, WeakFormConfig):
        raise TypeError("config must be a WeakFormConfig")
    values = np.asarray(field, dtype=np.float64)
    if values.ndim != 2 or not np.isfinite(values).all():
        raise ValueError("field must be a finite array of shape (n_times, n_space)")
    n_sensors = spec.check_grid(int(values.shape[0]), int(values.shape[1]))
    nodes = sensor_nodes(int(values.shape[1]), n_sensors)
    stride = int(values.shape[1]) // n_sensors
    advection: list[np.ndarray] = []
    diffusion: list[np.ndarray] = []
    for indexes in spec.series:
        frames = np.ascontiguousarray(values[list(indexes)][:, ::stride])
        times = np.asarray(indexes, dtype=np.float64) * float(spec.frame_dt)
        part_a, part_b = weak_form_terms(frames, nodes, times, settings)
        advection.append(part_a)
        diffusion.append(part_b)
    return solve_weak_normal(np.concatenate(advection), np.concatenate(diffusion), settings.projection_floor)


def solve_weak_normal(advection: np.ndarray, diffusion: np.ndarray, projection_floor: float = 1e-12) -> float:
    """``ν̂ = (a · b) / (b · b)`` after dropping numerically empty rows."""

    left = np.asarray(advection, dtype=np.float64).reshape(-1)
    right = np.asarray(diffusion, dtype=np.float64).reshape(-1)
    if left.shape != right.shape or left.size < 1:
        raise ValueError("weak-form rows are empty or mismatched")
    if not np.isfinite(left).all() or not np.isfinite(right).all():
        raise ValueError("weak-form rows must be finite")
    floor = float(projection_floor)
    scale = float(np.max(np.abs(right)))
    if not math.isfinite(scale) or scale <= 0.0:
        raise ValueError("diffusion projection vanishes, so nu is not identifiable")
    keep = np.abs(right) > floor * scale
    if not np.any(keep):
        raise ValueError("every weak-form row was below the projection floor")
    numerator = float(np.dot(left[keep], right[keep]))
    denominator = float(np.dot(right[keep], right[keep]))
    if not math.isfinite(denominator) or denominator <= 0.0:
        raise ValueError("weak-form normal equation is undefined")
    estimate = numerator / denominator
    if not math.isfinite(estimate):
        raise ValueError("weak-form viscosity is not finite")
    return estimate


def _time_weights(times: np.ndarray) -> np.ndarray:
    gaps = np.diff(times)
    if gaps.size < 1 or np.any(gaps <= 0.0):
        raise ValueError("times must be strictly increasing")
    step = float(gaps[0])
    if not np.allclose(gaps, step, rtol=0.0, atol=1e-12 * max(1.0, abs(step))):
        raise ValueError("weak-form time quadrature expects a uniform burst")
    weights = np.full(times.shape, step, dtype=np.float64)
    weights[0] *= 0.5
    weights[-1] *= 0.5
    return weights


def _temporal_bump(times: np.ndarray, degree: int) -> tuple[np.ndarray, np.ndarray]:
    start = float(times[0])
    end = float(times[-1])
    half = 0.5 * (end - start)
    if half <= 0.0:
        raise ValueError("temporal support must have positive length")
    midpoint = 0.5 * (start + end)
    xi = (times - midpoint) / half
    value, first, _second = bump_derivatives(xi, degree)
    return value, first / half


def _spatial_bump(
    nodes: np.ndarray,
    center: float,
    radius: float,
    degree: int,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    if not math.isfinite(radius) or radius <= 0.0:
        raise ValueError("bump radius must be > 0")
    delta = (nodes - float(center) + 0.5 * LENGTH) % LENGTH - 0.5 * LENGTH
    value, first, second = bump_derivatives(delta / radius, degree)
    return value, first / radius, second / (radius * radius)
