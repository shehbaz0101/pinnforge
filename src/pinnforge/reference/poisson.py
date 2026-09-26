"""Reference fields for the Poisson toy.

A field is returned only when it matches the named source and every
prescribed boundary condition on the spec's domain. The check uses the
closed form, including its first derivatives, on each face.

Source ``one`` in 1D is the quadratic ``u = -x²/2 + C x + D`` fixed by
the boundary data when those data determine ``C`` and ``D``. On ``[0, 1]``
with ``u(0) = u(1) = 0`` that quadratic is ``x(1 - x) / 2``. The
particular solution ``u = -x²/2`` is returned only when it itself meets
every prescribed condition. It does not meet zero values at both ends.

Sources ``zero``, ``sin_pi_x``, and ``sin_pi_x_sin_pi_y`` use their
manufactured fields when those fields meet the boundary data, including
on a domain other than the unit interval. Neumann values are outward
normal derivatives, the same convention as the boundary penalty.

When no candidate matches, :func:`reference_solution` raises
:class:`ReferenceUnavailable`. Evaluation then reports the reference as
unavailable instead of scoring against a field that breaks the boundary
conditions.
"""

from __future__ import annotations

import math
from collections.abc import Callable

import numpy as np

from pinnforge.equations.base import BoundaryCondition
from pinnforge.equations.poisson import PoissonSource, PoissonToySpec

_ATOL = 1e-8
_RTOL = 1e-8
_FACE_SAMPLES = 7


class ReferenceUnavailable(LookupError):
    """No reference field matches this source, domain, and boundary data."""


class _ClosedForm:
    """Value and coordinate derivative of one candidate field."""

    def __init__(
        self,
        value: Callable[[list[np.ndarray]], np.ndarray],
        derivative: Callable[[str, list[np.ndarray]], np.ndarray],
    ) -> None:
        self._value = value
        self._derivative = derivative

    def value(self, columns: list[np.ndarray]) -> np.ndarray:
        return self._value(columns)

    def derivative(self, variable: str, columns: list[np.ndarray]) -> np.ndarray:
        return self._derivative(variable, columns)


def reference_solution(spec: PoissonToySpec, *coords: object) -> np.ndarray:
    """Return the reference field u sampled at ``coords``.

    Raises:
        TypeError: ``spec`` is not a :class:`PoissonToySpec`.
        ValueError: the coordinate arrays do not match the spec, or a
            value is not finite.
        ReferenceUnavailable: no candidate matches the source, the
            domain, and every prescribed boundary condition.
    """

    if not isinstance(spec, PoissonToySpec):
        raise TypeError("spec must be a PoissonToySpec")
    if len(coords) != spec.dimensions:
        raise ValueError(
            f"poisson_toy in {spec.dimensions}D expects {spec.dimensions} coordinate arrays, "
            f"got {len(coords)}"
        )
    columns = [_coordinate(value, index=index) for index, value in enumerate(coords)]
    shape = columns[0].shape
    for column in columns[1:]:
        if column.shape != shape:
            raise ValueError("poisson coordinate arrays must have the same shape")
    field = _matched_field(spec)
    if field is None:
        raise ReferenceUnavailable(
            "poisson_toy has no reference field that matches the source, domain, "
            "and boundary conditions"
        )
    return field.value(columns)


def _matched_field(spec: PoissonToySpec) -> _ClosedForm | None:
    for candidate in _candidates(spec):
        if _satisfies(candidate, spec):
            return candidate
    return None


def _candidates(spec: PoissonToySpec) -> list[_ClosedForm]:
    if spec.source is PoissonSource.ZERO:
        return [_zeros()]
    if spec.source is PoissonSource.ONE and spec.dimensions == 1:
        candidates: list[_ClosedForm] = []
        solved = _solve_interval_source_one(spec)
        if solved is not None:
            candidates.append(solved)
        candidates.append(_quadratic_x(0.0, 0.0))
        return candidates
    if spec.source is PoissonSource.ONE:
        return [_quadratic_x(0.0, 0.0), _quadratic_y(), _radial_source_one()]
    if spec.source is PoissonSource.SIN_PI_X:
        return [_sin_pi_x()]
    if spec.source is PoissonSource.SIN_PI_X_SIN_PI_Y:
        return [_sin_product()]
    return []


def _solve_interval_source_one(spec: PoissonToySpec) -> _ClosedForm | None:
    """Fit ``u = -x²/2 + C x + D`` when two conditions determine it.

    The fit is still checked against every boundary condition, including
    any that were not used as rows. Neumann rows use the outward normal.
    """

    lower = float(spec.x.lower)
    upper = float(spec.x.upper)
    rows: list[tuple[float, float, float]] = []
    for condition in spec.boundary_conditions:
        if condition.variable != "x" or condition.kind == "periodic":
            continue
        if condition.side is None or condition.value is None:
            continue
        xs = lower if condition.side == "min" else upper
        value = float(condition.value)
        if condition.kind == "dirichlet":
            rows.append((xs, 1.0, value + 0.5 * xs * xs))
        elif condition.kind == "neumann" and condition.side == "max":
            rows.append((1.0, 0.0, value + xs))
        elif condition.kind == "neumann":
            rows.append((1.0, 0.0, xs - value))
    fitted = _fit_pair(rows)
    if fitted is None:
        return None
    slope, offset = fitted
    return _quadratic_x(slope, offset)


def _fit_pair(rows: list[tuple[float, float, float]]) -> tuple[float, float] | None:
    if len(rows) < 2:
        return None
    for index, left in enumerate(rows):
        for right in rows[index + 1 :]:
            a1, b1, c1 = left
            a2, b2, c2 = right
            det = a1 * b2 - a2 * b1
            if abs(det) < 1e-12:
                continue
            slope = (c1 * b2 - c2 * b1) / det
            offset = (a1 * c2 - a2 * c1) / det
            if math.isfinite(slope) and math.isfinite(offset):
                return slope, offset
    return None


def _satisfies(field: _ClosedForm, spec: PoissonToySpec) -> bool:
    for condition in spec.boundary_conditions:
        if not _satisfies_condition(field, spec, condition):
            return False
    return True


def _satisfies_condition(
    field: _ClosedForm,
    spec: PoissonToySpec,
    condition: BoundaryCondition,
) -> bool:
    if condition.kind == "periodic":
        left = _face_columns(spec, condition.variable, "min")
        right = _face_columns(spec, condition.variable, "max")
        values = _close(field.value(left), field.value(right))
        slopes = _close(
            field.derivative(condition.variable, left),
            field.derivative(condition.variable, right),
        )
        return values and slopes
    if condition.side is None or condition.value is None:
        return False
    columns = _face_columns(spec, condition.variable, condition.side)
    if condition.kind == "dirichlet":
        return _close(field.value(columns), float(condition.value))
    if condition.kind == "neumann":
        raw = field.derivative(condition.variable, columns)
        outward = raw if condition.side == "max" else -raw
        return _close(outward, float(condition.value))
    return False


def _face_columns(spec: PoissonToySpec, variable: str, side: str) -> list[np.ndarray]:
    grids: list[np.ndarray] = []
    for axis in spec.collocation_domain().axes:
        if axis.name == variable:
            endpoint = axis.bounds.lower if side == "min" else axis.bounds.upper
            grids.append(np.array([endpoint], dtype=np.float64))
        else:
            grids.append(
                np.linspace(axis.bounds.lower, axis.bounds.upper, _FACE_SAMPLES, dtype=np.float64)
            )
    mesh = np.meshgrid(*grids, indexing="ij")
    return [np.asarray(item, dtype=np.float64).reshape(-1) for item in mesh]


def _close(values: np.ndarray, target: np.ndarray | float) -> bool:
    array = np.asarray(values, dtype=np.float64)
    other = np.asarray(target, dtype=np.float64)
    if array.shape != other.shape and other.shape != ():
        return False
    if not np.isfinite(array).all() or not np.isfinite(other).all():
        return False
    return bool(np.allclose(array, other, rtol=_RTOL, atol=_ATOL))


def _zeros() -> _ClosedForm:
    def value(columns: list[np.ndarray]) -> np.ndarray:
        return np.zeros_like(columns[0], dtype=np.float64)

    def derivative(_variable: str, columns: list[np.ndarray]) -> np.ndarray:
        return np.zeros_like(columns[0], dtype=np.float64)

    return _ClosedForm(value, derivative)


def _quadratic_x(slope: float, offset: float) -> _ClosedForm:
    """``u = -x²/2 + slope * x + offset``. ``-u_xx = 1`` and ``u_yy = 0``."""

    def value(columns: list[np.ndarray]) -> np.ndarray:
        x = columns[0]
        return -0.5 * np.square(x) + slope * x + offset

    def derivative(variable: str, columns: list[np.ndarray]) -> np.ndarray:
        if variable == "x":
            return -columns[0] + slope
        return np.zeros_like(columns[0], dtype=np.float64)

    return _ClosedForm(value, derivative)


def _quadratic_y() -> _ClosedForm:
    """``u = -y²/2``. ``-u_yy = 1`` and ``u_xx = 0``."""

    def value(columns: list[np.ndarray]) -> np.ndarray:
        return -0.5 * np.square(columns[1])

    def derivative(variable: str, columns: list[np.ndarray]) -> np.ndarray:
        if variable == "y":
            return -columns[1]
        return np.zeros_like(columns[0], dtype=np.float64)

    return _ClosedForm(value, derivative)


def _radial_source_one() -> _ClosedForm:
    """``u = -(x² + y²) / 4``. ``-Δu = 1``."""

    def value(columns: list[np.ndarray]) -> np.ndarray:
        return -(np.square(columns[0]) + np.square(columns[1])) / 4.0

    def derivative(variable: str, columns: list[np.ndarray]) -> np.ndarray:
        if variable == "x":
            return -0.5 * columns[0]
        if variable == "y":
            return -0.5 * columns[1]
        return np.zeros_like(columns[0], dtype=np.float64)

    return _ClosedForm(value, derivative)


def _sin_pi_x() -> _ClosedForm:
    def value(columns: list[np.ndarray]) -> np.ndarray:
        return np.sin(math.pi * columns[0]) / (math.pi**2)

    def derivative(variable: str, columns: list[np.ndarray]) -> np.ndarray:
        if variable == "x":
            return np.cos(math.pi * columns[0]) / math.pi
        return np.zeros_like(columns[0], dtype=np.float64)

    return _ClosedForm(value, derivative)


def _sin_product() -> _ClosedForm:
    scale = 2.0 * math.pi**2

    def value(columns: list[np.ndarray]) -> np.ndarray:
        return np.sin(math.pi * columns[0]) * np.sin(math.pi * columns[1]) / scale

    def derivative(variable: str, columns: list[np.ndarray]) -> np.ndarray:
        if variable == "x":
            return math.pi * np.cos(math.pi * columns[0]) * np.sin(math.pi * columns[1]) / scale
        if variable == "y":
            return math.pi * np.sin(math.pi * columns[0]) * np.cos(math.pi * columns[1]) / scale
        return np.zeros_like(columns[0], dtype=np.float64)

    return _ClosedForm(value, derivative)


def _coordinate(value: object, *, index: int) -> np.ndarray:
    array = np.asarray(value, dtype=np.float64)
    if not np.isfinite(array).all():
        raise ValueError(f"poisson coordinate {index} must be finite")
    return array
