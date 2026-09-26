"""Uniform, Latin-hypercube, and stratified draws on a collocation domain.

Interior coordinates land in the half-open box ``[lower, upper)`` along
each free axis. A fixed axis is set to that value on every row and does
not consume random numbers, so an initial-condition or boundary face can
share one generator with the interior draw.

The generator is ``numpy.random.Generator`` from
``numpy.random.default_rng``. The legacy ``RandomState`` is rejected so a
seed cannot silently follow a different stream.
"""

from __future__ import annotations

import math
from collections.abc import Mapping, Sequence

import numpy as np

from pinnforge.equations.base import BoundaryCondition, CollocationDomain
from pinnforge.sampling.config import SAMPLE_METHODS, SampleMethod


def require_count(n: int, *, label: str) -> int:
    """Return ``n`` when it is a non-negative integer."""

    if isinstance(n, bool) or not isinstance(n, int) or n < 0:
        raise ValueError(f"{label} must be a non-negative integer")
    return n


def require_rng(rng: np.random.Generator) -> np.random.Generator:
    """Return ``rng`` when it is a ``numpy.random.Generator``."""

    if not isinstance(rng, np.random.Generator):
        raise TypeError("rng must be a numpy.random.Generator; use numpy.random.default_rng(seed)")
    return rng


def sample_collocation(
    domain: CollocationDomain,
    n: int,
    rng: np.random.Generator,
    method: SampleMethod = "uniform",
    *,
    fixed: Mapping[str, float] | None = None,
) -> np.ndarray:
    """Draw ``n`` points on ``domain``.

    Columns follow ``domain.axes`` order. Axes named in ``fixed`` are
    constant. The other axes are drawn with ``method`` on ``[lower, upper)``.
    ``n == 0`` returns shape ``(0, d)`` and does not advance ``rng``.

    Raises:
        TypeError: ``domain`` or ``rng`` has the wrong type.
        ValueError: ``n`` is negative, ``method`` is unknown, or a fixed
            coordinate is missing, non-finite, or outside the closed interval.
    """

    if not isinstance(domain, CollocationDomain):
        raise TypeError("domain must be a CollocationDomain")
    require_rng(rng)
    require_count(n, label="n")
    _require_method(method)
    names = [axis.name for axis in domain.axes]
    fixed_values = _fixed_values(domain, names, fixed)
    width = len(names)
    if n == 0:
        return np.empty((0, width), dtype=np.float64)
    coords = np.empty((n, width), dtype=np.float64)
    for name, value in fixed_values.items():
        coords[:, names.index(name)] = value
    free = [index for index, name in enumerate(names) if name not in fixed_values]
    if not free:
        return coords
    unit = _unit_sample(n, len(free), rng, method)
    for column, axis_index in enumerate(free):
        bounds = domain.axes[axis_index].bounds
        coords[:, axis_index] = bounds.lower + unit[:, column] * (bounds.upper - bounds.lower)
    return coords


def sample_boundary(
    domain: CollocationDomain,
    conditions: Sequence[BoundaryCondition],
    n: int,
    rng: np.random.Generator,
    method: SampleMethod = "uniform",
) -> tuple[np.ndarray, tuple[str, ...], tuple[str, ...]]:
    """Draw ``n`` points on the faces named by ``conditions``.

    Dirichlet and Neumann each contribute the face ``side`` names.
    Periodic contributes the minimum face and then the maximum face.
    Points are split across those faces in order; earlier faces receive
    the remainder. A face with a zero share is omitted and does not
    advance ``rng``.

    Returns coordinates of shape ``(n, d)``, the boundary variable of each
    row, and ``min`` or ``max`` for each row.

    Raises:
        ValueError: ``n > 0`` but ``conditions`` names no face, or a
            boundary variable is not an axis of ``domain``.
    """

    if not isinstance(domain, CollocationDomain):
        raise TypeError("domain must be a CollocationDomain")
    require_rng(rng)
    require_count(n, label="n_bc")
    _require_method(method)
    width = len(domain.axes)
    if n == 0:
        return np.empty((0, width), dtype=np.float64), (), ()
    faces = boundary_faces(conditions)
    if not faces:
        raise ValueError("n_bc > 0 but the spec has no boundary conditions")
    blocks: list[np.ndarray] = []
    variables: list[str] = []
    sides: list[str] = []
    for (variable, side), count in zip(faces, _split_count(n, len(faces)), strict=True):
        if count == 0:
            continue
        try:
            interval = domain.interval(variable)
        except KeyError:
            raise ValueError(
                f"boundary variable {variable!r} is not in the collocation domain"
            ) from None
        value = interval.lower if side == "min" else interval.upper
        block = sample_collocation(domain, count, rng, method, fixed={variable: value})
        blocks.append(block)
        variables.extend([variable] * count)
        sides.extend([side] * count)
    if not blocks:
        return np.empty((0, width), dtype=np.float64), (), ()
    return np.concatenate(blocks, axis=0), tuple(variables), tuple(sides)


def boundary_faces(conditions: Sequence[BoundaryCondition]) -> list[tuple[str, str]]:
    """Faces in sample order: ``(variable, side)`` with ``side`` ``min`` or ``max``."""

    faces: list[tuple[str, str]] = []
    for condition in conditions:
        if not isinstance(condition, BoundaryCondition):
            raise TypeError("boundary conditions must be BoundaryCondition descriptors")
        if condition.kind == "periodic":
            faces.append((condition.variable, "min"))
            faces.append((condition.variable, "max"))
            continue
        if condition.side is None:
            raise ValueError(f"{condition.kind} boundary conditions require side")
        faces.append((condition.variable, condition.side))
    return faces


def _require_method(method: str) -> None:
    if method not in SAMPLE_METHODS:
        known = ", ".join(SAMPLE_METHODS)
        raise ValueError(f"unknown sample method {method!r}; known: {known}")


def _fixed_values(
    domain: CollocationDomain,
    names: list[str],
    fixed: Mapping[str, float] | None,
) -> dict[str, float]:
    if fixed is None:
        return {}
    unknown = sorted(set(fixed) - set(names))
    if unknown:
        raise ValueError(f"fixed axes not in the domain: {', '.join(unknown)}")
    values: dict[str, float] = {}
    for name, raw in fixed.items():
        if isinstance(raw, bool) or not isinstance(raw, (int, float)) or not math.isfinite(raw):
            raise ValueError(f"fixed coordinate {name} must be finite")
        value = float(raw)
        bounds = domain.interval(name)
        if value < bounds.lower or value > bounds.upper:
            raise ValueError(f"fixed coordinate {name}={value} is outside the domain")
        values[name] = value
    return values


def _split_count(n: int, parts: int) -> list[int]:
    if parts < 1:
        raise ValueError("cannot split a count across zero faces")
    base, extra = divmod(n, parts)
    return [base + (1 if index < extra else 0) for index in range(parts)]


def _unit_sample(n: int, width: int, rng: np.random.Generator, method: SampleMethod) -> np.ndarray:
    if width < 1:
        raise ValueError("sample dimension must be positive")
    if method == "uniform":
        return rng.random((n, width))
    if method == "latin_hypercube":
        return _latin_hypercube(n, width, rng)
    if method == "stratified":
        return _stratified(n, width, rng)
    known = ", ".join(SAMPLE_METHODS)
    raise ValueError(f"unknown sample method {method!r}; known: {known}")


def _latin_hypercube(n: int, width: int, rng: np.random.Generator) -> np.ndarray:
    """One jittered point in each 1D stratum, independently permuted per axis."""

    unit = np.empty((n, width), dtype=np.float64)
    for axis in range(width):
        bins = rng.permutation(n).astype(np.float64)
        unit[:, axis] = (bins + rng.random(n)) / n
    return unit


def _stratified(n: int, width: int, rng: np.random.Generator) -> np.ndarray:
    """One jittered point in each cell of a nearly cubic partition.

    The grid is ``side`` cells along every axis, with ``side`` the smallest
    integer whose power covers ``n``. Extra cells are dropped with
    ``rng.choice``. When the grid is exact, only the jitter advances
    ``rng``. In one dimension the points stay in bin order, which is the
    difference from a Latin hypercube.
    """

    side = math.ceil(n ** (1.0 / width))
    cells = _cell_origins(side, width)
    total = int(cells.shape[0])
    if total < n:
        raise ValueError("stratified grid is smaller than the sample count")
    if total == n:
        chosen = cells
    else:
        pick = np.sort(rng.choice(total, size=n, replace=False))
        chosen = cells[pick]
    jitter = rng.random((n, width))
    return (chosen + jitter) / float(side)


def _cell_origins(side: int, width: int) -> np.ndarray:
    axes = [np.arange(side, dtype=np.float64) for _ in range(width)]
    mesh = np.meshgrid(*axes, indexing="ij")
    return np.stack([item.reshape(-1) for item in mesh], axis=1)
