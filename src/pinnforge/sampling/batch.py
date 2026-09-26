"""Equation-aware collocation batches.

One ``numpy.random.Generator`` draws the whole batch: interior points,
then the initial-time slice, then boundary faces in descriptor order.
The same spec, config, and numpy Generator stream reproduce the same
coordinates. Field values (Dirichlet numbers, initial profiles) are not
sampled; those stay on the spec for a later residual.
"""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass
from typing import TypeVar

import numpy as np

from pinnforge.equations.base import EquationSpec
from pinnforge.equations.burgers import Burgers1DSpec
from pinnforge.equations.harmonic import HarmonicOscillatorSpec
from pinnforge.equations.poisson import PoissonToySpec
from pinnforge.sampling.config import SAMPLE_METHODS, SampleConfig, SampleMethod
from pinnforge.sampling.draw import require_rng, sample_boundary, sample_collocation

SpecT = TypeVar("SpecT", bound=EquationSpec)

POINT_LABELS: tuple[str, ...] = ("interior", "ic", "bc")


@dataclass(frozen=True, eq=False, slots=True)
class CollocationBatch:
    """Interior, initial-condition, and boundary coordinates for one spec.

    ``interior``, ``ic``, and ``bc`` have shape ``(n, d)`` with columns in
    ``axis_names`` order. ``lower`` and ``upper`` are the closed domain
    bounds, shape ``(d,)``. ``bc_variable`` and ``bc_side`` describe each
    boundary row (``min`` or ``max``). They are empty when ``bc`` is empty.
    """

    equation_id: str
    axis_names: tuple[str, ...]
    lower: np.ndarray
    upper: np.ndarray
    method: SampleMethod
    seed: int
    interior: np.ndarray
    ic: np.ndarray
    bc: np.ndarray
    bc_variable: tuple[str, ...]
    bc_side: tuple[str, ...]

    def __post_init__(self) -> None:
        if not isinstance(self.equation_id, str) or self.equation_id == "":
            raise ValueError("equation_id must be a non-empty string")
        if isinstance(self.seed, bool) or not isinstance(self.seed, int) or self.seed < 0:
            raise ValueError("seed must be a non-negative integer")
        if self.method not in SAMPLE_METHODS:
            known = ", ".join(SAMPLE_METHODS)
            raise ValueError(f"unknown sample method {self.method!r}; known: {known}")
        names = tuple(self.axis_names)
        if not names or not all(isinstance(name, str) and name for name in names):
            raise ValueError("axis names must be non-empty strings")
        if len(set(names)) != len(names):
            raise ValueError("axis names must be unique")
        object.__setattr__(self, "axis_names", names)
        width = len(names)
        lower = _as_vector(self.lower, width, label="lower")
        upper = _as_vector(self.upper, width, label="upper")
        if np.any(upper <= lower):
            raise ValueError("upper bounds must be greater than lower bounds")
        object.__setattr__(self, "lower", lower)
        object.__setattr__(self, "upper", upper)
        interior = _as_points(self.interior, width, label="interior")
        ic = _as_points(self.ic, width, label="ic")
        bc = _as_points(self.bc, width, label="bc")
        _inside(interior, lower, upper, label="interior")
        _inside(ic, lower, upper, label="ic")
        _inside(bc, lower, upper, label="bc")
        object.__setattr__(self, "interior", interior)
        object.__setattr__(self, "ic", ic)
        object.__setattr__(self, "bc", bc)
        variables = tuple(self.bc_variable)
        sides = tuple(self.bc_side)
        if len(variables) != bc.shape[0] or len(sides) != bc.shape[0]:
            raise ValueError("bc_variable and bc_side must have one entry per boundary point")
        if any(not isinstance(name, str) or name == "" for name in variables):
            raise ValueError("bc_variable entries must be non-empty strings")
        if any(side not in ("min", "max") for side in sides):
            raise ValueError("bc_side entries must be 'min' or 'max'")
        object.__setattr__(self, "bc_variable", variables)
        object.__setattr__(self, "bc_side", sides)

    @property
    def counts(self) -> dict[str, int]:
        """Point counts keyed by ``interior``, ``ic``, and ``bc``."""

        return {
            "interior": int(self.interior.shape[0]),
            "ic": int(self.ic.shape[0]),
            "bc": int(self.bc.shape[0]),
        }

    def coordinates(self) -> np.ndarray:
        """Stack interior, then IC, then BC. Shape ``(n_interior + n_ic + n_bc, d)``."""

        return np.concatenate([self.interior, self.ic, self.bc], axis=0)

    def labels(self) -> np.ndarray:
        """Label per row of :meth:`coordinates`: ``interior``, ``ic``, or ``bc``."""

        counts = self.counts
        parts = (
            np.full(counts["interior"], "interior", dtype="U8"),
            np.full(counts["ic"], "ic", dtype="U8"),
            np.full(counts["bc"], "bc", dtype="U8"),
        )
        if counts["interior"] + counts["ic"] + counts["bc"] == 0:
            return np.array([], dtype="U8")
        return np.concatenate(parts)


def format_summary(batch: CollocationBatch) -> str:
    """Short text summary: equation, method, seed, counts, and bounds."""

    if not isinstance(batch, CollocationBatch):
        raise TypeError("batch must be a CollocationBatch")
    counts = batch.counts
    lines = [
        f"equation: {batch.equation_id}",
        f"method: {batch.method}",
        f"seed: {batch.seed}",
        f"counts: interior={counts['interior']} ic={counts['ic']} bc={counts['bc']}",
    ]
    for name, lower, upper in zip(batch.axis_names, batch.lower, batch.upper, strict=True):
        lines.append(f"bounds: {name}=[{float(lower)!r}, {float(upper)!r}]")
    return "\n".join(lines) + "\n"


def sample_equation(
    spec: EquationSpec,
    config: SampleConfig,
    *,
    rng: np.random.Generator | None = None,
) -> CollocationBatch:
    """Sample ``spec`` with ``config``.

    ``rng``, when set, draws the batch instead of
    ``numpy.random.default_rng(config.seed)``. Training and evaluation
    pass a named stream from :mod:`pinnforge.sampling.streams`. The
    batch still records ``config.seed``. Omitting ``rng`` keeps the
    historical single stream, including the prefix overlap between a
    short draw and a longer draw that share that seed.

    Raises:
        TypeError: ``spec`` is not a Day 1 equation spec, ``config`` is
            not a :class:`SampleConfig`, or ``rng`` is not a Generator.
        ValueError: the counts do not fit the spec.
    """

    if isinstance(spec, HarmonicOscillatorSpec):
        return sample_harmonic(spec, config, rng=rng)
    if isinstance(spec, Burgers1DSpec):
        return sample_burgers(spec, config, rng=rng)
    if isinstance(spec, PoissonToySpec):
        return sample_poisson(spec, config, rng=rng)
    raise TypeError("spec must be a HarmonicOscillatorSpec, Burgers1DSpec, or PoissonToySpec")


def sample_harmonic(
    spec: HarmonicOscillatorSpec,
    config: SampleConfig,
    *,
    rng: np.random.Generator | None = None,
) -> CollocationBatch:
    """Collocation batch for the harmonic oscillator.

    Initial-condition rows sit at ``t = time.lower``. Boundary rows follow
    ``boundary_conditions`` when that list is non-empty.
    """

    spec = _require_spec(spec, HarmonicOscillatorSpec)
    config = _require_config(config)
    fixed = {spec.initial_condition.variable: float(spec.time.lower)}
    return _sample_parts(spec, config, ic_fixed=fixed, rng=rng)


def sample_burgers(
    spec: Burgers1DSpec,
    config: SampleConfig,
    *,
    rng: np.random.Generator | None = None,
) -> CollocationBatch:
    """Collocation batch for viscous Burgers.

    Initial-condition rows sit at ``t = t.lower`` with ``x`` drawn on the
    spatial interval. Boundary rows lie on the ``x`` faces.
    """

    spec = _require_spec(spec, Burgers1DSpec)
    config = _require_config(config)
    fixed = {spec.initial_condition.variable: float(spec.t.lower)}
    return _sample_parts(spec, config, ic_fixed=fixed, rng=rng)


def sample_poisson(
    spec: PoissonToySpec,
    config: SampleConfig,
    *,
    rng: np.random.Generator | None = None,
) -> CollocationBatch:
    """Collocation batch for the Poisson toy.

    The problem is elliptic, so ``n_ic`` must be 0. Boundary rows lie on
    the spatial faces.

    Raises:
        ValueError: ``n_ic`` is not 0.
    """

    spec = _require_spec(spec, PoissonToySpec)
    config = _require_config(config)
    return _sample_parts(spec, config, ic_fixed=None, rng=rng)


def _sample_parts(
    spec: EquationSpec,
    config: SampleConfig,
    *,
    ic_fixed: Mapping[str, float] | None,
    rng: np.random.Generator | None = None,
) -> CollocationBatch:
    raw_conditions = getattr(spec, "boundary_conditions", None)
    if raw_conditions is None:
        raise TypeError(f"{type(spec).__name__} has no boundary conditions list")
    conditions = list(raw_conditions)
    if config.n_bc > 0 and not conditions:
        raise ValueError(f"{spec.equation_id} has no boundary conditions; n_bc must be 0")
    domain = spec.collocation_domain()
    generator = require_rng(rng) if rng is not None else np.random.default_rng(config.seed)
    interior = sample_collocation(domain, config.n_interior, generator, config.method)
    if ic_fixed is None:
        if config.n_ic != 0:
            raise ValueError(f"{spec.equation_id} has no initial condition; n_ic must be 0")
        ic = np.empty((0, len(domain.axes)), dtype=np.float64)
    else:
        ic = sample_collocation(domain, config.n_ic, generator, config.method, fixed=dict(ic_fixed))
    bc, variables, sides = sample_boundary(
        domain, conditions, config.n_bc, generator, config.method
    )
    lower = np.array([axis.bounds.lower for axis in domain.axes], dtype=np.float64)
    upper = np.array([axis.bounds.upper for axis in domain.axes], dtype=np.float64)
    return CollocationBatch(
        equation_id=spec.equation_id,
        axis_names=tuple(axis.name for axis in domain.axes),
        lower=lower,
        upper=upper,
        method=config.method,
        seed=config.seed,
        interior=interior,
        ic=ic,
        bc=bc,
        bc_variable=variables,
        bc_side=sides,
    )


def _require_spec(spec: SpecT, expected: type[SpecT]) -> SpecT:
    if not isinstance(spec, expected):
        raise TypeError(f"spec must be a {expected.__name__}")
    return spec


def _require_config(config: SampleConfig) -> SampleConfig:
    if not isinstance(config, SampleConfig):
        raise TypeError("config must be a SampleConfig")
    return config


def _as_vector(value: np.ndarray, width: int, *, label: str) -> np.ndarray:
    array = np.array(value, dtype=np.float64, copy=True)
    if array.shape != (width,):
        raise ValueError(f"{label} must have shape ({width},)")
    if not np.isfinite(array).all():
        raise ValueError(f"{label} must be finite")
    return array


def _as_points(value: np.ndarray, width: int, *, label: str) -> np.ndarray:
    array = np.array(value, dtype=np.float64, copy=True)
    if array.ndim != 2 or array.shape[1] != width:
        raise ValueError(f"{label} coordinates must have shape (n, {width})")
    if not np.isfinite(array).all():
        raise ValueError(f"{label} coordinates must be finite")
    return array


def _inside(array: np.ndarray, lower: np.ndarray, upper: np.ndarray, *, label: str) -> None:
    if array.size == 0:
        return
    if np.any(array < lower) or np.any(array > upper):
        raise ValueError(f"{label} coordinates fall outside the domain")
