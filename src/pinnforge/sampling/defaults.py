"""Built-in specs for the ``pinnforge sample`` command.

These are small textbook problems, not a trained model. The harmonic
default is an initial-value problem, so it has no boundary list. Burgers
uses the periodic interval ``x ∈ [-1, 1]``. Poisson defaults to one
dimension with homogeneous Dirichlet ends.
"""

from __future__ import annotations

import math
from typing import Literal

from pinnforge.equations.base import (
    BoundaryCondition,
    EquationSpec,
    Interval,
    ProfileInitialCondition,
    StateInitialCondition,
)
from pinnforge.equations.burgers import Burgers1DSpec
from pinnforge.equations.harmonic import HarmonicOscillatorSpec
from pinnforge.equations.poisson import PoissonSource, PoissonToySpec

EQUATION_ALIASES: dict[str, str] = {
    "harmonic": "harmonic_oscillator",
    "harmonic_oscillator": "harmonic_oscillator",
    "burgers": "burgers_1d",
    "burgers_1d": "burgers_1d",
    "poisson": "poisson_toy",
    "poisson_toy": "poisson_toy",
}

# Used when the CLI omits ``--n-ic`` or ``--n-bc``. The harmonic default
# spec has no boundary conditions, so its boundary count stays at zero.
CLI_COUNT_DEFAULTS: dict[str, dict[str, int]] = {
    "harmonic_oscillator": {"n_ic": 16, "n_bc": 0},
    "burgers_1d": {"n_ic": 16, "n_bc": 16},
    "poisson_toy": {"n_ic": 0, "n_bc": 16},
}


def resolve_equation_id(name: str) -> str:
    """Map a CLI name such as ``harmonic`` to an ``equation_id``."""

    try:
        return EQUATION_ALIASES[name]
    except KeyError:
        known = ", ".join(sorted(EQUATION_ALIASES))
        raise ValueError(f"unknown equation {name!r}; known: {known}") from None


def default_spec(name: str, *, dimensions: Literal[1, 2] | None = None) -> EquationSpec:
    """Return the built-in spec for ``name``.

    ``dimensions`` selects a 1D or 2D Poisson problem. It is rejected for
    the other equations.

    Raises:
        ValueError: ``name`` is unknown, or ``dimensions`` does not apply.
    """

    equation_id = resolve_equation_id(name)
    if equation_id != "poisson_toy" and dimensions is not None:
        raise ValueError("dimensions is only valid for poisson_toy")
    if equation_id == "harmonic_oscillator":
        return _harmonic()
    if equation_id == "burgers_1d":
        return _burgers()
    poisson_dimensions = 1 if dimensions is None else dimensions
    if poisson_dimensions not in (1, 2):
        raise ValueError("Poisson dimensions must be 1 or 2")
    return _poisson(poisson_dimensions)


def _harmonic() -> HarmonicOscillatorSpec:
    return HarmonicOscillatorSpec(
        omega=1.0,
        time=Interval(lower=0.0, upper=1.0),
        initial_condition=StateInitialCondition(components={"u": 1.0, "du_dt": 0.0}),
    )


def _burgers() -> Burgers1DSpec:
    return Burgers1DSpec(
        nu=0.01 / math.pi,
        x=Interval(lower=-1.0, upper=1.0),
        t=Interval(lower=0.0, upper=1.0),
        initial_condition=ProfileInitialCondition(profile="negative_sin_pi_x"),
        boundary_conditions=[BoundaryCondition(variable="x", kind="periodic")],
    )


def _poisson(dimensions: Literal[1, 2]) -> PoissonToySpec:
    if dimensions == 1:
        return PoissonToySpec(
            dimensions=1,
            source=PoissonSource.SIN_PI_X,
            x=Interval(lower=0.0, upper=1.0),
            boundary_conditions=[
                BoundaryCondition(variable="x", kind="dirichlet", side="min", value=0.0),
                BoundaryCondition(variable="x", kind="dirichlet", side="max", value=0.0),
            ],
        )
    return PoissonToySpec(
        dimensions=2,
        source=PoissonSource.SIN_PI_X_SIN_PI_Y,
        x=Interval(lower=0.0, upper=1.0),
        y=Interval(lower=0.0, upper=1.0),
        boundary_conditions=[
            BoundaryCondition(variable="x", kind="dirichlet", side="min", value=0.0),
            BoundaryCondition(variable="x", kind="dirichlet", side="max", value=0.0),
            BoundaryCondition(variable="y", kind="dirichlet", side="min", value=0.0),
            BoundaryCondition(variable="y", kind="dirichlet", side="max", value=0.0),
        ],
    )
