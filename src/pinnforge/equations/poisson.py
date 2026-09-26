"""Poisson toy on a 1D interval or a 2D rectangle.

The residual is -Δu - f = 0. ``source`` names f. It is an enum, not a
callable, so a spec stays serializable:

- ``zero`` and ``one`` are valid in 1D and 2D
- ``sin_pi_x`` is 1D only, f(x) = sin(π x)
- ``sin_pi_x_sin_pi_y`` is 2D only, f(x, y) = sin(π x) sin(π y)

There is no initial condition; the problem is elliptic. ``y`` is required
in 2D and forbidden in 1D. Boundary conditions name the spatial faces.
Neumann-only data leaves a constant null space; the schema still accepts
it and records the conditions as given.

:func:`pinnforge.reference.poisson.reference_solution` is the evaluation
hook and raises ``NotImplementedError`` on Day 1.
"""

from __future__ import annotations

from enum import Enum
from typing import Literal, Self

from pydantic import Field, model_validator

from pinnforge.equations.base import (
    Axis,
    BoundaryCondition,
    CollocationDomain,
    EquationSpec,
    Interval,
)
from pinnforge.equations.registry import register_equation


class PoissonSource(str, Enum):
    """Named source term f in -Δu = f."""

    ZERO = "zero"
    ONE = "one"
    SIN_PI_X = "sin_pi_x"
    SIN_PI_X_SIN_PI_Y = "sin_pi_x_sin_pi_y"


_SOURCE_DIMENSIONS: dict[PoissonSource, frozenset[int]] = {
    PoissonSource.ZERO: frozenset({1, 2}),
    PoissonSource.ONE: frozenset({1, 2}),
    PoissonSource.SIN_PI_X: frozenset({1}),
    PoissonSource.SIN_PI_X_SIN_PI_Y: frozenset({2}),
}


@register_equation
class PoissonToySpec(EquationSpec):
    """1D or 2D Poisson problem with a named source and boundary data."""

    equation_id: Literal["poisson_toy"] = "poisson_toy"
    dimensions: Literal[1, 2]
    source: PoissonSource
    x: Interval
    y: Interval | None = None
    boundary_conditions: list[BoundaryCondition] = Field(min_length=1)

    @model_validator(mode="after")
    def _problem(self) -> Self:
        if self.dimensions == 1:
            if self.y is not None:
                raise ValueError("1D Poisson omits y")
            allowed = frozenset({"x"})
        else:
            if self.y is None:
                raise ValueError("2D Poisson requires y")
            allowed = frozenset({"x", "y"})
        allowed_dims = _SOURCE_DIMENSIONS[self.source]
        if self.dimensions not in allowed_dims:
            raise ValueError(
                f"source {self.source.value!r} is not valid in {self.dimensions}D"
            )
        periodic: set[str] = set()
        valued: set[str] = set()
        seen: set[tuple[str, str, str, str | None]] = set()
        for condition in self.boundary_conditions:
            if condition.variable not in allowed:
                raise ValueError(
                    f"Poisson boundary variable must be one of {', '.join(sorted(allowed))}"
                )
            if condition.component != "u":
                raise ValueError("Poisson boundary component must be u")
            key = (condition.variable, condition.kind, condition.component, condition.side)
            if key in seen:
                raise ValueError("duplicate Poisson boundary condition")
            seen.add(key)
            if condition.kind == "periodic":
                periodic.add(condition.variable)
            else:
                valued.add(condition.variable)
        overlap = periodic & valued
        if overlap:
            names = ", ".join(sorted(overlap))
            raise ValueError(
                f"periodic Poisson data on {names} cannot be combined with Dirichlet or Neumann data"
            )
        return self

    def collocation_domain(self) -> CollocationDomain:
        """Spatial box. 1D is ``x``; 2D is ``x`` then ``y``."""

        axes = [Axis(name="x", bounds=self.x)]
        if self.dimensions == 2:
            assert self.y is not None
            axes.append(Axis(name="y", bounds=self.y))
        return CollocationDomain(axes=axes)
