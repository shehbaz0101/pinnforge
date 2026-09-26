"""Viscous Burgers equation in one space dimension.

The residual is u_t + u u_x - ν u_xx = 0 on ``x`` × ``t``. ``nu`` is the
viscosity ν and must be finite and positive. Inviscid Burgers (ν = 0) is
not a Day 1 spec.

The initial condition is a named profile at the lower end of ``t``, not a
callable. ``negative_sin_pi_x`` is the usual Raissi initial condition
u(x, 0) = -sin(π x), conventionally on x ∈ [-1, 1]; the schema does not
force that interval. Neither profile takes parameters.

Boundary conditions are on ``x`` only. A periodic condition covers both
ends and cannot be combined with Dirichlet or Neumann data on ``x``.

:func:`pinnforge.reference.burgers.reference_solution` raises
``NotImplementedError``. Evaluation scores Burgers with residual metrics
only.
"""

from __future__ import annotations

import math
from typing import Literal, Self

from pydantic import Field, model_validator

from pinnforge.equations.base import (
    Axis,
    BoundaryCondition,
    CollocationDomain,
    EquationSpec,
    Interval,
    Number,
    ProfileInitialCondition,
)
from pinnforge.equations.registry import register_equation

BURGERS_PROFILES = frozenset({"negative_sin_pi_x", "sin_pi_x"})


def _positive(value: float, label: str) -> None:
    if not math.isfinite(value) or value <= 0:
        raise ValueError(f"{label} must be finite and > 0")


@register_equation
class Burgers1DSpec(EquationSpec):
    """Viscous Burgers problem on a 1D interval times a time interval."""

    equation_id: Literal["burgers_1d"] = "burgers_1d"
    nu: Number
    x: Interval
    t: Interval
    initial_condition: ProfileInitialCondition
    boundary_conditions: list[BoundaryCondition] = Field(min_length=1)

    @model_validator(mode="after")
    def _problem(self) -> Self:
        _positive(self.nu, "nu")
        if self.initial_condition.variable != "t":
            raise ValueError("Burgers initial condition is on t")
        if self.initial_condition.profile not in BURGERS_PROFILES:
            known = ", ".join(sorted(BURGERS_PROFILES))
            raise ValueError(
                f"unknown Burgers profile {self.initial_condition.profile!r}; known: {known}"
            )
        if self.initial_condition.params:
            raise ValueError(f"{self.initial_condition.profile} does not take params")
        periodic = False
        valued = False
        seen: set[tuple[str, str, str | None]] = set()
        for condition in self.boundary_conditions:
            if condition.variable != "x":
                raise ValueError("Burgers boundary conditions are on x")
            if condition.component != "u":
                raise ValueError("Burgers boundary component must be u")
            key = (condition.kind, condition.component, condition.side)
            if key in seen:
                raise ValueError("duplicate Burgers boundary condition")
            seen.add(key)
            if condition.kind == "periodic":
                periodic = True
            else:
                valued = True
        if periodic and valued:
            raise ValueError("periodic Burgers data cannot be combined with Dirichlet or Neumann data")
        return self

    def collocation_domain(self) -> CollocationDomain:
        """Space then time, the order a later sampler should use."""

        return CollocationDomain(axes=[Axis(name="x", bounds=self.x), Axis(name="t", bounds=self.t)])
