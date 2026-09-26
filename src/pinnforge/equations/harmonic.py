"""Harmonic oscillator: u'' + ω² u = 0.

Pass ``omega``, or pass both spring stiffness ``k`` and mass ``m``.
Those are the same parameter: ω = sqrt(k / m). The initial condition is
the state ``(u, du_dt)`` at the lower end of ``time``. Boundary
conditions are optional descriptors for a later boundary-value form; the
initial-value problem does not need them.

The closed form lives in :mod:`pinnforge.reference.harmonic`. This module
only validates the problem statement.
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
    StateInitialCondition,
)
from pinnforge.equations.registry import register_equation

_STATE = frozenset({"u", "du_dt"})


def _positive(value: float, label: str) -> None:
    if not math.isfinite(value) or value <= 0:
        raise ValueError(f"{label} must be finite and > 0")


@register_equation
class HarmonicOscillatorSpec(EquationSpec):
    """Initial-value problem for the undamped harmonic oscillator."""

    equation_id: Literal["harmonic_oscillator"] = "harmonic_oscillator"
    omega: Number | None = None
    k: Number | None = None
    m: Number | None = None
    time: Interval
    initial_condition: StateInitialCondition
    boundary_conditions: list[BoundaryCondition] = Field(default_factory=list)

    @model_validator(mode="after")
    def _parameters(self) -> Self:
        has_omega = self.omega is not None
        has_k = self.k is not None
        has_m = self.m is not None
        if has_omega and (has_k or has_m):
            raise ValueError("pass omega, or both k and m, not both parameterizations")
        if has_omega:
            assert self.omega is not None
            _positive(self.omega, "omega")
        elif has_k and has_m:
            assert self.k is not None and self.m is not None
            _positive(self.k, "k")
            _positive(self.m, "m")
        else:
            raise ValueError("pass omega, or both k and m")
        if self.initial_condition.variable != "t":
            raise ValueError("harmonic oscillator initial condition is on t")
        names = set(self.initial_condition.components)
        missing = _STATE - names
        extra = names - _STATE
        if missing or extra:
            raise ValueError("harmonic oscillator initial condition requires u and du_dt only")
        seen: set[tuple[str, str, str | None]] = set()
        for condition in self.boundary_conditions:
            if condition.variable != "t":
                raise ValueError("harmonic oscillator boundary conditions are on t")
            if condition.component not in _STATE:
                raise ValueError("harmonic oscillator boundary component must be u or du_dt")
            key = (condition.kind, condition.component, condition.side)
            if key in seen:
                raise ValueError("duplicate harmonic oscillator boundary condition")
            seen.add(key)
        return self

    @property
    def angular_frequency(self) -> float:
        """ω from ``omega``, or sqrt(k / m) when stiffness and mass are set."""

        if self.omega is not None:
            return float(self.omega)
        assert self.k is not None and self.m is not None
        return math.sqrt(self.k / self.m)

    def collocation_domain(self) -> CollocationDomain:
        """Time interval the residual is enforced on."""

        return CollocationDomain(axes=[Axis(name="t", bounds=self.time)])
