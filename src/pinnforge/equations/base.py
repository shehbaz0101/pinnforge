"""Shared domain, initial-condition, and boundary-condition types.

Day 1 stores descriptors. It does not sample collocation points or build a
residual. Later days can extend these models; unknown fields are rejected
so a typo does not pass as a new parameter.
"""

from __future__ import annotations

import math
import re
from typing import Annotated, Literal, Self

from pydantic import BaseModel, BeforeValidator, ConfigDict, Field, model_validator

_NAME = re.compile(r"^[a-z][a-z0-9_]*$")


def _reject_bool(value: object) -> object:
    if isinstance(value, bool):
        raise ValueError("must be a real number")
    return value


Number = Annotated[float, BeforeValidator(_reject_bool)]


class _StrictModel(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True)


def _require_finite(value: float, *, label: str) -> float:
    if isinstance(value, bool) or not isinstance(value, (int, float)) or not math.isfinite(value):
        raise ValueError(f"{label} must be finite")
    return float(value)


def _require_name(value: str, *, label: str) -> str:
    if not isinstance(value, str) or _NAME.fullmatch(value) is None:
        raise ValueError(
            f"{label} must match {_NAME.pattern} (lowercase identifier, digits and underscores)"
        )
    return value


class Interval(_StrictModel):
    """Closed interval ``[lower, upper]`` with positive length.

    Collocation needs a non-empty range, so ``upper == lower`` is rejected.
    """

    lower: Number
    upper: Number

    @model_validator(mode="after")
    def _ordered(self) -> Self:
        _require_finite(self.lower, label="lower")
        _require_finite(self.upper, label="upper")
        if self.upper <= self.lower:
            raise ValueError("upper must be greater than lower")
        return self


class Axis(_StrictModel):
    """One independent variable and the interval it is sampled on."""

    name: str
    bounds: Interval

    @model_validator(mode="after")
    def _name(self) -> Self:
        _require_name(self.name, label="axis name")
        return self


class CollocationDomain(_StrictModel):
    """Axis-aligned box in the independent variables.

    ``axes`` order is the coordinate order a later sampler should use.
    Names are unique.
    """

    axes: list[Axis] = Field(min_length=1)

    @model_validator(mode="after")
    def _unique_names(self) -> Self:
        names = [axis.name for axis in self.axes]
        if len(names) != len(set(names)):
            raise ValueError("collocation axes must have unique names")
        return self

    def interval(self, name: str) -> Interval:
        """Return the bounds of ``name``.

        Raises:
            KeyError: ``name`` is not an axis of this domain.
        """

        for axis in self.axes:
            if axis.name == name:
                return axis.bounds
        raise KeyError(name)


class ConditionDescriptor(_StrictModel):
    """Named condition a later residual can attach a loss to.

    ``description`` is free text for logs. It is not parsed.
    """

    description: str = ""


class StateInitialCondition(ConditionDescriptor):
    """Finite ODE state at the start of ``variable``.

    ``components`` maps a state name to a value. The harmonic oscillator
    requires ``u`` and ``du_dt``.
    """

    kind: Literal["state"] = "state"
    variable: str = "t"
    components: dict[str, Number] = Field(min_length=1)

    @model_validator(mode="after")
    def _components_are_named_reals(self) -> Self:
        _require_name(self.variable, label="initial-condition variable")
        for name, value in self.components.items():
            _require_name(name, label="component name")
            _require_finite(value, label=name)
        return self


class ProfileInitialCondition(ConditionDescriptor):
    """Named spatial profile at the start of ``variable``.

    ``profile`` is a callable name, not a callable. Day 1 does not evaluate
    it. ``params`` are finite scalars a later implementation may read.
    """

    kind: Literal["profile"] = "profile"
    variable: str = "t"
    profile: str
    params: dict[str, Number] = Field(default_factory=dict)

    @model_validator(mode="after")
    def _profile_is_named(self) -> Self:
        _require_name(self.variable, label="initial-condition variable")
        _require_name(self.profile, label="profile")
        for name, value in self.params.items():
            _require_name(name, label="profile param")
            _require_finite(value, label=name)
        return self


class BoundaryCondition(ConditionDescriptor):
    """One condition on the boundary of an independent variable.

    ``variable`` is the axis the face is constant in (``x``, ``y``, or
    ``t``). ``side`` is which end of that variable's interval.

    Dirichlet prescribes the field value. Neumann prescribes the derivative
    with respect to ``variable``; Day 1 stores the number and does not fix a
    sign beyond that. Periodic identifies the two ends of ``variable`` and
    omits both ``side`` and ``value``.

    The descriptor does not evaluate a residual.
    """

    variable: str
    kind: Literal["dirichlet", "neumann", "periodic"]
    side: Literal["min", "max"] | None = None
    value: Number | None = None
    component: str = "u"

    @model_validator(mode="after")
    def _kind_shape(self) -> Self:
        _require_name(self.variable, label="boundary variable")
        _require_name(self.component, label="boundary component")
        if self.kind == "periodic":
            if self.side is not None:
                raise ValueError("periodic boundary conditions omit side")
            if self.value is not None:
                raise ValueError("periodic boundary conditions omit value")
            return self
        if self.side is None:
            raise ValueError(f"{self.kind} boundary conditions require side")
        if self.value is None:
            raise ValueError(f"{self.kind} boundary conditions require value")
        _require_finite(self.value, label="value")
        return self


class EquationSpec(_StrictModel):
    """One residual problem.

    Subclasses set ``equation_id`` and the parameters a later sampler and
    loss need. They do not build a network.
    """

    equation_id: str

    def collocation_domain(self) -> CollocationDomain:
        """Independent-variable box for a later sampler."""

        raise NotImplementedError(f"{type(self).__name__} does not define a collocation domain")
