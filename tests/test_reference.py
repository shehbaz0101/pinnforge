"""Burgers and Poisson reference hooks stay unimplemented on Day 1."""

from __future__ import annotations

import pytest

from pinnforge.equations import (
    BoundaryCondition,
    Burgers1DSpec,
    Interval,
    PoissonToySpec,
    ProfileInitialCondition,
)
from pinnforge.reference import burgers_reference, poisson_reference
from pinnforge.reference.burgers import reference_solution as burgers_solution
from pinnforge.reference.poisson import reference_solution as poisson_solution


def _burgers() -> Burgers1DSpec:
    return Burgers1DSpec(
        nu=0.01,
        x=Interval(lower=-1.0, upper=1.0),
        t=Interval(lower=0.0, upper=1.0),
        initial_condition=ProfileInitialCondition(profile="negative_sin_pi_x"),
        boundary_conditions=[BoundaryCondition(variable="x", kind="periodic")],
    )


def _poisson() -> PoissonToySpec:
    return PoissonToySpec(
        dimensions=2,
        source="zero",
        x=Interval(lower=0.0, upper=1.0),
        y=Interval(lower=0.0, upper=1.0),
        boundary_conditions=[BoundaryCondition(variable="x", kind="periodic")],
    )


def test_burgers_schema_validates_and_reference_is_unimplemented() -> None:
    spec = _burgers()
    assert spec.equation_id == "burgers_1d"
    with pytest.raises(NotImplementedError, match="burgers_1d"):
        burgers_reference(spec, [-1.0, 1.0], [0.0, 1.0])
    with pytest.raises(TypeError, match="Burgers1DSpec"):
        burgers_solution(spec.model_dump(), 0.0, 0.0)  # type: ignore[arg-type]


def test_poisson_schema_validates_and_reference_checks_rank() -> None:
    spec = _poisson()
    assert spec.dimensions == 2
    with pytest.raises(NotImplementedError, match="poisson_toy"):
        poisson_reference(spec, [0.0, 1.0], [0.0, 1.0])
    with pytest.raises(ValueError, match="2D"):
        poisson_solution(spec, [0.0, 1.0])
    with pytest.raises(TypeError, match="PoissonToySpec"):
        poisson_solution(object(), [0.0], [0.0])  # type: ignore[arg-type]
