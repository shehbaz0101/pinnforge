"""Burgers has no field reference. Poisson named sources have manufactured fields."""

from __future__ import annotations

import math

import numpy as np
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


def _poisson_1d(source: str) -> PoissonToySpec:
    return PoissonToySpec(
        dimensions=1,
        source=source,
        x=Interval(lower=0.0, upper=1.0),
        boundary_conditions=[
            BoundaryCondition(variable="x", kind="dirichlet", side="min", value=0.0),
        ],
    )


def _poisson_2d(source: str) -> PoissonToySpec:
    return PoissonToySpec(
        dimensions=2,
        source=source,
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
    spec = _poisson_2d("zero")
    assert spec.dimensions == 2
    with pytest.raises(ValueError, match="2D"):
        poisson_solution(spec, [0.0, 1.0])
    with pytest.raises(TypeError, match="PoissonToySpec"):
        poisson_solution(object(), [0.0], [0.0])  # type: ignore[arg-type]
    got = poisson_reference(spec, [0.0, 1.0], [0.25, 0.5])
    assert np.allclose(got, np.zeros(2))


def test_poisson_sin_pi_x_matches_the_manufactured_field() -> None:
    spec = _poisson_1d("sin_pi_x")
    x = np.linspace(0.0, 1.0, 6)
    got = poisson_solution(spec, x)
    assert got.shape == x.shape
    assert np.allclose(got, np.sin(math.pi * x) / (math.pi**2))
    scalar = poisson_solution(spec, 0.5)
    assert scalar.shape == ()
    assert float(scalar) == pytest.approx(1.0 / (math.pi**2))


def test_poisson_sin_pi_product_matches_the_unit_square_field() -> None:
    spec = _poisson_2d("sin_pi_x_sin_pi_y")
    x = np.linspace(0.0, 1.0, 4)
    y = np.linspace(0.0, 1.0, 5)
    xx, yy = np.meshgrid(x, y, indexing="xy")
    got = poisson_solution(spec, xx, yy)
    expect = np.sin(math.pi * xx) * np.sin(math.pi * yy) / (2.0 * math.pi**2)
    assert got.shape == xx.shape
    assert np.allclose(got, expect)
    assert got[0, 0] == pytest.approx(0.0)
    assert got[-1, -1] == pytest.approx(0.0)


def test_poisson_one_is_a_particular_solution() -> None:
    spec = _poisson_1d("one")
    x = np.array([0.0, 0.5, 2.0])
    assert np.allclose(poisson_solution(spec, x), -0.5 * np.square(x))
    spec_2d = _poisson_2d("one")
    y = np.array([0.1, 0.2, 0.3])
    assert np.allclose(poisson_solution(spec_2d, x, y), -0.5 * np.square(x))


def test_poisson_reference_rejects_bad_coordinates() -> None:
    spec = _poisson_2d("sin_pi_x_sin_pi_y")
    with pytest.raises(ValueError, match="same shape"):
        poisson_solution(spec, [0.0, 1.0], [0.0])
    with pytest.raises(ValueError, match="finite"):
        poisson_solution(_poisson_1d("zero"), [0.0, math.nan])
