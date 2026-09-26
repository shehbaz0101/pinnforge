"""Schema validation for the Day 1 equation specs."""

from __future__ import annotations

import math

import pytest
from pydantic import ValidationError

from pinnforge.equations import (
    BURGERS_PROFILES,
    Axis,
    BoundaryCondition,
    Burgers1DSpec,
    CollocationDomain,
    HarmonicOscillatorSpec,
    Interval,
    PoissonSource,
    PoissonToySpec,
    ProfileInitialCondition,
    StateInitialCondition,
    get_equation,
    parse_equation,
    registered_equations,
)


def _interval(lower: float = 0.0, upper: float = 1.0) -> Interval:
    return Interval(lower=lower, upper=upper)


def _harmonic(**overrides: object) -> HarmonicOscillatorSpec:
    payload: dict[str, object] = {
        "omega": 2.0,
        "time": _interval(),
        "initial_condition": StateInitialCondition(components={"u": 1.0, "du_dt": 0.0}),
    }
    payload.update(overrides)
    return HarmonicOscillatorSpec.model_validate(payload)


def _burgers(**overrides: object) -> Burgers1DSpec:
    payload: dict[str, object] = {
        "nu": 0.01 / math.pi,
        "x": _interval(-1.0, 1.0),
        "t": _interval(0.0, 1.0),
        "initial_condition": ProfileInitialCondition(profile="negative_sin_pi_x"),
        "boundary_conditions": [BoundaryCondition(variable="x", kind="periodic")],
    }
    payload.update(overrides)
    return Burgers1DSpec.model_validate(payload)


def _poisson(**overrides: object) -> PoissonToySpec:
    payload: dict[str, object] = {
        "dimensions": 1,
        "source": "sin_pi_x",
        "x": _interval(0.0, 1.0),
        "boundary_conditions": [
            BoundaryCondition(variable="x", kind="dirichlet", side="min", value=0.0),
            BoundaryCondition(variable="x", kind="dirichlet", side="max", value=0.0),
        ],
    }
    payload.update(overrides)
    return PoissonToySpec.model_validate(payload)


def test_interval_and_collocation_domain() -> None:
    domain = CollocationDomain(
        axes=[Axis(name="x", bounds=_interval(-1.0, 1.0)), Axis(name="t", bounds=_interval())]
    )
    assert domain.interval("x").lower == -1.0
    assert domain.interval("t").upper == 1.0
    with pytest.raises(KeyError):
        domain.interval("y")


@pytest.mark.parametrize(
    "payload",
    [
        {"lower": 1.0, "upper": 1.0},
        {"lower": 1.0, "upper": 0.0},
        {"lower": math.nan, "upper": 1.0},
        {"lower": 0.0, "upper": math.inf},
        {"lower": True, "upper": 1.0},
    ],
)
def test_interval_rejects_invalid_bounds(payload: dict[str, float]) -> None:
    with pytest.raises(ValidationError):
        Interval.model_validate(payload)


def test_collocation_domain_rejects_duplicate_axes() -> None:
    with pytest.raises(ValidationError, match="unique"):
        CollocationDomain(axes=[Axis(name="t", bounds=_interval()), Axis(name="t", bounds=_interval())])


def test_boundary_condition_kind_shape() -> None:
    periodic = BoundaryCondition(variable="x", kind="periodic")
    assert periodic.value is None
    valued = BoundaryCondition(variable="x", kind="neumann", side="min", value=-1.5)
    assert valued.value == -1.5
    with pytest.raises(ValidationError):
        BoundaryCondition(variable="x", kind="periodic", value=0.0)
    with pytest.raises(ValidationError):
        BoundaryCondition(variable="x", kind="dirichlet", side="min")
    with pytest.raises(ValidationError):
        BoundaryCondition(variable="x", kind="dirichlet", value=0.0)


def test_harmonic_omega_form_and_collocation() -> None:
    spec = _harmonic()
    assert spec.equation_id == "harmonic_oscillator"
    assert spec.angular_frequency == 2.0
    domain = spec.collocation_domain()
    assert [axis.name for axis in domain.axes] == ["t"]
    assert domain.interval("t") == spec.time
    restored = HarmonicOscillatorSpec.model_validate(spec.model_dump())
    assert restored == spec


def test_harmonic_stiffness_form() -> None:
    spec = _harmonic(omega=None, k=8.0, m=2.0)
    assert spec.angular_frequency == pytest.approx(2.0)
    assert spec.omega is None


def test_harmonic_accepts_optional_boundary_condition() -> None:
    spec = _harmonic(
        boundary_conditions=[
            BoundaryCondition(variable="t", kind="dirichlet", side="max", value=0.0),
        ]
    )
    assert spec.boundary_conditions[0].side == "max"


@pytest.mark.parametrize(
    "overrides",
    [
        {"omega": 0.0},
        {"omega": -1.0},
        {"omega": math.nan},
        {"omega": True},
        {"omega": None, "k": 1.0},
        {"omega": None, "m": 1.0},
        {"omega": 1.0, "k": 1.0, "m": 1.0},
        {"k": 1.0, "m": 1.0},
        {"omega": None, "k": -1.0, "m": 1.0},
        {"omega": None, "k": 1.0, "m": 0.0},
        {"time": {"lower": 1.0, "upper": 0.0}},
        {"initial_condition": {"kind": "state", "components": {"u": 1.0}}},
        {"initial_condition": {"kind": "state", "components": {"u": 1.0, "du_dt": 0.0, "a": 1.0}}},
        {"initial_condition": {"kind": "state", "variable": "x", "components": {"u": 1.0, "du_dt": 0.0}}},
        {"epochs": 3},
        {"equation_id": "burgers_1d"},
    ],
)
def test_harmonic_rejects_invalid_params(overrides: dict[str, object]) -> None:
    payload: dict[str, object] = {
        "omega": 2.0,
        "time": {"lower": 0.0, "upper": 1.0},
        "initial_condition": {"kind": "state", "components": {"u": 1.0, "du_dt": 0.0}},
    }
    payload.update(overrides)
    with pytest.raises(ValidationError):
        HarmonicOscillatorSpec.model_validate(payload)


def test_harmonic_rejects_duplicate_boundary_condition() -> None:
    condition = {
        "variable": "t",
        "kind": "dirichlet",
        "side": "max",
        "value": 0.0,
        "component": "u",
    }
    with pytest.raises(ValidationError, match="duplicate"):
        _harmonic(boundary_conditions=[condition, condition])


def test_harmonic_is_frozen() -> None:
    spec = _harmonic()
    with pytest.raises(ValidationError):
        spec.omega = 1.0  # type: ignore[misc]


def test_burgers_periodic_spec() -> None:
    spec = _burgers()
    assert spec.equation_id == "burgers_1d"
    assert spec.nu == pytest.approx(0.01 / math.pi)
    assert spec.initial_condition.profile in BURGERS_PROFILES
    domain = spec.collocation_domain()
    assert [axis.name for axis in domain.axes] == ["x", "t"]
    restored = parse_equation(spec.model_dump(mode="json"))
    assert isinstance(restored, Burgers1DSpec)
    assert restored == spec


def test_burgers_dirichlet_ends() -> None:
    spec = _burgers(
        initial_condition=ProfileInitialCondition(profile="sin_pi_x"),
        boundary_conditions=[
            BoundaryCondition(variable="x", kind="dirichlet", side="min", value=0.0),
            BoundaryCondition(variable="x", kind="dirichlet", side="max", value=0.0),
        ],
    )
    assert spec.initial_condition.profile == "sin_pi_x"
    assert len(spec.boundary_conditions) == 2


@pytest.mark.parametrize(
    "overrides",
    [
        {"nu": 0.0},
        {"nu": -0.1},
        {"nu": math.inf},
        {"nu": True},
        {"x": {"lower": 1.0, "upper": -1.0}},
        {"t": {"lower": 2.0, "upper": 2.0}},
        {"initial_condition": {"kind": "profile", "profile": "shock"}},
        {"initial_condition": {"kind": "profile", "profile": "sin_pi_x", "params": {"amp": 2.0}}},
        {"boundary_conditions": []},
        {
            "boundary_conditions": [
                {"variable": "y", "kind": "dirichlet", "side": "min", "value": 0.0}
            ]
        },
        {
            "boundary_conditions": [
                {"variable": "x", "kind": "periodic"},
                {"variable": "x", "kind": "dirichlet", "side": "min", "value": 0.0},
            ]
        },
        {"viscosity": 0.1},
    ],
)
def test_burgers_rejects_invalid_params(overrides: dict[str, object]) -> None:
    payload: dict[str, object] = {
        "nu": 0.01,
        "x": {"lower": -1.0, "upper": 1.0},
        "t": {"lower": 0.0, "upper": 1.0},
        "initial_condition": {"kind": "profile", "profile": "negative_sin_pi_x"},
        "boundary_conditions": [{"variable": "x", "kind": "periodic"}],
    }
    payload.update(overrides)
    with pytest.raises(ValidationError):
        Burgers1DSpec.model_validate(payload)


def test_poisson_1d_and_2d() -> None:
    one_d = _poisson()
    assert one_d.dimensions == 1
    assert one_d.y is None
    assert one_d.source is PoissonSource.SIN_PI_X
    assert [axis.name for axis in one_d.collocation_domain().axes] == ["x"]

    two_d = _poisson(
        dimensions=2,
        source="sin_pi_x_sin_pi_y",
        y=_interval(0.0, 1.0),
        boundary_conditions=[
            BoundaryCondition(variable="x", kind="dirichlet", side="min", value=0.0),
            BoundaryCondition(variable="y", kind="dirichlet", side="max", value=0.0),
        ],
    )
    assert two_d.source is PoissonSource.SIN_PI_X_SIN_PI_Y
    assert [axis.name for axis in two_d.collocation_domain().axes] == ["x", "y"]
    assert parse_equation(two_d.model_dump(mode="json")) == two_d


def test_poisson_constant_sources_in_both_dimensions() -> None:
    for source in ("zero", "one"):
        assert _poisson(source=source).source.value == source
        spec = _poisson(
            dimensions=2,
            source=source,
            y=_interval(),
            boundary_conditions=[BoundaryCondition(variable="x", kind="periodic")],
        )
        assert spec.dimensions == 2


@pytest.mark.parametrize(
    "overrides",
    [
        {"dimensions": 3},
        {"dimensions": 1, "y": {"lower": 0.0, "upper": 1.0}},
        {"dimensions": 2, "source": "zero"},
        {"source": "sin_pi_x_sin_pi_y"},
        {"dimensions": 2, "source": "sin_pi_x", "y": {"lower": 0.0, "upper": 1.0}},
        {"source": "callable"},
        {"boundary_conditions": []},
        {
            "boundary_conditions": [
                {"variable": "t", "kind": "dirichlet", "side": "min", "value": 0.0}
            ]
        },
        {
            "boundary_conditions": [
                {"variable": "x", "kind": "periodic"},
                {"variable": "x", "kind": "neumann", "side": "min", "value": 1.0},
            ]
        },
    ],
)
def test_poisson_rejects_invalid_params(overrides: dict[str, object]) -> None:
    payload: dict[str, object] = {
        "dimensions": 1,
        "source": "one",
        "x": {"lower": 0.0, "upper": 1.0},
        "boundary_conditions": [
            {"variable": "x", "kind": "dirichlet", "side": "min", "value": 0.0}
        ],
    }
    payload.update(overrides)
    with pytest.raises(ValidationError):
        PoissonToySpec.model_validate(payload)


def test_registry_lists_day1_equations() -> None:
    assert registered_equations() == ("burgers_1d", "harmonic_oscillator", "poisson_toy")
    assert get_equation("harmonic_oscillator") is HarmonicOscillatorSpec
    assert get_equation("burgers_1d") is Burgers1DSpec
    assert get_equation("poisson_toy") is PoissonToySpec
    with pytest.raises(KeyError, match="unknown equation"):
        get_equation("heat")


def test_parse_equation_requires_a_string_id() -> None:
    with pytest.raises(ValueError, match="equation_id"):
        parse_equation({})
    with pytest.raises(ValueError, match="string"):
        parse_equation({"equation_id": 1})
    spec = parse_equation(
        {
            "equation_id": "harmonic_oscillator",
            "omega": 1.0,
            "time": {"lower": 0.0, "upper": 2.0},
            "initial_condition": {"kind": "state", "components": {"u": 0.0, "du_dt": 1.0}},
        }
    )
    assert isinstance(spec, HarmonicOscillatorSpec)
    assert spec.angular_frequency == 1.0
