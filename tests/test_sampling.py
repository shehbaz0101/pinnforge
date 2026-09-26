"""Collocation, initial-condition, and boundary sampling."""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pytest
from pydantic import ValidationError

from pinnforge.equations import (
    Axis,
    BoundaryCondition,
    Burgers1DSpec,
    CollocationDomain,
    HarmonicOscillatorSpec,
    Interval,
    PoissonToySpec,
    ProfileInitialCondition,
    StateInitialCondition,
)
from pinnforge.sampling import (
    POINT_LABELS,
    SAMPLE_METHODS,
    STREAM_NAMES,
    SampleConfig,
    batch_from_dict,
    batch_to_dict,
    default_spec,
    load_sample_record,
    resolve_output_path,
    sample_boundary,
    sample_burgers,
    sample_collocation,
    sample_equation,
    sample_harmonic,
    sample_poisson,
    stream_generator,
    write_sample_record,
)

FIXTURES = Path(__file__).resolve().parent / "fixtures" / "sampling"
FIXTURE_NAMES = (
    "harmonic_seed0",
    "burgers_seed0",
    "poisson1d_seed0",
    "poisson2d_seed0",
)


def _unit_interval() -> Interval:
    return Interval(lower=0.0, upper=1.0)


def _domain_t() -> CollocationDomain:
    return CollocationDomain(axes=[Axis(name="t", bounds=_unit_interval())])


def _domain_xt() -> CollocationDomain:
    return CollocationDomain(
        axes=[
            Axis(name="x", bounds=Interval(lower=-1.0, upper=1.0)),
            Axis(name="t", bounds=_unit_interval()),
        ]
    )


def _harmonic(**overrides: object) -> HarmonicOscillatorSpec:
    payload: dict[str, object] = {
        "omega": 2.0,
        "time": Interval(lower=0.0, upper=1.0),
        "initial_condition": StateInitialCondition(components={"u": 1.0, "du_dt": 0.0}),
    }
    payload.update(overrides)
    return HarmonicOscillatorSpec.model_validate(payload)


def _burgers(**overrides: object) -> Burgers1DSpec:
    payload: dict[str, object] = {
        "nu": 0.1,
        "x": Interval(lower=-1.0, upper=1.0),
        "t": Interval(lower=0.0, upper=1.0),
        "initial_condition": ProfileInitialCondition(profile="negative_sin_pi_x"),
        "boundary_conditions": [
            BoundaryCondition(variable="x", kind="dirichlet", side="min", value=0.0),
            BoundaryCondition(variable="x", kind="dirichlet", side="max", value=0.0),
        ],
    }
    payload.update(overrides)
    return Burgers1DSpec.model_validate(payload)


def _poisson_1d(**overrides: object) -> PoissonToySpec:
    payload: dict[str, object] = {
        "dimensions": 1,
        "source": "sin_pi_x",
        "x": _unit_interval(),
        "boundary_conditions": [
            BoundaryCondition(variable="x", kind="dirichlet", side="min", value=0.0),
            BoundaryCondition(variable="x", kind="dirichlet", side="max", value=0.0),
        ],
    }
    payload.update(overrides)
    return PoissonToySpec.model_validate(payload)


def _poisson_2d() -> PoissonToySpec:
    return PoissonToySpec(
        dimensions=2,
        source="sin_pi_x_sin_pi_y",
        x=Interval(lower=0.0, upper=2.0),
        y=Interval(lower=-0.5, upper=0.5),
        boundary_conditions=[
            BoundaryCondition(variable="x", kind="dirichlet", side="min", value=1.0),
            BoundaryCondition(variable="x", kind="neumann", side="max", value=-2.0),
            BoundaryCondition(variable="y", kind="dirichlet", side="min", value=3.0),
            BoundaryCondition(variable="y", kind="dirichlet", side="max", value=4.0),
        ],
    )


def _inside(points: np.ndarray, lower: np.ndarray, upper: np.ndarray) -> None:
    assert points.dtype == np.float64
    if points.size == 0:
        return
    assert np.all(points >= lower)
    assert np.all(points <= upper)
    assert np.isfinite(points).all()


def test_config_rejects_bad_counts() -> None:
    with pytest.raises(ValidationError, match="at least one"):
        SampleConfig(n_interior=0, n_ic=0, n_bc=0)
    with pytest.raises(ValidationError, match="greater than or equal to 0"):
        SampleConfig(n_interior=-1)
    with pytest.raises(ValidationError, match="greater than or equal to 0"):
        SampleConfig(n_interior=1, seed=-1)
    with pytest.raises(ValidationError, match="integer"):
        SampleConfig(n_interior=True)  # type: ignore[arg-type]
    with pytest.raises(ValidationError, match="integer"):
        SampleConfig(n_interior=1.5)  # type: ignore[arg-type]
    with pytest.raises(ValidationError, match="Extra inputs"):
        SampleConfig.model_validate({"n_interior": 4, "epochs": 3})
    with pytest.raises(ValidationError):
        SampleConfig(n_interior=4, method="sobol")  # type: ignore[arg-type]
    config = SampleConfig(n_interior=4)
    with pytest.raises(ValidationError):
        config.n_interior = 8  # type: ignore[misc]


def test_config_accepts_each_method() -> None:
    for method in SAMPLE_METHODS:
        config = SampleConfig(n_interior=2, method=method)  # type: ignore[arg-type]
        assert config.method == method


def test_low_level_rejects_bad_arguments() -> None:
    domain = _domain_t()
    rng = np.random.default_rng(0)
    with pytest.raises(TypeError, match="Generator"):
        sample_collocation(domain, 1, np.random.RandomState(0))  # type: ignore[arg-type]
    with pytest.raises(ValueError, match="non-negative"):
        sample_collocation(domain, -1, rng)
    with pytest.raises(ValueError, match="non-negative"):
        sample_collocation(domain, True, rng)  # type: ignore[arg-type]
    with pytest.raises(ValueError, match="unknown sample method"):
        sample_collocation(domain, 1, rng, "sobol")  # type: ignore[arg-type]
    with pytest.raises(ValueError, match="not in the domain"):
        sample_collocation(domain, 1, rng, fixed={"z": 0.0})
    with pytest.raises(ValueError, match="outside"):
        sample_collocation(domain, 1, rng, fixed={"t": 2.0})
    with pytest.raises(ValueError, match="finite"):
        sample_collocation(domain, 1, rng, fixed={"t": float("nan")})
    with pytest.raises(ValueError, match="no boundary"):
        sample_boundary(domain, [], 2, rng)


def test_zero_count_does_not_advance_rng() -> None:
    domain = _domain_t()
    rng = np.random.default_rng(0)
    empty = sample_collocation(domain, 0, rng, "uniform")
    followed = sample_collocation(domain, 4, rng, "uniform")
    fresh = sample_collocation(domain, 4, np.random.default_rng(0), "uniform")
    assert empty.shape == (0, 1)
    np.testing.assert_array_equal(followed, fresh)


def test_fixed_axes_do_not_advance_rng_and_uniform_scale() -> None:
    domain = _domain_xt()
    rng = np.random.default_rng(0)
    held = sample_collocation(domain, 4, rng, "uniform", fixed={"x": -1.0, "t": 0.0})
    assert np.all(held[:, 0] == -1.0)
    assert np.all(held[:, 1] == 0.0)
    followed = sample_collocation(domain, 3, rng, "uniform")
    fresh = sample_collocation(domain, 3, np.random.default_rng(0), "uniform")
    np.testing.assert_array_equal(followed, fresh)

    partial = sample_collocation(domain, 4, np.random.default_rng(0), "uniform", fixed={"t": 0.0})
    unit = np.random.default_rng(0).random((4, 1))
    expected_x = -1.0 + unit[:, 0] * 2.0
    np.testing.assert_array_equal(partial[:, 0], expected_x)
    assert np.all(partial[:, 1] == 0.0)


def test_stratified_1d_keeps_bin_order() -> None:
    points = sample_collocation(_domain_t(), 8, np.random.default_rng(0), "stratified")
    assert points.shape == (8, 1)
    for index, value in enumerate(points[:, 0]):
        assert index / 8 <= value < (index + 1) / 8


def test_latin_hypercube_1d_hits_each_bin() -> None:
    points = sample_collocation(_domain_t(), 8, np.random.default_rng(0), "latin_hypercube")
    bins = np.sort((points[:, 0] * 8).astype(int))
    assert bins.tolist() == list(range(8))


def test_latin_hypercube_2d_stratifies_each_margin() -> None:
    domain = CollocationDomain(
        axes=[Axis(name="x", bounds=_unit_interval()), Axis(name="y", bounds=_unit_interval())]
    )
    points = sample_collocation(domain, 8, np.random.default_rng(1), "latin_hypercube")
    assert points.shape == (8, 2)
    for axis in range(2):
        bins = np.sort((points[:, axis] * 8).astype(int))
        assert bins.tolist() == list(range(8))


def test_stratified_2d_uses_one_point_per_cell() -> None:
    domain = CollocationDomain(
        axes=[Axis(name="x", bounds=_unit_interval()), Axis(name="y", bounds=_unit_interval())]
    )
    points = sample_collocation(domain, 4, np.random.default_rng(0), "stratified")
    expected_cells = [(0, 0), (0, 1), (1, 0), (1, 1)]
    for (cell_x, cell_y), row in zip(expected_cells, points, strict=True):
        assert cell_x / 2 <= row[0] < (cell_x + 1) / 2
        assert cell_y / 2 <= row[1] < (cell_y + 1) / 2


@pytest.mark.parametrize("method", SAMPLE_METHODS)
def test_methods_are_reproducible_and_distinct(method: str) -> None:
    domain = _domain_xt()
    first = sample_collocation(domain, 16, np.random.default_rng(0), method)  # type: ignore[arg-type]
    second = sample_collocation(domain, 16, np.random.default_rng(0), method)  # type: ignore[arg-type]
    other = sample_collocation(domain, 16, np.random.default_rng(1), method)  # type: ignore[arg-type]
    np.testing.assert_array_equal(first, second)
    assert not np.array_equal(first, other)
    _inside(first, np.array([-1.0, 0.0]), np.array([1.0, 1.0]))


def test_methods_do_not_match_each_other() -> None:
    domain = _domain_t()
    draws = [
        sample_collocation(domain, 8, np.random.default_rng(0), method)  # type: ignore[arg-type]
        for method in SAMPLE_METHODS
    ]
    assert not np.array_equal(draws[0], draws[1])
    assert not np.array_equal(draws[0], draws[2])
    assert not np.array_equal(draws[1], draws[2])


def test_harmonic_ic_matches_initial_time_and_bounds() -> None:
    spec = _harmonic(time=Interval(lower=0.1, upper=0.3))
    config = SampleConfig(n_interior=32, n_ic=7, n_bc=0, seed=3, method="uniform")
    first = sample_harmonic(spec, config)
    second = sample_equation(spec, config)
    assert first.axis_names == ("t",)
    assert first.interior.shape == (32, 1)
    assert first.ic.shape == (7, 1)
    assert first.bc.shape == (0, 1)
    assert np.all(first.ic[:, 0] == spec.time.lower)
    _inside(first.interior, first.lower, first.upper)
    assert np.min(first.interior) < spec.time.upper
    np.testing.assert_array_equal(first.coordinates(), second.coordinates())
    assert first.labels().tolist() == ["interior"] * 32 + ["ic"] * 7
    other = sample_harmonic(spec, config.model_copy(update={"seed": 4}))
    assert not np.array_equal(first.interior, other.interior)


def test_harmonic_boundary_uses_the_time_endpoint() -> None:
    spec = _harmonic(
        time=Interval(lower=0.25, upper=1.5),
        boundary_conditions=[
            BoundaryCondition(variable="t", kind="dirichlet", side="min", value=9.0),
            BoundaryCondition(variable="t", kind="dirichlet", side="max", value=-4.0, component="du_dt"),
        ],
    )
    batch = sample_harmonic(spec, SampleConfig(n_interior=8, n_ic=3, n_bc=5, seed=0))
    assert np.all(batch.ic[:, 0] == 0.25)
    assert batch.bc_side == ("min", "min", "min", "max", "max")
    assert np.all(batch.bc[:3, 0] == spec.time.lower)
    assert np.all(batch.bc[3:, 0] == spec.time.upper)
    assert 9.0 not in batch.bc[:, 0]
    assert -4.0 not in batch.bc[:, 0]


def test_harmonic_rejects_boundary_samples_without_conditions() -> None:
    spec = _harmonic()
    with pytest.raises(ValueError, match="no boundary conditions"):
        sample_harmonic(spec, SampleConfig(n_interior=4, n_ic=1, n_bc=2, seed=0))


def test_burgers_ic_slice_and_dirichlet_faces() -> None:
    spec = _burgers(
        x=Interval(lower=-1.0, upper=1.0),
        t=Interval(lower=0.2, upper=1.7),
    )
    config = SampleConfig(n_interior=16, n_ic=8, n_bc=5, seed=1, method="latin_hypercube")
    batch = sample_burgers(spec, config)
    assert batch.axis_names == ("x", "t")
    assert batch.interior.shape == (16, 2)
    assert batch.ic.shape == (8, 2)
    assert batch.bc.shape == (5, 2)
    assert np.all(batch.ic[:, 1] == spec.t.lower)
    _inside(batch.ic[:, :1], np.array([spec.x.lower]), np.array([spec.x.upper]))
    _inside(batch.interior, batch.lower, batch.upper)
    assert batch.bc_variable == ("x",) * 5
    assert batch.bc_side == ("min", "min", "min", "max", "max")
    assert np.all(batch.bc[:3, 0] == spec.x.lower)
    assert np.all(batch.bc[3:, 0] == spec.x.upper)
    assert np.all(batch.bc[:, 1] >= spec.t.lower)
    assert np.all(batch.bc[:, 1] <= spec.t.upper)
    assert set(np.unique(batch.bc[:, 0]).tolist()) == {spec.x.lower, spec.x.upper}
    labels = batch.labels()
    assert labels.tolist() == ["interior"] * 16 + ["ic"] * 8 + ["bc"] * 5
    assert set(labels.tolist()) <= set(POINT_LABELS)


def test_burgers_profile_does_not_move_points() -> None:
    config = SampleConfig(n_interior=8, n_ic=4, n_bc=4, seed=0)
    negative = sample_burgers(_burgers(), config)
    positive = sample_burgers(
        _burgers(initial_condition=ProfileInitialCondition(profile="sin_pi_x")),
        config,
    )
    np.testing.assert_array_equal(negative.coordinates(), positive.coordinates())


def test_burgers_periodic_samples_both_ends() -> None:
    spec = _burgers(boundary_conditions=[BoundaryCondition(variable="x", kind="periodic")])
    batch = sample_burgers(spec, SampleConfig(n_interior=4, n_ic=2, n_bc=1, seed=0))
    assert batch.bc.shape == (1, 2)
    assert batch.bc_side == ("min",)
    assert batch.bc[0, 0] == spec.x.lower
    both = sample_burgers(spec, SampleConfig(n_interior=4, n_ic=2, n_bc=4, seed=0))
    assert both.bc_side == ("min", "min", "max", "max")
    assert np.all(both.bc[:2, 0] == spec.x.lower)
    assert np.all(both.bc[2:, 0] == spec.x.upper)


def test_poisson_rejects_initial_condition_count() -> None:
    spec = _poisson_1d()
    with pytest.raises(ValueError, match="no initial condition"):
        sample_poisson(spec, SampleConfig(n_interior=4, n_ic=1, n_bc=2, seed=0))
    with pytest.raises(ValueError, match="no initial condition"):
        sample_equation(spec, SampleConfig(n_interior=4, n_ic=1, n_bc=2, seed=0))


def test_poisson_1d_boundaries_are_the_endpoints() -> None:
    spec = _poisson_1d(x=Interval(lower=-2.0, upper=3.0))
    batch = sample_poisson(spec, SampleConfig(n_interior=10, n_ic=0, n_bc=4, seed=2, method="stratified"))
    assert batch.ic.shape == (0, 1)
    assert batch.interior.shape == (10, 1)
    assert batch.bc_side == ("min", "min", "max", "max")
    assert np.all(batch.bc[:2, 0] == -2.0)
    assert np.all(batch.bc[2:, 0] == 3.0)
    _inside(batch.interior, batch.lower, batch.upper)
    assert batch.labels().tolist() == ["interior"] * 10 + ["bc"] * 4


def test_poisson_2d_faces_follow_descriptors_not_values() -> None:
    spec = _poisson_2d()
    batch = sample_equation(spec, SampleConfig(n_interior=9, n_ic=0, n_bc=5, seed=1, method="uniform"))
    assert batch.axis_names == ("x", "y")
    assert batch.interior.shape == (9, 2)
    assert batch.bc.shape == (5, 2)
    assert batch.bc_variable == ("x", "x", "x", "y", "y")
    assert batch.bc_side == ("min", "min", "max", "min", "max")
    assert np.all(batch.bc[:2, 0] == spec.x.lower)
    assert np.all(batch.bc[2, 0] == spec.x.upper)
    assert np.all(batch.bc[3, 1] == spec.y.lower)
    assert np.all(batch.bc[4, 1] == spec.y.upper)
    _inside(batch.interior, batch.lower, batch.upper)
    _inside(batch.bc, batch.lower, batch.upper)


def test_equation_dispatch_rejects_the_wrong_type() -> None:
    config = SampleConfig(n_interior=2, n_ic=0, n_bc=0, seed=0)
    with pytest.raises(TypeError, match="HarmonicOscillatorSpec"):
        sample_harmonic(_burgers(), config)  # type: ignore[arg-type]
    with pytest.raises(TypeError, match="Burgers1DSpec"):
        sample_burgers(_poisson_1d(), config)  # type: ignore[arg-type]
    with pytest.raises(TypeError, match="PoissonToySpec"):
        sample_poisson(_harmonic(), config)  # type: ignore[arg-type]
    with pytest.raises(TypeError, match="spec must be"):
        sample_equation(object(), config)  # type: ignore[arg-type]
    with pytest.raises(TypeError, match="SampleConfig"):
        sample_harmonic(_harmonic(), {"n_interior": 1})  # type: ignore[arg-type]


@pytest.mark.parametrize("method", SAMPLE_METHODS)
def test_batch_draw_order_is_interior_then_ic_then_bc(method: str) -> None:
    spec = default_spec("burgers")
    assert isinstance(spec, Burgers1DSpec)
    config = SampleConfig(n_interior=8, n_ic=4, n_bc=4, seed=0, method=method)  # type: ignore[arg-type]
    batch = sample_equation(spec, config)
    domain = spec.collocation_domain()
    rng = np.random.default_rng(0)
    interior = sample_collocation(domain, config.n_interior, rng, config.method)
    ic = sample_collocation(
        domain,
        config.n_ic,
        rng,
        config.method,
        fixed={spec.initial_condition.variable: float(spec.t.lower)},
    )
    bc, variables, sides = sample_boundary(
        domain,
        spec.boundary_conditions,
        config.n_bc,
        rng,
        config.method,
    )
    np.testing.assert_array_equal(batch.interior, interior)
    np.testing.assert_array_equal(batch.ic, ic)
    np.testing.assert_array_equal(batch.bc, bc)
    assert batch.bc_variable == variables
    assert batch.bc_side == sides


def test_default_specs_match_cli_problems() -> None:
    harmonic = default_spec("harmonic")
    burgers = default_spec("burgers_1d")
    poisson = default_spec("poisson")
    poisson_2d = default_spec("poisson_toy", dimensions=2)
    assert isinstance(harmonic, HarmonicOscillatorSpec)
    assert harmonic.boundary_conditions == []
    assert isinstance(burgers, Burgers1DSpec)
    assert burgers.boundary_conditions[0].kind == "periodic"
    assert isinstance(poisson, PoissonToySpec)
    assert poisson.dimensions == 1
    assert isinstance(poisson_2d, PoissonToySpec)
    assert poisson_2d.dimensions == 2
    with pytest.raises(ValueError, match="dimensions"):
        default_spec("harmonic", dimensions=2)
    with pytest.raises(ValueError, match="unknown equation"):
        default_spec("heat")


def test_record_roundtrip(tmp_path: Path) -> None:
    spec = _poisson_2d()
    config = SampleConfig(n_interior=5, n_ic=0, n_bc=3, seed=6, method="latin_hypercube")
    batch = sample_equation(spec, config)
    path = tmp_path / "nested" / "batch.json"
    write_sample_record(spec, config, batch, path)
    loaded_spec, loaded_config, loaded = load_sample_record(path)
    assert loaded_spec == spec
    assert loaded_config == config
    np.testing.assert_array_equal(loaded.coordinates(), batch.coordinates())
    assert loaded.labels().tolist() == batch.labels().tolist()
    payload = json.loads(path.read_text(encoding="utf-8"))
    assert payload["format"] == "pinnforge.sample.v1"
    assert "http://" not in path.read_text(encoding="utf-8")


def test_batch_from_dict_rejects_mismatched_labels() -> None:
    batch = sample_equation(_harmonic(), SampleConfig(n_interior=3, n_ic=1, n_bc=0, seed=0))
    data = batch_to_dict(batch)
    data["labels"] = ["bc"] * len(data["labels"])  # type: ignore[index]
    with pytest.raises(ValueError, match="labels"):
        batch_from_dict(data)


def test_resolve_output_path_stays_inside_the_working_directory(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.chdir(tmp_path)
    resolved = resolve_output_path("batches/a.json")
    assert resolved == (tmp_path / "batches" / "a.json").resolve()
    with pytest.raises(ValueError, match="relative"):
        resolve_output_path(tmp_path / "a.json")
    with pytest.raises(ValueError, match="working directory"):
        resolve_output_path("../a.json")
    with pytest.raises(ValueError, match=r"\.json"):
        resolve_output_path("a.txt")
    with pytest.raises(ValueError, match="file"):
        resolve_output_path(".")


@pytest.mark.parametrize("name", FIXTURE_NAMES)
def test_offline_fixture_matches_sampler(name: str) -> None:
    json_path = FIXTURES / f"{name}.json"
    npy_path = FIXTURES / f"{name}.npy"
    text = json_path.read_text(encoding="utf-8")
    assert "http://" not in text
    assert "https://" not in text
    spec, config, stored = load_sample_record(json_path)
    fresh = sample_equation(spec, config)
    np.testing.assert_array_equal(fresh.interior, stored.interior)
    np.testing.assert_array_equal(fresh.ic, stored.ic)
    np.testing.assert_array_equal(fresh.bc, stored.bc)
    assert fresh.bc_variable == stored.bc_variable
    assert fresh.bc_side == stored.bc_side
    assert fresh.labels().tolist() == stored.labels().tolist()
    saved = np.load(npy_path)
    assert saved.dtype == np.float64
    assert saved.ndim == 2
    np.testing.assert_array_equal(saved, fresh.coordinates())


def test_named_streams_differ_at_the_same_seed_and_count() -> None:
    """Train points must not be the eval points when the integer seed matches.

    The historical sampler uses one ``default_rng(seed)`` stream. A short
    uniform draw is then the prefix of a longer draw. Named streams break
    that overlap at the same count, so the check does not depend on drawing
    more evaluation points than training points.
    """

    spec = default_spec("harmonic")
    config = SampleConfig(n_interior=8, n_ic=4, n_bc=0, seed=0, method="uniform")
    draws = {
        name: sample_equation(spec, config, rng=stream_generator(0, name)).interior
        for name in STREAM_NAMES
    }
    assert len({array.tobytes() for array in draws.values()}) == 3
    shared = sample_equation(spec, config).interior
    longer = sample_equation(
        spec, config.model_copy(update={"n_interior": 16})
    ).interior
    np.testing.assert_array_equal(shared, longer[:8])
    for name, array in draws.items():
        assert array.shape == shared.shape
        assert not np.array_equal(array, shared), name
        assert not np.array_equal(array, longer[:8]), name


def test_harmonic_fixture_ic_is_the_initial_time() -> None:
    spec, _config, stored = load_sample_record(FIXTURES / "harmonic_seed0.json")
    assert isinstance(spec, HarmonicOscillatorSpec)
    assert stored.ic.shape[0] > 0
    assert np.all(stored.ic[:, 0] == spec.time.lower)
    assert stored.bc.shape[0] > 0
    assert np.all(stored.bc[:, 0] == spec.time.upper)
