"""Soft initial-condition and Dirichlet penalties."""

from __future__ import annotations

import math

import pytest

from pinnforge.equations import (
    BoundaryCondition,
    Burgers1DSpec,
    HarmonicOscillatorSpec,
    Interval,
    PoissonSource,
    PoissonToySpec,
    ProfileInitialCondition,
    StateInitialCondition,
)
from pinnforge.sampling import SampleConfig, sample_equation

pytestmark = pytest.mark.ml


@pytest.fixture(autouse=True)
def _need_torch() -> None:
    pytest.importorskip("torch")


def _constant(value: float):
    import torch

    def field(coords: torch.Tensor) -> torch.Tensor:
        return torch.full((coords.shape[0], 1), value, dtype=coords.dtype, device=coords.device)

    return field


def _value(tensor: object) -> float:
    import torch

    if not isinstance(tensor, torch.Tensor):
        raise TypeError("expected a tensor")
    return float(tensor.detach())


def test_harmonic_exact_initial_and_derivative_boundary_are_near_zero() -> None:
    from pinnforge.losses import soft_penalty

    spec = HarmonicOscillatorSpec(
        omega=1.0,
        time=Interval(lower=0.0, upper=1.0),
        initial_condition=StateInitialCondition(components={"u": 3.0, "du_dt": 0.0}),
        boundary_conditions=[
            BoundaryCondition(variable="t", kind="dirichlet", side="min", value=3.0, component="u"),
            BoundaryCondition(variable="t", kind="dirichlet", side="max", value=0.0, component="du_dt"),
        ],
    )
    batch = sample_equation(spec, SampleConfig(n_interior=4, n_ic=5, n_bc=4, seed=1))
    penalty = soft_penalty(_constant(3.0), batch, spec)
    assert _value(penalty.initial) < 1e-12
    assert _value(penalty.dirichlet) < 1e-12
    assert _value(penalty.total()) < 1e-12
    shifted = soft_penalty(_constant(0.0), batch, spec)
    assert _value(shifted.initial) > 1.0
    assert _value(shifted.dirichlet) > 1.0


def test_burgers_profile_penalty_and_periodic_dirichlet_is_zero() -> None:
    import torch

    from pinnforge.losses import initial_condition_loss, soft_penalty
    from pinnforge.sampling import default_spec

    spec = default_spec("burgers")
    batch = sample_equation(spec, SampleConfig(n_interior=4, n_ic=6, n_bc=4, seed=2))

    def profile(coords: torch.Tensor) -> torch.Tensor:
        return -torch.sin(coords[:, 0:1] * math.pi)

    matched = soft_penalty(profile, batch, spec)
    assert _value(matched.initial) < 1e-12
    assert _value(matched.dirichlet) == 0.0
    missed = initial_condition_loss(_constant(0.0), batch, spec)
    assert _value(missed) > 0.0


def test_poisson_dirichlet_uses_prescribed_values_and_skips_neumann() -> None:
    import torch

    from pinnforge.losses import dirichlet_boundary_loss, initial_condition_loss, soft_penalty

    spec = PoissonToySpec(
        dimensions=1,
        source=PoissonSource.SIN_PI_X,
        x=Interval(lower=0.0, upper=1.0),
        boundary_conditions=[
            BoundaryCondition(variable="x", kind="dirichlet", side="min", value=0.0),
            BoundaryCondition(variable="x", kind="neumann", side="max", value=5.0),
        ],
    )
    batch = sample_equation(spec, SampleConfig(n_interior=4, n_ic=0, n_bc=6, seed=3))

    def manufactured(coords: torch.Tensor) -> torch.Tensor:
        return torch.sin(coords[:, 0:1] * math.pi) / (math.pi**2)

    penalty = soft_penalty(manufactured, batch, spec)
    assert _value(penalty.initial) == 0.0
    assert _value(initial_condition_loss(manufactured, batch, spec)) == 0.0
    assert _value(penalty.dirichlet) < 1e-12
    # A constant 2 matches neither end of the manufactured field, but the
    # Neumann face must not be scored as Dirichlet data valued at 5.
    assert _value(dirichlet_boundary_loss(_constant(2.0), batch, spec)) == pytest.approx(4.0)


def test_dirichlet_burgers_ends_match_the_sine_profile() -> None:
    import torch

    from pinnforge.losses import dirichlet_boundary_loss

    spec = Burgers1DSpec(
        nu=0.1,
        x=Interval(lower=-1.0, upper=1.0),
        t=Interval(lower=0.0, upper=1.0),
        initial_condition=ProfileInitialCondition(profile="sin_pi_x"),
        boundary_conditions=[
            BoundaryCondition(variable="x", kind="dirichlet", side="min", value=0.0),
            BoundaryCondition(variable="x", kind="dirichlet", side="max", value=0.0),
        ],
    )
    batch = sample_equation(spec, SampleConfig(n_interior=2, n_ic=2, n_bc=4, seed=4))

    def profile(coords: torch.Tensor) -> torch.Tensor:
        return torch.sin(coords[:, 0:1] * math.pi)

    assert _value(dirichlet_boundary_loss(profile, batch, spec)) < 1e-12


def test_neumann_uses_the_outward_derivative_and_not_the_dirichlet_term() -> None:
    import torch

    from pinnforge.losses import neumann_boundary_loss, soft_penalty

    spec = PoissonToySpec(
        dimensions=1,
        source=PoissonSource.ONE,
        x=Interval(lower=0.0, upper=1.0),
        boundary_conditions=[
            BoundaryCondition(variable="x", kind="dirichlet", side="min", value=0.0),
            BoundaryCondition(variable="x", kind="neumann", side="max", value=5.0),
            BoundaryCondition(variable="x", kind="neumann", side="min", value=-5.0),
        ],
    )
    batch = sample_equation(spec, SampleConfig(n_interior=2, n_ic=0, n_bc=6, seed=5))

    def slope(coords: torch.Tensor) -> torch.Tensor:
        return 5.0 * coords[:, 0:1]

    matched = soft_penalty(slope, batch, spec)
    assert _value(matched.dirichlet) < 1e-12
    assert _value(matched.neumann) < 1e-10
    # A constant has zero derivative. Outward flux is 0, not the prescribed 5 and -5.
    missed = neumann_boundary_loss(_constant(2.0), batch, spec)
    assert _value(missed) == pytest.approx(25.0)


def test_periodic_burgers_matches_value_and_derivative_at_the_same_time() -> None:
    import numpy as np
    import torch

    from pinnforge.losses import periodic_boundary_loss, soft_penalty
    from pinnforge.sampling import CollocationBatch, default_spec

    spec = default_spec("burgers")
    batch = sample_equation(spec, SampleConfig(n_interior=4, n_ic=4, n_bc=4, seed=2))

    def linear(coords: torch.Tensor) -> torch.Tensor:
        return coords[:, 0:1]

    def squares(coords: torch.Tensor) -> torch.Tensor:
        return coords[:, 0:1] ** 2

    def periodic_profile(coords: torch.Tensor) -> torch.Tensor:
        return torch.sin(coords[:, 0:1] * math.pi)

    assert _value(periodic_boundary_loss(linear, batch, spec)) > 0.1
    assert _value(periodic_boundary_loss(squares, batch, spec)) > 0.1
    assert _value(periodic_boundary_loss(periodic_profile, batch, spec)) < 1e-10
    penalty = soft_penalty(periodic_profile, batch, spec)
    assert _value(penalty.dirichlet) == 0.0
    assert _value(penalty.periodic) < 1e-10
    # Min and max faces below do not share t. A correct penalty mirrors each
    # row onto both ends at that row's own t. Zipping the two faces compares
    # different times and does not vanish for u = sin(pi x) * (1 + t).
    paired = CollocationBatch(
        equation_id="burgers_1d",
        axis_names=("x", "t"),
        lower=np.array([-1.0, 0.0]),
        upper=np.array([1.0, 1.0]),
        method="uniform",
        seed=0,
        interior=np.array([[0.0, 0.3]]),
        ic=np.array([[0.2, 0.0]]),
        bc=np.array([[-1.0, 0.2], [1.0, 0.8]]),
        bc_variable=("x", "x"),
        bc_side=("min", "max"),
    )

    def same_time(coords: torch.Tensor) -> torch.Tensor:
        return torch.sin(coords[:, 0:1] * math.pi) * (1.0 + coords[:, 1:2])

    assert _value(periodic_boundary_loss(same_time, paired, spec)) < 1e-10
    assert _value(periodic_boundary_loss(linear, paired, spec)) > 0.1


def test_training_rejects_a_periodic_spec_with_no_boundary_samples(
    tmp_path, monkeypatch: pytest.MonkeyPatch
) -> None:
    from pinnforge.training import TrainConfig, train_loop

    monkeypatch.chdir(tmp_path)
    config = TrainConfig(
        equation_id="burgers",
        epochs=1,
        n_interior=4,
        n_ic=2,
        n_bc=0,
        hidden_widths=(4, 4),
        checkpoint_dir="ckpts",
        log_path="metrics.jsonl",
    )
    with pytest.raises(ValueError, match="periodic"):
        train_loop(config)


def test_penalty_rejects_a_mismatched_batch() -> None:
    from pinnforge.losses import soft_penalty
    from pinnforge.sampling import default_spec

    harmonic = default_spec("harmonic")
    burgers = default_spec("burgers")
    batch = sample_equation(burgers, SampleConfig(n_interior=2, n_ic=2, n_bc=2, seed=0))
    with pytest.raises(ValueError, match="does not match"):
        soft_penalty(_constant(0.0), batch, harmonic)
