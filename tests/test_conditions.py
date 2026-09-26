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
    assert float(penalty.initial) < 1e-12
    assert float(penalty.dirichlet) < 1e-12
    assert float(penalty.total()) < 1e-12
    shifted = soft_penalty(_constant(0.0), batch, spec)
    assert float(shifted.initial) > 1.0
    assert float(shifted.dirichlet) > 1.0


def test_burgers_profile_penalty_and_periodic_dirichlet_is_zero() -> None:
    import torch

    from pinnforge.losses import initial_condition_loss, soft_penalty
    from pinnforge.sampling import default_spec

    spec = default_spec("burgers")
    batch = sample_equation(spec, SampleConfig(n_interior=4, n_ic=6, n_bc=4, seed=2))

    def profile(coords: torch.Tensor) -> torch.Tensor:
        return -torch.sin(coords[:, 0:1] * math.pi)

    matched = soft_penalty(profile, batch, spec)
    assert float(matched.initial) < 1e-12
    assert float(matched.dirichlet) == 0.0
    missed = initial_condition_loss(_constant(0.0), batch, spec)
    assert float(missed) > 0.0


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
    assert float(penalty.initial) == 0.0
    assert float(initial_condition_loss(manufactured, batch, spec)) == 0.0
    assert float(penalty.dirichlet) < 1e-12
    # A constant 2 matches neither end of the manufactured field, but the
    # Neumann face must not be scored as Dirichlet data valued at 5.
    assert float(dirichlet_boundary_loss(_constant(2.0), batch, spec)) == pytest.approx(4.0)


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

    assert float(dirichlet_boundary_loss(profile, batch, spec)) < 1e-12


def test_penalty_rejects_a_mismatched_batch() -> None:
    from pinnforge.losses import soft_penalty
    from pinnforge.sampling import default_spec

    harmonic = default_spec("harmonic")
    burgers = default_spec("burgers")
    batch = sample_equation(burgers, SampleConfig(n_interior=2, n_ic=2, n_bc=2, seed=0))
    with pytest.raises(ValueError, match="does not match"):
        soft_penalty(_constant(0.0), batch, harmonic)
