"""Autograd residuals. The analytical checks use callables, not a network."""

from __future__ import annotations

import math

import pytest

from pinnforge.equations import (
    HarmonicOscillatorSpec,
    Interval,
    StateInitialCondition,
)

pytestmark = pytest.mark.ml


@pytest.fixture(autouse=True)
def _need_torch() -> None:
    pytest.importorskip("torch")


def _harmonic(*, omega: float | None = 2.0, k: float | None = None, m: float | None = None) -> HarmonicOscillatorSpec:
    payload: dict[str, object] = {
        "time": Interval(lower=0.0, upper=1.0),
        "initial_condition": StateInitialCondition(components={"u": 1.0, "du_dt": 0.5}),
    }
    if omega is None:
        payload["k"] = k
        payload["m"] = m
    else:
        payload["omega"] = omega
    return HarmonicOscillatorSpec.model_validate(payload)


def _exact_harmonic(spec: HarmonicOscillatorSpec):
    import torch

    omega = spec.angular_frequency
    t0 = float(spec.time.lower)
    u0 = float(spec.initial_condition.components["u"])
    v0 = float(spec.initial_condition.components["du_dt"])

    def field(coords: torch.Tensor) -> torch.Tensor:
        time = coords[:, 0:1]
        angle = omega * (time - t0)
        return u0 * torch.cos(angle) + (v0 / omega) * torch.sin(angle)

    return field


def test_harmonic_exact_callable_residual_is_near_zero() -> None:
    import torch

    from pinnforge.reference import displacement
    from pinnforge.residuals import residual

    spec = _harmonic()
    times = torch.linspace(0.0, 1.0, 9, dtype=torch.float64).unsqueeze(-1)
    field = _exact_harmonic(spec)
    values = residual(field, times, spec)
    assert values.shape == (9, 1)
    assert torch.max(torch.abs(values)).item() < 1e-8
    import numpy as np

    numpy_u = displacement(times.detach().numpy().reshape(-1), spec)
    got = field(times).detach().numpy().reshape(-1)
    assert np.allclose(numpy_u, got)


def test_harmonic_closed_form_tensor_residual_is_near_zero() -> None:
    import torch

    from pinnforge.residuals import harmonic_residual_from_field

    spec = _harmonic(omega=None, k=8.0, m=2.0)
    assert spec.angular_frequency == pytest.approx(2.0)
    time = torch.linspace(0.05, 0.95, 7, dtype=torch.float64).unsqueeze(-1).requires_grad_(True)
    field = _exact_harmonic(spec)(time)
    values = harmonic_residual_from_field(field, time, spec)
    assert values.shape == (7, 1)
    assert torch.max(torch.abs(values)).item() < 1e-8


def test_burgers_linear_field_residual_equals_x() -> None:
    import torch

    from pinnforge.residuals import burgers_residual, residual_from_field
    from pinnforge.sampling import default_spec

    spec = default_spec("burgers")
    coords = torch.tensor([[-0.3, 0.2], [0.0, 0.5], [0.8, 0.9]], dtype=torch.float64)
    from_model = burgers_residual(lambda rows: rows[:, 0:1], coords, spec)
    assert from_model.shape == (3, 1)
    assert torch.allclose(from_model, coords[:, 0:1])

    leaf = coords.detach().requires_grad_(True)
    from_field = residual_from_field(leaf[:, 0:1], leaf, spec)
    assert torch.allclose(from_field, leaf[:, 0:1])


def test_poisson_manufactured_residuals_are_near_zero() -> None:
    import torch

    from pinnforge.residuals import poisson_residual, poisson_source
    from pinnforge.sampling import default_spec

    spec_1d = default_spec("poisson")
    x = torch.linspace(0.0, 1.0, 8, dtype=torch.float64).unsqueeze(-1)
    source = poisson_source(spec_1d, x)
    assert torch.allclose(source, torch.sin(x * math.pi))

    def field_1d(coords: torch.Tensor) -> torch.Tensor:
        column = coords[:, 0:1]
        return torch.sin(column * math.pi) / (math.pi**2)

    values_1d = poisson_residual(field_1d, x, spec_1d)
    assert values_1d.shape == (8, 1)
    assert torch.max(torch.abs(values_1d)).item() < 1e-8

    spec_2d = default_spec("poisson", dimensions=2)
    xy = torch.tensor([[0.2, 0.3], [0.5, 0.5], [0.8, 0.1]], dtype=torch.float64)

    def field_2d(coords: torch.Tensor) -> torch.Tensor:
        column_x = coords[:, 0:1]
        column_y = coords[:, 1:2]
        return torch.sin(column_x * math.pi) * torch.sin(column_y * math.pi) / (2.0 * math.pi**2)

    values_2d = poisson_residual(field_2d, xy, spec_2d)
    assert values_2d.shape == (3, 1)
    assert torch.max(torch.abs(values_2d)).item() < 1e-8


@pytest.mark.parametrize(
    ("name", "dimensions", "width"),
    [
        ("harmonic", None, 1),
        ("burgers", None, 2),
        ("poisson", 1, 1),
        ("poisson", 2, 2),
    ],
)
def test_untrained_network_residual_shape_and_backward(
    name: str,
    dimensions: int | None,
    width: int,
) -> None:
    import torch

    from pinnforge.models import mlp_from_spec
    from pinnforge.residuals import mean_squared_residual
    from pinnforge.sampling import default_spec

    spec = default_spec(name) if dimensions is None else default_spec(name, dimensions=dimensions)
    torch.manual_seed(0)
    model = mlp_from_spec(spec, (8, 8))
    coords = torch.rand(5, width)
    loss = mean_squared_residual(model, coords, spec)
    assert loss.ndim == 0
    assert torch.isfinite(loss)
    loss.backward()
    gradients = [parameter.grad for parameter in model.parameters()]
    assert all(gradient is not None for gradient in gradients)
    assert sum(float(gradient.abs().sum()) for gradient in gradients if gradient is not None) > 0.0


def test_residual_rejects_the_wrong_coordinate_width() -> None:
    import torch

    from pinnforge.models import MLP, mlp_from_spec
    from pinnforge.residuals import residual
    from pinnforge.sampling import default_spec

    spec = default_spec("harmonic")
    model = mlp_from_spec(spec, (4,))
    with pytest.raises(ValueError, match="shape"):
        residual(model, torch.rand(4, 2), spec)
    with pytest.raises(ValueError, match="in_features"):
        residual(MLP(2, (4,)), torch.rand(4, 1), spec)
    empty = residual(model, torch.empty(0, 1), spec)
    assert tuple(empty.shape) == (0, 1)
    with pytest.raises(ValueError, match="at least one"):
        from pinnforge.residuals import mean_squared_residual

        mean_squared_residual(model, torch.empty(0, 1), spec)
