"""MLP shapes and the spec factory."""

from __future__ import annotations

import pytest

pytestmark = pytest.mark.ml


@pytest.fixture(autouse=True)
def _need_torch() -> None:
    pytest.importorskip("torch")


def test_mlp_output_shape_and_default_activation() -> None:
    import torch

    from pinnforge.models import ACTIVATIONS, MLP

    model = MLP(2, (4, 5))
    assert model.in_features == 2
    assert model.out_features == 1
    assert model.hidden_widths == (4, 5)
    assert model.activation == "tanh"
    assert "tanh" in ACTIVATIONS
    assert isinstance(model.network[-1], torch.nn.Linear)
    assert model.network[-1].out_features == 1
    assert tuple(model.network[0].weight.shape) == (4, 2)
    batch = model(torch.zeros(7, 2))
    assert batch.shape == (7, 1)
    single = model(torch.zeros(2))
    assert single.shape == (1, 1)


def test_mlp_rejects_bad_widths_and_activation() -> None:
    from pinnforge.models import MLP

    with pytest.raises(ValueError, match="hidden width"):
        MLP(1, (8, 0))
    with pytest.raises(ValueError, match="hidden width"):
        MLP(1, (True,))  # type: ignore[arg-type]
    with pytest.raises(ValueError, match="activation"):
        MLP(1, (4,), activation="gelu")
    linear = MLP(1, ())
    import torch

    assert linear(torch.zeros(3, 1)).shape == (3, 1)


def test_factory_input_width_follows_the_spec() -> None:
    from pinnforge.models import input_features, mlp_from_spec
    from pinnforge.sampling import default_spec

    harmonic = default_spec("harmonic")
    burgers = default_spec("burgers")
    poisson_1d = default_spec("poisson")
    poisson_2d = default_spec("poisson", dimensions=2)
    assert input_features(harmonic) == 1
    assert input_features(burgers) == 2
    assert input_features(poisson_1d) == 1
    assert input_features(poisson_2d) == 2
    assert mlp_from_spec(harmonic, (8,)).in_features == 1
    assert mlp_from_spec(burgers, (8,)).in_features == 2
    assert mlp_from_spec(poisson_2d, (8,), activation="silu").activation == "silu"
    assert mlp_from_spec(poisson_1d).activation == "tanh"
