"""TrainConfig validation. These tests need torch because importing
``pinnforge.training`` loads the training loop.
"""

from __future__ import annotations

import pytest
from pydantic import ValidationError

pytestmark = pytest.mark.ml


@pytest.fixture(autouse=True)
def _need_torch() -> None:
    pytest.importorskip("torch")


def test_harmonic_defaults_and_aliases() -> None:
    from pinnforge.models import ACTIVATIONS
    from pinnforge.training import TrainConfig
    from pinnforge.training.config import ACTIVATION_NAMES

    config = TrainConfig()
    assert config.equation_id == "harmonic_oscillator"
    assert config.n_interior == 32
    assert config.n_ic == 16
    assert config.n_bc == 0
    assert config.hidden_widths == (16, 16)
    assert config.activation == "tanh"
    assert config.epochs == 50
    assert config.lr == 1e-3
    assert config.w_pde == config.w_ic == config.w_bc == 1.0
    assert config.seed == 0
    assert config.device == "cpu"
    assert config.method == "uniform"
    assert config.checkpoint_dir == "checkpoints"
    assert config.log_path == "metrics.jsonl"
    assert TrainConfig(equation_id="harmonic").equation_id == "harmonic_oscillator"
    assert set(ACTIVATION_NAMES) == set(ACTIVATIONS)

    poisson = TrainConfig(equation_id="poisson")
    assert poisson.equation_id == "poisson_toy"
    assert poisson.n_ic == 0
    assert poisson.n_bc == 16

    burgers = TrainConfig(equation_id="burgers")
    assert burgers.n_ic == 16
    assert burgers.n_bc == 16
    explicit = TrainConfig(equation_id="burgers", n_bc=0)
    assert explicit.n_bc == 0


def test_config_round_trip_keeps_explicit_counts() -> None:
    from pinnforge.training import TrainConfig

    config = TrainConfig(equation_id="poisson", n_interior=8, epochs=2, hidden_widths=[8, 4])
    assert config.hidden_widths == (8, 4)
    restored = TrainConfig.model_validate(config.model_dump(mode="json"))
    assert restored == config


def test_config_rejects_unknown_fields_and_is_frozen() -> None:
    from pinnforge.training import TrainConfig

    with pytest.raises(ValidationError):
        TrainConfig(not_a_field=1)
    config = TrainConfig(epochs=2)
    with pytest.raises(ValidationError):
        config.epochs = 3  # type: ignore[misc]


@pytest.mark.parametrize(
    "kwargs",
    [
        {"epochs": 0},
        {"epochs": True},
        {"lr": 0},
        {"lr": -1.0},
        {"lr": float("inf")},
        {"device": "cuda"},
        {"checkpoint_dir": "/tmp/checkpoints"},
        {"checkpoint_dir": "."},
        {"checkpoint_dir": "~/checkpoints"},
        {"log_path": "/tmp/metrics.jsonl"},
        {"log_path": "metrics.json"},
        {"log_path": "."},
        {"equation_id": "wave"},
        {"seed": True},
        {"seed": -1},
        {"w_pde": -0.1},
        {"w_ic": float("nan")},
        {"w_pde": True},
        {"hidden_widths": ()},
        {"hidden_widths": (0,)},
        {"hidden_widths": (True,)},
        {"hidden_widths": "16,16"},
        {"n_interior": 0},
        {"equation_id": "poisson", "n_ic": 4},
        {"n_bc": 4},
        {"w_pde": 0, "w_ic": 0, "w_bc": 0},
        {"activation": "relu"},
        {"checkpoint_interval": 0},
    ],
)
def test_config_rejects_invalid_values(kwargs: dict[str, object]) -> None:
    from pinnforge.training import TrainConfig

    with pytest.raises(ValidationError):
        TrainConfig(**kwargs)
