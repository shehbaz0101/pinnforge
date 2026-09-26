"""Interior residual stats, field error, and checkpoint evaluation."""

from __future__ import annotations

import math
from pathlib import Path

import pytest

pytestmark = pytest.mark.ml


@pytest.fixture(autouse=True)
def _need_torch() -> None:
    pytest.importorskip("torch")


def _harmonic_field():
    import torch

    from pinnforge.sampling import default_spec

    spec = default_spec("harmonic")
    omega = spec.angular_frequency
    t0 = float(spec.time.lower)
    u0 = float(spec.initial_condition.components["u"])
    v0 = float(spec.initial_condition.components["du_dt"])

    def field(coords: torch.Tensor) -> torch.Tensor:
        time = coords[:, 0:1]
        angle = omega * (time - t0)
        return u0 * torch.cos(angle) + (v0 / omega) * torch.sin(angle)

    return spec, field


def test_harmonic_exact_field_residual_is_near_zero() -> None:
    import numpy as np
    import torch

    from pinnforge.reference import displacement
    from pinnforge.residuals import residual_from_field

    spec, field = _harmonic_field()
    time = torch.linspace(0.0, 1.0, 11, dtype=torch.float64).unsqueeze(-1).requires_grad_(True)
    values = field(time)
    numpy_u = displacement(time.detach().numpy().reshape(-1), spec)
    assert np.allclose(values.detach().numpy().reshape(-1), numpy_u)
    residual = residual_from_field(values, time, spec)
    assert residual.shape == (11, 1)
    assert torch.max(torch.abs(residual.detach())).item() < 1e-8


def test_exact_harmonic_eval_matches_the_closed_form() -> None:
    from pinnforge.evaluation import EvalConfig, evaluate_model

    spec, field = _harmonic_field()
    result = evaluate_model(field, spec, EvalConfig(n_interior=16, seed=1, bins=5))
    assert result.equation_id == "harmonic_oscillator"
    assert result.reference == "analytical"
    assert result.checkpoint is None
    assert result.n_interior == 16
    assert result.l2 is not None and result.l2 < 1e-8
    assert result.relative_l2 is not None and result.relative_l2 < 1e-8
    assert result.residual_max_abs < 1e-6
    assert result.residual_mean_abs <= result.residual_max_abs
    assert sum(result.histogram.counts) == 16
    assert len(result.histogram.edges) == len(result.histogram.counts) + 1


def test_random_network_eval_is_finite() -> None:
    import torch

    from pinnforge.evaluation import EvalConfig, evaluate_model
    from pinnforge.models import mlp_from_spec
    from pinnforge.sampling import default_spec

    spec = default_spec("harmonic")
    torch.manual_seed(0)
    model = mlp_from_spec(spec, (8, 8))
    result = evaluate_model(model, spec, EvalConfig(n_interior=8, seed=2, bins=4))
    assert result.reference == "analytical"
    scores = (
        result.l2,
        result.relative_l2,
        result.residual_mean_abs,
        result.residual_max_abs,
    )
    assert all(score is not None and math.isfinite(score) for score in scores)
    assert sum(result.histogram.counts) == 8
    assert all(math.isfinite(edge) for edge in result.histogram.edges)
    payload = result.as_dict()
    assert payload["format"] == "pinnforge.eval.v1"
    assert payload["histogram"]["counts"] == list(result.histogram.counts)


def test_burgers_eval_is_residual_only() -> None:
    import torch

    from pinnforge.evaluation import EvalConfig, evaluate_model
    from pinnforge.models import mlp_from_spec
    from pinnforge.sampling import default_spec

    spec = default_spec("burgers")
    torch.manual_seed(0)
    model = mlp_from_spec(spec, (4, 4))
    result = evaluate_model(model, spec, EvalConfig(n_interior=6, seed=0))
    assert result.reference == "unavailable"
    assert result.l2 is None
    assert result.relative_l2 is None
    assert math.isfinite(result.residual_mean_abs)
    assert math.isfinite(result.residual_max_abs)
    assert result.as_dict()["l2"] is None


def test_poisson_manufactured_eval_is_near_zero() -> None:
    import math

    import torch

    from pinnforge.equations import BoundaryCondition, Interval, PoissonToySpec
    from pinnforge.evaluation import EvalConfig, evaluate_model
    from pinnforge.sampling import default_spec

    spec = default_spec("poisson")

    def field(coords: torch.Tensor) -> torch.Tensor:
        column = coords[:, 0:1]
        return torch.sin(column * math.pi) / (math.pi**2)

    result = evaluate_model(field, spec, EvalConfig(n_interior=12, seed=3))
    assert result.reference == "analytical"
    assert result.l2 is not None and result.l2 < 1e-8
    assert result.relative_l2 is not None and result.relative_l2 < 1e-8
    assert result.residual_max_abs < 1e-6

    zero = PoissonToySpec(
        dimensions=1,
        source="zero",
        x=Interval(lower=0.0, upper=1.0),
        boundary_conditions=[
            BoundaryCondition(variable="x", kind="dirichlet", side="min", value=0.0),
        ],
    )

    def zeros(coords: torch.Tensor) -> torch.Tensor:
        return torch.zeros(coords.shape[0], 1, dtype=coords.dtype, device=coords.device)

    zero_result = evaluate_model(zeros, zero, EvalConfig(n_interior=5, seed=0))
    assert zero_result.l2 == pytest.approx(0.0)
    assert zero_result.relative_l2 is None
    assert zero_result.residual_max_abs == pytest.approx(0.0)


def test_checkpoint_eval_after_a_tiny_train(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    from pinnforge.evaluation import EvalConfig, evaluate_checkpoint
    from pinnforge.training import TrainConfig, train_loop

    monkeypatch.chdir(tmp_path)
    config = TrainConfig(
        equation_id="harmonic",
        epochs=2,
        seed=0,
        n_interior=8,
        n_ic=4,
        hidden_widths=(8, 8),
        checkpoint_dir="ckpts",
        log_path="metrics.jsonl",
    )
    train_loop(config)
    result = evaluate_checkpoint(
        Path("ckpts/checkpoint.pt"),
        EvalConfig(n_interior=8, seed=4),
        equation="harmonic",
    )
    assert result.equation_id == "harmonic_oscillator"
    assert result.checkpoint == "ckpts/checkpoint.pt"
    assert result.l2 is not None and math.isfinite(result.l2)
    assert math.isfinite(result.residual_mean_abs)
    assert math.isfinite(result.residual_max_abs)
    tagged = evaluate_checkpoint("ckpts/epoch_0002.pt", equation="harmonic_oscillator")
    assert tagged.checkpoint == "ckpts/epoch_0002.pt"
    assert tagged.l2 is not None and math.isfinite(tagged.l2)


def test_checkpoint_equation_must_match(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    from pinnforge.evaluation import evaluate_checkpoint
    from pinnforge.training import TrainConfig, train_loop

    monkeypatch.chdir(tmp_path)
    train_loop(
        TrainConfig(
            equation_id="harmonic",
            epochs=1,
            n_interior=4,
            n_ic=2,
            hidden_widths=(4, 4),
            checkpoint_dir="ckpts",
            log_path="metrics.jsonl",
        )
    )
    with pytest.raises(ValueError, match="harmonic_oscillator"):
        evaluate_checkpoint("ckpts/checkpoint.pt", equation="burgers")


def test_eval_paths_stay_inside_the_working_directory(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    import torch

    from pinnforge.evaluation import (
        EvalConfig,
        evaluate_checkpoint,
        evaluate_model,
        write_eval_json,
    )
    from pinnforge.models import mlp_from_spec
    from pinnforge.sampling import default_spec

    monkeypatch.chdir(tmp_path)
    with pytest.raises(ValueError, match="relative"):
        evaluate_checkpoint(Path("/tmp/checkpoint.pt"))
    with pytest.raises(ValueError, match="working directory"):
        evaluate_checkpoint(Path("../checkpoint.pt"))
    with pytest.raises(ValueError, match=r"\.pt"):
        evaluate_checkpoint(Path("checkpoint.txt"))
    with pytest.raises(ValueError, match="not found"):
        evaluate_checkpoint(Path("missing.pt"))

    torch.manual_seed(0)
    spec = default_spec("harmonic")
    result = evaluate_model(mlp_from_spec(spec, (4, 4)), spec, EvalConfig(n_interior=4, seed=0))
    with pytest.raises(ValueError, match="relative"):
        write_eval_json(result, Path("/tmp/eval.json"))
    with pytest.raises(ValueError, match="working directory"):
        write_eval_json(result, Path("../eval.json"))
    with pytest.raises(ValueError, match=r"\.json"):
        write_eval_json(result, Path("eval.txt"))
    written = write_eval_json(result, Path("runs/eval.json"))
    assert written == (tmp_path / "runs" / "eval.json").resolve()
    text = written.read_text(encoding="utf-8")
    assert "pinnforge.eval.v1" in text
    assert "residual_mean_abs" in text


def test_zero_field_has_zero_residual_and_full_relative_error() -> None:
    """u ≡ 0 solves u'' + ω² u = 0 but misses u(0) = 1.

    Residual-only evaluation would report a perfect model. The relative
    field error is 1 (100%) and the held-out initial-condition error is 1.
    """

    import torch

    from pinnforge.evaluation import EvalConfig, evaluate_model
    from pinnforge.sampling import default_spec

    spec = default_spec("harmonic")

    def zeros(coords: torch.Tensor) -> torch.Tensor:
        return torch.zeros(coords.shape[0], 1, dtype=coords.dtype, device=coords.device)

    result = evaluate_model(zeros, spec, EvalConfig(n_interior=16, seed=0))
    assert result.residual_max_abs < 1e-8
    assert result.relative_l2 == pytest.approx(1.0)
    assert result.l2 is not None and result.l2 > 0.5
    assert result.max_abs_error is not None and result.max_abs_error > 0.5
    assert result.ic_error == pytest.approx(1.0)
    assert result.bc_error is None
    assert result.rng is not None and result.rng["name"] == "test"


def test_poisson_boundary_error_is_reported_on_held_out_faces() -> None:
    import math

    import torch

    from pinnforge.evaluation import EvalConfig, evaluate_model
    from pinnforge.sampling import default_spec

    spec = default_spec("poisson")

    def manufactured(coords: torch.Tensor) -> torch.Tensor:
        return torch.sin(coords[:, 0:1] * math.pi) / (math.pi**2)

    exact = evaluate_model(manufactured, spec, EvalConfig(n_interior=8, seed=1))
    assert exact.bc_error is not None and exact.bc_error < 1e-8
    assert exact.bc_errors is not None and exact.bc_errors["dirichlet"] < 1e-8
    assert exact.ic_error is None
    assert exact.max_abs_error is not None and exact.max_abs_error < 1e-8

    def wrong(coords: torch.Tensor) -> torch.Tensor:
        return torch.ones(coords.shape[0], 1, dtype=coords.dtype, device=coords.device)

    missed = evaluate_model(wrong, spec, EvalConfig(n_interior=8, seed=1))
    assert missed.bc_error is not None and missed.bc_error > 0.5
    assert missed.reference == "analytical"
    assert missed.max_abs_error is not None and missed.max_abs_error > 0.5


def test_unmatched_poisson_reference_is_unavailable() -> None:
    import torch

    from pinnforge.equations import BoundaryCondition, Interval, PoissonToySpec
    from pinnforge.evaluation import EvalConfig, evaluate_model

    spec = PoissonToySpec(
        dimensions=1,
        source="sin_pi_x",
        x=Interval(lower=0.0, upper=0.5),
        boundary_conditions=[
            BoundaryCondition(variable="x", kind="dirichlet", side="min", value=0.0),
            BoundaryCondition(variable="x", kind="dirichlet", side="max", value=0.0),
        ],
    )

    def zeros(coords: torch.Tensor) -> torch.Tensor:
        return torch.zeros(coords.shape[0], 1, dtype=coords.dtype, device=coords.device)

    result = evaluate_model(zeros, spec, EvalConfig(n_interior=4, n_bc=2, seed=0))
    assert result.reference == "unavailable"
    assert result.l2 is None
    assert result.relative_l2 is None
    assert result.max_abs_error is None


def test_same_eval_seed_repeats() -> None:
    import torch

    from pinnforge.evaluation import EvalConfig, evaluate_model
    from pinnforge.models import mlp_from_spec
    from pinnforge.sampling import default_spec

    spec = default_spec("poisson")
    torch.manual_seed(1)
    first_model = mlp_from_spec(spec, (4, 4))
    torch.manual_seed(1)
    second_model = mlp_from_spec(spec, (4, 4))
    config = EvalConfig(n_interior=5, seed=7, bins=3)
    assert evaluate_model(first_model, spec, config) == evaluate_model(second_model, spec, config)
