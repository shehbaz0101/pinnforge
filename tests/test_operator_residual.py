"""Discrete Burgers residual on operator windows. No torch."""

from __future__ import annotations

import math

import numpy as np
import pytest

from pinnforge.operator.defaults import PREREGISTERED_HYBRID_WEIGHTS
from pinnforge.operator.residual import (
    LossConfig,
    central_burgers_residual,
    objective_from_parts,
    prediction_window_residual,
    residual_stats,
)
from pinnforge.operator.windows import FieldNorm
from pinnforge.reference.numerical.checks import cole_hopf
from pinnforge.reference.numerical.solver import grid


def test_preregistered_hybrid_weights_stay_fixed() -> None:
    assert PREREGISTERED_HYBRID_WEIGHTS == (1e-6, 1e-4, 1e-2)


def test_stationary_sine_matches_the_hand_residual() -> None:
    n_space = 64
    nu = 0.05
    x = grid(n_space)
    field = np.sin(math.pi * x)
    frames = np.broadcast_to(field, (1, 3, n_space)).copy()
    residual = central_burgers_residual(frames, np.array([nu]), dt=0.01)
    expected = 0.5 * math.pi * np.sin(2.0 * math.pi * x) + nu * (math.pi**2) * np.sin(math.pi * x)
    assert residual.shape == (1, 1, n_space)
    assert np.allclose(residual[0, 0], expected, rtol=1e-10, atol=1e-12)


def test_constant_field_has_zero_residual() -> None:
    frames = np.full((2, 5, 32), 0.3, dtype=np.float64)
    frames[1] = -0.2
    residual = central_burgers_residual(frames, np.array([0.04, 0.08]), dt=0.01)
    assert residual.shape == (2, 3, 32)
    assert np.allclose(residual, 0.0, atol=1e-12)


def test_cole_hopf_central_residual_shrinks_as_dt_shrinks() -> None:
    n_space = 64
    nu = 0.05
    x = grid(n_space)
    coarse = _cole_stack(x, nu=nu, dt=1e-2)
    fine = _cole_stack(x, nu=nu, dt=1e-4)
    coarse_abs = residual_stats(central_burgers_residual(coarse, np.array([nu]), dt=1e-2)).mean_abs
    fine_abs = residual_stats(central_burgers_residual(fine, np.array([nu]), dt=1e-4)).mean_abs
    assert fine_abs < 1e-6
    assert fine_abs < coarse_abs * 1e-3


def test_normalized_residual_is_the_physical_residual_over_u_std() -> None:
    norm = FieldNorm(u_mean=0.1, u_std=0.4, nu_mean=0.05, nu_std=0.02)
    rng = np.random.default_rng(0)
    physical = rng.normal(size=(2, 6, 32))
    normalized = norm.normalize_u(physical)
    nu = np.array([0.03, 0.07])
    physical_residual = central_burgers_residual(physical, nu, dt=0.01, space="physical")
    normalized_residual = central_burgers_residual(
        normalized,
        nu,
        dt=0.01,
        space="normalized",
        u_mean=norm.u_mean,
        u_std=norm.u_std,
    )
    assert np.allclose(physical_residual, norm.u_std * normalized_residual, rtol=1e-10, atol=1e-12)


def test_with_input_uses_the_last_two_frames_and_the_forecast() -> None:
    norm = FieldNorm(u_mean=0.0, u_std=1.0, nu_mean=0.05, nu_std=0.02)
    config = LossConfig(mode="residual", residual_scope="with_input")
    inputs = np.zeros((1, 4, 16))
    prediction = np.zeros((1, 4, 16))
    x = grid(16)
    inputs[0, -1] = np.sin(math.pi * x)
    inputs[0, -2] = 0.5 * np.sin(math.pi * x)
    base = prediction_window_residual(inputs, prediction, np.array([0.0]), norm, config)
    untouched = inputs.copy()
    untouched[0, 0] = 3.0
    same = prediction_window_residual(untouched, prediction, np.array([0.0]), norm, config)
    assert np.allclose(base, same)
    shifted = prediction.copy()
    shifted[0, 0] = 0.2
    changed = prediction_window_residual(inputs, shifted, np.array([0.0]), norm, config)
    assert not np.allclose(base, changed)
    interior = LossConfig(mode="residual", residual_scope="target_interior")
    interior_residual = prediction_window_residual(inputs, prediction, np.array([0.0]), norm, interior)
    assert interior_residual.shape[1] == 2
    assert base.shape[1] == 4


def test_data_loss_rejects_a_residual_weight_and_hybrid_requires_one() -> None:
    with pytest.raises(ValueError, match="residual_weight"):
        LossConfig(mode="data", residual_weight=1e-3)
    with pytest.raises(ValueError, match="residual_weight"):
        LossConfig(mode="residual", residual_weight=1e-3)
    with pytest.raises(ValueError, match="hybrid residual_weight"):
        LossConfig(mode="hybrid", residual_weight=0.0)
    hybrid = LossConfig(mode="hybrid", residual_weight=1e-4)
    assert hybrid.describe() == "normalized data MSE plus weighted Burgers residual"
    assert objective_from_parts(0.5, 20.0, hybrid) == pytest.approx(0.5 + 1e-4 * 20.0)
    assert objective_from_parts(0.5, 20.0, LossConfig()) == pytest.approx(0.5)
    assert LossConfig().describe() == "mean squared error in normalized u space"
    restored = LossConfig.from_dict(hybrid.to_dict())
    assert restored == hybrid


def test_operator_cli_rejects_a_weight_on_the_data_loss() -> None:
    from pinnforge.operator.__main__ import main as operator_main

    with pytest.raises(SystemExit) as caught:
        operator_main(["train", "--loss", "data", "--residual-weight", "1"])
    assert caught.value.code == 2
    with pytest.raises(SystemExit) as hybrid:
        operator_main(["train", "--loss", "hybrid"])
    assert hybrid.value.code == 2


def _cole_stack(x: np.ndarray, *, nu: float, dt: float) -> np.ndarray:
    center = 0.4
    times = np.array([center - dt, center, center + dt])
    field = cole_hopf(x[None, :], times[:, None], nu=nu, amplitude=0.5)
    return field[None, :, :]
