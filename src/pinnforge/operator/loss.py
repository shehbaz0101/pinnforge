"""Differentiable copy of the window Burgers residual.

The stencil matches :mod:`pinnforge.operator.residual`. Spatial derivatives
run in float64 and are cast back to the prediction dtype, so a float32 FNO
still trains against the same Fourier multiplier as the pilot solver.
``data`` mode does not build that graph. The returned scalar is the mean
over the batch that was passed in.
"""

from __future__ import annotations

import math

from pinnforge.ml_import import require_torch
from pinnforge.operator.residual import RESIDUAL_SCOPES, LossConfig
from pinnforge.operator.windows import FieldNorm
from pinnforge.reference.numerical.solver import LENGTH

torch = require_torch()


def window_operator_loss(
    prediction: torch.Tensor,
    targets: torch.Tensor,
    inputs: torch.Tensor,
    nu_normalized: torch.Tensor,
    norm: FieldNorm,
    config: LossConfig,
) -> torch.Tensor:
    """Normalized data MSE, residual MSE, or the weighted sum.

    ``prediction`` and ``targets`` are normalized ``u`` with shape
    ``(batch, output_frames, n_space)``. ``inputs`` is the normalized
    history. ``nu_normalized`` has shape ``(batch,)``.
    """

    _require_config(norm, config)
    data = torch.mean((prediction - targets) ** 2)
    if config.mode == "data":
        return data
    residual = torch_window_residual(inputs, prediction, nu_normalized, norm, config)
    physics = torch.mean(residual ** 2)
    if config.mode == "residual":
        return physics
    return data + float(config.residual_weight) * physics


def torch_window_residual(
    inputs: torch.Tensor,
    prediction: torch.Tensor,
    nu_normalized: torch.Tensor,
    norm: FieldNorm,
    config: LossConfig,
) -> torch.Tensor:
    """Residual tensor for one batch. Shape is ``(batch, n_residual, n_space)``."""

    _require_config(norm, config)
    if prediction.ndim != 3 or inputs.ndim != 3:
        raise ValueError("inputs and prediction must have shape (batch, frames, n_space)")
    if nu_normalized.ndim != 1 or nu_normalized.shape[0] != prediction.shape[0]:
        raise ValueError("nu must have shape (batch,)")
    nu = nu_normalized.to(dtype=prediction.dtype) * float(norm.nu_std) + float(norm.nu_mean)
    if config.residual_space == "physical":
        history = inputs * float(norm.u_std) + float(norm.u_mean)
        forecast = prediction * float(norm.u_std) + float(norm.u_mean)
        frames = _stack(history, forecast, config.residual_scope)
        return _central(frames, nu, dt=config.dt, space="physical", u_mean=0.0, u_std=1.0)
    frames = _stack(inputs, prediction, config.residual_scope)
    return _central(
        frames,
        nu,
        dt=config.dt,
        space="normalized",
        u_mean=float(norm.u_mean),
        u_std=float(norm.u_std),
    )


def _central(
    frames: torch.Tensor,
    nu: torch.Tensor,
    *,
    dt: float,
    space: str,
    u_mean: float,
    u_std: float,
) -> torch.Tensor:
    if frames.shape[1] < 3:
        raise ValueError("central Burgers residual needs at least 3 frames")
    u_t = (frames[:, 2:, :] - frames[:, :-2, :]) / (2.0 * float(dt))
    mid = frames[:, 1:-1, :]
    u_x = _spectral_derivative(mid, order=1)
    u_xx = _spectral_derivative(mid, order=2)
    column = nu.to(dtype=mid.dtype).view(-1, 1, 1)
    if space == "physical":
        return u_t + mid * u_x - column * u_xx
    amplitude = mid * float(u_std) + float(u_mean)
    return u_t + amplitude * u_x - column * u_xx


def _spectral_derivative(field: torch.Tensor, *, order: int) -> torch.Tensor:
    """``(i k)^order`` multiplier with the Nyquist entry removed.

    ``field`` keeps its dtype. The transform itself is float64, matching
    :func:`pinnforge.reference.numerical.solver.spectral_derivative`.
    """

    n_space = int(field.shape[-1])
    if n_space < 4 or n_space % 2 != 0:
        raise ValueError("n_space must be an even integer >= 4")
    if order < 1:
        raise ValueError("order must be >= 1")
    dx = LENGTH / n_space
    frequencies = torch.fft.fftfreq(n_space, d=dx, device=field.device)
    k = (2.0 * math.pi) * frequencies.to(dtype=torch.float64)
    k = k.clone()
    k[n_space // 2] = 0.0
    multiplier = (1j * k) ** order
    spectrum = torch.fft.fft(field.to(dtype=torch.float64), dim=-1)
    derivative = torch.fft.ifft(multiplier * spectrum, dim=-1).real
    return derivative.to(dtype=field.dtype)


def _stack(inputs: torch.Tensor, prediction: torch.Tensor, scope: str) -> torch.Tensor:
    if scope not in RESIDUAL_SCOPES:
        raise ValueError(f"unknown residual scope {scope!r}")
    if scope == "target_interior":
        return prediction
    if inputs.shape[1] < 2:
        raise ValueError("with_input residual needs at least 2 input frames")
    return torch.cat((inputs[:, -2:, :], prediction), dim=1)


def _require_config(norm: FieldNorm, config: LossConfig) -> None:
    if not isinstance(norm, FieldNorm):
        raise TypeError("norm must be a FieldNorm")
    if not isinstance(config, LossConfig):
        raise TypeError("config must be a LossConfig")
    if not math.isfinite(config.dt) or config.dt <= 0.0:
        raise ValueError("dt must be finite and > 0")
