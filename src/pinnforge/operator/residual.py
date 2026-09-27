"""Discrete viscous Burgers residual on FNO windows.

The pilot solves ``u_t + u u_x = ν u_xx`` on ``x ∈ [-1, 1]`` with the
Fourier method in :mod:`pinnforge.reference.numerical.solver`. This module
uses that same spatial operator and a second-order central difference in
time, on the saved frames. It does not import torch. The training loss in
:mod:`pinnforge.operator.loss` is the differentiable copy of the same
stencil.

Spatial derivatives are :func:`spectral_derivative`: wave numbers
``k_m = 2 π m / L`` with ``L = 2``, Nyquist multiplier set to 0, and
``(i k)^p`` applied to the unnormalized FFT. Adjacent saved frames are
``Δt`` apart. The pilot uses ``save_dt = 0.01``, which is the default.

At an interior index ``n``

    u_t^n = (u^{n+1} - u^{n-1}) / (2 Δt),
    R^n = u_t^n + u^n u_x^n - ν u_xx^n.

``ν`` is the physical viscosity of that instance, not the normalized
channel. The network predicts normalized ``u``. ``physical`` denormalizes
with the manifest mean and standard deviation before forming ``R``.
``normalized`` is the same residual divided by ``σ_u``:

    R / σ_u = û_t + (σ_u û + μ_u) û_x - ν û_xx.

Those two fields have the same zeros. Their mean squares differ by
``σ_u²``, which changes the balance against the normalized data MSE.

Two index sets are supported. ``T`` is the number of predicted frames.

- ``target_interior`` stacks only the prediction. ``R`` is formed at
  predicted indexes ``1 .. T-2``. Both neighbors are predictions. The
  first and last predicted frames are not residual nodes.
- ``with_input`` prepends the last two input frames, which are data.
  ``R`` is then formed at the last input time and at predicted indexes
  ``0 .. T-2``. The node at the last input time uses the first prediction
  as its forward neighbor, so the rollout has to stay on the discrete
  Burgers stencil of the known history. The last predicted frame is still
  not a residual node, because there is no future frame. It does appear
  in the backward neighbor of the previous node.

Both sets need at least three frames in the stack, so ``target_interior``
needs at least three predicted frames and ``with_input`` needs at least
two input frames. The loss is the mean of ``R²`` over the residual nodes.
It is not a relative residual and it is not an inverse problem for ``ν``.
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Any

import numpy as np

from pinnforge.operator.windows import FieldNorm
from pinnforge.reference.numerical.solver import spectral_derivative

LOSS_MODES = ("data", "residual", "hybrid")
RESIDUAL_SCOPES = ("target_interior", "with_input")
RESIDUAL_SPACES = ("physical", "normalized")
DEFAULT_FRAME_DT = 0.01

_DATA_LOSS = "mean squared error in normalized u space"
_RESIDUAL_LOSS = "mean squared Burgers residual"
_HYBRID_LOSS = "normalized data MSE plus weighted Burgers residual"


@dataclass(frozen=True, slots=True)
class LossConfig:
    """How an FNO step mixes normalized data MSE and the Burgers residual.

    ``data`` is the Stage 3 loss. ``residual_weight`` must be 0.
    ``residual`` is the mean square residual alone. ``residual_weight``
    must be 0, because the weight is not applied. ``hybrid`` is
    data MSE plus ``residual_weight`` times residual MSE, and the weight
    must be finite and positive. Selection of the checkpoint is not this
    loss. Callers still select on validation mean relative L2.
    """

    mode: str = "data"
    residual_weight: float = 0.0
    residual_scope: str = "with_input"
    residual_space: str = "physical"
    dt: float = DEFAULT_FRAME_DT

    def __post_init__(self) -> None:
        if self.mode not in LOSS_MODES:
            known = ", ".join(LOSS_MODES)
            raise ValueError(f"loss mode must be one of {known}")
        if self.residual_scope not in RESIDUAL_SCOPES:
            known = ", ".join(RESIDUAL_SCOPES)
            raise ValueError(f"residual scope must be one of {known}")
        if self.residual_space not in RESIDUAL_SPACES:
            known = ", ".join(RESIDUAL_SPACES)
            raise ValueError(f"residual space must be one of {known}")
        dt = _require_finite(self.dt, "dt")
        if dt <= 0.0:
            raise ValueError("dt must be > 0")
        object.__setattr__(self, "dt", dt)
        if self.mode == "hybrid":
            weight = _require_finite(self.residual_weight, "residual_weight")
            if weight <= 0.0:
                raise ValueError("hybrid residual_weight must be > 0")
            object.__setattr__(self, "residual_weight", weight)
            return
        weight = _require_finite(self.residual_weight, "residual_weight")
        if weight != 0.0:
            raise ValueError("residual_weight must be 0 unless the loss mode is hybrid")
        object.__setattr__(self, "residual_weight", 0.0)

    def uses_physics(self) -> bool:
        """True when the optimized loss includes the Burgers residual."""

        return self.mode != "data"

    def describe(self) -> str:
        """Short loss name stored in the run manifest."""

        if self.mode == "data":
            return _DATA_LOSS
        if self.mode == "residual":
            return _RESIDUAL_LOSS
        return _HYBRID_LOSS

    def to_dict(self) -> dict[str, Any]:
        return {
            "mode": self.mode,
            "residual_weight": self.residual_weight,
            "residual_scope": self.residual_scope,
            "residual_space": self.residual_space,
            "dt": self.dt,
        }

    @classmethod
    def from_dict(cls, payload: dict[str, Any]) -> LossConfig:
        """Rebuild a config from the checkpoint or run manifest block."""

        try:
            return cls(
                mode=str(payload["mode"]),
                residual_weight=float(payload["residual_weight"]),
                residual_scope=str(payload["residual_scope"]),
                residual_space=str(payload["residual_space"]),
                dt=float(payload["dt"]),
            )
        except KeyError as exc:
            raise ValueError(f"loss config is missing {exc.args[0]}") from exc
        except (TypeError, ValueError) as exc:
            raise ValueError(f"loss config is invalid: {exc}") from exc


@dataclass(frozen=True, slots=True)
class ResidualStats:
    """Mean magnitude and mean square of one residual array."""

    mean_abs: float
    mean_square: float
    n_points: int


def central_burgers_residual(
    frames: np.ndarray,
    nu: np.ndarray,
    *,
    dt: float,
    space: str = "physical",
    u_mean: float = 0.0,
    u_std: float = 1.0,
) -> np.ndarray:
    """Central residual at indexes ``1 .. n_times-2``.

    ``frames`` has shape ``(batch, n_times, n_space)`` and ``nu`` has
    shape ``(batch,)``. Both are finite. ``nu`` is physical viscosity and
    must be positive. ``n_space`` must be even and at least 4, because
    that is the spectral grid. The returned array has shape
    ``(batch, n_times - 2, n_space)``.
    """

    field = np.asarray(frames, dtype=np.float64)
    viscosity = np.asarray(nu, dtype=np.float64)
    step = _require_finite(dt, "dt")
    if step <= 0.0:
        raise ValueError("dt must be > 0")
    if space not in RESIDUAL_SPACES:
        raise ValueError(f"residual space must be one of {', '.join(RESIDUAL_SPACES)}")
    if field.ndim != 3 or field.shape[1] < 3 or field.shape[2] < 4:
        raise ValueError("frames must have shape (batch, n_times >= 3, n_space >= 4)")
    if viscosity.shape != (field.shape[0],):
        raise ValueError("nu must have shape (batch,)")
    if not np.isfinite(field).all() or not np.isfinite(viscosity).all():
        raise ValueError("frames and nu must be finite")
    if np.any(viscosity <= 0.0):
        raise ValueError("nu must be > 0")
    mean = _require_finite(u_mean, "u_mean")
    std = _require_finite(u_std, "u_std")
    if std <= 0.0:
        raise ValueError("u_std must be > 0")
    u_t = (field[:, 2:, :] - field[:, :-2, :]) / (2.0 * step)
    mid = field[:, 1:-1, :]
    u_x = spectral_derivative(mid, order=1)
    u_xx = spectral_derivative(mid, order=2)
    column = viscosity[:, None, None]
    if space == "physical":
        residual = u_t + mid * u_x - column * u_xx
    else:
        amplitude = std * mid + mean
        residual = u_t + amplitude * u_x - column * u_xx
    if not np.isfinite(residual).all():
        raise ValueError("Burgers residual is not finite")
    return np.asarray(residual, dtype=np.float64)


def prediction_window_residual(
    inputs: np.ndarray,
    prediction: np.ndarray,
    nu_normalized: np.ndarray,
    norm: FieldNorm,
    config: LossConfig,
) -> np.ndarray:
    """Residual of one predicted window under ``config``.

    ``inputs``, ``prediction``, and ``nu_normalized`` are in the normalized
    space stored on a :class:`~pinnforge.operator.windows.WindowDataset`.
    ``inputs`` is ``(batch, input_frames, n_space)`` and ``prediction`` is
    ``(batch, output_frames, n_space)``. Viscosity is denormalized with
    ``norm`` before it enters the PDE. Input frames used by ``with_input``
    are data. They are not model outputs.
    """

    if not isinstance(norm, FieldNorm):
        raise TypeError("norm must be a FieldNorm")
    if not isinstance(config, LossConfig):
        raise TypeError("config must be a LossConfig")
    history = np.asarray(inputs, dtype=np.float64)
    forecast = np.asarray(prediction, dtype=np.float64)
    nu_hat = np.asarray(nu_normalized, dtype=np.float64)
    _require_pair(history, forecast, nu_hat)
    nu = norm.denormalize_nu(nu_hat)
    if config.residual_space == "physical":
        frames = _stack(norm.denormalize_u(history), norm.denormalize_u(forecast), config.residual_scope)
        return central_burgers_residual(frames, nu, dt=config.dt, space="physical")
    frames = _stack(history, forecast, config.residual_scope)
    return central_burgers_residual(
        frames,
        nu,
        dt=config.dt,
        space="normalized",
        u_mean=norm.u_mean,
        u_std=norm.u_std,
    )


def residual_stats(residual: np.ndarray) -> ResidualStats:
    """Mean of ``|R|`` and mean of ``R²``."""

    values = np.asarray(residual, dtype=np.float64)
    if values.size < 1 or not np.isfinite(values).all():
        raise ValueError("residual must be non-empty and finite")
    return ResidualStats(
        mean_abs=float(np.mean(np.abs(values))),
        mean_square=float(np.mean(values * values)),
        n_points=int(values.size),
    )


def objective_from_parts(data_mse: float, residual_mse: float, config: LossConfig) -> float:
    """Scalar the optimizer minimizes, from full-set means.

    ``data`` returns the normalized data MSE. ``residual`` returns the
    residual MSE. ``hybrid`` returns the weighted sum. The parts are the
    means over the windows being scored, not means of per-batch means.
    """

    if not isinstance(config, LossConfig):
        raise TypeError("config must be a LossConfig")
    data = _require_finite(data_mse, "data_mse")
    physics = _require_finite(residual_mse, "residual_mse")
    if data < 0.0 or physics < 0.0:
        raise ValueError("mean squared terms must be >= 0")
    if config.mode == "data":
        return data
    if config.mode == "residual":
        return physics
    return data + config.residual_weight * physics


def _stack(inputs: np.ndarray, prediction: np.ndarray, scope: str) -> np.ndarray:
    if scope == "target_interior":
        return prediction
    if scope != "with_input":
        raise ValueError(f"unknown residual scope {scope!r}")
    if inputs.shape[1] < 2:
        raise ValueError("with_input residual needs at least 2 input frames")
    return np.concatenate([inputs[:, -2:, :], prediction], axis=1)


def _require_pair(inputs: np.ndarray, prediction: np.ndarray, nu: np.ndarray) -> None:
    if inputs.ndim != 3 or prediction.ndim != 3:
        raise ValueError("inputs and prediction must have shape (batch, frames, n_space)")
    if inputs.shape[0] != prediction.shape[0] or inputs.shape[2] != prediction.shape[2]:
        raise ValueError("inputs and prediction must share batch and n_space")
    if inputs.shape[1] < 1 or prediction.shape[1] < 1:
        raise ValueError("inputs and prediction need at least one frame")
    if nu.shape != (inputs.shape[0],) or not np.isfinite(nu).all():
        raise ValueError("nu must be a finite vector of length batch")
    if not np.isfinite(inputs).all() or not np.isfinite(prediction).all():
        raise ValueError("inputs and prediction must be finite")


def _require_finite(value: object, label: str) -> float:
    if isinstance(value, bool) or not isinstance(value, (int, float, np.floating)):
        raise ValueError(f"{label} must be a finite number")
    number = float(value)
    if not math.isfinite(number):
        raise ValueError(f"{label} must be a finite number")
    return number
