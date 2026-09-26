"""Shared autograd helpers for residual operators.

Coordinates are independent variables. :func:`prepare_coords` detaches
them and marks the copy as requiring grad, which is the leaf a later
training step differentiates through. :func:`derivative_wrt` uses
``torch.autograd.grad(..., create_graph=True)`` so the residual itself
can be differentiated with respect to the network weights.

A first derivative that does not depend on the coordinates (a linear
field has a constant slope) has a zero second derivative. That case
returns zeros instead of asking autograd to differentiate a constant.
"""

from __future__ import annotations

from pinnforge.ml_import import require_torch

torch = require_torch()


def dtype_device(model: object) -> tuple[torch.dtype, torch.device]:
    """Dtype and device of ``model`` parameters, or float64 on CPU.

    A plain function has no parameters. Float64 matches the collocation
    arrays from Day 2.
    """

    if isinstance(model, torch.nn.Module):
        try:
            parameter = next(model.parameters())
        except StopIteration:
            return torch.float64, torch.device("cpu")
        return parameter.dtype, parameter.device
    return torch.float64, torch.device("cpu")


def prepare_coords(coords: torch.Tensor, model: object) -> torch.Tensor:
    """Return a float leaf of shape ``(n, d)`` on the model's device.

    The returned tensor is detached from any graph the caller had. Empty
    batches stay empty and do not require grad.

    Raises:
        TypeError: ``coords`` is not a floating-point tensor.
        ValueError: ``coords`` is not a finite matrix.
    """

    if not isinstance(coords, torch.Tensor):
        raise TypeError("coords must be a torch.Tensor")
    if coords.ndim != 2:
        raise ValueError(f"coords must have shape (n, d), got {tuple(coords.shape)}")
    if not torch.is_floating_point(coords):
        raise TypeError("coords must be floating point")
    if not torch.isfinite(coords).all():
        raise ValueError("coords must be finite")
    dtype, device = dtype_device(model)
    prepared = coords.detach().to(dtype=dtype, device=device)
    if prepared.shape[0] == 0:
        return prepared
    return prepared.requires_grad_(True)


def evaluate_field(model: object, coords: torch.Tensor) -> torch.Tensor:
    """Evaluate ``model(coords)`` as a column of shape ``(n, 1)``."""

    if not callable(model):
        raise TypeError("model must be callable")
    value = model(coords)
    if not isinstance(value, torch.Tensor):
        raise TypeError("model must return a torch.Tensor")
    if value.ndim == 1:
        value = value.unsqueeze(-1)
    expected = (coords.shape[0], 1)
    if tuple(value.shape) != expected:
        raise ValueError(f"model must return shape {expected}, got {tuple(value.shape)}")
    if not torch.isfinite(value).all():
        raise ValueError("model output must be finite")
    return value


def derivative_wrt(outputs: torch.Tensor, inputs: torch.Tensor) -> torch.Tensor:
    """Row-wise partial derivatives of ``outputs`` with respect to ``inputs``.

    ``outputs`` has shape ``(n, 1)`` and ``inputs`` has shape ``(n, d)``.
    The result has shape ``(n, d)``. ``create_graph=True`` keeps the
    derivative in the graph. A column that does not depend on ``inputs``
    contributes zeros.
    """

    if outputs.ndim != 2 or outputs.shape[1] != 1:
        raise ValueError(f"outputs must have shape (n, 1), got {tuple(outputs.shape)}")
    if inputs.ndim != 2 or inputs.shape[0] != outputs.shape[0]:
        raise ValueError("inputs must have shape (n, d) with the same row count as outputs")
    zeros = torch.zeros(
        outputs.shape[0],
        inputs.shape[1],
        dtype=outputs.dtype,
        device=outputs.device,
    )
    if outputs.shape[0] == 0 or not outputs.requires_grad:
        return zeros
    grads = torch.autograd.grad(
        outputs,
        inputs,
        grad_outputs=torch.ones_like(outputs),
        create_graph=True,
        retain_graph=True,
        allow_unused=True,
    )[0]
    if grads is None:
        return zeros
    return grads


def zero_scalar(model: object) -> torch.Tensor:
    """A 0-d zero on the dtype and device :func:`dtype_device` selects."""

    dtype, device = dtype_device(model)
    return torch.zeros((), dtype=dtype, device=device)
