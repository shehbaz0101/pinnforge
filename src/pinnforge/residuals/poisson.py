"""Poisson residual −Δu − f for a named source.

The Day 1 spec states −Δu = f. The residual is zero where that equation
holds. It is the same set of solutions as Δu + f = 0. ``f`` is the
registered :class:`~pinnforge.equations.poisson.PoissonSource`:

- ``zero`` → 0
- ``one`` → 1
- ``sin_pi_x`` → sin(π x), 1D
- ``sin_pi_x_sin_pi_y`` → sin(π x) sin(π y), 2D

Columns follow the collocation domain: ``x`` in 1D, ``x`` then ``y`` in 2D.
"""

from __future__ import annotations

import math

from pinnforge.equations.poisson import PoissonSource, PoissonToySpec
from pinnforge.ml_import import require_torch
from pinnforge.residuals.field import derivative_wrt

torch = require_torch()


def poisson_source(spec: PoissonToySpec, coords: torch.Tensor) -> torch.Tensor:
    """Evaluate the named source ``f`` at ``coords``. Shape ``(n, 1)``."""

    _require_poisson(spec, coords)
    if spec.source is PoissonSource.ZERO:
        return torch.zeros(coords.shape[0], 1, dtype=coords.dtype, device=coords.device)
    if spec.source is PoissonSource.ONE:
        return torch.ones(coords.shape[0], 1, dtype=coords.dtype, device=coords.device)
    x = coords[:, 0:1]
    if spec.source is PoissonSource.SIN_PI_X:
        return torch.sin(x * math.pi)
    y = coords[:, 1:2]
    return torch.sin(x * math.pi) * torch.sin(y * math.pi)


def poisson_residual_from_field(
    u: torch.Tensor,
    coords: torch.Tensor,
    spec: PoissonToySpec,
) -> torch.Tensor:
    """Return −Δu − f.

    ``u`` has shape ``(n, 1)`` and must already depend on ``coords``.
    """

    _require_poisson(spec, coords)
    if u.shape != (coords.shape[0], 1):
        raise ValueError(f"u must have shape {(coords.shape[0], 1)}, got {tuple(u.shape)}")
    return -_laplacian(u, coords) - poisson_source(spec, coords)


def _laplacian(u: torch.Tensor, coords: torch.Tensor) -> torch.Tensor:
    total = torch.zeros_like(u)
    first = derivative_wrt(u, coords)
    for index in range(coords.shape[1]):
        slope = first[:, index : index + 1]
        total = total + derivative_wrt(slope, coords)[:, index : index + 1]
    return total


def _require_poisson(spec: PoissonToySpec, coords: torch.Tensor) -> None:
    if not isinstance(spec, PoissonToySpec):
        raise TypeError("spec must be a PoissonToySpec")
    names = tuple(axis.name for axis in spec.collocation_domain().axes)
    expected = ("x",) if spec.dimensions == 1 else ("x", "y")
    if names != expected:
        raise ValueError(f"Poisson coordinates are {expected}, got {names}")
    if coords.ndim != 2 or coords.shape[1] != len(expected):
        raise ValueError(
            f"Poisson expects coordinates of shape (n, {len(expected)}), got {tuple(coords.shape)}"
        )
