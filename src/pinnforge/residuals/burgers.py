"""Viscous Burgers residual u_t + u u_x − ν u_xx.

Columns follow the Day 1 collocation domain: ``x`` then ``t``. ν is
``spec.nu`` and is finite and positive.
"""

from __future__ import annotations

from pinnforge.equations.burgers import Burgers1DSpec
from pinnforge.ml_import import require_torch
from pinnforge.residuals.field import derivative_wrt

torch = require_torch()


def burgers_residual_from_field(
    u: torch.Tensor,
    coords: torch.Tensor,
    spec: Burgers1DSpec,
) -> torch.Tensor:
    """Return u_t + u u_x − ν u_xx.

    ``u`` has shape ``(n, 1)`` and ``coords`` has shape ``(n, 2)`` with
    columns ``(x, t)``. ``u`` must already depend on ``coords``.
    """

    _require_burgers(spec, coords)
    if u.shape != (coords.shape[0], 1):
        raise ValueError(f"u must have shape {(coords.shape[0], 1)}, got {tuple(u.shape)}")
    first = derivative_wrt(u, coords)
    u_x = first[:, 0:1]
    u_t = first[:, 1:2]
    u_xx = derivative_wrt(u_x, coords)[:, 0:1]
    return u_t + u * u_x - float(spec.nu) * u_xx


def _require_burgers(spec: Burgers1DSpec, coords: torch.Tensor) -> None:
    if not isinstance(spec, Burgers1DSpec):
        raise TypeError("spec must be a Burgers1DSpec")
    names = tuple(axis.name for axis in spec.collocation_domain().axes)
    if names != ("x", "t"):
        raise ValueError(f"Burgers coordinates are (x, t), got {names}")
    if coords.ndim != 2 or coords.shape[1] != 2:
        raise ValueError(f"Burgers expects coordinates of shape (n, 2), got {tuple(coords.shape)}")
