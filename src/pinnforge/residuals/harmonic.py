"""Harmonic oscillator residual ü + ω² u.

ω is :attr:`HarmonicOscillatorSpec.angular_frequency` (the ``omega``
field, or ``sqrt(k / m)``). Coordinates are a single column ``t``, the
collocation-domain order from Day 1.
"""

from __future__ import annotations

from pinnforge.equations.harmonic import HarmonicOscillatorSpec
from pinnforge.ml_import import require_torch
from pinnforge.residuals.field import derivative_wrt

torch = require_torch()


def harmonic_residual_from_field(
    u: torch.Tensor,
    coords: torch.Tensor,
    spec: HarmonicOscillatorSpec,
) -> torch.Tensor:
    """Return ü + ω² u for a field that already depends on ``coords``.

    ``u`` has shape ``(n, 1)`` and ``coords`` has shape ``(n, 1)``. This
    does not rebuild the coordinate leaf: the closed-form tensor and the
    coordinates must already share a graph. :func:`pinnforge.residuals.residual`
    is the entry point that evaluates a model first.
    """

    _require_harmonic(spec, coords)
    if u.shape != (coords.shape[0], 1):
        raise ValueError(f"u must have shape {(coords.shape[0], 1)}, got {tuple(u.shape)}")
    speed = derivative_wrt(u, coords)
    acceleration = derivative_wrt(speed, coords)
    omega = spec.angular_frequency
    return acceleration + (omega**2) * u


def _require_harmonic(spec: HarmonicOscillatorSpec, coords: torch.Tensor) -> None:
    if not isinstance(spec, HarmonicOscillatorSpec):
        raise TypeError("spec must be a HarmonicOscillatorSpec")
    names = tuple(axis.name for axis in spec.collocation_domain().axes)
    if names != ("t",):
        raise ValueError(f"harmonic oscillator coordinates are (t,), got {names}")
    if coords.ndim != 2 or coords.shape[1] != 1:
        raise ValueError(
            f"harmonic oscillator expects coordinates of shape (n, 1), got {tuple(coords.shape)}"
        )
