"""Closed form for the harmonic oscillator.

The initial-value solution on the whole line is

    u(t) = A cos(ω (t - t0)) + B sin(ω (t - t0))

with A = u(t0), B = u'(t0) / ω, and t0 the lower end of the spec's time
interval. The spec interval is the training window; the formula is the
same outside it. ω is ``spec.angular_frequency`` (the ``omega`` field, or
sqrt(k / m)).
"""

from __future__ import annotations

from collections.abc import Sequence

import numpy as np

from pinnforge.equations.harmonic import HarmonicOscillatorSpec

ArrayLike = Sequence[float] | float | np.ndarray


def displacement(t: ArrayLike, spec: HarmonicOscillatorSpec) -> np.ndarray:
    """Displacement u(t) for a validated harmonic-oscillator spec."""

    state, _velocity = _state(t, spec)
    return state


def velocity(t: ArrayLike, spec: HarmonicOscillatorSpec) -> np.ndarray:
    """Time derivative u'(t) for a validated harmonic-oscillator spec."""

    _state_value, speed = _state(t, spec)
    return speed


def _state(t: ArrayLike, spec: HarmonicOscillatorSpec) -> tuple[np.ndarray, np.ndarray]:
    if not isinstance(spec, HarmonicOscillatorSpec):
        raise TypeError("spec must be a HarmonicOscillatorSpec")
    time = np.asarray(t, dtype=np.float64)
    if not np.isfinite(time).all():
        raise ValueError("t must be finite")
    omega = spec.angular_frequency
    t0 = spec.time.lower
    u0 = spec.initial_condition.components["u"]
    v0 = spec.initial_condition.components["du_dt"]
    angle = omega * (time - t0)
    state = u0 * np.cos(angle) + (v0 / omega) * np.sin(angle)
    speed = -omega * u0 * np.sin(angle) + v0 * np.cos(angle)
    return state, speed
