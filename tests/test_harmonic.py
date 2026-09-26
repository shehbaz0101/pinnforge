"""Closed form for the harmonic oscillator."""

from __future__ import annotations

import math

import numpy as np
import pytest

from pinnforge.equations import (
    HarmonicOscillatorSpec,
    Interval,
    StateInitialCondition,
)
from pinnforge.reference import displacement, velocity


def _spec(
    *,
    omega: float | None = 2.0,
    k: float | None = None,
    m: float | None = None,
    t0: float = 0.0,
    t1: float = 1.0,
    u0: float = 1.0,
    v0: float = 0.0,
) -> HarmonicOscillatorSpec:
    payload: dict[str, object] = {
        "time": Interval(lower=t0, upper=t1),
        "initial_condition": StateInitialCondition(components={"u": u0, "du_dt": v0}),
    }
    if omega is None:
        payload["k"] = k
        payload["m"] = m
    else:
        payload["omega"] = omega
    return HarmonicOscillatorSpec.model_validate(payload)


def test_rest_displacement_is_cosine() -> None:
    spec = _spec(omega=2.0, u0=1.0, v0=0.0)
    times = np.array([0.0, math.pi / 4.0, math.pi / 2.0])
    got = displacement(times, spec)
    expected = np.cos(2.0 * times)
    assert got.shape == times.shape
    assert np.allclose(got, expected)
    assert np.allclose(velocity(times, spec), -2.0 * np.sin(2.0 * times))
    # u(π/2) = cos(π) = -1, and u'(π/4) = -ω sin(π/2) = -2.
    assert float(displacement(math.pi / 2.0, spec)) == pytest.approx(-1.0)
    assert float(velocity(math.pi / 4.0, spec)) == pytest.approx(-2.0)


def test_zero_displacement_is_sine() -> None:
    spec = _spec(omega=3.0, u0=0.0, v0=3.0)
    times = np.array([0.0, math.pi / 6.0])
    assert np.allclose(displacement(times, spec), np.sin(3.0 * times))
    assert np.allclose(velocity(times, spec), 3.0 * np.cos(3.0 * times))


def test_matches_a_cos_plus_b_sin() -> None:
    omega = 1.7
    u0 = -0.4
    v0 = 0.8
    t0 = 0.3
    spec = _spec(omega=omega, t0=t0, t1=t0 + 2.0, u0=u0, v0=v0)
    times = np.linspace(t0 - 0.5, t0 + 2.5, 9)
    amplitude_cos = u0
    amplitude_sin = v0 / omega
    angle = omega * (times - t0)
    expected = amplitude_cos * np.cos(angle) + amplitude_sin * np.sin(angle)
    assert np.allclose(displacement(times, spec), expected)


def test_initial_state_is_recovered_at_t0() -> None:
    spec = _spec(omega=math.pi, t0=1.5, t1=3.0, u0=0.25, v0=-0.5)
    assert displacement(1.5, spec) == pytest.approx(0.25)
    assert velocity(spec.time.lower, spec) == pytest.approx(-0.5)


def test_stiffness_matches_omega_and_conserves_energy() -> None:
    by_omega = _spec(omega=2.0, u0=0.5, v0=-1.25)
    by_stiffness = _spec(omega=None, k=8.0, m=2.0, u0=0.5, v0=-1.25)
    times = np.linspace(0.0, 1.7, 6)
    assert np.allclose(displacement(times, by_omega), displacement(times, by_stiffness))
    state = displacement(times, by_stiffness)
    speed = velocity(times, by_stiffness)
    energy = 0.5 * 2.0 * speed**2 + 0.5 * 8.0 * state**2
    assert np.allclose(energy, energy[0])


def test_scalar_and_vector_shapes() -> None:
    spec = _spec()
    scalar = displacement(0.0, spec)
    assert np.shape(scalar) == ()
    column = displacement(np.zeros((2, 1)), spec)
    assert column.shape == (2, 1)


def test_displacement_rejects_bad_time_and_spec() -> None:
    spec = _spec()
    with pytest.raises(ValueError, match="finite"):
        displacement([0.0, math.nan], spec)
    with pytest.raises(TypeError, match="HarmonicOscillatorSpec"):
        displacement(0.0, {"omega": 1.0})  # type: ignore[arg-type]
