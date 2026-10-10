"""Weak-form viscosity estimator. No torch and no pilot files."""

from __future__ import annotations

import numpy as np

from pinnforge.operator.inverse import ObservationSpec
from pinnforge.operator.weak_form import (
    WeakFormConfig,
    bump_derivatives,
    sensor_nodes,
    solve_weak_normal,
    weak_form_terms,
    weak_form_viscosity,
)
from pinnforge.reference.numerical.solver import grid, solve


def _smooth_trajectory(nu: float = 0.04):
    n = 64
    nodes = grid(n)
    u0 = np.sin(np.pi * nodes) + 0.25 * np.sin(2.0 * np.pi * nodes)
    u0 = u0 / np.max(np.abs(u0))
    trajectory = solve(u0, nu, dt=1.0e-3, t_final=0.4, save_dt=0.01)
    return trajectory, nu


def test_bump_derivatives_match_a_finite_difference() -> None:
    xi = np.linspace(-0.8, 0.8, 17)
    value, first, second = bump_derivatives(xi, 4)
    step = 1.0e-6
    value_right, _, _ = bump_derivatives(xi + step, 4)
    value_left, _, _ = bump_derivatives(xi - step, 4)
    assert np.allclose(first, (value_right - value_left) / (2.0 * step), rtol=1e-5, atol=1e-5)
    first_right = bump_derivatives(xi + step, 4)[1]
    first_left = bump_derivatives(xi - step, 4)[1]
    assert np.allclose(second, (first_right - first_left) / (2.0 * step), rtol=1e-4, atol=1e-4)
    outside, outside_d, outside_dd = bump_derivatives(np.array([-1.0, 1.0, 2.0]), 4)
    assert np.all(outside == 0.0)
    assert np.all(outside_d == 0.0)
    assert np.all(outside_dd == 0.0)


def test_resolved_weak_form_recovers_viscosity() -> None:
    trajectory, nu = _smooth_trajectory()
    spec = ObservationSpec(
        name="dense_sensors",
        n_sensors=64,
        series=(tuple(range(trajectory.u.shape[0])),),
        frame_dt=0.01,
    )
    hat = weak_form_viscosity(trajectory.u, spec, WeakFormConfig())
    assert abs(hat - nu) / nu < 3.0e-3


def test_burst_weak_form_stays_near_the_truth_on_a_smooth_field() -> None:
    trajectory, nu = _smooth_trajectory()
    spec = ObservationSpec(
        name="sensors32_bursts",
        n_sensors=32,
        series=tuple(tuple(range(start, start + 5)) for start in (0, 10, 20, 30)),
        frame_dt=0.01,
    )
    hat = weak_form_viscosity(trajectory.u, spec, WeakFormConfig())
    assert abs(hat - nu) / nu < 0.1


def test_off_sensor_values_are_not_read() -> None:
    trajectory, _nu = _smooth_trajectory()
    spec = ObservationSpec(
        name="sensors32_bursts",
        n_sensors=32,
        series=tuple(tuple(range(start, start + 5)) for start in (0, 10, 20, 30)),
        frame_dt=0.01,
    )
    config = WeakFormConfig()
    original = weak_form_viscosity(trajectory.u, spec, config)
    scrambled = np.array(trajectory.u, copy=True)
    scrambled[:, 1::2] = 50.0
    assert weak_form_viscosity(scrambled, spec, config) == original


def test_normal_equation_is_exact_for_a_single_coefficient() -> None:
    diffusion = np.array([1.0, 2.0, -0.5])
    advection = 0.03 * diffusion
    assert abs(solve_weak_normal(advection, diffusion) - 0.03) < 1e-12
    nodes = sensor_nodes(64, 32)
    assert nodes.shape == (32,)
    samples = np.ones((5, 32))
    times = np.arange(5, dtype=np.float64) * 0.01
    left, right = weak_form_terms(samples, nodes, times, WeakFormConfig())
    assert left.shape == right.shape
    assert left.size >= 1
