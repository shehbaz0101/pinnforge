"""Solver-in-the-loop viscosity fit on a small grid. No pilot files."""

from __future__ import annotations

import numpy as np

from pinnforge.operator.inverse import ObservationSpec
from pinnforge.operator.solver_fit import (
    SolverSearch,
    build_solver_problem,
    fit_solver_problem,
    minimize_scalar,
    sensor_mismatch,
)
from pinnforge.reference.numerical.solver import grid, solve


def test_golden_section_finds_a_quadratic_minimum() -> None:
    search = SolverSearch(nu_min=0.005, nu_max=0.1, n_coarse=7, refine_width=1.0e-4, max_refine=12)
    found = minimize_scalar(lambda nu: (nu - 0.037) ** 2, search)
    assert abs(found.nu_hat - 0.037) <= search.refine_width
    assert found.n_evaluations <= search.n_coarse + 2 + search.max_refine
    assert found.objective <= (search.refine_width) ** 2


def test_oracle_solver_recovers_viscosity_on_a_short_trajectory() -> None:
    n = 32
    nodes = grid(n)
    u0 = np.sin(np.pi * nodes)
    nu = 0.05
    trajectory = solve(u0, nu, dt=5.0e-3, t_final=0.3, save_dt=0.05)
    spec = ObservationSpec(
        name="sensors32_two_bursts",
        n_sensors=32,
        series=((0, 1, 2), (3, 4, 5)),
        frame_dt=0.05,
    )
    search = SolverSearch(
        nu_min=0.02,
        nu_max=0.08,
        n_coarse=5,
        refine_width=1.0e-3,
        max_refine=8,
        n=32,
        dt=5.0e-3,
        save_dt=0.05,
    )
    problem = build_solver_problem(trajectory.u, spec, search, initial_condition="oracle_ic")
    assert np.allclose(problem.u0, trajectory.u[0])
    assert sensor_mismatch(problem, nu, search) < 1.0e-20
    found = fit_solver_problem(problem, search)
    assert abs(found["nu_hat"] - nu) / nu < 0.05


def test_sensor_lift_ignores_off_sensor_samples() -> None:
    n = 32
    nodes = grid(n)
    u0 = np.sin(np.pi * nodes)
    nu = 0.05
    trajectory = solve(u0, nu, dt=5.0e-3, t_final=0.3, save_dt=0.05)
    spec = ObservationSpec(
        name="sensors16_two_bursts",
        n_sensors=16,
        series=((0, 1, 2), (3, 4, 5)),
        frame_dt=0.05,
    )
    search = SolverSearch(
        nu_min=0.02,
        nu_max=0.08,
        n_coarse=3,
        refine_width=0.05,
        max_refine=0,
        n=32,
        dt=5.0e-3,
        save_dt=0.05,
    )
    clean = build_solver_problem(trajectory.u, spec, search, initial_condition="sensor_lift")
    scrambled = np.array(trajectory.u, copy=True)
    scrambled[:, 1::2] = 7.0
    changed = build_solver_problem(scrambled, spec, search, initial_condition="sensor_lift")
    assert np.allclose(clean.u0, changed.u0)
    assert np.allclose(clean.observed, changed.observed)
    oracle = build_solver_problem(scrambled, spec, search, initial_condition="oracle_ic")
    assert not np.allclose(oracle.u0, trajectory.u[0])
