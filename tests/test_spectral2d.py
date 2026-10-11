"""2D spectral Burgers and vorticity-form Navier-Stokes."""

from __future__ import annotations

import math

import numpy as np
import pytest

from pinnforge.reference.spectral2d.burgers import solve_burgers2d
from pinnforge.reference.spectral2d.common import Grid2D, spectral_laplacian
from pinnforge.reference.spectral2d.dataset import relative_l2, restrict_field
from pinnforge.reference.spectral2d.initial import bandlimited_velocity, bandlimited_vorticity
from pinnforge.reference.spectral2d.navier_stokes import (
    solve_ns2d,
    streamfunction,
    velocity_from_vorticity,
)
from pinnforge.v04.sensors import SensorMask, lift_sensors, read_sensors


def test_laplacian_of_a_fourier_mode() -> None:
    grid = Grid2D(16)
    x, y = grid.mesh()
    field = np.sin(x) * np.cos(2.0 * y)
    lap = spectral_laplacian(field, grid)
    expected = -(1.0 + 4.0) * field
    assert np.allclose(lap, expected, atol=1e-10)


def test_biot_savart_recovers_a_streamfunction() -> None:
    grid = Grid2D(16)
    x, y = grid.mesh()
    psi = np.sin(x) * np.sin(y)
    omega = 2.0 * psi
    recovered = streamfunction(omega, grid)
    recovered = recovered - recovered.mean()
    assert np.allclose(recovered, psi - psi.mean(), atol=1e-10)
    velocity = velocity_from_vorticity(omega, grid)
    assert np.allclose(velocity[0], -np.sin(x) * np.cos(y), atol=1e-10)
    assert np.allclose(velocity[1], np.cos(x) * np.sin(y), atol=1e-10)


def test_burgers_short_run_stays_finite() -> None:
    grid = Grid2D(16)
    rng = np.random.default_rng(1)
    u0 = bandlimited_velocity(grid, rng, max_mode=2)
    traj = solve_burgers2d(u0, 0.04, dt=0.005, t_final=0.02, save_dt=0.01)
    assert traj.u.shape == (3, 2, 16, 16)
    assert np.isfinite(traj.u).all()
    assert traj.t[-1] == pytest.approx(0.02)


def test_ns_decaying_energy_does_not_grow() -> None:
    grid = Grid2D(16)
    rng = np.random.default_rng(2)
    omega0 = bandlimited_vorticity(grid, rng, max_mode=2)
    traj = solve_ns2d(omega0, 0.05, dt=0.005, t_final=0.02, save_dt=0.01, forcing="decaying")
    energy = [float(np.mean(frame**2)) for frame in traj.omega]
    assert energy[-1] <= energy[0] * 1.05


def test_sensor_lift_recovers_a_low_mode() -> None:
    grid = Grid2D(16)
    x, y = grid.mesh()
    field = np.sin(x) * np.cos(y)
    stacked = np.stack([field, field], axis=0)
    mask = SensorMask(n=16, n_side=8, burst_starts=(0,), burst_length=1, frame_dt=0.01)
    coarse = read_sensors(stacked, mask)
    lifted = lift_sensors(coarse, mask)
    assert relative_l2(lifted, stacked) < 1e-10


def test_restrict_field_keeps_a_resolved_mode() -> None:
    fine = Grid2D(16)
    x, y = fine.mesh()
    field = np.sin(x)
    coarse = restrict_field(field, 8)
    axis = (2.0 * math.pi / 8.0) * np.arange(8)
    expected = np.sin(axis)[:, None] * np.ones((8, 8))
    # sin(x) is independent of y.
    expected = np.broadcast_to(np.sin(axis)[:, None], (8, 8)).copy()
    assert relative_l2(coarse, expected) < 1e-10
