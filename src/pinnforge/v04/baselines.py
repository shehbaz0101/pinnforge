"""Closed-form residual least squares and a 2D weak-form identifier."""

from __future__ import annotations

import math

import numpy as np

from pinnforge.reference.spectral2d.burgers import burgers_rhs_physical
from pinnforge.reference.spectral2d.common import Grid2D, spectral_laplacian
from pinnforge.reference.spectral2d.navier_stokes import (
    advection_vorticity,
    kolmogorov_forcing,
    velocity_from_vorticity,
)
from pinnforge.v04.sensors import SensorMask, lift_sensors, read_sensors


def closed_form_ls(
    field: np.ndarray,
    mask: SensorMask,
    *,
    problem: str,
    dt: float,
    forcing: str = "none",
    length: float,
) -> float:
    """Normal equation of the lifted discrete residual. ``field`` is clean or noisy observations on the full grid or already the stored field; sensors are read first."""

    sensors = read_sensors(field, mask)
    lifted = lift_sensors(sensors, mask)
    grid = Grid2D(mask.n, length)
    rows_b: list[np.ndarray] = []
    rows_a: list[np.ndarray] = []
    for start in mask.burst_starts:
        block = lifted[start : start + mask.burst_length]
        if block.shape[0] < 3:
            continue
        mid = block[1]
        time_deriv = (block[2] - block[0]) / (2.0 * dt)
        if problem == "burgers2d":
            adv = burgers_rhs_physical(mid, grid)
            lap = np.stack([spectral_laplacian(mid[c], grid) for c in range(2)], axis=0)
            rows_b.append((time_deriv + adv).ravel())
            rows_a.append(lap.ravel())
        else:
            omega = mid[0]
            force = kolmogorov_forcing(grid) if forcing == "kolmogorov" else 0.0
            rhs = time_deriv[0] + advection_vorticity(omega, grid) - force
            rows_b.append(np.ravel(rhs))
            rows_a.append(np.ravel(spectral_laplacian(omega, grid)))
    if not rows_a:
        return float("nan")
    a = np.concatenate(rows_a)
    b = np.concatenate(rows_b)
    denom = float(np.dot(a, a))
    if denom <= 1e-18:
        return float("nan")
    return float(np.dot(a, b) / denom)


def weak_form(
    field: np.ndarray,
    mask: SensorMask,
    *,
    problem: str,
    dt: float,
    forcing: str = "none",
    length: float,
) -> float:
    """Compact test functions on the sensor lattice. The field is not lifted."""

    sensors = read_sensors(field, mask)
    n_side = mask.n_side
    dx = length / n_side
    coarse = Grid2D(n_side, length)
    degree = 4
    radius = max(2, n_side // 4)
    centers = range(0, n_side, 2)
    a_rows: list[float] = []
    b_rows: list[float] = []
    for start in mask.burst_starts:
        block = sensors[start : start + mask.burst_length]
        if block.shape[0] < 3:
            continue
        eta = _time_bump(block.shape[0])
        eta_t = np.gradient(eta, dt)
        for ix in centers:
            for iy in centers:
                bump = _space_bump(n_side, ix, iy, radius, degree)
                weight = bump * dx * dx
                if problem == "burgers2d":
                    # u_t · φ + (u·∇u) · φ = ν u · Δφ, derivatives of φ only for diffusion.
                    lap = spectral_laplacian(bump, coarse)
                    for component in range(2):
                        series = block[:, component]
                        u_mid = series[1]
                        time_term = -float(np.sum(series * eta_t[:, None, None] * weight))
                        vel = np.stack([block[1, 0], block[1, 1]], axis=0)
                        adv = burgers_rhs_physical(vel, coarse)
                        adv_term = float(np.sum(adv[component] * bump * dx * dx))
                        diff = float(np.sum(u_mid * lap * dx * dx))
                        a_rows.append(diff)
                        b_rows.append(time_term + adv_term)
                else:
                    omega = block[:, 0]
                    lap = spectral_laplacian(bump, coarse)
                    time_term = -float(np.sum(omega * eta_t[:, None, None] * weight))
                    vel = velocity_from_vorticity(omega[1], coarse)
                    gx, gy = np.gradient(bump, dx, dx)
                    adv_term = -float(np.sum(omega[1] * (vel[0] * gx + vel[1] * gy) * dx * dx))
                    force = 0.0
                    if forcing == "kolmogorov":
                        force = float(np.sum(kolmogorov_forcing(coarse) * bump * dx * dx) * float(np.sum(eta)))
                    diff = float(np.sum(omega[1] * lap * dx * dx))
                    a_rows.append(diff)
                    b_rows.append(time_term + adv_term - force)
    a = np.asarray(a_rows, dtype=np.float64)
    b = np.asarray(b_rows, dtype=np.float64)
    keep = np.abs(a) >= 1e-12 * max(1.0, float(np.max(np.abs(a))))
    if not np.any(keep):
        return float("nan")
    denom = float(np.dot(a[keep], a[keep]))
    if denom <= 1e-18:
        return float("nan")
    return float(np.dot(a[keep], b[keep]) / denom)


def _space_bump(n: int, cx: int, cy: int, radius: int, degree: int) -> np.ndarray:
    xs = (np.arange(n) - cx + n // 2) % n - n // 2
    ys = (np.arange(n) - cy + n // 2) % n - n // 2
    xx, yy = np.meshgrid(xs, ys, indexing="ij")
    xi = np.sqrt(xx * xx + yy * yy) / radius
    bump = np.clip(1.0 - xi * xi, 0.0, None) ** degree
    return bump


def _time_bump(n_frames: int) -> np.ndarray:
    if n_frames < 3:
        raise ValueError("weak-form burst must have at least 3 frames")
    axis = np.linspace(-1.0, 1.0, n_frames)
    return np.clip(1.0 - axis * axis, 0.0, None) ** 4


def relative_error(nu_hat: float, nu_true: float) -> float:
    if not math.isfinite(nu_hat) or nu_true <= 0.0:
        return float("inf")
    return abs(nu_hat - nu_true) / nu_true
