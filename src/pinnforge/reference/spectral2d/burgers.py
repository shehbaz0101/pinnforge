"""2D periodic viscous Burgers, vector form.

    u_t + (u · ∇)u = ν Δu

on a periodic square. The nonlinear term is dealiased with the two-thirds
rule. Time stepping is ETDRK4 with exact treatment of ``−ν Δ``.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import numpy as np

from pinnforge.reference.spectral2d.common import (
    Grid2D,
    fft2,
    ifft2,
    integrate_etdrk4,
    product_dealiased,
    require_viscosity,
    spectral_grad,
    time_nodes,
)


@dataclass(frozen=True)
class Burgers2DTrajectory:
    """Saved velocity field ``u[s, c, i, j]`` at time ``t[s]``."""

    x: np.ndarray
    y: np.ndarray
    t: np.ndarray
    u: np.ndarray
    nu: float
    dt: float
    grid: Grid2D

    def to_meta(self) -> dict[str, Any]:
        return {
            "equation": "burgers2d",
            "nu": self.nu,
            "dt": self.dt,
            "n": self.grid.n,
            "length": self.grid.length,
            "n_times": int(self.t.size),
        }


def burgers_rhs_physical(velocity: np.ndarray, grid: Grid2D) -> np.ndarray:
    """Return ``(u · ∇)u`` on a velocity field of shape ``(2, n, n)``."""

    field = np.asarray(velocity, dtype=np.float64)
    if field.shape != (2, grid.n, grid.n):
        raise ValueError("velocity must have shape (2, n, n)")
    adv = np.empty_like(field)
    for component in range(2):
        dx, dy = spectral_grad(field[component], grid)
        adv[component] = product_dealiased(field[0], dx, grid) + product_dealiased(field[1], dy, grid)
    return adv


def solve_burgers2d(
    u0: np.ndarray,
    nu: float,
    *,
    dt: float,
    t_final: float,
    save_dt: float,
    length: float = 2.0 * np.pi,
) -> Burgers2DTrajectory:
    field = np.asarray(u0, dtype=np.float64)
    if field.ndim != 3 or field.shape[0] != 2 or field.shape[1] != field.shape[2]:
        raise ValueError("u0 must have shape (2, n, n)")
    grid = Grid2D(field.shape[1], length)
    viscosity = require_viscosity(nu)
    _, _, k2 = grid.wavenumbers()
    mask = grid.dealias_mask()

    def nonlinear(spectrum: np.ndarray) -> np.ndarray:
        velocity = ifft2(spectrum * mask)
        return -fft2(burgers_rhs_physical(velocity, grid)) * mask

    linear = -viscosity * k2
    stored = integrate_etdrk4(
        field,
        linear,
        nonlinear,
        dt=dt,
        t_final=t_final,
        save_dt=save_dt,
    )
    x, y = grid.mesh()
    return Burgers2DTrajectory(
        x=x,
        y=y,
        t=time_nodes(t_final, save_dt),
        u=stored,
        nu=viscosity,
        dt=dt,
        grid=grid,
    )
