"""2D incompressible Navier–Stokes in periodic vorticity form.

    ω_t + u · ∇ω = ν Δω + f,   u = ∇^⊥ ψ,   −Δψ = ω

Kolmogorov forcing is the curl of velocity forcing ``(sin(k y), 0)``:

    f(x, y) = −k cos(k y)

Decaying turbulence uses ``f = 0``.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Literal

import numpy as np

from pinnforge.reference.spectral2d.common import (
    DEFAULT_LENGTH,
    Grid2D,
    fft2,
    ifft2,
    integrate_etdrk4,
    product_dealiased,
    require_viscosity,
    spectral_grad,
    time_nodes,
)

ForcingName = Literal["kolmogorov", "decaying"]
KOLMOGOROV_K = 4


@dataclass(frozen=True)
class NS2DTrajectory:
    """Saved vorticity ``omega[s, i, j]`` at time ``t[s]``."""

    x: np.ndarray
    y: np.ndarray
    t: np.ndarray
    omega: np.ndarray
    nu: float
    dt: float
    forcing: str
    grid: Grid2D

    def to_meta(self) -> dict[str, Any]:
        return {
            "equation": "ns2d",
            "nu": self.nu,
            "dt": self.dt,
            "forcing": self.forcing,
            "n": self.grid.n,
            "length": self.grid.length,
            "n_times": int(self.t.size),
        }


def streamfunction(omega: np.ndarray, grid: Grid2D) -> np.ndarray:
    """Periodic solution of ``−Δψ = ω`` with zero mean."""

    _, _, k2 = grid.wavenumbers()
    spectrum = fft2(omega)
    psi_hat = np.zeros_like(spectrum)
    nonzero = k2 > 0.0
    psi_hat[nonzero] = spectrum[nonzero] / k2[nonzero]
    psi_hat[0, 0] = 0.0
    return ifft2(psi_hat)


def velocity_from_vorticity(omega: np.ndarray, grid: Grid2D) -> np.ndarray:
    """Return ``(u, v) = (−ψ_y, ψ_x)``."""

    psi = streamfunction(omega, grid)
    psi_x, psi_y = spectral_grad(psi, grid)
    return np.stack((-psi_y, psi_x), axis=0)


def kolmogorov_forcing(grid: Grid2D, wavenumber: int = KOLMOGOROV_K) -> np.ndarray:
    if isinstance(wavenumber, bool) or not isinstance(wavenumber, (int, np.integer)) or int(wavenumber) < 1:
        raise ValueError("wavenumber must be an integer >= 1")
    wave = int(wavenumber)
    _, y = grid.mesh()
    return -wave * np.cos(wave * y)


def advection_vorticity(omega: np.ndarray, grid: Grid2D) -> np.ndarray:
    velocity = velocity_from_vorticity(omega, grid)
    dx, dy = spectral_grad(omega, grid)
    return product_dealiased(velocity[0], dx, grid) + product_dealiased(velocity[1], dy, grid)


def solve_ns2d(
    omega0: np.ndarray,
    nu: float,
    *,
    dt: float,
    t_final: float,
    save_dt: float,
    forcing: ForcingName = "kolmogorov",
    length: float = DEFAULT_LENGTH,
) -> NS2DTrajectory:
    field = np.asarray(omega0, dtype=np.float64)
    if field.ndim != 2 or field.shape[0] != field.shape[1]:
        raise ValueError("omega0 must have shape (n, n)")
    if forcing not in {"kolmogorov", "decaying"}:
        raise ValueError("forcing must be 'kolmogorov' or 'decaying'")
    grid = Grid2D(field.shape[0], length)
    viscosity = require_viscosity(nu)
    _, _, k2 = grid.wavenumbers()
    mask = grid.dealias_mask()
    force = kolmogorov_forcing(grid) if forcing == "kolmogorov" else np.zeros((grid.n, grid.n))
    force_hat = fft2(force) * mask

    def nonlinear(spectrum: np.ndarray) -> np.ndarray:
        omega = ifft2(spectrum * mask)
        return (-fft2(advection_vorticity(omega, grid)) + force_hat) * mask

    stored = integrate_etdrk4(
        field,
        -viscosity * k2,
        nonlinear,
        dt=dt,
        t_final=t_final,
        save_dt=save_dt,
    )
    x, y = grid.mesh()
    return NS2DTrajectory(
        x=x,
        y=y,
        t=time_nodes(t_final, save_dt),
        omega=stored,
        nu=viscosity,
        dt=dt,
        forcing=forcing,
        grid=grid,
    )
