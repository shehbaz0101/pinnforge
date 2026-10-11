"""CPU numpy pseudo-spectral 2D shear flow + passive tracer.

Matches The Well ``shear_flow`` setup documented at
https://polymathic-ai.org/the_well/datasets/shear_flow/ and the Dedalus
generator ``RudyMorel/the-well-rbc-sf`` ``src/generate_sf.py`` (Burns et al.,
Phys. Rev. Research 2, 023068, 2020). Periodic Fourier on
``[0, 1] x [-1, 1]``. Equations:

    u_t + u·∇u + ∇p = nu Δu
    s_t + u·∇s = kappa Δs
    div u = 0,  mean(p) = 0

with ``nu = 1/Re`` and ``kappa = nu/Sc``. Time stepping is vorticity-form
ETDRK4 with the two-thirds dealias. The Well generator used Dedalus SBDF3
at ``1024 x 2048`` then Fourier-downsampled to ``256 x 512``.
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Any

import numpy as np

from pinnforge.reference.numerical.solver import etdrk4_coefficients
from pinnforge.v06.grid import GridRect, fft2, ifft2, spectral_grad

LENGTH_X = 1.0
LENGTH_Y = 2.0
Y_MIN = -1.0


@dataclass(frozen=True)
class ShearIC:
    n_shear: int
    n_blobs: int
    width: float


SF_IC_FAMILY = [
    ShearIC(n_shear, n_blobs, width)
    for n_shear in (2, 4)
    for n_blobs in (2, 3, 4, 5)
    for width in (0.25, 0.5, 1.0, 2.0, 4.0)
]


def axes(nx: int, ny: int) -> tuple[np.ndarray, np.ndarray]:
    dx = LENGTH_X / int(nx)
    dy = LENGTH_Y / int(ny)
    x = dx * np.arange(int(nx), dtype=np.float64)
    y = Y_MIN + dy * np.arange(int(ny), dtype=np.float64)
    return x, y


def init_standard_shear(
    nx: int,
    ny: int,
    *,
    n_shear: int = 2,
    n_blobs: int = 2,
    width: float = 1.0,
) -> tuple[np.ndarray, np.ndarray]:
    """Dedalus ``init_standard_shear`` from ``the-well-rbc-sf``."""

    x, y = axes(nx, ny)
    xx, yy = np.meshgrid(x, y, indexing="ij")
    shear = np.zeros((nx, ny), dtype=np.float64)
    velocity = np.zeros((nx, ny), dtype=np.float64)
    z_shear = np.linspace(-1.0, 1.0, int(n_shear), endpoint=False) + 1.0 / float(n_shear)
    for i, z1 in enumerate(z_shear):
        sign = 2 * (i % 2) - 1
        zs = float(n_shear) * (yy - float(z1)) / 2.0 / float(width)
        shear += sign * 0.5 * np.tanh(zs / 0.1)
        velocity += 0.1 * np.sin(sign * float(n_blobs) * math.pi * xx / LENGTH_X) * np.exp(-(zs**2) / 0.01)
    shear += 0.5
    return shear, velocity


def vorticity(ux: np.ndarray, uy: np.ndarray, grid: GridRect) -> np.ndarray:
    ux_x, ux_y = spectral_grad(ux, grid)
    uy_x, uy_y = spectral_grad(uy, grid)
    _ = (ux_x, uy_y)
    return uy_x - ux_y


def velocity_from_vorticity(omega: np.ndarray, grid: GridRect) -> tuple[np.ndarray, np.ndarray]:
    kx, ky, k2 = grid.wavenumbers()
    spectrum = fft2(omega)
    psi_hat = np.zeros_like(spectrum)
    nonzero = k2 > 0.0
    # curl u = uy_x - ux_y = Δψ, so ψ_hat = ω_hat / (-k²)
    psi_hat[nonzero] = spectrum[nonzero] / (-k2[nonzero])
    psi = ifft2(psi_hat)
    psi_x, psi_y = spectral_grad(psi, grid)
    return -psi_y, psi_x


def pressure_from_velocity(ux: np.ndarray, uy: np.ndarray, grid: GridRect) -> np.ndarray:
    """Recover ``p`` from ``Δp = -∇·(u·∇u)`` with zero mean."""

    ux_x, ux_y = spectral_grad(ux, grid)
    uy_x, uy_y = spectral_grad(uy, grid)
    adv_x = ux * ux_x + uy * ux_y
    adv_y = ux * uy_x + uy * uy_y
    ax_x, _ax_y = spectral_grad(adv_x, grid)
    _ay_x, ay_y = spectral_grad(adv_y, grid)
    rhs = -(ax_x + ay_y)
    kx, ky, k2 = grid.wavenumbers()
    _ = (kx, ky)
    hat = fft2(rhs)
    p_hat = np.zeros_like(hat)
    nonzero = k2 > 0.0
    p_hat[nonzero] = hat[nonzero] / (-k2[nonzero])
    return ifft2(p_hat)


def _dealias_mask(grid: GridRect) -> np.ndarray:
    mx = np.fft.fftfreq(grid.nx) * grid.nx
    my = np.fft.fftfreq(grid.ny) * grid.ny
    mx2, my2 = np.meshgrid(mx, my, indexing="ij")
    return (np.abs(mx2) <= grid.nx / 3.0) & (np.abs(my2) <= grid.ny / 3.0)


def _product(left: np.ndarray, right: np.ndarray, grid: GridRect, mask: np.ndarray) -> np.ndarray:
    left_hat = fft2(left) * mask
    right_hat = fft2(right) * mask
    return ifft2(fft2(ifft2(left_hat) * ifft2(right_hat)) * mask)


def solve_shear(
    ux0: np.ndarray,
    uy0: np.ndarray,
    s0: np.ndarray,
    *,
    nu: float,
    kappa: float,
    dt: float,
    t_final: float,
    save_dt: float,
) -> dict[str, Any]:
    """Integrate vorticity and tracer. Returns stacked ``(T, 4, nx, ny)``."""

    ux0 = np.asarray(ux0, dtype=np.float64)
    uy0 = np.asarray(uy0, dtype=np.float64)
    s0 = np.asarray(s0, dtype=np.float64)
    nx, ny = ux0.shape
    grid = GridRect(nx, ny, LENGTH_X, LENGTH_Y)
    mask = _dealias_mask(grid)
    _kx, _ky, k2 = grid.wavenumbers()
    omega0 = vorticity(ux0, uy0, grid)
    linear_w = -float(nu) * k2
    linear_s = -float(kappa) * k2
    coeff_w = etdrk4_coefficients(linear_w, dt)
    coeff_s = etdrk4_coefficients(linear_s, dt)

    def nonlinear_w(spectrum: np.ndarray) -> np.ndarray:
        omega = ifft2(spectrum * mask)
        ux, uy = velocity_from_vorticity(omega, grid)
        wx, wy = spectral_grad(omega, grid)
        adv = _product(ux, wx, grid, mask) + _product(uy, wy, grid, mask)
        return (-fft2(adv)) * mask

    def nonlinear_s(spectrum: np.ndarray, ux: np.ndarray, uy: np.ndarray) -> np.ndarray:
        tracer = ifft2(spectrum * mask)
        sx, sy = spectral_grad(tracer, grid)
        adv = _product(ux, sx, grid, mask) + _product(uy, sy, grid, mask)
        return (-fft2(adv)) * mask

    w_hat = fft2(omega0)
    s_hat = fft2(s0)
    step_count = _n_steps(t_final, dt)
    save_stride = _n_steps(save_dt, dt)
    if step_count % save_stride != 0:
        raise ValueError("save_dt must divide t_final")
    n_save = step_count // save_stride + 1
    stored = np.empty((n_save, 4, nx, ny), dtype=np.float64)
    stored[0, 0] = ux0
    stored[0, 1] = uy0
    stored[0, 2] = pressure_from_velocity(ux0, uy0, grid)
    stored[0, 3] = s0
    frame = 1
    for step in range(1, step_count + 1):
        ux, uy = velocity_from_vorticity(ifft2(w_hat), grid)
        w_hat = _etdrk4_step(w_hat, nonlinear_w, *coeff_w)

        def _ns(spec: np.ndarray, ux: np.ndarray = ux, uy: np.ndarray = uy) -> np.ndarray:
            return nonlinear_s(spec, ux, uy)

        s_hat = _etdrk4_step(s_hat, _ns, *coeff_s)
        if step % save_stride == 0:
            omega = ifft2(w_hat)
            ux, uy = velocity_from_vorticity(omega, grid)
            stored[frame, 0] = ux
            stored[frame, 1] = uy
            stored[frame, 2] = pressure_from_velocity(ux, uy, grid)
            stored[frame, 3] = ifft2(s_hat)
            if not np.isfinite(stored[frame]).all():
                raise RuntimeError(f"non-finite field at step {step}")
            frame += 1
    return {
        "fields": stored,
        "t": np.arange(n_save, dtype=np.float64) * save_dt,
        "nu": float(nu),
        "kappa": float(kappa),
        "dt": float(dt),
        "nx": nx,
        "ny": ny,
    }


def _etdrk4_step(
    u_hat: np.ndarray,
    nonlinear: Any,
    e: np.ndarray,
    e2: np.ndarray,
    q: np.ndarray,
    f1: np.ndarray,
    f2: np.ndarray,
    f3: np.ndarray,
) -> np.ndarray:
    n0 = nonlinear(u_hat)
    a = e2 * u_hat + q * n0
    n1 = nonlinear(a)
    b = e2 * u_hat + q * n1
    n2 = nonlinear(b)
    c = e2 * a + q * (2.0 * n2 - n0)
    n3 = nonlinear(c)
    return e * u_hat + n0 * f1 + 2.0 * (n1 + n2) * f2 + n3 * f3


def _n_steps(total: float, step: float) -> int:
    count = int(round(float(total) / float(step)))
    if count < 1 or not math.isclose(count * float(step), float(total), rel_tol=0.0, abs_tol=1e-12):
        raise ValueError(f"{total} / {step} is not an integer step count")
    return count


def rel_l2(hat: np.ndarray, truth: np.ndarray) -> float:
    num = float(np.linalg.norm(np.asarray(hat) - np.asarray(truth)))
    den = float(np.linalg.norm(np.asarray(truth)))
    return num / max(den, 1e-18)
