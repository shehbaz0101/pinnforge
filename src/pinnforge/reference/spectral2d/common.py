"""Shared 2D periodic Fourier helpers and ETDRK4 coefficients.

The domain is a square of length ``L = 2π`` unless a caller passes another
length. Nodes are ``x_j = L * j / N``. Wave numbers use NumPy FFT order
with ``k_m = 2π m / L``. The Nyquist mode of an even grid is zeroed in
every derivative multiplier. Dealiasing is the two-thirds rule: modes
with ``|m| > N/3`` are dropped before every product.
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Any

import numpy as np

from pinnforge.reference.numerical.solver import etdrk4_coefficients

TWO_PI = 2.0 * math.pi
DEFAULT_LENGTH = TWO_PI
_SERIES_ORDER = 18


@dataclass(frozen=True)
class Grid2D:
    """Periodic square grid."""

    n: int
    length: float = DEFAULT_LENGTH

    def __post_init__(self) -> None:
        if isinstance(self.n, bool) or not isinstance(self.n, (int, np.integer)) or int(self.n) < 8 or int(self.n) % 2:
            raise ValueError("n must be an even integer >= 8")
        object.__setattr__(self, "n", int(self.n))
        length = float(self.length)
        if not math.isfinite(length) or length <= 0.0:
            raise ValueError("length must be finite and > 0")
        object.__setattr__(self, "length", length)

    @property
    def dx(self) -> float:
        return self.length / self.n

    def axes(self) -> np.ndarray:
        return self.dx * np.arange(self.n, dtype=np.float64)

    def mesh(self) -> tuple[np.ndarray, np.ndarray]:
        axis = self.axes()
        return np.meshgrid(axis, axis, indexing="ij")

    def wavenumbers(self) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        k1 = TWO_PI * np.fft.fftfreq(self.n, d=self.dx)
        k1 = np.asarray(k1, dtype=np.float64)
        k1[self.n // 2] = 0.0
        kx, ky = np.meshgrid(k1, k1, indexing="ij")
        return kx, ky, kx * kx + ky * ky

    def dealias_mask(self) -> np.ndarray:
        cutoff = self.n / 3.0
        modes = np.fft.fftfreq(self.n) * self.n
        mx, my = np.meshgrid(modes, modes, indexing="ij")
        return (np.abs(mx) <= cutoff) & (np.abs(my) <= cutoff)

    def to_dict(self) -> dict[str, Any]:
        return {
            "n": self.n,
            "length": self.length,
            "dealiasing": "two_thirds",
            "wavenumber": "k_m = 2*pi/L * m",
        }


def fft2(field: np.ndarray) -> np.ndarray:
    return np.fft.fft2(np.asarray(field, dtype=np.float64), axes=(-2, -1))


def ifft2(spectrum: np.ndarray) -> np.ndarray:
    return np.fft.ifft2(spectrum, axes=(-2, -1)).real.astype(np.float64, copy=False)


def spectral_grad(field: np.ndarray, grid: Grid2D) -> tuple[np.ndarray, np.ndarray]:
    kx, ky, _ = grid.wavenumbers()
    spectrum = fft2(field)
    return ifft2(1j * kx * spectrum), ifft2(1j * ky * spectrum)


def spectral_laplacian(field: np.ndarray, grid: Grid2D) -> np.ndarray:
    _, _, k2 = grid.wavenumbers()
    return ifft2(-k2 * fft2(field))


def product_dealiased(left: np.ndarray, right: np.ndarray, grid: Grid2D) -> np.ndarray:
    mask = grid.dealias_mask()
    left_hat = fft2(left) * mask
    right_hat = fft2(right) * mask
    return ifft2(fft2(ifft2(left_hat) * ifft2(right_hat)) * mask)


def require_viscosity(nu: float) -> float:
    value = float(nu)
    if not math.isfinite(value) or value <= 0.0:
        raise ValueError("nu must be finite and > 0")
    return value


def time_nodes(t_final: float, save_dt: float) -> np.ndarray:
    steps = _multiple(t_final, save_dt, "t_final / save_dt")
    return np.arange(steps + 1, dtype=np.float64) * save_dt


def _multiple(total: float, step: float, label: str) -> int:
    if not math.isfinite(total) or not math.isfinite(step) or total <= 0.0 or step <= 0.0:
        raise ValueError(f"{label} must use finite positive values")
    count = int(round(total / step))
    if not math.isclose(count * step, total, rel_tol=0.0, abs_tol=1e-12):
        raise ValueError(f"{label} must be an integer")
    return count


def integrate_etdrk4(
    field0: np.ndarray,
    linear: np.ndarray,
    nonlinear: Any,
    *,
    dt: float,
    t_final: float,
    save_dt: float,
) -> np.ndarray:
    """Advance a Fourier field with diagonal linear part ``linear``.

    ``field0`` is physical space, shape ``(..., n, n)``. ``linear`` broadcasts
    to the spectrum. ``nonlinear(spectrum)`` returns the Fourier increment
    of the explicit term (already the ``N(U)`` in ``U_t = L U + N(U)``).
    """

    spectrum = fft2(field0)
    coefficients = etdrk4_coefficients(linear, dt)
    step_count = _multiple(t_final, dt, "t_final / dt")
    save_stride = _multiple(save_dt, dt, "save_dt / dt")
    if step_count % save_stride != 0:
        raise ValueError("save_dt must divide t_final in an integer number of steps")
    n_save = step_count // save_stride + 1
    stored = np.empty((n_save, *field0.shape), dtype=np.float64)
    stored[0] = field0
    frame = 1
    for step in range(1, step_count + 1):
        spectrum = _etdrk4_step(spectrum, nonlinear, *coefficients)
        if step % save_stride == 0:
            stored[frame] = ifft2(spectrum)
            if not np.isfinite(stored[frame]).all():
                raise RuntimeError(f"spectral solver produced a non-finite field at step {step}")
            frame += 1
    return stored


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
