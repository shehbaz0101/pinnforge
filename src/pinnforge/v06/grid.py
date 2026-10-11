"""Rectangular periodic grid for the downsampled shear-flow fields."""

from __future__ import annotations

import math
from dataclasses import dataclass

import numpy as np

TWO_PI = 2.0 * math.pi


@dataclass(frozen=True)
class GridRect:
    nx: int
    ny: int
    length_x: float
    length_y: float

    def __post_init__(self) -> None:
        for name, value in (("nx", self.nx), ("ny", self.ny)):
            if isinstance(value, bool) or not isinstance(value, (int, np.integer)) or int(value) < 4 or int(value) % 2:
                raise ValueError(f"{name} must be an even integer >= 4")
            object.__setattr__(self, name, int(value))
        for name in ("length_x", "length_y"):
            length = float(getattr(self, name))
            if not math.isfinite(length) or length <= 0.0:
                raise ValueError(f"{name} must be finite and > 0")
            object.__setattr__(self, name, length)

    @property
    def dx(self) -> float:
        return self.length_x / self.nx

    @property
    def dy(self) -> float:
        return self.length_y / self.ny

    def wavenumbers(self) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        kx1 = TWO_PI * np.fft.fftfreq(self.nx, d=self.dx).astype(np.float64)
        ky1 = TWO_PI * np.fft.fftfreq(self.ny, d=self.dy).astype(np.float64)
        kx1[self.nx // 2] = 0.0
        ky1[self.ny // 2] = 0.0
        kx, ky = np.meshgrid(kx1, ky1, indexing="ij")
        return kx, ky, kx * kx + ky * ky


def fft2(field: np.ndarray) -> np.ndarray:
    return np.fft.fft2(np.asarray(field, dtype=np.float64), axes=(-2, -1))


def ifft2(spectrum: np.ndarray) -> np.ndarray:
    return np.fft.ifft2(spectrum, axes=(-2, -1)).real.astype(np.float64, copy=False)


def spectral_grad(field: np.ndarray, grid: GridRect) -> tuple[np.ndarray, np.ndarray]:
    kx, ky, _k2 = grid.wavenumbers()
    spectrum = fft2(field)
    return ifft2(1j * kx * spectrum), ifft2(1j * ky * spectrum)


def spectral_lap(field: np.ndarray, grid: GridRect) -> np.ndarray:
    _kx, _ky, k2 = grid.wavenumbers()
    return ifft2(-k2 * fft2(field))


def downsample_fourier(field: np.ndarray, nx: int, ny: int) -> np.ndarray:
    """Keep the lowest Fourier modes and scale. ``field`` is ``(..., n_x, n_y)``."""

    values = np.asarray(field, dtype=np.float64)
    n_x, n_y = values.shape[-2], values.shape[-1]
    if n_x % nx or n_y % ny:
        raise ValueError("fine grid must be a multiple of the coarse grid")
    spectrum = np.fft.fft2(values, axes=(-2, -1))
    kept = np.zeros(values.shape[:-2] + (nx, ny), dtype=np.complex128)
    hx, hy = nx // 2, ny // 2
    kept[..., :hx, :hy] = spectrum[..., :hx, :hy]
    kept[..., :hx, -hy:] = spectrum[..., :hx, n_y - hy :]
    kept[..., -hx:, :hy] = spectrum[..., n_x - hx :, :hy]
    kept[..., -hx:, -hy:] = spectrum[..., n_x - hx :, n_y - hy :]
    kept *= (nx / n_x) * (ny / n_y)
    return np.fft.ifft2(kept, axes=(-2, -1)).real.astype(np.float64, copy=False)
