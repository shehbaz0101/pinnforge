"""Bandlimited initial data for the v0.4 2D pilots."""

from __future__ import annotations

import math

import numpy as np

from pinnforge.reference.spectral2d.common import Grid2D, fft2, ifft2


def bandlimited_velocity(grid: Grid2D, rng: np.random.Generator, *, max_mode: int = 3) -> np.ndarray:
    """Two-component field with modes through ``max_mode``, max-abs 1."""

    field = np.stack(
        [_random_modes(grid, rng, max_mode, spectrum="inv_sqrt") for _ in range(2)],
        axis=0,
    )
    peak = float(np.max(np.abs(field)))
    if peak <= 0.0:
        raise RuntimeError("bandlimited velocity was identically zero")
    return field / peak


def bandlimited_vorticity(grid: Grid2D, rng: np.random.Generator, *, max_mode: int = 4) -> np.ndarray:
    """Zero-mean vorticity with a decaying spectrum, max-abs 1."""

    field = _random_modes(grid, rng, max_mode, spectrum="inv_k2")
    peak = float(np.max(np.abs(field)))
    if peak <= 0.0:
        raise RuntimeError("bandlimited vorticity was identically zero")
    return field / peak


def _random_modes(
    grid: Grid2D,
    rng: np.random.Generator,
    max_mode: int,
    *,
    spectrum: str,
) -> np.ndarray:
    if isinstance(max_mode, bool) or not isinstance(max_mode, (int, np.integer)) or int(max_mode) < 1:
        raise ValueError("max_mode must be an integer >= 1")
    limit = int(max_mode)
    hat = np.zeros((grid.n, grid.n), dtype=np.complex128)
    for mx in range(-limit, limit + 1):
        for my in range(-limit, limit + 1):
            if mx == 0 and my == 0:
                continue
            radius = math.sqrt(mx * mx + my * my)
            if spectrum == "inv_sqrt":
                scale = 1.0 / radius
            elif spectrum == "inv_k2":
                scale = 1.0 / (1.0 + mx * mx + my * my)
            else:
                raise ValueError(f"unknown spectrum {spectrum}")
            amp = rng.uniform(-1.0, 1.0) * scale
            phase = rng.uniform(0.0, 2.0 * math.pi)
            coeff = amp * np.exp(1j * phase)
            hat[mx % grid.n, my % grid.n] += coeff
    field = ifft2(hat)
    field = field - float(field.mean())
    # Hermitian cleanup: take the real inverse already used.
    _ = fft2(field)
    return field
