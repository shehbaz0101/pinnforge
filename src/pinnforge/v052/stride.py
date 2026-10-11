"""Bump-center stride relative to a monochromatic forcing period."""

from __future__ import annotations


def center_stride(n_grid: int, fraction: float) -> int:
    return max(1, int(round(float(fraction) * int(n_grid))))


def stride_is_incommensurate(n_grid: int, fraction: float, wavenumber: int) -> bool:
    """True when bump centers do not all share one phase of ``cos(k x)``."""

    stride = center_stride(n_grid, fraction)
    cycles = (int(wavenumber) * stride) % int(n_grid)
    return cycles != 0
