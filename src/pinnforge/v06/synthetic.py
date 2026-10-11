"""Tiny manufactured shear-flow-like fields for CI smoke."""

from __future__ import annotations

from typing import Any

import numpy as np

from pinnforge.v06.grid import GridRect


def manufactured_pack(protocol: dict[str, Any], *, n_instances: int = 4) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Periodic viscous mode plus a decaying tracer. ``(N, T, 4, nx, ny)``."""

    nx = int(protocol["downsample"]["nx"])
    ny = int(protocol["downsample"]["ny"])
    # smoke overrides to 8 by 16
    n_frames = int(protocol["time_window"]["n_frames"])
    dt = float(protocol["time_window"]["dt"])
    grid = GridRect(nx, ny, float(protocol["downsample"]["length_x"]), float(protocol["downsample"]["length_y"]))
    kx, ky, k2 = grid.wavenumbers()
    xs = (grid.length_x / nx) * np.arange(nx)[:, None]
    ys = (grid.length_y / ny) * np.arange(ny)[None, :]
    fields = np.zeros((n_instances, n_frames, 4, nx, ny), dtype=np.float64)
    nu = np.array([0.0001, 0.0001, 0.00002, 0.00001][:n_instances], dtype=np.float64)
    kappa = np.array([0.0001, 0.0002, 0.00002, 0.00001][:n_instances], dtype=np.float64)
    for i in range(n_instances):
        for t in range(n_frames):
            time = t * dt
            decay_u = np.exp(-float(nu[i]) * ((2.0 * np.pi / grid.length_x) ** 2) * time)
            decay_s = np.exp(-float(kappa[i]) * ((2.0 * np.pi / grid.length_x) ** 2 + (2.0 * np.pi / grid.length_y) ** 2) * time)
            fields[i, t, 0] = decay_u * np.sin(2.0 * np.pi * xs / grid.length_x)
            fields[i, t, 1] = 0.0
            fields[i, t, 2] = 0.0
            fields[i, t, 3] = decay_s * np.sin(2.0 * np.pi * xs / grid.length_x) * np.sin(2.0 * np.pi * ys / grid.length_y)
    _ = (kx, ky, k2)
    return fields, nu, kappa
