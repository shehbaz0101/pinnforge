"""Kappa-only golden-section search on the 64 by 128 solver."""

from __future__ import annotations

from typing import Any

import numpy as np

from pinnforge.v06.masks import ShearMask, read_sensors
from pinnforge.v061.fno_inverse import pack_lattice
from pinnforge.v062.solver import solve_shear


def lift_to_sim(sensors: np.ndarray, nx: int, ny: int) -> np.ndarray:
    """Fourier-pad a lattice ``(..., n_side_x, n_side_y)`` onto ``nx, ny``."""

    values = np.asarray(sensors, dtype=np.float64)
    n_x, n_y = values.shape[-2], values.shape[-1]
    spectrum = np.fft.fft2(values, axes=(-2, -1))
    padded = np.zeros(values.shape[:-2] + (nx, ny), dtype=np.complex128)
    hx, hy = n_x // 2, n_y // 2
    padded[..., :hx, :hy] = spectrum[..., :hx, :hy]
    padded[..., :hx, -hy:] = spectrum[..., :hx, -hy:]
    padded[..., -hx:, :hy] = spectrum[..., -hx:, :hy]
    padded[..., -hx:, -hy:] = spectrum[..., -hx:, -hy:]
    padded *= (nx / n_x) * (ny / n_y)
    return np.fft.ifft2(padded, axes=(-2, -1)).real.astype(np.float64, copy=False)


def search_kappa(
    observed: np.ndarray,
    *,
    mask: ShearMask,
    nu: float,
    lo: float,
    hi: float,
    protocol: dict[str, Any],
) -> float:
    sim_nx = int(protocol["sim_grid"]["nx"])
    sim_ny = int(protocol["sim_grid"]["ny"])
    dt = float(protocol["compute_reduction"]["sitl_dt"])
    window = float(protocol["compute_reduction"]["sitl_window"])
    n_steps = int(protocol["compute_reduction"]["sitl_steps"])
    sensors = read_sensors(observed, mask)
    start = lift_to_sim(sensors[0], sim_nx, sim_ny)
    work = _mask_on(sim_nx, sim_ny, mask)

    def loss(kappa: float) -> float:
        rolled = solve_shear(
            start[0],
            start[1],
            start[3],
            nu=float(nu),
            kappa=float(kappa),
            dt=dt,
            t_final=window,
            save_dt=0.1,
        )
        pred = read_sensors(rolled["fields"], work)
        n = min(pred.shape[0], sensors.shape[0])
        return float(np.mean((pred[:n] - sensors[:n]) ** 2))

    return _golden(loss, float(lo), float(hi), n_steps)


def _mask_on(nx: int, ny: int, mask: ShearMask) -> ShearMask:
    return ShearMask(
        nx=nx,
        ny=ny,
        n_side_x=mask.n_side_x,
        n_side_y=mask.n_side_y,
        burst_starts=mask.burst_starts,
        burst_length=mask.burst_length,
    )


def _golden(fn: Any, lo: float, hi: float, n_steps: int) -> float:
    phi = 0.5 * (3.0 - np.sqrt(5.0))
    a, b = float(lo), float(hi)
    c = b - phi * (b - a)
    d = a + phi * (b - a)
    fc, fd = fn(c), fn(d)
    for _ in range(max(1, int(n_steps))):
        if fc < fd:
            b, d, fd = d, c, fc
            c = b - phi * (b - a)
            fc = fn(c)
        else:
            a, c, fc = c, d, fd
            d = a + phi * (b - a)
            fd = fn(d)
    return float(0.5 * (a + b))


def pack_obs_lattice(field: np.ndarray, mask: ShearMask) -> np.ndarray:
    return pack_lattice(field, mask)
