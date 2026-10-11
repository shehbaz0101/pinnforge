"""Equispaced rectangular probe lattice and bursts."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import numpy as np


@dataclass(frozen=True)
class ShearMask:
    nx: int
    ny: int
    n_side_x: int
    n_side_y: int
    burst_starts: tuple[int, ...]
    burst_length: int

    @property
    def stride_x(self) -> int:
        return self.nx // self.n_side_x

    @property
    def stride_y(self) -> int:
        return self.ny // self.n_side_y

    @property
    def index(self) -> tuple[np.ndarray, np.ndarray]:
        xs = np.arange(0, self.nx, self.stride_x)
        ys = np.arange(0, self.ny, self.stride_y)
        return np.meshgrid(xs, ys, indexing="ij")


def mask_from_protocol(protocol: dict[str, Any]) -> ShearMask:
    down = protocol["downsample"]
    sensors = protocol["sensors"]
    window = protocol["time_window"]
    return ShearMask(
        nx=int(down["nx"]),
        ny=int(down["ny"]),
        n_side_x=int(sensors["n_side_x"]),
        n_side_y=int(sensors["n_side_y"]),
        burst_starts=tuple(int(item) for item in window["burst_starts"]),
        burst_length=int(window["burst_length"]),
    )


def read_sensors(field: np.ndarray, mask: ShearMask) -> np.ndarray:
    ii, jj = mask.index
    return np.asarray(field, dtype=np.float64)[..., ii, jj]


def lift_sensors(coarse: np.ndarray, mask: ShearMask) -> np.ndarray:
    values = np.asarray(coarse, dtype=np.float64)
    if values.shape[-2:] != (mask.n_side_x, mask.n_side_y):
        raise ValueError("coarse field must match the sensor lattice")
    # Inverse of a stride subsample is a Fourier pad from the lattice onto the coarse grid.
    n_x, n_y = mask.n_side_x, mask.n_side_y
    spectrum = np.fft.fft2(values, axes=(-2, -1))
    padded = np.zeros(values.shape[:-2] + (mask.nx, mask.ny), dtype=np.complex128)
    hx, hy = n_x // 2, n_y // 2
    padded[..., :hx, :hy] = spectrum[..., :hx, :hy]
    padded[..., :hx, -hy:] = spectrum[..., :hx, -hy:]
    padded[..., -hx:, :hy] = spectrum[..., -hx:, :hy]
    padded[..., -hx:, -hy:] = spectrum[..., -hx:, -hy:]
    padded *= (mask.nx / n_x) * (mask.ny / n_y)
    return np.fft.ifft2(padded, axes=(-2, -1)).real.astype(np.float64, copy=False)


def add_noise(field: np.ndarray, fraction: float, rng: np.random.Generator) -> np.ndarray:
    values = np.asarray(field, dtype=np.float64)
    if fraction < 0.0 or not np.isfinite(fraction):
        raise ValueError("noise fraction must be finite and >= 0")
    if fraction == 0.0:
        return values.copy()
    scale = float(values.std()) * float(fraction)
    return values + rng.normal(0.0, scale, size=values.shape)


def burst_slices(mask: ShearMask) -> list[slice]:
    return [slice(start, start + mask.burst_length) for start in mask.burst_starts]


def replay_noise(fields: np.ndarray, instance_id: int, noise: float, seed: int, protocol: dict[str, Any]) -> np.ndarray:
    if float(noise) == 0.0:
        return np.asarray(fields[instance_id], dtype=np.float64)
    noise_seed = int(protocol["noise"]["noise_seed"]) + 100 * int(seed)
    rng = np.random.default_rng(noise_seed + int(instance_id))
    return add_noise(fields[instance_id], float(noise), rng)
