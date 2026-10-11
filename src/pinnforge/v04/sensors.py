"""Frozen 8x8 lattice sensors, bursts, noise, and a trigonometric lift."""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Any

import numpy as np

from pinnforge.reference.spectral2d.common import Grid2D


@dataclass(frozen=True)
class SensorMask:
    """Equispaced lattice plus burst windows."""

    n: int
    n_side: int
    burst_starts: tuple[int, ...]
    burst_length: int
    frame_dt: float

    @property
    def stride(self) -> int:
        return self.n // self.n_side

    @property
    def index(self) -> tuple[np.ndarray, np.ndarray]:
        nodes = np.arange(0, self.n, self.stride)
        return np.meshgrid(nodes, nodes, indexing="ij")

    def to_dict(self) -> dict[str, Any]:
        return {
            "n": self.n,
            "n_side": self.n_side,
            "n_sensors": self.n_side * self.n_side,
            "burst_starts": list(self.burst_starts),
            "burst_length": self.burst_length,
            "frame_dt": self.frame_dt,
        }


def mask_from_protocol(protocol: dict[str, Any], n: int | None = None) -> SensorMask:
    block = protocol["sensors"]
    grid_n = int(protocol["domain"]["n"] if n is None else n)
    side = int(round(math.sqrt(int(block["n_sensors"]))))
    if side * side != int(block["n_sensors"]):
        raise ValueError("n_sensors must be a square lattice")
    if grid_n % side:
        raise ValueError("grid n must be divisible by the sensor lattice side")
    return SensorMask(
        n=grid_n,
        n_side=side,
        burst_starts=tuple(int(item) for item in block["burst_starts"]),
        burst_length=int(block["burst_length"]),
        frame_dt=float(block["frame_dt"]),
    )


def read_sensors(field: np.ndarray, mask: SensorMask) -> np.ndarray:
    """``field`` is ``(..., n, n)``. Returns ``(..., n_side, n_side)``."""

    ii, jj = mask.index
    return np.asarray(field, dtype=np.float64)[..., ii, jj]


def lift_sensors(coarse: np.ndarray, mask: SensorMask) -> np.ndarray:
    """Trigonometric interpolant from the lattice onto the solver grid."""

    values = np.asarray(coarse, dtype=np.float64)
    if values.shape[-2:] != (mask.n_side, mask.n_side):
        raise ValueError("coarse field must match the sensor lattice")
    n_coarse = mask.n_side
    n_fine = mask.n
    spectrum = np.fft.fft2(values, axes=(-2, -1))
    padded = np.zeros(values.shape[:-2] + (n_fine, n_fine), dtype=np.complex128)
    half = n_coarse // 2
    padded[..., :half, :half] = spectrum[..., :half, :half]
    padded[..., :half, -half:] = spectrum[..., :half, -half:]
    padded[..., -half:, :half] = spectrum[..., -half:, :half]
    padded[..., -half:, -half:] = spectrum[..., -half:, -half:]
    padded *= (n_fine / n_coarse) ** 2
    return np.fft.ifft2(padded, axes=(-2, -1)).real.astype(np.float64, copy=False)


def add_noise(field: np.ndarray, fraction: float, rng: np.random.Generator) -> np.ndarray:
    values = np.asarray(field, dtype=np.float64)
    if fraction < 0.0 or not np.isfinite(fraction):
        raise ValueError("noise fraction must be finite and >= 0")
    if fraction == 0.0:
        return values.copy()
    scale = float(values.std()) * float(fraction)
    return values + rng.normal(0.0, scale, size=values.shape)


def burst_slices(mask: SensorMask) -> list[slice]:
    return [slice(start, start + mask.burst_length) for start in mask.burst_starts]


def target_frames(mask: SensorMask) -> list[int]:
    frames: list[int] = []
    for start in mask.burst_starts[1:]:
        frames.extend(range(start, start + mask.burst_length))
    return frames


def grid_from_mask(mask: SensorMask, length: float) -> Grid2D:
    return Grid2D(mask.n, length)
