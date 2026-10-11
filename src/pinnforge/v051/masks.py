"""Lattice masks for the v0.5.1 density sweep."""

from __future__ import annotations

from typing import Any

from pinnforge.v04.sensors import SensorMask


def lattices(protocol: dict[str, Any]) -> list[dict[str, Any]]:
    return list(protocol["lattices"])


def mask_for_lattice(protocol: dict[str, Any], n_side: int, n: int | None = None) -> SensorMask:
    grid_n = int(protocol["domain"]["n"] if n is None else n)
    side = int(n_side)
    if side < 1 or grid_n % side:
        raise ValueError("lattice n_side must be a positive divisor of the grid")
    block = protocol["sensors"]
    return SensorMask(
        n=grid_n,
        n_side=side,
        burst_starts=tuple(int(item) for item in block["burst_starts"]),
        burst_length=int(block["burst_length"]),
        frame_dt=float(block["frame_dt"]),
    )
