"""Read The Well shear_flow HDF5 and pack a downsampled window."""

from __future__ import annotations

from pathlib import Path
from typing import Any

import numpy as np

from pinnforge.v06.grid import downsample_fourier

UX, UY, P, S = 0, 1, 2, 3


def inspect_hdf5(path: Path) -> dict[str, Any]:
    import h5py

    with h5py.File(path, "r") as handle:
        t0 = list(handle["t0_fields"].keys()) if "t0_fields" in handle else []
        t1 = list(handle["t1_fields"].keys()) if "t1_fields" in handle else []
        dims = {}
        if "dimensions" in handle:
            for name in handle["dimensions"]:
                dims[name] = [int(n) for n in handle["dimensions"][name].shape]
        n_traj = int(handle.attrs.get("n_trajectories", -1))
        return {
            "t0_fields": t0,
            "t1_fields": t1,
            "dimensions": dims,
            "n_trajectories": n_traj,
            "attr_keys": [str(key) for key in handle.attrs],
        }


def load_window(
    path: Path,
    *,
    protocol: dict[str, Any],
    trajectory_ids: list[int] | None = None,
) -> tuple[np.ndarray, dict[str, Any]]:
    """Return ``(n_traj, n_frames, 4, nx, ny)`` on the coarse grid."""

    import h5py

    window = protocol["time_window"]
    start = int(window["frame_start"])
    n_frames = int(window["n_frames"])
    stop = start + n_frames
    nx = int(protocol["downsample"]["nx"])
    ny = int(protocol["downsample"]["ny"])
    with h5py.File(path, "r") as handle:
        tracer = _dataset(handle, "t0_fields", ("tracer", "scalar", "s"))
        pressure = _dataset(handle, "t0_fields", ("pressure", "p"))
        velocity = _dataset(handle, "t1_fields", ("velocity", "u"))
        n_traj = int(tracer.shape[0])
        ids = list(range(n_traj)) if trajectory_ids is None else [int(i) for i in trajectory_ids]
        # Expected (B, T, X, Y) or velocity (B, T, X, Y, 2)
        frames = []
        for tid in ids:
            s = np.asarray(tracer[tid, start:stop], dtype=np.float64)
            p = np.asarray(pressure[tid, start:stop], dtype=np.float64)
            vel = np.asarray(velocity[tid, start:stop], dtype=np.float64)
            ux, uy, s, p = _split_fields(s, p, vel)
            stacked = np.stack([ux, uy, p, s], axis=1)
            frames.append(downsample_fourier(stacked, nx, ny))
        packed = np.stack(frames, axis=0)
        meta = {
            "path": str(path),
            "trajectory_ids": ids,
            "n_native": [int(tracer.shape[0]), int(tracer.shape[1]), int(tracer.shape[2]), int(tracer.shape[3])],
            "packed_shape": list(packed.shape),
        }
    return packed, meta


def _dataset(handle: Any, group: str, names: tuple[str, ...]) -> Any:
    block = handle[group]
    for name in names:
        if name in block:
            return block[name]
    raise KeyError(f"none of {names} in {group}: {list(block.keys())}")


def _split_fields(
    tracer: np.ndarray,
    pressure: np.ndarray,
    velocity: np.ndarray,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """Return ux, uy, tracer, pressure as ``(T, X, Y)``."""

    if velocity.ndim == 4:
        # (T, X, Y, 2)
        ux, uy = velocity[..., 0], velocity[..., 1]
    elif velocity.ndim == 5:
        # (T, 2, X, Y) or (T, X, Y, 2) already handled
        if velocity.shape[1] == 2:
            ux, uy = velocity[:, 0], velocity[:, 1]
        else:
            ux, uy = velocity[..., 0], velocity[..., 1]
    else:
        raise ValueError(f"velocity shape {velocity.shape} is not supported")
    if tracer.ndim != 3 or pressure.ndim != 3:
        raise ValueError("tracer and pressure must be (T, X, Y)")
    return ux, uy, tracer, pressure
