"""Read named shear_flow frames and downsample without editing v06."""

from __future__ import annotations

from pathlib import Path
from typing import Any

import numpy as np

from pinnforge.v06.grid import downsample_fourier
from pinnforge.v06.io import _dataset, _split_fields


def parse_reynolds_schmidt(relative: str) -> tuple[float, float]:
    name = Path(relative).name.replace(".hdf5", "")
    parts = name.split("_")
    re_tok = parts[parts.index("Reynolds") + 1]
    sc_tok = parts[parts.index("Schmidt") + 1]
    return float(re_tok), float(sc_tok)


def coefficients(reynolds: float, schmidt: float) -> tuple[float, float]:
    nu = 1.0 / float(reynolds)
    kappa = 1.0 / (float(reynolds) * float(schmidt))
    return nu, kappa


def resolve_raw(relative: str, roots: list[Path]) -> Path:
    for root in roots:
        direct = root / relative
        if direct.is_file():
            return direct
        named = root / Path(relative).name
        if named.is_file():
            return named
        nested = root / "data" / Path(relative).parent.name / Path(relative).name
        if nested.is_file():
            return nested
    raise FileNotFoundError(relative)


def load_frames(
    path: Path,
    *,
    frame_ids: list[int],
    nx: int,
    ny: int,
    trajectory_ids: list[int] | None = None,
) -> tuple[np.ndarray, dict[str, Any]]:
    """Return ``(n_traj, n_frames, 4, nx, ny)``."""

    import h5py

    ids_time = [int(i) for i in frame_ids]
    with h5py.File(path, "r") as handle:
        tracer = _dataset(handle, "t0_fields", ("tracer", "scalar", "s"))
        pressure = _dataset(handle, "t0_fields", ("pressure", "p"))
        velocity = _dataset(handle, "t1_fields", ("velocity", "u"))
        n_traj = int(tracer.shape[0])
        ids = list(range(n_traj)) if trajectory_ids is None else [int(i) for i in trajectory_ids]
        native_x, native_y = int(tracer.shape[2]), int(tracer.shape[3])
        frames = []
        for tid in ids:
            s = np.asarray(tracer[tid, ids_time], dtype=np.float64)
            p = np.asarray(pressure[tid, ids_time], dtype=np.float64)
            vel = np.asarray(velocity[tid, ids_time], dtype=np.float64)
            ux, uy, s, p = _split_fields(s, p, vel)
            stacked = np.stack([ux, uy, p, s], axis=1)
            if nx == native_x and ny == native_y:
                frames.append(stacked)
            else:
                frames.append(downsample_fourier(stacked, nx, ny))
        packed = np.stack(frames, axis=0)
        meta = {
            "path": str(path),
            "trajectory_ids": ids,
            "frame_ids": ids_time,
            "n_native": [n_traj, int(tracer.shape[1]), native_x, native_y],
            "packed_shape": list(packed.shape),
        }
    return packed, meta
