"""Generate the v0.4 2D pilots and record a committed manifest."""

from __future__ import annotations

import hashlib
import json
import math
from pathlib import Path
from typing import Any

import numpy as np

from pinnforge.reference.spectral2d.burgers import solve_burgers2d
from pinnforge.reference.spectral2d.common import Grid2D, time_nodes
from pinnforge.reference.spectral2d.initial import bandlimited_velocity, bandlimited_vorticity
from pinnforge.reference.spectral2d.navier_stokes import solve_ns2d

PROTOCOL_PATH = Path("docs/v04/protocol.json")


def load_protocol(path: Path | None = None) -> dict[str, Any]:
    target = PROTOCOL_PATH if path is None else path
    document = json.loads(target.read_text(encoding="utf-8"))
    fmt = document.get("format")
    if fmt not in {
        "pinnforge.v04_2d_inverse_protocol.v1",
        "pinnforge.v041_2d_inverse_protocol.v1",
    }:
        raise ValueError("2D inverse protocol format is not recognized")
    return document


def protocol_sha256(path: Path | None = None) -> str:
    target = PROTOCOL_PATH if path is None else path
    return hashlib.sha256(target.read_bytes()).hexdigest()


def assign_splits(n_train: int, n_val: int, n_test: int) -> dict[str, list[int]]:
    total = n_train + n_val + n_test
    return {
        "train": list(range(0, n_train)),
        "val": list(range(n_train, n_train + n_val)),
        "test": list(range(n_train + n_val, total)),
    }


def draw_viscosities(protocol: dict[str, Any], n_instances: int, *, seed: int) -> np.ndarray:
    block = protocol["viscosity"]
    rng = np.random.default_rng(seed)
    return rng.uniform(block["nu_min"], block["nu_max"], size=n_instances).astype(np.float64)


def hard_ood_threshold(train_nu: np.ndarray) -> float:
    return float(np.quantile(np.asarray(train_nu, dtype=np.float64), 0.25, method="linear"))


def generate_problem(
    problem: str,
    protocol: dict[str, Any],
    output: Path,
    *,
    n: int | None = None,
    n_train: int | None = None,
    n_val: int | None = None,
    n_test: int | None = None,
    t_final: float | None = None,
    dt: float | None = None,
    save_dt: float | None = None,
    protocol_path: Path | None = None,
) -> dict[str, Any]:
    if problem not in {"burgers2d", "ns2d"}:
        raise ValueError("problem must be burgers2d or ns2d")
    splits_spec = protocol["splits"]
    time_spec = protocol["time"]
    domain = protocol["domain"]
    n_grid = int(domain["n"] if n is None else n)
    train_n = int(splits_spec["n_train"] if n_train is None else n_train)
    val_n = int(splits_spec["n_val"] if n_val is None else n_val)
    test_n = int(splits_spec["n_test"] if n_test is None else n_test)
    t_end = float(time_spec["t_final"] if t_final is None else t_final)
    step = float(time_spec["dt"] if dt is None else dt)
    store = float(time_spec["save_dt"] if save_dt is None else save_dt)
    splits = assign_splits(train_n, val_n, test_n)
    n_instances = train_n + val_n + test_n
    viscosities = draw_viscosities(protocol, n_instances, seed=int(protocol["viscosity"]["draw_seed_offset"]))
    grid = Grid2D(n_grid, float(domain["length"]))
    times = time_nodes(t_end, store)
    n_comp = 2 if problem == "burgers2d" else 1
    fields = np.empty((n_instances, times.size, n_comp, n_grid, n_grid), dtype=np.float64)
    forcings: list[str] = []
    for instance_id in range(n_instances):
        rng = np.random.default_rng(10_000 * (0 if problem == "burgers2d" else 1) + instance_id)
        nu = float(viscosities[instance_id])
        if problem == "burgers2d":
            u0 = bandlimited_velocity(grid, rng, max_mode=int(protocol["problems"]["burgers2d"]["initial"]["max_mode"]))
            traj = solve_burgers2d(u0, nu, dt=step, t_final=t_end, save_dt=store, length=grid.length)
            fields[instance_id] = traj.u
            forcings.append("none")
        else:
            omega0 = bandlimited_vorticity(grid, rng, max_mode=int(protocol["problems"]["ns2d"]["initial"]["max_mode"]))
            forcing = "kolmogorov" if instance_id % 2 == 0 else "decaying"
            traj = solve_ns2d(
                omega0,
                nu,
                dt=step,
                t_final=t_end,
                save_dt=store,
                forcing=forcing,
                length=grid.length,
            )
            fields[instance_id, :, 0] = traj.omega
            forcings.append(forcing)
    output.mkdir(parents=True, exist_ok=True)
    field_path = output / f"{problem}_fields.npy"
    nu_path = output / f"{problem}_nu.npy"
    np.save(field_path, fields)
    np.save(nu_path, viscosities)
    train_ids = splits["train"]
    threshold = hard_ood_threshold(viscosities[train_ids])
    manifest = {
        "format": "pinnforge.v04_pilot.v1",
        "problem": problem,
        "protocol": str(protocol_path or PROTOCOL_PATH),
        "protocol_sha256": protocol_sha256(protocol_path),
        "n": n_grid,
        "length": grid.length,
        "t_final": t_end,
        "dt": step,
        "save_dt": store,
        "n_frames": int(times.size),
        "n_instances": n_instances,
        "splits": splits,
        "nu": viscosities.tolist(),
        "forcing": forcings,
        "hard_ood": {
            "threshold_nu": threshold,
            "comparison": "nu <= threshold_nu",
            "source": "training-split lower quartile",
            "refit": False,
            "n_train_hard_ood": int(np.sum(viscosities[train_ids] <= threshold)),
        },
        "field_sha256": _file_sha256(field_path),
        "nu_sha256": _file_sha256(nu_path),
        "files": {
            "fields": field_path.name,
            "nu": nu_path.name,
        },
    }
    (output / f"{problem}_manifest.json").write_text(json.dumps(manifest, indent=2) + "\n", encoding="utf-8")
    return manifest


def _file_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1 << 20), b""):
            digest.update(chunk)
    return digest.hexdigest()


def relative_l2(left: np.ndarray, right: np.ndarray) -> float:
    num = float(np.linalg.norm(np.asarray(left) - np.asarray(right)))
    den = float(np.linalg.norm(right))
    if den == 0.0:
        raise ValueError("relative L2 denominator is zero")
    return num / den


def restrict_field(field: np.ndarray, n_coarse: int) -> np.ndarray:
    """Keep the lowest Fourier modes of a periodic 2D field."""

    values = np.asarray(field, dtype=np.float64)
    if values.shape[-1] != values.shape[-2]:
        raise ValueError("field must be square on the last two axes")
    n_fine = values.shape[-1]
    if n_fine % n_coarse:
        raise ValueError("fine grid must be a multiple of the coarse grid")
    spectrum = np.fft.fft2(values, axes=(-2, -1))
    half = n_coarse // 2
    kept = np.zeros(values.shape[:-2] + (n_coarse, n_coarse), dtype=np.complex128)
    for block in (slice(0, half), slice(n_fine - half, n_fine)):
        dest = slice(0, half) if block.start == 0 else slice(n_coarse - half, n_coarse)
        kept[..., dest, :half] = spectrum[..., block, :half]
        kept[..., dest, n_coarse - half :] = spectrum[..., block, n_fine - half :]
    kept *= (n_coarse / n_fine) ** 2
    return np.fft.ifft2(kept, axes=(-2, -1)).real.astype(np.float64, copy=False)


def convergence_record(
    problem: str,
    *,
    nu: float,
    seed: int,
    grids: tuple[int, ...] = (32, 64, 128),
    t_final: float = 0.1,
    dt_at_64: float = 0.0025,
) -> dict[str, Any]:
    """Compare N=64 against a restriction of N=128 on a short trajectory."""

    if 64 not in grids or 128 not in grids:
        raise ValueError("convergence grids must include 64 and 128")
    stored: dict[int, np.ndarray] = {}
    for n in grids:
        dt = dt_at_64 * (64 / n)
        save_dt = t_final
        grid = Grid2D(n)
        rng = np.random.default_rng(seed)
        if problem == "burgers2d":
            u0 = bandlimited_velocity(grid, rng, max_mode=2)
            traj = solve_burgers2d(u0, nu, dt=dt, t_final=t_final, save_dt=save_dt)
            stored[n] = traj.u[-1]
        else:
            omega0 = bandlimited_vorticity(grid, rng, max_mode=3)
            traj = solve_ns2d(omega0, nu, dt=dt, t_final=t_final, save_dt=save_dt, forcing="decaying")
            stored[n] = traj.omega[-1]
    reference = stored[128]
    coarse = stored[64]
    rel = relative_l2(coarse, restrict_field(reference, 64))
    return {
        "problem": problem,
        "nu": nu,
        "seed": seed,
        "grids": list(grids),
        "t_final": t_final,
        "relative_l2_64_vs_128": rel,
        "finite": bool(math.isfinite(rel)),
    }
