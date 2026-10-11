"""Data-only FNO on the probe lattice (not a lift onto the native grid)."""

from __future__ import annotations

from pathlib import Path
from typing import Any

import numpy as np

from pinnforge.v04.fno2d import FNO2d
from pinnforge.v06.fno_inverse import _forward, _torch
from pinnforge.v06.masks import ShearMask, read_sensors


def pack_lattice(field: np.ndarray, mask: ShearMask) -> np.ndarray:
    """``field`` is ``(T, 4, nx, ny)``. Returns ``(T*4, n_side_x, n_side_y)``."""

    sensors = read_sensors(field, mask)
    frames, n_comp, n_x, n_y = sensors.shape
    return sensors.reshape(frames * n_comp, n_x, n_y)


def train_fno(
    fields: np.ndarray,
    nu: np.ndarray,
    kappa: np.ndarray,
    train_ids: list[int],
    val_ids: list[int],
    *,
    protocol: dict[str, Any],
    mask: ShearMask,
    seed: int,
    checkpoint: Path,
) -> dict[str, Any]:
    torch = _torch()
    spec = protocol["methods"]["fno_data"]
    rng = np.random.default_rng(int(seed))
    torch.manual_seed(int(seed))
    sample = pack_lattice(fields[train_ids[0]], mask)
    model = FNO2d(
        int(sample.shape[0]),
        2,
        width=int(spec["width"]),
        modes=int(spec["modes"]),
        n_layers=int(spec["layers"]),
    )
    optimizer = torch.optim.Adam(model.parameters(), lr=float(spec["lr"]))
    batch = int(spec["batch_size"])
    best = float("inf")
    history: list[float] = []
    train_x, train_y = _stack(fields, nu, kappa, train_ids, mask)
    val_x, val_y = _stack(fields, nu, kappa, val_ids, mask)
    for _epoch in range(int(spec["epochs"])):
        order = rng.permutation(train_x.shape[0])
        model.train()
        for start in range(0, order.size, batch):
            sl = order[start : start + batch]
            pred = _forward(model, torch.tensor(train_x[sl], dtype=torch.float32))
            target = torch.tensor(train_y[sl], dtype=torch.float32)
            loss = torch.mean((pred - target) ** 2)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
        model.eval()
        with torch.no_grad():
            val_pred = _forward(model, torch.tensor(val_x, dtype=torch.float32))
            val_loss = float(torch.mean((val_pred - torch.tensor(val_y, dtype=torch.float32)) ** 2).item())
        history.append(val_loss)
        if val_loss < best:
            best = val_loss
            checkpoint.parent.mkdir(parents=True, exist_ok=True)
            torch.save({"state": model.state_dict(), "in_channels": int(sample.shape[0]), "config": spec}, checkpoint)
    return {"best_val_mse": best, "history": history, "checkpoint": str(checkpoint)}


_MODEL_CACHE: dict[str, Any] = {}


def predict_logs(field: np.ndarray, *, mask: ShearMask, checkpoint: Path) -> tuple[float, float]:
    torch = _torch()
    key = str(checkpoint.resolve())
    model = _MODEL_CACHE.get(key)
    if model is None:
        try:
            payload = torch.load(checkpoint, map_location="cpu", weights_only=False)
        except TypeError:
            payload = torch.load(checkpoint, map_location="cpu")
        model = FNO2d(
            int(payload["in_channels"]),
            2,
            width=int(payload["config"]["width"]),
            modes=int(payload["config"]["modes"]),
            n_layers=int(payload["config"]["layers"]),
        )
        model.load_state_dict(payload["state"])
        model.eval()
        _MODEL_CACHE[key] = model
    packed = pack_lattice(field, mask)[None]
    with torch.no_grad():
        logs = _forward(model, torch.tensor(packed, dtype=torch.float32)).numpy()[0]
    return float(logs[0]), float(logs[1])


def _stack(
    fields: np.ndarray,
    nu: np.ndarray,
    kappa: np.ndarray,
    ids: list[int],
    mask: ShearMask,
) -> tuple[np.ndarray, np.ndarray]:
    xs = [pack_lattice(fields[i], mask) for i in ids]
    ys = np.stack([np.log([float(nu[i]), float(kappa[i])]) for i in ids], axis=0)
    return np.stack(xs, axis=0), ys
