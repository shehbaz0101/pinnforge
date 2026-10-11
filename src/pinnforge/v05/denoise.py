"""Train and apply the v0.4 FNO2d class as a full-field denoiser."""

from __future__ import annotations

import json
import math
from pathlib import Path
from typing import Any

import numpy as np

from pinnforge.ml_import import require_torch
from pinnforge.v04.fno2d import FNO2d
from pinnforge.v04.sensors import SensorMask, lift_sensors, read_sensors
from pinnforge.v04.train import field_stats, relative_l2

torch = require_torch()


def pack_field(field: np.ndarray) -> Any:
    values = np.asarray(field, dtype=np.float32)
    n_times, n_comp, height, width = values.shape
    return torch.from_numpy(values.reshape(1, n_times * n_comp, height, width))


def unpack_field(tensor: Any, n_times: int, n_comp: int) -> np.ndarray:
    array = tensor.detach().cpu().numpy().astype(np.float64, copy=False)
    _batch, channels, height, width = array.shape
    if channels != n_times * n_comp:
        raise ValueError("denoiser output channels do not match frames * components")
    return array.reshape(n_times, n_comp, height, width)


def normalize(field: np.ndarray, mean: float, std: float) -> np.ndarray:
    return (np.asarray(field, dtype=np.float64) - float(mean)) / float(std)


def denormalize(field: np.ndarray, mean: float, std: float) -> np.ndarray:
    return np.asarray(field, dtype=np.float64) * float(std) + float(mean)


def lift_observation(field: np.ndarray, mask: SensorMask) -> np.ndarray:
    return lift_sensors(read_sensors(field, mask), mask)


def train_denoiser(
    fields: np.ndarray,
    *,
    train_ids: list[int],
    val_ids: list[int],
    protocol: dict[str, Any],
    mask: SensorMask,
    seed: int,
    output: Path,
    noisy_field,
) -> dict[str, Any]:
    spec = protocol["fno_denoiser"]
    torch.manual_seed(int(seed))
    n_times = int(fields.shape[1])
    n_comp = int(fields.shape[2])
    u_mean, u_std = field_stats(fields[train_ids])
    model = FNO2d(
        n_times * n_comp,
        n_times * n_comp,
        width=int(spec["width"]),
        modes=int(spec["modes"]),
        n_layers=int(spec["layers"]),
    )
    optimizer = torch.optim.Adam(model.parameters(), lr=float(spec["lr"]))
    train_pairs = _pairs(fields, train_ids, protocol, mask, seed, noisy_field)
    val_pairs = _pairs(fields, val_ids, protocol, mask, seed, noisy_field)
    best_val = math.inf
    history: list[dict[str, float]] = []
    output.mkdir(parents=True, exist_ok=True)
    checkpoint = output / "checkpoint.pt"
    for epoch in range(1, int(spec["epochs"]) + 1):
        model.train()
        order = np.random.default_rng(int(seed) + epoch).permutation(len(train_pairs))
        losses: list[float] = []
        batch_size = int(spec["batch_size"])
        for begin in range(0, len(order), batch_size):
            batch = [train_pairs[int(item)] for item in order[begin : begin + batch_size]]
            pred, target = _forward_batch(model, batch, u_mean, u_std)
            loss = torch.mean((pred - target) ** 2)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            losses.append(float(loss.detach().cpu()))
        val_rel = _split_rel_l2(model, val_pairs, u_mean, u_std)
        history.append({"epoch": float(epoch), "train_mse": float(np.mean(losses)), "val_rel_l2": val_rel})
        if val_rel < best_val:
            best_val = val_rel
            torch.save(
                {
                    "model": model.state_dict(),
                    "config": model.config_dict(),
                    "u_mean": u_mean,
                    "u_std": u_std,
                    "n_frames": n_times,
                    "n_comp": n_comp,
                    "seed": int(seed),
                    "val_rel_l2": val_rel,
                },
                checkpoint,
            )
    record = {
        "seed": int(seed),
        "u_mean": u_mean,
        "u_std": u_std,
        "best_val_rel_l2": best_val,
        "history": history,
        "parameter_count": model.parameter_count(),
        "checkpoint": str(checkpoint),
    }
    (output / "train.json").write_text(json.dumps(record, indent=2) + "\n", encoding="utf-8")
    return record


def load_denoiser(path: Path) -> tuple[Any, dict[str, Any]]:
    payload = torch.load(path, map_location="cpu", weights_only=False)
    config = payload["config"]
    model = FNO2d(
        int(config["in_channels"]),
        int(config["out_channels"]),
        width=int(config["width"]),
        modes=int(config["modes"]),
        n_layers=int(config["n_layers"]),
    )
    model.load_state_dict(payload["model"])
    model.eval()
    return model, payload


def predict_denoised(
    field: np.ndarray,
    mask: SensorMask,
    *,
    model: Any,
    payload: dict[str, Any],
) -> np.ndarray:
    lifted = lift_observation(field, mask)
    mean = float(payload["u_mean"])
    std = float(payload["u_std"])
    with torch.no_grad():
        pred = model(pack_field(normalize(lifted, mean, std)))
    return denormalize(unpack_field(pred, int(payload["n_frames"]), int(payload["n_comp"])), mean, std)


def _pairs(
    fields: np.ndarray,
    instance_ids: list[int],
    protocol: dict[str, Any],
    mask: SensorMask,
    seed: int,
    noisy_field,
) -> list[tuple[np.ndarray, np.ndarray]]:
    pairs: list[tuple[np.ndarray, np.ndarray]] = []
    for instance_id in instance_ids:
        clean = np.asarray(fields[int(instance_id)], dtype=np.float64)
        for noise in protocol["fno_denoiser"]["train_noise_fractions"]:
            observed = noisy_field(fields, int(instance_id), float(noise), int(seed), protocol)
            pairs.append((lift_observation(observed, mask), clean))
    return pairs


def _forward_batch(model: Any, batch: list[tuple[np.ndarray, np.ndarray]], mean: float, std: float) -> tuple[Any, Any]:
    inputs = torch.cat([pack_field(normalize(item[0], mean, std)) for item in batch], dim=0)
    targets = torch.cat([pack_field(normalize(item[1], mean, std)) for item in batch], dim=0)
    return model(inputs), targets


def _split_rel_l2(model: Any, pairs: list[tuple[np.ndarray, np.ndarray]], mean: float, std: float) -> float:
    model.eval()
    scores: list[float] = []
    with torch.no_grad():
        for lifted, clean in pairs:
            pred = denormalize(
                unpack_field(model(pack_field(normalize(lifted, mean, std))), clean.shape[0], clean.shape[1]),
                mean,
                std,
            )
            scores.append(relative_l2(pred, clean))
    return float(np.mean(scores)) if scores else float("inf")
