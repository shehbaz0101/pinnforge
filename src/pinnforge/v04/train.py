"""Train a 2D FNO on v0.4 windows. Torch is imported here."""

from __future__ import annotations

import json
import math
from pathlib import Path
from typing import Any

import numpy as np

from pinnforge.ml_import import require_torch
from pinnforge.reference.spectral2d.common import Grid2D
from pinnforge.v04.fno2d import FNO2d, pack_inputs, unpack_outputs

torch = require_torch()


def field_stats(fields: np.ndarray) -> tuple[float, float]:
    return float(fields.mean()), float(fields.std() + 1e-12)


def viscosity_stats(nu: np.ndarray) -> tuple[float, float]:
    return float(nu.mean()), float(nu.std() + 1e-12)


def cut_windows(
    fields: np.ndarray,
    viscosities: np.ndarray,
    *,
    input_frames: int,
    output_frames: int,
    stride: int,
    instance_ids: list[int],
    forcings: list[str] | None = None,
) -> list[dict[str, Any]]:
    windows: list[dict[str, Any]] = []
    n_times = fields.shape[1]
    last = n_times - input_frames - output_frames
    for instance_id in instance_ids:
        for start in range(0, last + 1, stride):
            windows.append(
                {
                    "instance_id": instance_id,
                    "start": start,
                    "inputs": fields[instance_id, start : start + input_frames],
                    "targets": fields[instance_id, start + input_frames : start + input_frames + output_frames],
                    "nu": float(viscosities[instance_id]),
                    "forcing": None if forcings is None else forcings[instance_id],
                }
            )
    return windows


def relative_l2(pred: np.ndarray, target: np.ndarray) -> float:
    num = float(np.linalg.norm(pred - target))
    den = float(np.linalg.norm(target))
    return num / den if den > 0 else float("inf")


def train_fno(
    windows_train: list[dict[str, Any]],
    windows_val: list[dict[str, Any]],
    *,
    problem: str,
    protocol: dict[str, Any],
    seed: int,
    loss: str,
    output: Path,
    length: float,
    dt: float,
    u_mean: float,
    u_std: float,
    nu_mean: float,
    nu_std: float,
) -> dict[str, Any]:
    if loss not in {"data", "hybrid"}:
        raise ValueError("loss must be data or hybrid")
    spec = protocol["fno"]
    rng_seed = int(seed)
    torch.manual_seed(rng_seed)
    np.random.default_rng(rng_seed)
    n_comp = 2 if problem == "burgers2d" else 1
    in_channels = spec["input_frames"] * n_comp + 1
    out_channels = spec["output_frames"] * n_comp
    model = FNO2d(
        in_channels,
        out_channels,
        width=int(spec["width"]),
        modes=int(spec["modes"]),
        n_layers=int(spec["layers"]),
    )
    optimizer = torch.optim.Adam(model.parameters(), lr=float(spec["lr"]))
    residual_weight = float(protocol["methods"]["fno_forward_hybrid"]["residual_weight"]) if loss == "hybrid" else 0.0
    grid = Grid2D(int(windows_train[0]["inputs"].shape[-1]), length)
    best_val = math.inf
    history: list[dict[str, float]] = []
    output.mkdir(parents=True, exist_ok=True)
    for epoch in range(1, int(spec["epochs"]) + 1):
        model.train()
        order = np.random.default_rng(rng_seed + epoch).permutation(len(windows_train))
        losses: list[float] = []
        for begin in range(0, len(order), int(spec["batch_size"])):
            batch_ids = order[begin : begin + int(spec["batch_size"])]
            batch = [windows_train[int(i)] for i in batch_ids]
            data_loss, resid_loss = _batch_losses(
                model,
                batch,
                u_mean=u_mean,
                u_std=u_std,
                nu_mean=nu_mean,
                nu_std=nu_std,
                problem=problem,
                grid=grid,
                dt=dt,
            )
            total = data_loss + residual_weight * resid_loss
            optimizer.zero_grad()
            total.backward()
            optimizer.step()
            losses.append(float(total.detach()))
        val = _mean_relative(model, windows_val, u_mean, u_std, nu_mean, nu_std)
        history.append({"epoch": epoch, "train_loss": float(np.mean(losses)), "val_relative_l2": val})
        if val < best_val:
            best_val = val
            _save_checkpoint(
                output / "checkpoint.pt",
                model,
                {
                    "problem": problem,
                    "loss": loss,
                    "seed": seed,
                    "u_mean": u_mean,
                    "u_std": u_std,
                    "nu_mean": nu_mean,
                    "nu_std": nu_std,
                    "val_relative_l2": val,
                    "epoch": epoch,
                    **model.config_dict(),
                },
            )
    record = {
        "problem": problem,
        "loss": loss,
        "seed": seed,
        "best_val_relative_l2": best_val,
        "history": history,
        "parameter_count": model.parameter_count(),
    }
    (output / "train.json").write_text(json.dumps(record, indent=2) + "\n", encoding="utf-8")
    return record


def load_model(path: Path) -> tuple[FNO2d, dict[str, Any]]:
    payload = torch.load(path, map_location="cpu", weights_only=True)
    config = payload["config"]
    model = FNO2d(
        int(config["in_channels"]),
        int(config["out_channels"]),
        width=int(config["width"]),
        modes=int(config["modes"]),
        n_layers=int(config["n_layers"]),
    )
    model.load_state_dict(payload["state_dict"])
    model.eval()
    return model, config


def predict_window(
    model: FNO2d,
    inputs: np.ndarray,
    nu: float,
    config: dict[str, Any],
) -> np.ndarray:
    n_comp = inputs.shape[1]
    tensor = torch.as_tensor(inputs[None], dtype=torch.float32)
    norm_u = (tensor - float(config["u_mean"])) / float(config["u_std"])
    norm_nu = torch.tensor([(nu - float(config["nu_mean"])) / float(config["nu_std"])], dtype=torch.float32)
    with torch.no_grad():
        pred = model(pack_inputs(norm_u, norm_nu))
    frames = unpack_outputs(pred, pred.shape[1] // n_comp, n_comp).numpy()[0]
    return frames * float(config["u_std"]) + float(config["u_mean"])


def _batch_losses(
    model: FNO2d,
    batch: list[dict[str, Any]],
    *,
    u_mean: float,
    u_std: float,
    nu_mean: float,
    nu_std: float,
    problem: str,
    grid: Grid2D,
    dt: float,
) -> tuple[torch.Tensor, torch.Tensor]:
    inputs = torch.as_tensor(np.stack([item["inputs"] for item in batch]), dtype=torch.float32)
    targets = torch.as_tensor(np.stack([item["targets"] for item in batch]), dtype=torch.float32)
    nu = torch.as_tensor([item["nu"] for item in batch], dtype=torch.float32)
    n_comp = inputs.shape[2]
    packed = pack_inputs((inputs - u_mean) / u_std, (nu - nu_mean) / nu_std)
    pred = unpack_outputs(model(packed), inputs.shape[1], n_comp)
    target_norm = (targets - u_mean) / u_std
    data_loss = torch.mean((pred - target_norm) ** 2)
    physical = pred * u_std + u_mean
    residual = _torch_residual(physical, nu, problem, grid, dt, [item.get("forcing") for item in batch])
    return data_loss, residual


def _torch_residual(
    pred: torch.Tensor,
    nu: torch.Tensor,
    problem: str,
    grid: Grid2D,
    dt: float,
    forcings: list[str | None],
) -> torch.Tensor:
    last = pred[:, -1]
    prev = pred[:, 0]
    time_deriv = (last - prev) / max(dt, 1e-8)
    terms = []
    for i in range(last.shape[0]):
        visc = nu[i]
        if problem == "burgers2d":
            adv = _torch_burgers_adv(last[i], grid)
            lap = torch.stack([_torch_laplacian(last[i, c], grid) for c in range(2)], dim=0)
            terms.append(torch.mean((time_deriv[i] + adv - visc * lap) ** 2))
        else:
            omega = last[i, 0]
            force = _torch_kolmogorov(grid, last.device, last.dtype) if forcings[i] == "kolmogorov" else 0.0
            resid = time_deriv[i, 0] + _torch_ns_adv(omega, grid) - force - visc * _torch_laplacian(omega, grid)
            terms.append(torch.mean(resid**2))
    return torch.stack(terms).mean()


def _wavenumbers(grid: Grid2D, device: torch.device, dtype: torch.dtype) -> tuple[torch.Tensor, torch.Tensor]:
    k1 = 2.0 * math.pi * torch.fft.fftfreq(grid.n, d=grid.dx, device=device, dtype=dtype)
    k1[grid.n // 2] = 0.0
    kx, ky = torch.meshgrid(k1, k1, indexing="ij")
    return kx, ky


def _torch_laplacian(field: torch.Tensor, grid: Grid2D) -> torch.Tensor:
    kx, ky = _wavenumbers(grid, field.device, field.dtype)
    hat = torch.fft.fft2(field)
    return torch.fft.ifft2(-(kx * kx + ky * ky) * hat).real


def _torch_grad(field: torch.Tensor, grid: Grid2D) -> tuple[torch.Tensor, torch.Tensor]:
    kx, ky = _wavenumbers(grid, field.device, field.dtype)
    hat = torch.fft.fft2(field)
    return torch.fft.ifft2(1j * kx * hat).real, torch.fft.ifft2(1j * ky * hat).real


def _torch_burgers_adv(velocity: torch.Tensor, grid: Grid2D) -> torch.Tensor:
    adv = []
    for component in range(2):
        dx, dy = _torch_grad(velocity[component], grid)
        adv.append(velocity[0] * dx + velocity[1] * dy)
    return torch.stack(adv, dim=0)


def _torch_streamfunction(omega: torch.Tensor, grid: Grid2D) -> torch.Tensor:
    kx, ky = _wavenumbers(grid, omega.device, omega.dtype)
    k2 = kx * kx + ky * ky
    hat = torch.fft.fft2(omega)
    psi_hat = torch.zeros_like(hat)
    nonzero = k2 > 0.0
    psi_hat = torch.where(nonzero, hat / torch.clamp(k2, min=1e-30), torch.zeros_like(hat))
    psi_hat[0, 0] = 0.0
    psi = torch.fft.ifft2(psi_hat).real
    return psi - psi.mean()


def _torch_ns_adv(omega: torch.Tensor, grid: Grid2D) -> torch.Tensor:
    psi = _torch_streamfunction(omega, grid)
    psi_x, psi_y = _torch_grad(psi, grid)
    u, v = -psi_y, psi_x
    dx, dy = _torch_grad(omega, grid)
    return u * dx + v * dy


def _torch_kolmogorov(grid: Grid2D, device: torch.device, dtype: torch.dtype) -> torch.Tensor:
    axis = torch.arange(grid.n, device=device, dtype=dtype) * grid.dx
    yy = axis.view(1, -1).expand(grid.n, grid.n)
    return -4.0 * torch.cos(4.0 * yy)


def _mean_relative(
    model: FNO2d,
    windows: list[dict[str, Any]],
    u_mean: float,
    u_std: float,
    nu_mean: float,
    nu_std: float,
) -> float:
    if not windows:
        return float("inf")
    model.eval()
    scores: list[float] = []
    n_comp = windows[0]["inputs"].shape[1]
    with torch.no_grad():
        for item in windows:
            inputs = torch.as_tensor(item["inputs"][None], dtype=torch.float32)
            packed = pack_inputs((inputs - u_mean) / u_std, torch.tensor([(item["nu"] - nu_mean) / nu_std]))
            pred = unpack_outputs(model(packed), inputs.shape[1], n_comp).numpy()[0] * u_std + u_mean
            scores.append(relative_l2(pred, item["targets"]))
    return float(np.mean(scores))


def _save_checkpoint(path: Path, model: FNO2d, config: dict[str, Any]) -> None:
    torch.save({"state_dict": model.state_dict(), "config": config}, path)
