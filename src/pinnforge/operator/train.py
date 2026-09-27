"""Supervised training for the Burgers FNO.

The default loss is mean squared error between the network output and the
target window in normalized ``u`` space. ``LossConfig`` can replace that
term with the discrete Burgers residual, or add the residual with a fixed
weight. The checkpoint is still the epoch with the lowest mean per-window
relative L2 on the validation instances, after denormalizing. Test
instances are not loaded, and they are not used to choose the weight.

Epoch 0 is the network before any Adam step. Later epochs are full passes
over the training windows.
"""

from __future__ import annotations

import json
import math
import time
from collections.abc import Callable
from dataclasses import dataclass
from pathlib import Path

from pinnforge.ml_import import require_torch
from pinnforge.operator.checkpoint import save_fno_checkpoint
from pinnforge.operator.data import load_split_windows
from pinnforge.operator.defaults import (
    DEFAULT_BATCH_SIZE,
    DEFAULT_EPOCHS,
    DEFAULT_LAYERS,
    DEFAULT_LR,
    DEFAULT_MODES,
    DEFAULT_SEED,
    DEFAULT_WIDTH,
)
from pinnforge.operator.evaluate import score_dataset
from pinnforge.operator.fno import FNO1d, pack_inputs
from pinnforge.operator.loss import window_operator_loss
from pinnforge.operator.metrics import normalized_mse
from pinnforge.operator.residual import (
    LossConfig,
    objective_from_parts,
    prediction_window_residual,
    residual_stats,
)
from pinnforge.operator.windows import (
    DEFAULT_INPUT_FRAMES,
    DEFAULT_OUTPUT_FRAMES,
    DEFAULT_STRIDE,
    WindowDataset,
    WindowSpec,
)
from pinnforge.training.manifest import environment

torch = require_torch()

METRICS_FORMAT = "pinnforge.fno_metrics.v1"
MANIFEST_FORMAT = "pinnforge.fno_manifest.v1"


@dataclass(frozen=True, slots=True)
class EpochRow:
    """One training epoch, including epoch 0 before the optimizer runs."""

    epoch: int
    train_mse: float
    val_mse: float
    val_relative_l2: float
    lr: float
    train_objective: float
    train_residual_mse: float
    val_mean_abs_residual: float

    def as_dict(self) -> dict[str, object]:
        return {
            "format": METRICS_FORMAT,
            "epoch": self.epoch,
            "train_mse": self.train_mse,
            "val_mse": self.val_mse,
            "val_relative_l2": self.val_relative_l2,
            "lr": self.lr,
            "train_objective": self.train_objective,
            "train_residual_mse": self.train_residual_mse,
            "val_mean_abs_residual": self.val_mean_abs_residual,
        }


@dataclass(frozen=True, slots=True)
class TrainResult:
    """Paths and the validation score of the selected epoch."""

    checkpoint: Path
    metrics_path: Path
    manifest_path: Path
    selected_epoch: int
    val_relative_l2: float
    val_mse: float
    train_mse: float
    parameter_count: int
    wall_clock_seconds: float


def fit_fno(
    train: WindowDataset,
    val: WindowDataset,
    *,
    width: int = DEFAULT_WIDTH,
    modes: int = DEFAULT_MODES,
    n_layers: int = DEFAULT_LAYERS,
    epochs: int = DEFAULT_EPOCHS,
    batch_size: int = DEFAULT_BATCH_SIZE,
    lr: float = DEFAULT_LR,
    seed: int = DEFAULT_SEED,
    loss: LossConfig | None = None,
    log: Callable[[EpochRow], None] | None = None,
) -> tuple[FNO1d, list[EpochRow]]:
    """Train on ``train`` and record validation relative L2 each epoch.

    ``train`` and ``val`` must use the same window spec, normalization, and
    spatial size, and their instance ids must be disjoint. The returned
    network is the epoch with the lowest validation relative L2, loaded in
    ``eval`` mode. History includes epoch 0. ``loss`` defaults to normalized
    data MSE. A physics loss does not change that selection rule.
    """

    loss_config = loss if loss is not None else LossConfig()
    if not isinstance(loss_config, LossConfig):
        raise TypeError("loss must be a LossConfig")
    _require_pair(train, val)
    _require_positive_int(epochs, "epochs")
    _require_positive_int(batch_size, "batch_size")
    _require_positive_int(seed, "seed", allow_zero=True)
    if isinstance(lr, bool) or not isinstance(lr, (int, float)) or not math.isfinite(float(lr)) or float(lr) <= 0.0:
        raise ValueError("lr must be a finite number > 0")
    torch.manual_seed(seed)
    model = FNO1d(
        in_channels=train.spec.input_frames + 1,
        out_channels=train.spec.output_frames,
        width=width,
        modes=modes,
        n_layers=n_layers,
    )
    optimizer = torch.optim.Adam(model.parameters(), lr=float(lr))
    generator = torch.Generator(device="cpu")
    generator.manual_seed(seed)
    history: list[EpochRow] = []
    best_score = float("inf")
    best_state: dict[str, torch.Tensor] | None = None
    for epoch in range(epochs + 1):
        if epoch > 0:
            _adam_epoch(
                model,
                optimizer,
                train,
                batch_size=batch_size,
                generator=generator,
                loss_config=loss_config,
            )
        row = _measure(
            model,
            train,
            val,
            epoch=epoch,
            lr=float(lr),
            batch_size=batch_size,
            loss_config=loss_config,
        )
        history.append(row)
        if log is not None:
            log(row)
        if row.val_relative_l2 < best_score:
            best_score = row.val_relative_l2
            best_state = {key: value.detach().cpu().clone() for key, value in model.state_dict().items()}
    if best_state is None:
        raise RuntimeError("training produced no checkpoint state")
    model.load_state_dict(best_state)
    model.eval()
    return model, history


def train_from_paths(
    *,
    pilot_dir: Path,
    manifest_path: Path,
    output_dir: Path,
    epochs: int = DEFAULT_EPOCHS,
    batch_size: int = DEFAULT_BATCH_SIZE,
    lr: float = DEFAULT_LR,
    width: int = DEFAULT_WIDTH,
    modes: int = DEFAULT_MODES,
    n_layers: int = DEFAULT_LAYERS,
    input_frames: int = DEFAULT_INPUT_FRAMES,
    output_frames: int = DEFAULT_OUTPUT_FRAMES,
    stride: int = DEFAULT_STRIDE,
    seed: int = DEFAULT_SEED,
    loss: LossConfig | None = None,
    check_field_hash: bool = True,
    log: Callable[[EpochRow], None] | None = None,
) -> TrainResult:
    """Train on pilot windows and write checkpoint, metrics, and a manifest.

    Only the train and validation splits are read.
    """

    spec = WindowSpec(input_frames=input_frames, output_frames=output_frames, stride=stride)
    loss_config = loss if loss is not None else LossConfig()
    if not isinstance(loss_config, LossConfig):
        raise TypeError("loss must be a LossConfig")
    datasets = load_split_windows(
        pilot_dir,
        manifest_path,
        spec,
        ("train", "val"),
        check_field_hash=check_field_hash,
    )
    started = time.perf_counter()
    model, history = fit_fno(
        datasets["train"],
        datasets["val"],
        width=width,
        modes=modes,
        n_layers=n_layers,
        epochs=epochs,
        batch_size=batch_size,
        lr=lr,
        seed=seed,
        loss=loss_config,
        log=log,
    )
    elapsed = time.perf_counter() - started
    selected = min(history, key=lambda row: (row.val_relative_l2, row.epoch))
    destination = Path(output_dir)
    destination.mkdir(parents=True, exist_ok=True)
    metrics_path = destination / "metrics.jsonl"
    _write_metrics(metrics_path, history)
    checkpoint = save_fno_checkpoint(
        destination / "checkpoint.pt",
        model,
        epoch=selected.epoch,
        spec=spec,
        norm=datasets["train"].norm,
        seed=seed,
        val_relative_l2=selected.val_relative_l2,
        loss=loss_config,
    )
    manifest_file = destination / "manifest.json"
    _write_run_manifest(
        manifest_file,
        datasets=datasets,
        manifest_path=Path(manifest_path),
        spec=spec,
        model=model,
        history=history,
        selected=selected,
        seed=seed,
        epochs=epochs,
        batch_size=batch_size,
        lr=float(lr),
        loss=loss_config,
        wall_clock_seconds=elapsed,
    )
    return TrainResult(
        checkpoint=checkpoint,
        metrics_path=metrics_path,
        manifest_path=manifest_file,
        selected_epoch=selected.epoch,
        val_relative_l2=selected.val_relative_l2,
        val_mse=selected.val_mse,
        train_mse=selected.train_mse,
        parameter_count=model.parameter_count(),
        wall_clock_seconds=elapsed,
    )


def _adam_epoch(
    model: FNO1d,
    optimizer: torch.optim.Optimizer,
    dataset: WindowDataset,
    *,
    batch_size: int,
    generator: torch.Generator,
    loss_config: LossConfig,
) -> None:
    model.train()
    inputs = torch.as_tensor(dataset.inputs, dtype=torch.float32)
    nu = torch.as_tensor(dataset.nu, dtype=torch.float32)
    targets = torch.as_tensor(dataset.targets, dtype=torch.float32)
    order = torch.randperm(inputs.shape[0], generator=generator)
    for start in range(0, int(order.shape[0]), batch_size):
        index = order[start : start + batch_size]
        optimizer.zero_grad(set_to_none=True)
        prediction = model(pack_inputs(inputs[index], nu[index]))
        if loss_config.mode == "data":
            loss = torch.mean((prediction - targets[index]) ** 2)
        else:
            loss = window_operator_loss(
                prediction,
                targets[index],
                inputs[index],
                nu[index],
                dataset.norm,
                loss_config,
            )
        if not torch.isfinite(loss):
            raise RuntimeError("training loss is not finite")
        loss.backward()
        optimizer.step()


def _measure(
    model: FNO1d,
    train: WindowDataset,
    val: WindowDataset,
    *,
    epoch: int,
    lr: float,
    batch_size: int,
    loss_config: LossConfig,
) -> EpochRow:
    model.eval()
    train_prediction = _predict(model, train, batch_size)
    train_mse = normalized_mse(train_prediction, train.targets)
    train_residual = residual_stats(
        prediction_window_residual(
            train.inputs,
            train_prediction,
            train.nu,
            train.norm,
            loss_config,
        )
    )
    val_score = score_dataset(model, val, batch_size=batch_size, residual=loss_config)
    return EpochRow(
        epoch=epoch,
        train_mse=train_mse,
        val_mse=val_score.normalized_mse,
        val_relative_l2=val_score.mean_relative_l2,
        lr=lr,
        train_objective=objective_from_parts(train_mse, train_residual.mean_square, loss_config),
        train_residual_mse=train_residual.mean_square,
        val_mean_abs_residual=val_score.mean_abs_residual,
    )


def _predict(model: FNO1d, dataset: WindowDataset, batch_size: int) -> object:
    from pinnforge.operator.evaluate import predict_normalized

    return predict_normalized(model, dataset, batch_size=batch_size).numpy()


def _require_pair(train: WindowDataset, val: WindowDataset) -> None:
    if train.split != "train" or val.split != "val":
        raise ValueError("fit_fno expects a train dataset and a val dataset")
    if train.spec != val.spec:
        raise ValueError("train and val windows must use the same window spec")
    if train.norm != val.norm:
        raise ValueError("train and val windows must use the same normalization")
    if train.n_space() != val.n_space():
        raise ValueError("train and val spatial sizes differ")
    shared = train.instance_id_set() & val.instance_id_set()
    if shared:
        raise ValueError(f"train and val share instance ids {sorted(shared)[:5]}")


def _write_metrics(path: Path, history: list[EpochRow]) -> None:
    lines = [json.dumps(row.as_dict(), allow_nan=False) for row in history]
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def _write_run_manifest(
    path: Path,
    *,
    datasets: dict[str, WindowDataset],
    manifest_path: Path,
    spec: WindowSpec,
    model: FNO1d,
    history: list[EpochRow],
    selected: EpochRow,
    seed: int,
    epochs: int,
    batch_size: int,
    lr: float,
    loss: LossConfig,
    wall_clock_seconds: float,
) -> None:
    train = datasets["train"]
    val = datasets["val"]
    payload = {
        "format": MANIFEST_FORMAT,
        "task": "1D FNO on Burgers windows",
        "loss": loss.describe(),
        "loss_config": loss.to_dict(),
        "selection": "minimum validation mean relative L2, ties take the earliest epoch",
        "test_used_for_training": False,
        "test_used_for_selection": False,
        "physics_residual": loss.uses_physics(),
        "wall_clock_seconds": wall_clock_seconds,
        "selected_val_mean_abs_residual": selected.val_mean_abs_residual,
        "device": "cpu",
        "seed": seed,
        "epochs_requested": epochs,
        "selected_epoch": selected.epoch,
        "selected_val_relative_l2": selected.val_relative_l2,
        "selected_val_mse": selected.val_mse,
        "selected_train_mse": selected.train_mse,
        "lr": lr,
        "batch_size": batch_size,
        "optimizer": "Adam",
        "schedule": "constant",
        "parameter_count": model.parameter_count(),
        "model": model.config_dict(),
        "window": {
            "input_frames": spec.input_frames,
            "output_frames": spec.output_frames,
            "stride": spec.stride,
        },
        "viscosity": "spatially constant channel, normalized by the training-split mean and population std",
        "normalization": train.norm.to_dict(),
        "pilot_manifest": str(manifest_path),
        "n_train_instances": train.n_instances(),
        "n_val_instances": val.n_instances(),
        "n_train_windows": train.n_windows(),
        "n_val_windows": val.n_windows(),
        "train_instance_ids": sorted(train.instance_id_set()),
        "val_instance_ids": sorted(val.instance_id_set()),
        "environment": environment(),
        "history": [row.as_dict() for row in history],
    }
    path.write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")


def _require_positive_int(value: object, label: str, *, allow_zero: bool = False) -> int:
    if isinstance(value, bool) or not isinstance(value, int):
        raise ValueError(f"{label} must be an integer")
    if value < 0 or (value == 0 and not allow_zero):
        raise ValueError(f"{label} must be >= {0 if allow_zero else 1}")
    return value
