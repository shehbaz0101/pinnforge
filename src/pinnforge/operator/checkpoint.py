"""Checkpoint for the Burgers FNO.

The file is a ``torch.save`` dict with a format tag that is not the
coordinate-PINN checkpoint tag. ``load_fno_checkpoint`` reads it with
``weights_only=True`` and rebuilds :class:`~pinnforge.operator.fno.FNO1d`
on CPU. ``loss_config`` records the training loss. Checkpoints written
before that block load as normalized data MSE.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

from pinnforge.ml_import import require_torch
from pinnforge.operator.fno import FNO1d, fno_from_config
from pinnforge.operator.residual import LossConfig
from pinnforge.operator.windows import FieldNorm, WindowSpec

torch = require_torch()

FNO_CHECKPOINT_FORMAT = "pinnforge.fno_checkpoint.v1"


@dataclass(frozen=True, slots=True)
class LoadedFNO:
    """Selected FNO weights and the data contract stored beside them."""

    model: FNO1d
    epoch: int
    path: Path
    spec: WindowSpec
    norm: FieldNorm
    seed: int
    val_relative_l2: float
    loss: LossConfig


def save_fno_checkpoint(
    path: Path,
    model: FNO1d,
    *,
    epoch: int,
    spec: WindowSpec,
    norm: FieldNorm,
    seed: int,
    val_relative_l2: float,
    loss: LossConfig | None = None,
) -> Path:
    """Write one checkpoint file. Parent directories are created."""

    if not isinstance(model, FNO1d):
        raise TypeError("model must be an FNO1d")
    if isinstance(epoch, bool) or not isinstance(epoch, int) or epoch < 0:
        raise ValueError("epoch must be a non-negative integer")
    loss_config = loss if loss is not None else LossConfig()
    if not isinstance(loss_config, LossConfig):
        raise TypeError("loss must be a LossConfig")
    destination = Path(path)
    destination.parent.mkdir(parents=True, exist_ok=True)
    payload = {
        "format": FNO_CHECKPOINT_FORMAT,
        "epoch": epoch,
        "seed": seed,
        "val_relative_l2": float(val_relative_l2),
        "model": model.config_dict(),
        "window": {
            "input_frames": spec.input_frames,
            "output_frames": spec.output_frames,
            "stride": spec.stride,
        },
        "normalization": norm.to_dict(),
        "loss_config": loss_config.to_dict(),
        "state_dict": {key: value.detach().cpu() for key, value in model.state_dict().items()},
    }
    torch.save(payload, destination)
    return destination


def load_fno_checkpoint(path: Path) -> LoadedFNO:
    """Load a checkpoint written by :func:`save_fno_checkpoint`."""

    source = Path(path)
    if not source.is_file():
        raise ValueError(f"checkpoint does not exist: {source}")
    payload = _load_payload(source)
    if not isinstance(payload, dict):
        raise ValueError("checkpoint must be a dict")
    if payload.get("format") != FNO_CHECKPOINT_FORMAT:
        raise ValueError(f"unsupported checkpoint format {payload.get('format')!r}")
    epoch = payload.get("epoch")
    seed = payload.get("seed")
    if isinstance(epoch, bool) or not isinstance(epoch, int) or epoch < 0:
        raise ValueError("checkpoint epoch must be a non-negative integer")
    if isinstance(seed, bool) or not isinstance(seed, int) or seed < 0:
        raise ValueError("checkpoint seed must be a non-negative integer")
    score = payload.get("val_relative_l2")
    if isinstance(score, bool) or not isinstance(score, (int, float)):
        raise ValueError("checkpoint val_relative_l2 must be a float")
    model_config = payload.get("model")
    window = payload.get("window")
    normalization = payload.get("normalization")
    state = payload.get("state_dict")
    if not isinstance(model_config, dict) or not isinstance(window, dict):
        raise ValueError("checkpoint model and window must be dicts")
    if not isinstance(normalization, dict) or not isinstance(state, dict):
        raise ValueError("checkpoint normalization and state_dict must be dicts")
    spec = WindowSpec(
        input_frames=int(window["input_frames"]),
        output_frames=int(window["output_frames"]),
        stride=int(window["stride"]),
    )
    norm = FieldNorm(
        u_mean=float(normalization["u_mean"]),
        u_std=float(normalization["u_std"]),
        nu_mean=float(normalization["nu_mean"]),
        nu_std=float(normalization["nu_std"]),
    )
    model = fno_from_config(model_config)
    model.load_state_dict(state)
    model.eval()
    return LoadedFNO(
        model=model,
        epoch=epoch,
        path=source,
        spec=spec,
        norm=norm,
        seed=seed,
        val_relative_l2=float(score),
        loss=_loss_from_payload(payload),
    )


def _loss_from_payload(payload: dict[str, object]) -> LossConfig:
    raw = payload.get("loss_config")
    if raw is None:
        return LossConfig()
    if not isinstance(raw, dict):
        raise ValueError("checkpoint loss_config must be a dict")
    return LossConfig.from_dict(raw)


def _load_payload(path: Path) -> object:
    try:
        return torch.load(path, map_location="cpu", weights_only=True)
    except TypeError as exc:
        if "weights_only" not in str(exc):
            raise
        return torch.load(path, map_location="cpu")
