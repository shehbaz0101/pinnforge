"""CPU checkpoints for a later evaluation step.

A checkpoint is a small ``torch.save`` dict: a format tag, the epoch,
the :class:`~pinnforge.specs.train.TrainConfig` as JSON-ready
data, the model ``state_dict``, and, for checkpoints written by the
current trainer, the equation spec. Optimizer state is not stored.
``load_checkpoint`` rebuilds the MLP from that spec (or the built-in
spec when an older file omitted it) and the saved widths, then loads
the weights onto CPU.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

from pinnforge.equations.base import EquationSpec
from pinnforge.equations.registry import parse_equation
from pinnforge.ml_import import require_torch
from pinnforge.models import mlp_from_spec
from pinnforge.sampling import default_spec
from pinnforge.specs.paths import resolve_inside_cwd
from pinnforge.specs.train import TrainConfig

torch = require_torch()

CHECKPOINT_FORMAT = "pinnforge.checkpoint.v1"


@dataclass(frozen=True, slots=True)
class LoadedCheckpoint:
    """Model, config, and epoch read from one checkpoint file.

    ``model`` is in ``eval`` mode on CPU. ``path`` is the resolved file
    that was loaded.
    """

    model: torch.nn.Module
    config: TrainConfig
    epoch: int
    path: Path
    spec: EquationSpec


def save_checkpoint(
    model: torch.nn.Module,
    config: TrainConfig,
    epoch: int,
    directory: str | Path,
    *,
    spec: EquationSpec | None = None,
) -> Path:
    """Write ``checkpoint.pt`` and ``epoch_XXXX.pt`` under ``directory``.

    ``directory`` must be a relative path inside the working directory.
    Both files hold the same payload. The return value is ``checkpoint.pt``,
    which always matches the latest epoch this function wrote.

    ``spec``, when set, is stored as JSON so a later load rebuilds the
    problem that was trained, including parameter overrides. The format
    tag stays ``pinnforge.checkpoint.v1``. Checkpoints written before
    that field existed omit it and load the built-in spec.

    Raises:
        TypeError: ``model`` or ``config`` has the wrong type, or
            ``spec`` is not a Day 1 equation spec.
        ValueError: ``epoch`` is negative, ``spec`` does not match
            ``config.equation_id``, or ``directory`` escapes the working
            directory.
    """

    if not isinstance(model, torch.nn.Module):
        raise TypeError("model must be a torch.nn.Module")
    if not isinstance(config, TrainConfig):
        raise TypeError("config must be a TrainConfig")
    if spec is not None:
        _require_matching_spec(spec, config)
    if isinstance(epoch, bool) or not isinstance(epoch, int) or epoch < 0:
        raise ValueError("epoch must be a non-negative integer")
    resolved = resolve_inside_cwd(directory)
    if resolved.exists() and not resolved.is_dir():
        raise ValueError("checkpoint dir must be a directory")
    resolved.mkdir(parents=True, exist_ok=True)
    payload = {
        "format": CHECKPOINT_FORMAT,
        "epoch": epoch,
        "config": config.model_dump(mode="json"),
        "state_dict": model.state_dict(),
    }
    if spec is not None:
        payload["spec"] = spec.model_dump(mode="json")
    latest = resolved / "checkpoint.pt"
    tagged = resolved / f"epoch_{epoch:04d}.pt"
    torch.save(payload, latest)
    torch.save(payload, tagged)
    return latest


def load_checkpoint(path: str | Path) -> LoadedCheckpoint:
    """Load a checkpoint written by :func:`save_checkpoint`.

    The path must be a relative ``.pt`` file inside the working
    directory. Weights are mapped onto CPU. The returned model is in
    ``eval`` mode.

    Raises:
        ValueError: the path escapes the working directory, the format
            tag is missing, or the payload is not a training checkpoint.
        RuntimeError: the ``state_dict`` does not match the rebuilt MLP.
    """

    resolved = resolve_inside_cwd(path, suffix=".pt")
    payload = torch.load(resolved, map_location="cpu", weights_only=False)
    if not isinstance(payload, dict):
        raise ValueError("checkpoint must be a dict")
    if payload.get("format") != CHECKPOINT_FORMAT:
        raise ValueError(f"unsupported checkpoint format {payload.get('format')!r}")
    epoch = payload.get("epoch")
    if isinstance(epoch, bool) or not isinstance(epoch, int) or epoch < 0:
        raise ValueError("checkpoint epoch must be a non-negative integer")
    config_data = payload.get("config")
    state = payload.get("state_dict")
    if not isinstance(config_data, dict):
        raise ValueError("checkpoint config must be a dict")
    if not isinstance(state, dict):
        raise ValueError("checkpoint state_dict must be a dict")
    config = TrainConfig.model_validate(config_data)
    spec = _spec_from_payload(payload.get("spec"), config)
    model = mlp_from_spec(spec, config.hidden_widths, activation=config.activation)
    model.load_state_dict(state)
    model.to(torch.device("cpu"))
    model.eval()
    return LoadedCheckpoint(model=model, config=config, epoch=epoch, path=resolved, spec=spec)


def _require_matching_spec(spec: EquationSpec, config: TrainConfig) -> None:
    if not isinstance(spec, EquationSpec):
        raise TypeError("spec must be an EquationSpec")
    if spec.equation_id != config.equation_id:
        raise ValueError(
            f"spec equation_id {spec.equation_id} does not match "
            f"config equation_id {config.equation_id}"
        )


def _spec_from_payload(raw: object, config: TrainConfig) -> EquationSpec:
    if raw is None:
        return default_spec(config.equation_id)
    if not isinstance(raw, dict):
        raise ValueError("checkpoint spec must be a dict")
    spec = parse_equation(raw)
    _require_matching_spec(spec, config)
    return spec
