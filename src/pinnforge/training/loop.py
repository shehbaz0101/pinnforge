"""Fixed-batch Adam loop over a residual and soft penalties.

The collocation batch is drawn once from the training RNG stream.
``torch.manual_seed`` uses ``TrainConfig.seed`` before a new network is
built. A resumed run loads weights, Adam state, and the torch RNG state
instead of building a new network. Every saved epoch scores that batch,
appends one ``metrics.jsonl`` line, writes a checkpoint, and then takes
one Adam step. Epoch 0 is the initial weights, so the logged losses stay
on one point set.

The validation stream is drawn and recorded in ``manifest.json``. It is
not added to the loss. Evaluation uses the test stream and its own seed.

Re-drawing each epoch with ``seed + epoch`` would also be reproducible,
but the loss would mix optimization progress with a new sample. This
loop keeps the batch fixed so a drop in ``loss`` or ``loss_pde`` is the
optimizer on the same collocation problem.
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from pathlib import Path

from pinnforge.equations import EquationSpec
from pinnforge.equations.burgers import Burgers1DSpec
from pinnforge.equations.harmonic import HarmonicOscillatorSpec
from pinnforge.equations.poisson import PoissonToySpec
from pinnforge.losses import boundary_coverage_gaps, soft_penalty
from pinnforge.ml_import import require_torch
from pinnforge.models import mlp_from_spec
from pinnforge.residuals import mean_squared_residual
from pinnforge.sampling import (
    CollocationBatch,
    SampleConfig,
    default_spec,
    points_sha256,
    sample_equation,
    stream_generator,
    stream_identity,
)
from pinnforge.specs.paths import resolve_inside_cwd
from pinnforge.specs.train import TrainConfig
from pinnforge.training.checkpoint import load_checkpoint, save_checkpoint
from pinnforge.training.manifest import MANIFEST_FORMAT, environment, write_manifest
from pinnforge.training.metrics import EpochMetrics, append_metrics, write_metrics

torch = require_torch()


@dataclass(frozen=True, slots=True)
class TrainResult:
    """Weights and files left by :func:`train_loop`.

    ``epoch`` is the last completed epoch, equal to ``config.epochs``.
    ``history[0]`` is the loss before any Adam step. ``model`` is still
    in train mode. ``log_path`` and ``checkpoint_path`` are resolved
    under the working directory.
    """

    model: torch.nn.Module
    config: TrainConfig
    epoch: int
    log_path: Path
    checkpoint_path: Path
    history: tuple[EpochMetrics, ...]


def train_loop(config: TrainConfig, *, spec: EquationSpec | None = None) -> TrainResult:
    """Train on one seeded batch and return the final weights.

    The total loss is ``w_pde * MSE(residual) + w_ic * initial + w_bc *
    boundary``, where ``boundary`` is the Dirichlet, Neumann, and
    periodic penalties. ``loss_pde``, ``loss_ic``, and ``loss_bc`` in
    the metrics file are the unweighted mean squares. ``lr`` is the
    Adam learning rate and does not change.

    ``spec`` overrides the built-in problem for ``config.equation_id``.
    It is the spec an experiment config resolved, including parameter
    overrides. When it is omitted, the built-in spec is used. Either
    way the checkpoint stores the spec that was trained.

    Checkpoints hold the weights that produced that epoch's line,
    plus Adam state and the torch RNG state. ``checkpoint.pt`` matches
    the last saved epoch. Tagged files are written at epoch 0, at the
    last epoch, and every ``checkpoint_interval`` epochs.
    ``manifest.json`` records the three RNG stream identities.

    Raises:
        TypeError: ``config`` is not a :class:`TrainConfig`, or ``spec``
            is not one of the three Day 1 specs.
        ValueError: a path escapes the working directory, the sample
            counts do not fit the spec, or ``spec.equation_id`` does not
            match the config.
        RuntimeError: the loss does not depend on the model parameters.
    """

    if not isinstance(config, TrainConfig):
        raise TypeError("config must be a TrainConfig")
    spec = _resolve_spec(config, spec)
    log_path = resolve_inside_cwd(config.log_path, suffix=".jsonl")
    checkpoint_dir = resolve_inside_cwd(config.checkpoint_dir)
    if checkpoint_dir.exists() and not checkpoint_dir.is_dir():
        raise ValueError("checkpoint dir must be a directory")
    sample_config = _sample_config(config)
    batch = sample_equation(spec, sample_config, rng=stream_generator(config.seed, "train"))
    _require_boundary_samples(batch, spec, config)
    validation = sample_equation(
        spec, sample_config, rng=stream_generator(config.seed, "validation")
    )
    model, optimizer, history, start_epoch = _initial_state(config, spec)
    if start_epoch > 0:
        # The saved epoch logged the loss and then stopped before its Adam
        # step. A run that continued would have taken that step. Take it
        # now so the next logged epoch matches an uninterrupted run.
        catch_up, _pde, _initial, _boundary = _loss_terms(model, batch, spec, config)
        if not catch_up.requires_grad:
            raise RuntimeError("training loss is not connected to the model parameters")
        optimizer.zero_grad(set_to_none=True)
        catch_up.backward()
        optimizer.step()
    log_path.parent.mkdir(parents=True, exist_ok=True)
    checkpoint_dir.mkdir(parents=True, exist_ok=True)
    if history:
        write_metrics(log_path, history)
    else:
        log_path.write_text("", encoding="utf-8")
    _write_run_manifest(
        checkpoint_dir,
        config,
        validation_digest=points_sha256(validation.interior),
    )
    checkpoint_path: Path | None = None
    with torch.enable_grad():
        for epoch in range(start_epoch, config.epochs + 1):
            total, pde, initial, boundary = _loss_terms(model, batch, spec, config)
            row = EpochMetrics(
                epoch=epoch,
                loss=_scalar(total),
                loss_pde=_scalar(pde),
                loss_ic=_scalar(initial),
                loss_bc=_scalar(boundary),
                lr=float(optimizer.param_groups[0]["lr"]),
            )
            append_metrics(log_path, row)
            history.append(row)
            if _checkpoint_due(epoch, config):
                checkpoint_path = save_checkpoint(
                    model,
                    config,
                    epoch,
                    config.checkpoint_dir,
                    spec=spec,
                    optimizer=optimizer,
                    history=tuple(item.as_dict() for item in history),
                )
            if epoch == config.epochs:
                break
            if not total.requires_grad:
                raise RuntimeError("training loss is not connected to the model parameters")
            optimizer.zero_grad(set_to_none=True)
            total.backward()
            optimizer.step()
    if checkpoint_path is None:
        raise RuntimeError("training loop did not write a checkpoint")
    return TrainResult(
        model=model,
        config=config,
        epoch=config.epochs,
        log_path=log_path,
        checkpoint_path=checkpoint_path,
        history=tuple(history),
    )


def _resolve_spec(config: TrainConfig, spec: EquationSpec | None) -> EquationSpec:
    if spec is None:
        return default_spec(config.equation_id)
    if not isinstance(spec, (HarmonicOscillatorSpec, Burgers1DSpec, PoissonToySpec)):
        raise TypeError("spec must be a HarmonicOscillatorSpec, Burgers1DSpec, or PoissonToySpec")
    if spec.equation_id != config.equation_id:
        raise ValueError(
            f"spec equation_id {spec.equation_id} does not match "
            f"config equation_id {config.equation_id}"
        )
    return spec


def _initial_state(
    config: TrainConfig,
    spec: EquationSpec,
) -> tuple[torch.nn.Module, torch.optim.Optimizer, list[EpochMetrics], int]:
    if config.resume_from is None:
        torch.manual_seed(config.seed)
        model = mlp_from_spec(spec, config.hidden_widths, activation=config.activation)
        model.to(torch.device(config.device))
        model.train()
        optimizer = torch.optim.Adam(model.parameters(), lr=config.lr)
        return model, optimizer, [], 0
    loaded = load_checkpoint(config.resume_from)
    _require_resume_match(loaded.config, config, loaded.spec, spec)
    if loaded.epoch >= config.epochs:
        raise ValueError(
            f"checkpoint epoch {loaded.epoch} is already at or past epochs {config.epochs}"
        )
    if loaded.optimizer_state is None or loaded.torch_rng is None or not loaded.history:
        raise ValueError("checkpoint has no optimizer, RNG, and metrics history to resume")
    model = loaded.model
    model.to(torch.device(config.device))
    model.train()
    optimizer = torch.optim.Adam(model.parameters(), lr=config.lr)
    optimizer.load_state_dict(loaded.optimizer_state)
    torch.set_rng_state(loaded.torch_rng)
    history = [_metrics_row(row) for row in loaded.history]
    if [row.epoch for row in history] != list(range(loaded.epoch + 1)):
        raise ValueError("checkpoint metrics history does not match its epoch")
    return model, optimizer, history, loaded.epoch + 1


def _require_resume_match(
    saved: TrainConfig,
    config: TrainConfig,
    saved_spec: EquationSpec,
    spec: EquationSpec,
) -> None:
    fields = (
        "equation_id",
        "n_interior",
        "n_ic",
        "n_bc",
        "hidden_widths",
        "activation",
        "lr",
        "seed",
        "method",
        "device",
        "w_pde",
        "w_ic",
        "w_bc",
    )
    for name in fields:
        if getattr(saved, name) != getattr(config, name):
            raise ValueError(f"resume checkpoint {name} does not match the training config")
    if saved_spec.model_dump(mode="json") != spec.model_dump(mode="json"):
        raise ValueError("resume checkpoint spec does not match the training spec")


def _require_boundary_samples(
    batch: CollocationBatch,
    spec: EquationSpec,
    config: TrainConfig,
) -> None:
    if config.w_bc == 0:
        return
    gaps = boundary_coverage_gaps(batch, spec)
    if not gaps:
        return
    names = ", ".join(gaps)
    raise ValueError(
        f"prescribed boundary conditions are not enforced ({names}). "
        "Increase n_bc so every face is sampled, or set w_bc to 0 to drop "
        "boundary penalties explicitly."
    )


def _checkpoint_due(epoch: int, config: TrainConfig) -> bool:
    if epoch == 0 or epoch == config.epochs:
        return True
    return epoch % config.checkpoint_interval == 0


def _write_run_manifest(directory: Path, config: TrainConfig, *, validation_digest: str) -> None:
    payload: dict[str, object] = {
        "format": MANIFEST_FORMAT,
        "equation_id": config.equation_id,
        "seed": config.seed,
        "checkpoint_interval": config.checkpoint_interval,
        "resume_from": config.resume_from,
        "streams": {
            "train": stream_identity(config.seed, "train"),
            "validation": stream_identity(config.seed, "validation"),
            "test": stream_identity(config.seed, "test"),
        },
        "validation_interior_sha256": validation_digest,
        "test_stream_note": (
            "Evaluation draws the test stream from EvalConfig.seed. "
            "The integer above is the training seed. The streams stay distinct "
            "when those integers are equal."
        ),
        "environment": environment(),
    }
    write_manifest(directory, payload)


def _metrics_row(payload: dict[str, object]) -> EpochMetrics:
    return EpochMetrics(
        epoch=_as_int(payload.get("epoch"), label="epoch"),
        loss=_as_float(payload.get("loss"), label="loss"),
        loss_pde=_as_float(payload.get("loss_pde"), label="loss_pde"),
        loss_ic=_as_float(payload.get("loss_ic"), label="loss_ic"),
        loss_bc=_as_float(payload.get("loss_bc"), label="loss_bc"),
        lr=_as_float(payload.get("lr"), label="lr"),
    )


def _as_int(value: object, *, label: str) -> int:
    if isinstance(value, bool) or not isinstance(value, int):
        raise ValueError(f"checkpoint history {label} must be an integer")
    return value


def _as_float(value: object, *, label: str) -> float:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise ValueError(f"checkpoint history {label} must be a number")
    return float(value)


def _sample_config(config: TrainConfig) -> SampleConfig:
    return SampleConfig(
        n_interior=config.n_interior,
        n_ic=config.n_ic,
        n_bc=config.n_bc,
        seed=config.seed,
        method=config.method,
    )


def _loss_terms(
    model: torch.nn.Module,
    batch: CollocationBatch,
    spec: EquationSpec,
    config: TrainConfig,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    coords = torch.tensor(batch.interior)
    pde = mean_squared_residual(model, coords, spec)
    penalty = soft_penalty(model, batch, spec)
    boundary = penalty.boundary()
    total = config.w_pde * pde + config.w_ic * penalty.initial + config.w_bc * boundary
    return total, pde, penalty.initial, boundary


def _scalar(value: torch.Tensor) -> float:
    number = float(value.detach())
    if not math.isfinite(number):
        raise ValueError("training loss must be finite")
    return number
