"""Fixed-batch Adam loop over a Day 3 residual and soft penalties.

The collocation batch is drawn once. ``SampleConfig.seed`` is
``TrainConfig.seed``, and ``torch.manual_seed`` uses that same seed
before the network is built. Every epoch scores that batch, appends one
``metrics.jsonl`` line, writes a checkpoint, and then takes one Adam
step. Epoch 0 is the initial weights, so the logged losses stay on one
point set and can be compared from the start of the run to the end.

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
from pinnforge.losses import soft_penalty
from pinnforge.ml_import import require_torch
from pinnforge.models import mlp_from_spec
from pinnforge.residuals import mean_squared_residual
from pinnforge.sampling import CollocationBatch, SampleConfig, default_spec, sample_equation
from pinnforge.training.checkpoint import save_checkpoint
from pinnforge.training.config import TrainConfig
from pinnforge.training.metrics import EpochMetrics, append_metrics
from pinnforge.training.paths import resolve_inside_cwd

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


def train_loop(config: TrainConfig) -> TrainResult:
    """Train on one seeded batch and return the final weights.

    The total loss is ``w_pde * MSE(residual) + w_ic * initial + w_bc *
    dirichlet``. ``loss_pde``, ``loss_ic``, and ``loss_bc`` in the
    metrics file are the unweighted mean squares. ``lr`` is the Adam
    learning rate and does not change.

    Checkpoints hold the weights that produced that epoch's line.
    ``checkpoint.pt`` matches the last epoch. Optimizer state is not
    saved.

    Raises:
        TypeError: ``config`` is not a :class:`TrainConfig`.
        ValueError: a path escapes the working directory, or the sample
            counts do not fit the built-in spec.
        RuntimeError: the loss does not depend on the model parameters.
    """

    if not isinstance(config, TrainConfig):
        raise TypeError("config must be a TrainConfig")
    log_path = resolve_inside_cwd(config.log_path, suffix=".jsonl")
    checkpoint_dir = resolve_inside_cwd(config.checkpoint_dir)
    if checkpoint_dir.exists() and not checkpoint_dir.is_dir():
        raise ValueError("checkpoint dir must be a directory")
    torch.manual_seed(config.seed)
    spec = default_spec(config.equation_id)
    batch = sample_equation(spec, _sample_config(config))
    model = mlp_from_spec(spec, config.hidden_widths, activation=config.activation)
    model.to(torch.device(config.device))
    model.train()
    optimizer = torch.optim.Adam(model.parameters(), lr=config.lr)
    log_path.parent.mkdir(parents=True, exist_ok=True)
    log_path.write_text("", encoding="utf-8")
    history: list[EpochMetrics] = []
    checkpoint_path: Path | None = None
    with torch.enable_grad():
        for epoch in range(config.epochs + 1):
            total, pde, initial, dirichlet = _loss_terms(model, batch, spec, config)
            row = EpochMetrics(
                epoch=epoch,
                loss=_scalar(total),
                loss_pde=_scalar(pde),
                loss_ic=_scalar(initial),
                loss_bc=_scalar(dirichlet),
                lr=float(optimizer.param_groups[0]["lr"]),
            )
            append_metrics(log_path, row)
            history.append(row)
            checkpoint_path = save_checkpoint(model, config, epoch, config.checkpoint_dir)
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
    total = config.w_pde * pde + config.w_ic * penalty.initial + config.w_bc * penalty.dirichlet
    return total, pde, penalty.initial, penalty.dirichlet


def _scalar(value: torch.Tensor) -> float:
    number = float(value.detach())
    if not math.isfinite(number):
        raise ValueError("training loss must be finite")
    return number
