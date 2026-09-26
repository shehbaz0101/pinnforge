"""Score a model on an interior batch.

:func:`evaluate_model` accepts any callable with the residual signature,
including an in-memory :class:`~pinnforge.models.MLP`.
:func:`evaluate_checkpoint` loads a Day 4 CPU checkpoint and scores the
rebuilt network against the built-in spec for that equation id. Training
stores the equation id, not a custom spec, so the built-in problem is
the one that was trained.

Field error uses the harmonic closed form or a Poisson manufactured
field. Burgers has no field reference, so only the residual stats are
filled in.
"""

from __future__ import annotations

from dataclasses import replace
from pathlib import Path

import numpy as np

from pinnforge.equations.base import EquationSpec
from pinnforge.equations.burgers import Burgers1DSpec
from pinnforge.equations.harmonic import HarmonicOscillatorSpec
from pinnforge.equations.poisson import PoissonToySpec
from pinnforge.evaluation.config import EvalConfig
from pinnforge.evaluation.record import EvalResult, ResidualHistogram
from pinnforge.ml_import import require_torch
from pinnforge.reference import burgers_reference, displacement, poisson_reference
from pinnforge.residuals import residual_from_field
from pinnforge.residuals.field import evaluate_field, prepare_coords
from pinnforge.sampling import SampleConfig, default_spec, resolve_equation_id, sample_equation
from pinnforge.training import load_checkpoint

torch = require_torch()


def evaluate_model(
    model: object,
    spec: EquationSpec,
    config: EvalConfig | None = None,
) -> EvalResult:
    """Score ``model`` on a fresh interior batch for ``spec``.

    The batch uses ``config`` (or :class:`EvalConfig` defaults) and draws
    no initial or boundary points. Mean and max absolute residual are
    taken on those interior points. ``l2`` and ``relative_l2`` compare
    the predicted field with the reference on the same points when one
    exists.

    The model is set to ``eval`` for the forward pass when it is a
    ``torch.nn.Module``, then restored to the mode it had on entry.

    Raises:
        TypeError: ``spec`` is not a Day 1 equation spec, ``config`` is
            not an :class:`EvalConfig`, or ``model`` is not callable.
        ValueError: the sample does not fit the spec, or a score is not
            finite.
    """

    if config is None:
        config = EvalConfig()
    if not isinstance(config, EvalConfig):
        raise TypeError("config must be an EvalConfig")
    _require_spec(spec)
    if not callable(model):
        raise TypeError("model must be callable")
    sample = SampleConfig(
        n_interior=config.n_interior,
        n_ic=0,
        n_bc=0,
        seed=config.seed,
        method=config.method,
    )
    coords = sample_equation(spec, sample).interior
    field, prepared = _forward(model, coords)
    residual = residual_from_field(field, prepared, spec)
    residual_np = residual.detach().cpu().numpy().reshape(-1)
    predicted = field.detach().cpu().numpy().reshape(-1)
    if not np.isfinite(residual_np).all() or not np.isfinite(predicted).all():
        raise ValueError("evaluation values must be finite")
    reference_name, l2, relative_l2 = _field_scores(spec, coords, predicted)
    absolute = np.abs(residual_np)
    counts, edges = np.histogram(absolute, bins=config.bins)
    return EvalResult(
        equation_id=spec.equation_id,
        n_interior=int(coords.shape[0]),
        seed=config.seed,
        method=config.method,
        bins=config.bins,
        reference=reference_name,
        l2=l2,
        relative_l2=relative_l2,
        residual_mean_abs=float(np.mean(absolute)),
        residual_max_abs=float(np.max(absolute)),
        histogram=ResidualHistogram(
            counts=tuple(int(value) for value in counts),
            edges=tuple(float(value) for value in edges),
        ),
    )


def evaluate_checkpoint(
    path: str | Path,
    config: EvalConfig | None = None,
    *,
    equation: str | None = None,
) -> EvalResult:
    """Load ``path`` and score the checkpointed network.

    ``equation``, when set, is a registry id or a CLI alias. It must
    match the equation stored in the checkpoint. The returned
    ``checkpoint`` field is ``path`` in POSIX form, after the file loads.

    Raises:
        ValueError: the path escapes the working directory, the file is
            missing or not a training checkpoint, or ``equation`` does
            not match.
    """

    expected = None if equation is None else resolve_equation_id(equation)
    try:
        loaded = load_checkpoint(path)
    except FileNotFoundError as exc:
        raise ValueError(f"checkpoint not found: {path}") from exc
    if expected is not None and expected != loaded.config.equation_id:
        raise ValueError(
            f"checkpoint equation is {loaded.config.equation_id}, not {expected}"
        )
    spec = default_spec(loaded.config.equation_id)
    result = evaluate_model(loaded.model, spec, config)
    return replace(result, checkpoint=Path(path).as_posix())


def _forward(model: object, coords: np.ndarray) -> tuple[torch.Tensor, torch.Tensor]:
    tensor = torch.tensor(coords, dtype=torch.float64)
    module = model if isinstance(model, torch.nn.Module) else None
    training = bool(module.training) if module is not None else False
    if module is not None:
        module.eval()
    try:
        prepared = prepare_coords(tensor, model)
        return evaluate_field(model, prepared), prepared
    finally:
        if module is not None and training:
            module.train()


def _field_scores(
    spec: EquationSpec,
    coords: np.ndarray,
    predicted: np.ndarray,
) -> tuple[str, float | None, float | None]:
    reference = _reference_values(spec, coords)
    if reference is None:
        return "unavailable", None, None
    if reference.shape != predicted.shape or not np.isfinite(reference).all():
        raise ValueError("reference values must be finite and match the prediction")
    difference = predicted - reference
    l2 = float(np.sqrt(np.mean(np.square(difference))))
    denom = float(np.sqrt(np.mean(np.square(reference))))
    if not np.isfinite(l2):
        raise ValueError("l2 must be finite")
    if denom == 0.0:
        return "analytical", l2, None
    relative = l2 / denom
    if not np.isfinite(relative):
        raise ValueError("relative l2 must be finite")
    return "analytical", l2, relative


def _reference_values(spec: EquationSpec, coords: np.ndarray) -> np.ndarray | None:
    expected = len(spec.collocation_domain().axes)
    if coords.ndim != 2 or coords.shape[1] != expected:
        names = ", ".join(axis.name for axis in spec.collocation_domain().axes)
        raise ValueError(
            f"{spec.equation_id} expects coordinates of shape (n, {expected}) "
            f"in order ({names}), got {tuple(coords.shape)}"
        )
    if isinstance(spec, HarmonicOscillatorSpec):
        values = displacement(coords[:, 0], spec)
        return np.asarray(values, dtype=np.float64).reshape(-1)
    if isinstance(spec, PoissonToySpec):
        columns = [coords[:, index] for index in range(coords.shape[1])]
        values = poisson_reference(spec, *columns)
        return np.asarray(values, dtype=np.float64).reshape(-1)
    if isinstance(spec, Burgers1DSpec):
        return _burgers_reference(spec, coords)
    raise TypeError("spec must be a HarmonicOscillatorSpec, Burgers1DSpec, or PoissonToySpec")


def _burgers_reference(spec: Burgers1DSpec, coords: np.ndarray) -> np.ndarray | None:
    try:
        values = burgers_reference(spec, coords[:, 0], coords[:, 1])
    except NotImplementedError:
        return None
    return np.asarray(values, dtype=np.float64).reshape(-1)


def _require_spec(spec: EquationSpec) -> None:
    if not isinstance(spec, (HarmonicOscillatorSpec, Burgers1DSpec, PoissonToySpec)):
        raise TypeError("spec must be a HarmonicOscillatorSpec, Burgers1DSpec, or PoissonToySpec")
