"""Score a model on a held-out batch.

:func:`evaluate_model` accepts any callable with the residual signature,
including an in-memory :class:`~pinnforge.models.MLP`.
:func:`evaluate_checkpoint` loads a CPU checkpoint and scores the rebuilt
network against the spec stored in that file. Checkpoints written before
the spec field existed use the built-in problem for the equation id.

Interior points use the test RNG stream, not the training stream, so a
shared integer seed does not redraw the training rows. Field error uses
the harmonic closed form or a Poisson field that matches the boundary
data. Burgers has no field reference. Held-out initial-condition and
boundary errors are reported beside the residual even when the residual
is zero.
"""

from __future__ import annotations

import math
from collections.abc import Iterator
from contextlib import contextmanager
from dataclasses import replace
from pathlib import Path

import numpy as np

from pinnforge.equations.base import EquationSpec
from pinnforge.equations.burgers import Burgers1DSpec
from pinnforge.equations.harmonic import HarmonicOscillatorSpec
from pinnforge.equations.poisson import PoissonToySpec
from pinnforge.evaluation.record import EvalResult, ResidualHistogram
from pinnforge.losses.conditions import (
    boundary_coverage_gaps,
    dirichlet_boundary_loss,
    initial_condition_loss,
    neumann_boundary_loss,
    periodic_boundary_loss,
)
from pinnforge.ml_import import require_torch
from pinnforge.reference import (
    ReferenceUnavailable,
    burgers_reference,
    displacement,
    poisson_reference,
)
from pinnforge.residuals import residual_from_field
from pinnforge.residuals.field import evaluate_field, prepare_coords
from pinnforge.sampling import (
    SampleConfig,
    resolve_equation_id,
    sample_equation,
    stream_generator,
    stream_identity,
)
from pinnforge.sampling.draw import boundary_faces
from pinnforge.specs.eval import EvalConfig
from pinnforge.training import load_checkpoint

torch = require_torch()


def evaluate_model(
    model: object,
    spec: EquationSpec,
    config: EvalConfig | None = None,
) -> EvalResult:
    """Score ``model`` on a fresh interior batch for ``spec``.

    The batch uses ``config`` (or :class:`EvalConfig` defaults) and the
    test RNG stream. Mean and max absolute residual are taken on the
    interior points. ``l2``, ``relative_l2``, and ``max_abs_error``
    compare the predicted field with the reference on those points when
    a matched reference exists. ``ic_error`` and ``bc_error`` use
    held-out condition rows from the same stream.

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
    n_ic, n_bc = _held_out_counts(spec, config)
    sample = SampleConfig(
        n_interior=config.n_interior,
        n_ic=n_ic,
        n_bc=n_bc,
        seed=config.seed,
        method=config.method,
    )
    batch = sample_equation(spec, sample, rng=stream_generator(config.seed, "test"))
    if n_bc > 0:
        gaps = boundary_coverage_gaps(batch, spec)
        if gaps:
            names = ", ".join(gaps)
            raise ValueError(
                f"evaluation did not sample every prescribed boundary face ({names}). "
                "Increase n_bc or leave it unset."
            )
    coords = batch.interior
    with _eval_mode(model):
        field, prepared = _forward(model, coords)
        residual = residual_from_field(field, prepared, spec)
        ic_error, bc_error, bc_errors = _condition_errors(model, batch, spec)
    residual_np = residual.detach().cpu().numpy().reshape(-1)
    predicted = field.detach().cpu().numpy().reshape(-1)
    if not np.isfinite(residual_np).all() or not np.isfinite(predicted).all():
        raise ValueError("evaluation values must be finite")
    reference_name, l2, relative_l2, max_abs_error = _field_scores(spec, coords, predicted)
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
        max_abs_error=max_abs_error,
        ic_error=ic_error,
        bc_error=bc_error,
        bc_errors=bc_errors,
        rng=stream_identity(config.seed, "test"),
    )


def evaluate_checkpoint(
    path: str | Path,
    config: EvalConfig | None = None,
    *,
    equation: str | None = None,
) -> EvalResult:
    """Load ``path`` and score the checkpointed network.

    ``equation``, when set, is a registry id or a CLI alias. It must
    match the equation stored in the checkpoint. The scored spec is the
    one stored with the checkpoint, or the built-in spec when an older
    file omitted it. The returned ``checkpoint`` field is ``path`` in
    POSIX form, after the file loads.

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
    result = evaluate_model(loaded.model, loaded.spec, config)
    return replace(result, checkpoint=Path(path).as_posix())


def _forward(model: object, coords: np.ndarray) -> tuple[torch.Tensor, torch.Tensor]:
    tensor = torch.tensor(coords, dtype=torch.float64)
    prepared = prepare_coords(tensor, model)
    return evaluate_field(model, prepared), prepared


@contextmanager
def _eval_mode(model: object) -> Iterator[None]:
    module = model if isinstance(model, torch.nn.Module) else None
    training = bool(module.training) if module is not None else False
    if module is not None:
        module.eval()
    try:
        yield
    finally:
        if module is not None and training:
            module.train()


def _held_out_counts(spec: EquationSpec, config: EvalConfig) -> tuple[int, int]:
    if config.n_ic is None:
        n_ic = 0 if isinstance(spec, PoissonToySpec) else 8
    else:
        n_ic = config.n_ic
    conditions = tuple(getattr(spec, "boundary_conditions", ()))
    if config.n_bc is None:
        n_bc = len(boundary_faces(conditions))
    else:
        n_bc = config.n_bc
    return n_ic, n_bc


def _condition_errors(
    model: object,
    batch: object,
    spec: EquationSpec,
) -> tuple[float | None, float | None, dict[str, float] | None]:
    from pinnforge.sampling import CollocationBatch

    if not isinstance(batch, CollocationBatch):
        raise TypeError("batch must be a CollocationBatch")
    has_ic = not isinstance(spec, PoissonToySpec)
    if has_ic and batch.ic.shape[0] > 0:
        ic_error = _sqrt_penalty(initial_condition_loss(model, batch, spec))
    else:
        ic_error = None
    conditions = tuple(getattr(spec, "boundary_conditions", ()))
    kinds = {condition.kind for condition in conditions}
    if not kinds or batch.bc.shape[0] == 0:
        return ic_error, None, None
    detail: dict[str, float] = {}
    total: torch.Tensor | None = None
    for kind, penalty in (
        ("dirichlet", dirichlet_boundary_loss),
        ("neumann", neumann_boundary_loss),
        ("periodic", periodic_boundary_loss),
    ):
        if kind not in kinds:
            continue
        term = penalty(model, batch, spec)
        detail[kind] = _sqrt_penalty(term)
        total = term if total is None else total + term
    if total is None:
        return ic_error, None, None
    return ic_error, _sqrt_penalty(total), detail


def _sqrt_penalty(penalty: torch.Tensor) -> float:
    value = float(torch.sqrt(penalty.detach()).cpu())
    if not math.isfinite(value):
        raise ValueError("condition error must be finite")
    return value


def _field_scores(
    spec: EquationSpec,
    coords: np.ndarray,
    predicted: np.ndarray,
) -> tuple[str, float | None, float | None, float | None]:
    reference = _reference_values(spec, coords)
    if reference is None:
        return "unavailable", None, None, None
    if reference.shape != predicted.shape or not np.isfinite(reference).all():
        raise ValueError("reference values must be finite and match the prediction")
    difference = predicted - reference
    l2 = float(np.sqrt(np.mean(np.square(difference))))
    max_abs = float(np.max(np.abs(difference)))
    denom = float(np.sqrt(np.mean(np.square(reference))))
    if not np.isfinite(l2) or not np.isfinite(max_abs):
        raise ValueError("field error must be finite")
    if denom == 0.0:
        return "analytical", l2, None, max_abs
    relative = l2 / denom
    if not np.isfinite(relative):
        raise ValueError("relative l2 must be finite")
    return "analytical", l2, relative, max_abs


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
        try:
            values = poisson_reference(spec, *columns)
        except ReferenceUnavailable:
            return None
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
