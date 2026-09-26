"""Soft initial-condition and Dirichlet penalties.

These are mean-squared differences between the network field and the
values stored on the Day 1 spec. Rows come from a
:class:`~pinnforge.sampling.CollocationBatch`: ``ic`` for the initial
condition, and ``bc`` rows whose face is Dirichlet. Neumann and periodic
faces are not penalized.

Day 4 adds these terms to the residual loss and runs the optimizer. This
module only returns the penalties.
"""

from __future__ import annotations

import math
from dataclasses import dataclass

from pinnforge.equations.base import BoundaryCondition, EquationSpec
from pinnforge.equations.burgers import Burgers1DSpec
from pinnforge.equations.harmonic import HarmonicOscillatorSpec
from pinnforge.equations.poisson import PoissonToySpec
from pinnforge.ml_import import require_torch
from pinnforge.residuals.field import (
    derivative_wrt,
    dtype_device,
    evaluate_field,
    prepare_coords,
    zero_scalar,
)
from pinnforge.sampling.batch import CollocationBatch

torch = require_torch()


@dataclass(frozen=True, slots=True)
class SoftPenalty:
    """Unweighted initial and Dirichlet penalties.

    Both tensors are 0-d. :meth:`total` adds them with weight 1 each.
    Day 4 chooses training weights and does not have to use :meth:`total`.
    Poisson has no initial condition, so ``initial`` is zero. A batch with
    no Dirichlet rows has ``dirichlet`` zero.
    """

    initial: torch.Tensor
    dirichlet: torch.Tensor

    def total(self) -> torch.Tensor:
        """Sum of the two penalties."""

        return self.initial + self.dirichlet


def soft_penalty(model: object, batch: CollocationBatch, spec: EquationSpec) -> SoftPenalty:
    """Initial-condition and Dirichlet penalties for one batch.

    ``batch`` selects the rows. Prescribed numbers come from ``spec``:
    harmonic ``(u, du_dt)``, a Burgers profile name, and Dirichlet
    ``value`` fields. The two penalties are separate so a later trainer
    can weight them.
    """

    return SoftPenalty(
        initial=initial_condition_loss(model, batch, spec),
        dirichlet=dirichlet_boundary_loss(model, batch, spec),
    )


def initial_condition_loss(
    model: object,
    batch: CollocationBatch,
    spec: EquationSpec,
) -> torch.Tensor:
    """Mean-squared penalty on the initial condition.

    The harmonic term is ``mse(u - u0) + mse(u_t - v0)`` at the ``ic``
    rows (the sampler places those rows at ``t = t0``). Burgers compares
    ``u`` with the named profile. Poisson returns zero. An empty ``ic``
    block returns zero.
    """

    _require_pair(batch, spec)
    if isinstance(spec, PoissonToySpec) or batch.ic.shape[0] == 0:
        return zero_scalar(model)
    coords = _tensor(batch.ic, model)
    prepared = prepare_coords(coords, model)
    field = evaluate_field(model, prepared)
    if isinstance(spec, HarmonicOscillatorSpec):
        return _harmonic_initial(field, prepared, spec)
    if isinstance(spec, Burgers1DSpec):
        return _burgers_initial(field, prepared, batch, spec)
    raise TypeError("spec must be a HarmonicOscillatorSpec, Burgers1DSpec, or PoissonToySpec")


def dirichlet_boundary_loss(
    model: object,
    batch: CollocationBatch,
    spec: EquationSpec,
) -> torch.Tensor:
    """Mean-squared penalty on Dirichlet faces.

    Each Dirichlet component contributes its own mean square, and those
    means are added. ``u`` is the field. ``du_dt`` (harmonic boundary
    data) is the time derivative. Neumann and periodic rows are skipped.
    No Dirichlet rows yields zero.
    """

    _require_pair(batch, spec)
    grouped = _dirichlet_rows(batch, spec)
    if not grouped or batch.bc.shape[0] == 0:
        return zero_scalar(model)
    prepared = prepare_coords(_tensor(batch.bc, model), model)
    field = evaluate_field(model, prepared)
    total = zero_scalar(model)
    if "u" in grouped:
        total = total + _indexed_mse(field, grouped["u"])
    if "du_dt" in grouped:
        time_axis = _axis_index(spec, "t")
        speed = derivative_wrt(field, prepared)[:, time_axis : time_axis + 1]
        total = total + _indexed_mse(speed, grouped["du_dt"])
    unknown = set(grouped) - {"u", "du_dt"}
    if unknown:
        names = ", ".join(sorted(unknown))
        raise ValueError(f"unsupported Dirichlet component: {names}")
    return total


def _harmonic_initial(
    field: torch.Tensor,
    coords: torch.Tensor,
    spec: HarmonicOscillatorSpec,
) -> torch.Tensor:
    components = spec.initial_condition.components
    target_u = torch.full_like(field, float(components["u"]))
    speed = derivative_wrt(field, coords)
    target_v = torch.full_like(speed, float(components["du_dt"]))
    return _mse(field, target_u) + _mse(speed, target_v)


def _burgers_initial(
    field: torch.Tensor,
    coords: torch.Tensor,
    batch: CollocationBatch,
    spec: Burgers1DSpec,
) -> torch.Tensor:
    x = coords[:, batch.axis_names.index("x") : batch.axis_names.index("x") + 1]
    return _mse(field, _burgers_profile(spec, x))


def _burgers_profile(spec: Burgers1DSpec, x: torch.Tensor) -> torch.Tensor:
    profile = spec.initial_condition.profile
    if profile == "negative_sin_pi_x":
        return -torch.sin(x * math.pi)
    if profile == "sin_pi_x":
        return torch.sin(x * math.pi)
    raise ValueError(f"unknown Burgers profile {profile!r}")


def _dirichlet_rows(
    batch: CollocationBatch,
    spec: EquationSpec,
) -> dict[str, list[tuple[int, float]]]:
    lookup = _dirichlet_lookup(spec)
    grouped: dict[str, list[tuple[int, float]]] = {}
    rows = zip(range(len(batch.bc_variable)), batch.bc_variable, batch.bc_side, strict=True)
    for index, variable, side in rows:
        for component, value in lookup.get((variable, side), ()):
            grouped.setdefault(component, []).append((index, value))
    return grouped


def _dirichlet_lookup(spec: EquationSpec) -> dict[tuple[str, str], tuple[tuple[str, float], ...]]:
    conditions = getattr(spec, "boundary_conditions", None)
    if conditions is None:
        raise TypeError(f"{type(spec).__name__} has no boundary conditions list")
    found: dict[tuple[str, str], list[tuple[str, float]]] = {}
    for condition in conditions:
        if not isinstance(condition, BoundaryCondition):
            raise TypeError("boundary conditions must be BoundaryCondition descriptors")
        if condition.kind != "dirichlet":
            continue
        if condition.side is None or condition.value is None:
            raise ValueError("Dirichlet boundary conditions require side and value")
        found.setdefault((condition.variable, condition.side), []).append(
            (condition.component, float(condition.value))
        )
    return {key: tuple(values) for key, values in found.items()}


def _indexed_mse(prediction: torch.Tensor, rows: list[tuple[int, float]]) -> torch.Tensor:
    indices = torch.tensor([index for index, _value in rows], dtype=torch.long, device=prediction.device)
    target = torch.tensor(
        [value for _index, value in rows],
        dtype=prediction.dtype,
        device=prediction.device,
    ).unsqueeze(-1)
    return _mse(prediction.index_select(0, indices), target)


def _mse(prediction: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
    return torch.mean((prediction - target) ** 2)


def _tensor(values: object, model: object) -> torch.Tensor:
    dtype, device = dtype_device(model)
    return torch.tensor(values, dtype=dtype, device=device)


def _axis_index(spec: EquationSpec, name: str) -> int:
    names = [axis.name for axis in spec.collocation_domain().axes]
    try:
        return names.index(name)
    except ValueError:
        raise ValueError(f"{spec.equation_id} has no axis {name!r}") from None


def _require_pair(batch: CollocationBatch, spec: EquationSpec) -> None:
    if not isinstance(batch, CollocationBatch):
        raise TypeError("batch must be a CollocationBatch")
    if not isinstance(spec, (HarmonicOscillatorSpec, Burgers1DSpec, PoissonToySpec)):
        raise TypeError("spec must be a HarmonicOscillatorSpec, Burgers1DSpec, or PoissonToySpec")
    if batch.equation_id != spec.equation_id:
        raise ValueError(
            f"batch equation {batch.equation_id!r} does not match spec {spec.equation_id!r}"
        )
    expected = tuple(axis.name for axis in spec.collocation_domain().axes)
    if batch.axis_names != expected:
        raise ValueError(f"batch axes {batch.axis_names} do not match spec axes {expected}")
