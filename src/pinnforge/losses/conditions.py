"""Soft initial-condition and boundary penalties.

These are mean-squared differences between the network field and the
values stored on the spec. Rows come from a
:class:`~pinnforge.sampling.CollocationBatch`: ``ic`` for the initial
condition, and ``bc`` rows for Dirichlet and Neumann faces.

Periodic conditions do not zip the independently sampled min and max
faces. Those faces do not share the other coordinates. Each boundary
row is mirrored onto both endpoints of its periodic axis, and the
penalty matches the field and its derivative with respect to that axis
at that same free coordinate (the same time, for Burgers).

Neumann values are outward normal derivatives. On side ``max`` that is
``∂/∂variable``. On side ``min`` it is ``-∂/∂variable``. A condition
whose kind or component this module cannot enforce raises
``ValueError`` instead of being omitted.
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
    """Unweighted initial and boundary penalties.

    All four tensors are 0-d. :meth:`boundary` adds the Dirichlet,
    Neumann, and periodic terms. :meth:`total` adds that sum to the
    initial-condition term. Poisson has no initial condition, so
    ``initial`` is zero. A term is zero when the spec prescribes none
    of that kind or the batch has no rows for it.
    """

    initial: torch.Tensor
    dirichlet: torch.Tensor
    neumann: torch.Tensor
    periodic: torch.Tensor

    def boundary(self) -> torch.Tensor:
        """Sum of the Dirichlet, Neumann, and periodic penalties."""

        return self.dirichlet + self.neumann + self.periodic

    def total(self) -> torch.Tensor:
        """Sum of the initial-condition penalty and :meth:`boundary`."""

        return self.initial + self.boundary()


def soft_penalty(model: object, batch: CollocationBatch, spec: EquationSpec) -> SoftPenalty:
    """Initial-condition and boundary penalties for one batch.

    ``batch`` selects the rows. Prescribed numbers come from ``spec``.
    The four penalties are separate so a trainer can weight the initial
    condition apart from the boundary terms.

    Raises:
        ValueError: a boundary condition kind or component is not
            enforced by this module.
    """

    _require_supported_conditions(spec)
    return SoftPenalty(
        initial=initial_condition_loss(model, batch, spec),
        dirichlet=dirichlet_boundary_loss(model, batch, spec),
        neumann=neumann_boundary_loss(model, batch, spec),
        periodic=periodic_boundary_loss(model, batch, spec),
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
    data) is the time derivative. Neumann and periodic rows are not
    scored here. No Dirichlet rows yields zero.
    """

    _require_pair(batch, spec)
    _require_supported_conditions(spec)
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


def neumann_boundary_loss(
    model: object,
    batch: CollocationBatch,
    spec: EquationSpec,
) -> torch.Tensor:
    """Mean-squared outward-normal penalty on Neumann faces.

    The stored value is the outward flux. Side ``max`` compares
    ``∂component/∂variable`` with that value. Side ``min`` compares
    ``-∂component/∂variable`` with it. Component ``u`` is the field.
    Component ``du_dt`` is the time derivative, so the flux is a second
    derivative. No Neumann rows yields zero.
    """

    _require_pair(batch, spec)
    _require_supported_conditions(spec)
    grouped = _neumann_groups(batch, spec)
    if not grouped or batch.bc.shape[0] == 0:
        return zero_scalar(model)
    prepared = prepare_coords(_tensor(batch.bc, model), model)
    field = evaluate_field(model, prepared)
    grads = derivative_wrt(field, prepared)
    total = zero_scalar(model)
    for component, rows in grouped.items():
        slope = _component_jacobian(component, grads, prepared, spec)
        total = total + _neumann_mse(slope, rows, spec)
    return total


def periodic_boundary_loss(
    model: object,
    batch: CollocationBatch,
    spec: EquationSpec,
) -> torch.Tensor:
    """Match the field and its axial derivative at both periodic ends.

    The sampler draws the min face and the max face independently, so
    those rows are not pairs. Every boundary row whose variable is the
    periodic axis is copied onto both endpoints while the other
    coordinates stay fixed. For Burgers that other coordinate is time,
    so the match is at one time. The penalty is the mean square of the
    value gap plus the mean square of the derivative gap. Component
    ``u`` uses the field and ``∂u/∂variable``. Component ``du_dt`` uses
    ``u_t`` and its derivative along the periodic axis.
    """

    _require_pair(batch, spec)
    _require_supported_conditions(spec)
    conditions = [item for item in _boundary_conditions(spec) if item.kind == "periodic"]
    if not conditions or batch.bc.shape[0] == 0:
        return zero_scalar(model)
    coords = _tensor(batch.bc, model)
    bounds = {axis.name: axis.bounds for axis in spec.collocation_domain().axes}
    total = zero_scalar(model)
    for condition in conditions:
        indices = [
            index
            for index, variable in enumerate(batch.bc_variable)
            if variable == condition.variable
        ]
        if not indices:
            continue
        chosen = coords.index_select(
            0, torch.tensor(indices, dtype=torch.long, device=coords.device)
        )
        axis = _axis_index(spec, condition.variable)
        left = chosen.clone()
        right = chosen.clone()
        left[:, axis] = float(bounds[condition.variable].lower)
        right[:, axis] = float(bounds[condition.variable].upper)
        left_value, left_slope = _endpoint_state(model, left, spec, condition.component, condition.variable)
        right_value, right_slope = _endpoint_state(
            model, right, spec, condition.component, condition.variable
        )
        total = total + _mse(left_value, right_value) + _mse(left_slope, right_slope)
    return total


def boundary_coverage_gaps(batch: CollocationBatch, spec: EquationSpec) -> tuple[str, ...]:
    """Names of prescribed faces that ``batch`` does not sample.

    A periodic condition needs one row on its variable. Dirichlet and
    Neumann each need one row on that variable and side. An empty tuple
    means every prescribed face has a sample. This does not score the
    field.
    """

    _require_pair(batch, spec)
    _require_supported_conditions(spec)
    gaps: list[str] = []
    for condition in _boundary_conditions(spec):
        if condition.kind == "periodic":
            covered = any(variable == condition.variable for variable in batch.bc_variable)
            if not covered:
                gaps.append(f"periodic {condition.variable}")
            continue
        covered = any(
            variable == condition.variable and side == condition.side
            for variable, side in zip(batch.bc_variable, batch.bc_side, strict=True)
        )
        if not covered:
            gaps.append(f"{condition.kind} {condition.variable} {condition.side}")
    return tuple(gaps)


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


def _boundary_conditions(spec: EquationSpec) -> tuple[BoundaryCondition, ...]:
    conditions = getattr(spec, "boundary_conditions", None)
    if conditions is None:
        raise TypeError(f"{type(spec).__name__} has no boundary conditions list")
    found = tuple(conditions)
    for condition in found:
        if not isinstance(condition, BoundaryCondition):
            raise TypeError("boundary conditions must be BoundaryCondition descriptors")
    return found


def _require_supported_conditions(spec: EquationSpec) -> None:
    names = _axis_names(spec)
    for condition in _boundary_conditions(spec):
        if condition.kind not in {"dirichlet", "neumann", "periodic"}:
            raise ValueError(
                f"unsupported boundary condition {condition.kind!r}; "
                "prescribed physics is not skipped"
            )
        if condition.component not in {"u", "du_dt"}:
            raise ValueError(f"unsupported boundary component {condition.component!r}")
        if condition.component == "du_dt" and "t" not in names:
            raise ValueError("du_dt boundary data requires a time axis")


def _neumann_groups(
    batch: CollocationBatch,
    spec: EquationSpec,
) -> dict[str, list[tuple[int, float, str, str]]]:
    lookup: dict[tuple[str, str], list[tuple[str, float]]] = {}
    for condition in _boundary_conditions(spec):
        if condition.kind != "neumann":
            continue
        if condition.side is None or condition.value is None:
            raise ValueError("Neumann boundary conditions require side and value")
        lookup.setdefault((condition.variable, condition.side), []).append(
            (condition.component, float(condition.value))
        )
    grouped: dict[str, list[tuple[int, float, str, str]]] = {}
    rows = zip(range(len(batch.bc_variable)), batch.bc_variable, batch.bc_side, strict=True)
    for index, variable, side in rows:
        for component, value in lookup.get((variable, side), ()):
            grouped.setdefault(component, []).append((index, value, variable, side))
    return grouped


def _component_jacobian(
    component: str,
    grads: torch.Tensor,
    prepared: torch.Tensor,
    spec: EquationSpec,
) -> torch.Tensor:
    """Jacobian of the boundary component. Shape ``(n, d)``."""

    if component == "u":
        return grads
    if component == "du_dt":
        time_axis = _axis_index(spec, "t")
        speed = grads[:, time_axis : time_axis + 1]
        return derivative_wrt(speed, prepared)
    raise ValueError(f"unsupported boundary component {component!r}")


def _neumann_mse(
    slope: torch.Tensor,
    rows: list[tuple[int, float, str, str]],
    spec: EquationSpec,
) -> torch.Tensor:
    outward = []
    targets: list[float] = []
    for index, value, variable, side in rows:
        axis = _axis_index(spec, variable)
        raw = slope[index, axis]
        outward.append(raw if side == "max" else -raw)
        targets.append(value)
    prediction = torch.stack(outward).unsqueeze(-1)
    target = torch.tensor(targets, dtype=prediction.dtype, device=prediction.device).unsqueeze(-1)
    return _mse(prediction, target)


def _endpoint_state(
    model: object,
    coords: torch.Tensor,
    spec: EquationSpec,
    component: str,
    variable: str,
) -> tuple[torch.Tensor, torch.Tensor]:
    prepared = prepare_coords(coords, model)
    field = evaluate_field(model, prepared)
    grads = derivative_wrt(field, prepared)
    axis = _axis_index(spec, variable)
    if component == "u":
        return field, grads[:, axis : axis + 1]
    if component == "du_dt":
        time_axis = _axis_index(spec, "t")
        speed = grads[:, time_axis : time_axis + 1]
        second = derivative_wrt(speed, prepared)
        return speed, second[:, axis : axis + 1]
    raise ValueError(f"unsupported periodic component {component!r}")


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


def _axis_names(spec: EquationSpec) -> list[str]:
    return [axis.name for axis in spec.collocation_domain().axes]


def _axis_index(spec: EquationSpec, name: str) -> int:
    names = _axis_names(spec)
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
