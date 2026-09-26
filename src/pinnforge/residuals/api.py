"""Model residual for a Day 1 equation spec.

:func:`residual` evaluates ``model`` at collocation coordinates and
returns the pointwise PDE residual. The closed-form helpers in the
equation modules are for a tensor that already depends on those
coordinates. Day 4 sums a loss of this residual with the soft penalties
in :mod:`pinnforge.losses`. This module does not train.
"""

from __future__ import annotations

from pinnforge.equations.base import EquationSpec
from pinnforge.equations.burgers import Burgers1DSpec
from pinnforge.equations.harmonic import HarmonicOscillatorSpec
from pinnforge.equations.poisson import PoissonToySpec
from pinnforge.ml_import import require_torch
from pinnforge.models.mlp import input_features
from pinnforge.residuals.burgers import burgers_residual_from_field
from pinnforge.residuals.field import evaluate_field, prepare_coords
from pinnforge.residuals.harmonic import harmonic_residual_from_field
from pinnforge.residuals.poisson import poisson_residual_from_field

torch = require_torch()


def residual(model: object, coords: torch.Tensor, spec: EquationSpec) -> torch.Tensor:
    """Pointwise residual of ``spec`` at ``coords``.

    ``coords`` has shape ``(n, d)`` in collocation-domain order. ``model``
    maps that matrix to the scalar field ``u`` (shape ``(n,)`` or
    ``(n, 1)``). The result has shape ``(n, 1)``.

    The operator is ü + ω² u, u_t + u u_x − ν u_xx, or −Δu − f, matching
    the Day 1 specs. Derivatives are taken with ``create_graph=True``.

    Raises:
        TypeError: ``spec`` is not a Day 1 equation spec, or ``model``
            does not return a tensor.
        ValueError: the coordinate width does not match the spec, or the
            model declares a different ``in_features``.
    """

    _require_spec(spec)
    prepared = prepare_coords(coords, model)
    _check_width(model, prepared, spec)
    if prepared.shape[0] == 0:
        return prepared.new_empty((0, 1))
    field = evaluate_field(model, prepared)
    return residual_from_field(field, prepared, spec)


def residual_from_field(
    u: torch.Tensor,
    coords: torch.Tensor,
    spec: EquationSpec,
) -> torch.Tensor:
    """Residual of a field tensor that already depends on ``coords``.

    Use this when ``u`` is a closed form built from ``coords`` with grad
    enabled. :func:`residual` is the model entry point; it prepares a
    fresh coordinate leaf and would drop that graph.
    """

    if isinstance(spec, HarmonicOscillatorSpec):
        return harmonic_residual_from_field(u, coords, spec)
    if isinstance(spec, Burgers1DSpec):
        return burgers_residual_from_field(u, coords, spec)
    if isinstance(spec, PoissonToySpec):
        return poisson_residual_from_field(u, coords, spec)
    raise TypeError("spec must be a HarmonicOscillatorSpec, Burgers1DSpec, or PoissonToySpec")


def mean_squared_residual(model: object, coords: torch.Tensor, spec: EquationSpec) -> torch.Tensor:
    """Mean of the squared residual. A 0-d tensor.

    Raises:
        ValueError: ``coords`` has no rows.
    """

    values = residual(model, coords, spec)
    if values.numel() == 0:
        raise ValueError("residual MSE needs at least one collocation point")
    return torch.mean(values**2)


def harmonic_residual(model: object, coords: torch.Tensor, spec: HarmonicOscillatorSpec) -> torch.Tensor:
    """Oscillator residual. ``spec`` must be a :class:`HarmonicOscillatorSpec`."""

    if not isinstance(spec, HarmonicOscillatorSpec):
        raise TypeError("spec must be a HarmonicOscillatorSpec")
    return residual(model, coords, spec)


def burgers_residual(model: object, coords: torch.Tensor, spec: Burgers1DSpec) -> torch.Tensor:
    """Burgers residual. ``spec`` must be a :class:`Burgers1DSpec`."""

    if not isinstance(spec, Burgers1DSpec):
        raise TypeError("spec must be a Burgers1DSpec")
    return residual(model, coords, spec)


def poisson_residual(model: object, coords: torch.Tensor, spec: PoissonToySpec) -> torch.Tensor:
    """Poisson residual. ``spec`` must be a :class:`PoissonToySpec`."""

    if not isinstance(spec, PoissonToySpec):
        raise TypeError("spec must be a PoissonToySpec")
    return residual(model, coords, spec)


def _require_spec(spec: EquationSpec) -> None:
    if not isinstance(spec, (HarmonicOscillatorSpec, Burgers1DSpec, PoissonToySpec)):
        raise TypeError("spec must be a HarmonicOscillatorSpec, Burgers1DSpec, or PoissonToySpec")


def _check_width(model: object, coords: torch.Tensor, spec: EquationSpec) -> None:
    expected = input_features(spec)
    if coords.shape[1] != expected:
        names = ", ".join(axis.name for axis in spec.collocation_domain().axes)
        raise ValueError(
            f"{spec.equation_id} expects coordinates of shape (n, {expected}) "
            f"in order ({names}), got {tuple(coords.shape)}"
        )
    declared = getattr(model, "in_features", None)
    if isinstance(declared, int) and declared != expected:
        raise ValueError(
            f"model in_features is {declared} but {spec.equation_id} has input dimension {expected}"
        )
