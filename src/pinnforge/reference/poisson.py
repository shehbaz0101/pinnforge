"""Manufactured fields for the Poisson toy.

The residual is −Δu = f for a named source. These fields satisfy that
equation. They are not a fit to arbitrary boundary data:

- ``zero`` is ``u = 0`` in 1D and 2D.
- ``one`` is the particular solution ``u = -x² / 2``. Then −u_xx = 1 and
  the other second derivatives are zero. It does not match homogeneous
  Dirichlet data.
- ``sin_pi_x`` is ``u = sin(π x) / π²``. It is zero at integer ``x``,
  which matches the default 1D Dirichlet ends on ``[0, 1]``.
- ``sin_pi_x_sin_pi_y`` is ``u = sin(π x) sin(π y) / (2 π²)``. It is zero
  on the faces of the unit square, which matches the default 2D
  Dirichlet spec.

``coords`` is one array in 1D and two arrays in 2D, in the same order as
:meth:`PoissonToySpec.collocation_domain` (``x``, then ``y``). The arrays
must share a shape. The result has that shape.
"""

from __future__ import annotations

import math

import numpy as np

from pinnforge.equations.poisson import PoissonSource, PoissonToySpec


def reference_solution(spec: PoissonToySpec, *coords: object) -> np.ndarray:
    """Return the manufactured field u sampled at ``coords``.

    Raises:
        TypeError: ``spec`` is not a :class:`PoissonToySpec`.
        ValueError: the number of coordinate arrays does not match
            ``spec.dimensions``, the arrays have different shapes, or a
            value is not finite.
    """

    if not isinstance(spec, PoissonToySpec):
        raise TypeError("spec must be a PoissonToySpec")
    if len(coords) != spec.dimensions:
        raise ValueError(
            f"poisson_toy in {spec.dimensions}D expects {spec.dimensions} coordinate arrays, "
            f"got {len(coords)}"
        )
    columns = [_coordinate(value, index=index) for index, value in enumerate(coords)]
    shape = columns[0].shape
    for column in columns[1:]:
        if column.shape != shape:
            raise ValueError("poisson coordinate arrays must have the same shape")
    if spec.source is PoissonSource.ZERO:
        return np.zeros_like(columns[0], dtype=np.float64)
    if spec.source is PoissonSource.ONE:
        return -0.5 * np.square(columns[0])
    if spec.source is PoissonSource.SIN_PI_X:
        return np.sin(math.pi * columns[0]) / (math.pi**2)
    if spec.source is PoissonSource.SIN_PI_X_SIN_PI_Y:
        return np.sin(math.pi * columns[0]) * np.sin(math.pi * columns[1]) / (2.0 * math.pi**2)
    raise ValueError(f"unknown Poisson source {spec.source.value!r}")


def _coordinate(value: object, *, index: int) -> np.ndarray:
    array = np.asarray(value, dtype=np.float64)
    if not np.isfinite(array).all():
        raise ValueError(f"poisson coordinate {index} must be finite")
    return array
