"""Reference hook for the Poisson toy.

Manufactured solutions exist for the named sources (for example
``sin_pi_x`` on a 1D interval with matching Dirichlet data). Day 1 does
not evaluate them. The schema still validates. A later eval day can fill
this function in without changing the signature.

``coords`` is one array in 1D and two arrays in 2D, in the same order as
:meth:`PoissonToySpec.collocation_domain` (``x``, then ``y``).
"""

from __future__ import annotations

from pinnforge.equations.poisson import PoissonToySpec


def reference_solution(spec: PoissonToySpec, *coords: object) -> object:
    """Return the field u sampled at ``coords``.

    Raises:
        TypeError: ``spec`` is not a :class:`PoissonToySpec`.
        ValueError: the number of coordinate arrays does not match
            ``spec.dimensions``.
        NotImplementedError: the spec and the coordinate rank are valid,
            and Day 1 has no manufactured evaluation yet.
    """

    if not isinstance(spec, PoissonToySpec):
        raise TypeError("spec must be a PoissonToySpec")
    if len(coords) != spec.dimensions:
        raise ValueError(
            f"poisson_toy in {spec.dimensions}D expects {spec.dimensions} coordinate arrays, "
            f"got {len(coords)}"
        )
    raise NotImplementedError(
        "poisson_toy has no reference solution on Day 1. "
        "The schema validates the problem; a manufactured reference is a later eval day."
    )
