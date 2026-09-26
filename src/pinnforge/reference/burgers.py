"""Reference hook for viscous Burgers.

Special initial data have a Hopf-Cole formula. Day 1 does not evaluate it.
The schema still validates. A later eval day can return a fixed grid or
that quadrature without changing this signature.
"""

from __future__ import annotations

from pinnforge.equations.burgers import Burgers1DSpec


def reference_solution(spec: Burgers1DSpec, x: object, t: object) -> object:
    """Return u(x, t) for a validated Burgers problem.

    ``x`` and ``t`` are the sample coordinates a later implementation will
    accept as numpy arrays. They are not read on Day 1.

    Raises:
        TypeError: ``spec`` is not a :class:`Burgers1DSpec`.
        NotImplementedError: always, on Day 1, after the type check.
    """

    if not isinstance(spec, Burgers1DSpec):
        raise TypeError("spec must be a Burgers1DSpec")
    del x, t
    raise NotImplementedError(
        "burgers_1d has no reference solution on Day 1. "
        "The schema validates the problem; a fixed reference is a later eval day."
    )
