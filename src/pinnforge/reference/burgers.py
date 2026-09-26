"""Reference hook for viscous Burgers.

Special initial data have a Hopf-Cole formula. This package does not
evaluate it. The schema still validates. Day 5 scores a Burgers model
with residual metrics only. A later day can return a fixed grid or that
quadrature without changing this signature.
"""

from __future__ import annotations

from pinnforge.equations.burgers import Burgers1DSpec


def reference_solution(spec: Burgers1DSpec, x: object, t: object) -> object:
    """Return u(x, t) for a validated Burgers problem.

    ``x`` and ``t`` are the sample coordinates a later implementation will
    accept as numpy arrays. They are not read yet.

    Raises:
        TypeError: ``spec`` is not a :class:`Burgers1DSpec`.
        NotImplementedError: always, after the type check. Evaluation
            then reports residual metrics only.
    """

    if not isinstance(spec, Burgers1DSpec):
        raise TypeError("spec must be a Burgers1DSpec")
    del x, t
    raise NotImplementedError(
        "burgers_1d has no reference solution. "
        "Eval reports residual mean and max absolute residual only."
    )
