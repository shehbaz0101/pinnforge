"""Reference hook for viscous Burgers on the coordinate-PINN path.

Evaluation calls this function and gets ``NotImplementedError``, then
scores Burgers with residual metrics only. Periodic trajectory labels
are produced by :mod:`pinnforge.reference.numerical` and are not returned
here.
"""

from __future__ import annotations

from pinnforge.equations.burgers import Burgers1DSpec


def reference_solution(spec: Burgers1DSpec, x: object, t: object) -> object:
    """Return u(x, t) for a validated Burgers problem.

``x`` and ``t`` are the sample coordinates passed by evaluation. They
are not read. Spectral trajectories are a separate module. A data-only
Fourier neural operator on windows of those trajectories is
:mod:`pinnforge.operator`. This function does not call it.

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
