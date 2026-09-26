"""Analytical and manufactured reference hooks.

The harmonic oscillator is a closed form. Poisson returns a field only
when that field matches the source and the prescribed boundary data, and
raises :class:`~pinnforge.reference.poisson.ReferenceUnavailable`
otherwise. Burgers validates the spec and then raises
``NotImplementedError``.
"""

from pinnforge.reference.burgers import reference_solution as burgers_reference
from pinnforge.reference.harmonic import displacement, velocity
from pinnforge.reference.poisson import ReferenceUnavailable
from pinnforge.reference.poisson import reference_solution as poisson_reference

__all__ = [
    "ReferenceUnavailable",
    "burgers_reference",
    "displacement",
    "poisson_reference",
    "velocity",
]
