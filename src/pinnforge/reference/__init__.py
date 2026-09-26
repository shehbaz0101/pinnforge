"""Analytical and manufactured reference hooks.

The harmonic oscillator is a closed form. Poisson named sources have
manufactured fields. Burgers validates the spec and then raises
``NotImplementedError``.
"""

from pinnforge.reference.burgers import reference_solution as burgers_reference
from pinnforge.reference.harmonic import displacement, velocity
from pinnforge.reference.poisson import reference_solution as poisson_reference

__all__ = [
    "burgers_reference",
    "displacement",
    "poisson_reference",
    "velocity",
]
