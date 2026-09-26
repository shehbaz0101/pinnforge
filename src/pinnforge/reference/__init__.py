"""Analytical and fixed reference hooks.

The harmonic oscillator is implemented. Burgers and Poisson validate their
specs and then raise ``NotImplementedError``.
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
