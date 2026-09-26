"""Autograd residuals for the Day 1 equations.

Importing this package requires torch (the optional ``ml`` extra).
Day 4 consumes :func:`residual` inside a training loop. Nothing here
updates weights.
"""

from pinnforge.residuals.api import (
    burgers_residual,
    harmonic_residual,
    mean_squared_residual,
    poisson_residual,
    residual,
    residual_from_field,
)
from pinnforge.residuals.burgers import burgers_residual_from_field
from pinnforge.residuals.harmonic import harmonic_residual_from_field
from pinnforge.residuals.poisson import poisson_residual_from_field, poisson_source

__all__ = [
    "burgers_residual",
    "burgers_residual_from_field",
    "harmonic_residual",
    "harmonic_residual_from_field",
    "mean_squared_residual",
    "poisson_residual",
    "poisson_residual_from_field",
    "poisson_source",
    "residual",
    "residual_from_field",
]
