"""Soft IC and Dirichlet penalties.

Importing this package requires torch (the optional ``ml`` extra).
Day 4 wires :func:`soft_penalty` into the training loop.
"""

from pinnforge.losses.conditions import (
    SoftPenalty,
    dirichlet_boundary_loss,
    initial_condition_loss,
    soft_penalty,
)

__all__ = [
    "SoftPenalty",
    "dirichlet_boundary_loss",
    "initial_condition_loss",
    "soft_penalty",
]
