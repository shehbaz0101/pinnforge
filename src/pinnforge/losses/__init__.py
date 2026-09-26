"""Soft IC and boundary penalties.

Importing this package requires torch (the optional ``ml`` extra).
Training adds :func:`soft_penalty` to the residual loss. Dirichlet,
Neumann, and periodic terms are separate. An unsupported boundary
kind raises ``ValueError``.
"""

from pinnforge.losses.conditions import (
    SoftPenalty,
    boundary_coverage_gaps,
    dirichlet_boundary_loss,
    initial_condition_loss,
    neumann_boundary_loss,
    periodic_boundary_loss,
    soft_penalty,
)

__all__ = [
    "SoftPenalty",
    "boundary_coverage_gaps",
    "dirichlet_boundary_loss",
    "initial_condition_loss",
    "neumann_boundary_loss",
    "periodic_boundary_loss",
    "soft_penalty",
]
