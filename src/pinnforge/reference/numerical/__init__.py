"""Numerical reference for periodic viscous Burgers.

This package is not the coordinate-PINN reference hook. ``pinnforge``
evaluation still calls :func:`pinnforge.reference.burgers.reference_solution`,
which raises ``NotImplementedError``. Trajectories here are a separate
Fourier reference. The data-only operator trained on windows of those
trajectories is :mod:`pinnforge.operator`. This package does not train it.

The problem is

    u_t + u u_x = ν u_xx,    x ∈ [-1, 1],    t ∈ [0, 1],

with periodic boundary conditions in ``x``. The solver is a dealiased
Fourier spectral method with ETDRK4 time stepping. See
:mod:`pinnforge.reference.numerical.solver`.
"""

from pinnforge.reference.numerical.checks import (
    cole_hopf,
    energy,
    finite_difference_solve,
    relative_l2,
    spatial_mean,
)
from pinnforge.reference.numerical.dataset import PilotConfig, generate_pilot
from pinnforge.reference.numerical.initial import (
    NU_MAX,
    NU_MIN,
    InitialCondition,
    draw_initial_condition,
)
from pinnforge.reference.numerical.solver import (
    LENGTH,
    X_LOWER,
    X_UPPER,
    SolverConfig,
    Trajectory,
    grid,
    solve,
    spectral_derivative,
    wavenumbers,
)

__all__ = [
    "LENGTH",
    "NU_MAX",
    "NU_MIN",
    "X_LOWER",
    "X_UPPER",
    "InitialCondition",
    "PilotConfig",
    "SolverConfig",
    "Trajectory",
    "cole_hopf",
    "draw_initial_condition",
    "energy",
    "finite_difference_solve",
    "generate_pilot",
    "grid",
    "relative_l2",
    "solve",
    "spatial_mean",
    "spectral_derivative",
    "wavenumbers",
]
