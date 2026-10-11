"""2D periodic spectral solvers for Burgers and vorticity-form NS."""

from pinnforge.reference.spectral2d.burgers import Burgers2DTrajectory, solve_burgers2d
from pinnforge.reference.spectral2d.common import Grid2D
from pinnforge.reference.spectral2d.dataset import (
    convergence_record,
    generate_problem,
    load_protocol,
    protocol_sha256,
)
from pinnforge.reference.spectral2d.navier_stokes import (
    NS2DTrajectory,
    solve_ns2d,
    velocity_from_vorticity,
)

__all__ = [
    "Burgers2DTrajectory",
    "Grid2D",
    "NS2DTrajectory",
    "convergence_record",
    "generate_problem",
    "load_protocol",
    "protocol_sha256",
    "solve_burgers2d",
    "solve_ns2d",
    "velocity_from_vorticity",
]
