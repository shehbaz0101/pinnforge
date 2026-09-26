"""Equation specs and the in-process registry."""

from pinnforge.equations.base import (
    Axis,
    BoundaryCondition,
    CollocationDomain,
    ConditionDescriptor,
    EquationSpec,
    Interval,
    ProfileInitialCondition,
    StateInitialCondition,
)
from pinnforge.equations.burgers import BURGERS_PROFILES, Burgers1DSpec
from pinnforge.equations.harmonic import HarmonicOscillatorSpec
from pinnforge.equations.poisson import PoissonSource, PoissonToySpec
from pinnforge.equations.registry import get_equation, parse_equation, registered_equations

__all__ = [
    "BURGERS_PROFILES",
    "Axis",
    "BoundaryCondition",
    "Burgers1DSpec",
    "CollocationDomain",
    "ConditionDescriptor",
    "EquationSpec",
    "HarmonicOscillatorSpec",
    "Interval",
    "PoissonSource",
    "PoissonToySpec",
    "ProfileInitialCondition",
    "StateInitialCondition",
    "get_equation",
    "parse_equation",
    "registered_equations",
]
