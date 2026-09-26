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
from pinnforge.equations.registry import (
    EquationInfo,
    build_equation,
    get_equation,
    list_equations,
    parse_equation,
    registered_equations,
    resolve_equation_id,
)

__all__ = [
    "BURGERS_PROFILES",
    "Axis",
    "BoundaryCondition",
    "Burgers1DSpec",
    "CollocationDomain",
    "ConditionDescriptor",
    "EquationInfo",
    "EquationSpec",
    "HarmonicOscillatorSpec",
    "Interval",
    "PoissonSource",
    "PoissonToySpec",
    "ProfileInitialCondition",
    "StateInitialCondition",
    "build_equation",
    "get_equation",
    "list_equations",
    "parse_equation",
    "registered_equations",
    "resolve_equation_id",
]
