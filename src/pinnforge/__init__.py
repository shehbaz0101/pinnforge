"""PINNForge: physics-informed neural nets for classic ODE and PDE residuals.

Day 1 ships equation schemas, a harmonic-oscillator closed form, and a CLI
stub. Day 2 adds seeded collocation, initial-condition, and boundary
samplers. The MLP, training, and the local API are later days.
Importing this package does not import torch. The optional ``ml`` extra is
where torch will live.
"""

from pinnforge.equations import (
    Burgers1DSpec,
    HarmonicOscillatorSpec,
    PoissonToySpec,
    get_equation,
    parse_equation,
    registered_equations,
)
from pinnforge.sampling import CollocationBatch, SampleConfig, sample_equation

__version__ = "0.1.0"

__all__ = [
    "Burgers1DSpec",
    "CollocationBatch",
    "HarmonicOscillatorSpec",
    "PoissonToySpec",
    "SampleConfig",
    "__version__",
    "get_equation",
    "parse_equation",
    "registered_equations",
    "sample_equation",
]
