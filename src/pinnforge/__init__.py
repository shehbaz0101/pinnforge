"""PINNForge: physics-informed neural nets for classic ODE and PDE residuals.

Day 1 ships equation schemas, a harmonic-oscillator closed form, and a CLI
stub. Day 2 adds seeded collocation, initial-condition, and boundary
samplers. Day 3 adds an MLP and residual operators. Day 4 trains that
residual with Adam. Day 5 scores a checkpoint or an in-memory model
against a reference field and the residual. The model, residual, loss,
training, and evaluation modules import torch and raise
:class:`~pinnforge.ml_import.InstallHint` when the optional ``ml`` extra
is missing. Importing this package does not import torch.
"""

from pinnforge.equations import (
    Burgers1DSpec,
    HarmonicOscillatorSpec,
    PoissonToySpec,
    get_equation,
    parse_equation,
    registered_equations,
)
from pinnforge.ml_import import InstallHint
from pinnforge.sampling import CollocationBatch, SampleConfig, sample_equation

__version__ = "0.1.0"

__all__ = [
    "Burgers1DSpec",
    "CollocationBatch",
    "HarmonicOscillatorSpec",
    "InstallHint",
    "PoissonToySpec",
    "SampleConfig",
    "__version__",
    "get_equation",
    "parse_equation",
    "registered_equations",
    "sample_equation",
]
