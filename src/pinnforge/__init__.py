"""PINNForge: physics-informed neural nets for classic ODE and PDE residuals.

Day 1 ships equation schemas, a harmonic-oscillator closed form, and a CLI
stub. Day 2 adds seeded collocation, initial-condition, and boundary
samplers. Day 3 adds an MLP and residual operators. Day 4 trains that
residual with Adam. Day 5 scores a checkpoint or an in-memory model
against a reference field and the residual. Day 6 loads an experiment
config that selects a built-in equation, optionally overrides its
parameters, and drives train then eval. Day 7 serves that catalog and
the train-then-eval path on localhost. Day 8 keeps user paths inside a
sandbox root, rate-limits the train, eval, and run routes, and refuses
non-loopback TCP connects on those commands. Day 9 adds ``pinnforge demo``,
which trains a checked-in sample on CPU and prints the reference error.
Version 0.2.0 adds ``pinnforge demo burgers-inverse``, a small CPU
viscosity recovery from 32 sensors. Version 0.2.1 draws that demo's
plots with Matplotlib's DejaVu Sans and prints a short terminal table.
Version 0.2.2 puts that table and its one-line reading at the top of
``result.md`` and ``index.html``. The model, residual, loss,
training, and evaluation modules import torch and raise
:class:`~pinnforge.ml_import.InstallHint` when the optional ``ml`` extra
is missing. The HTTP app imports FastAPI when the optional ``api`` extra
is installed. Importing this package does not import torch or FastAPI.
"""

from pinnforge.equations import (
    Burgers1DSpec,
    EquationInfo,
    HarmonicOscillatorSpec,
    PoissonToySpec,
    build_equation,
    get_equation,
    list_equations,
    parse_equation,
    registered_equations,
)
from pinnforge.ml_import import InstallHint
from pinnforge.sampling import CollocationBatch, SampleConfig, sample_equation

__version__ = "0.2.2"

__all__ = [
    "Burgers1DSpec",
    "CollocationBatch",
    "EquationInfo",
    "HarmonicOscillatorSpec",
    "InstallHint",
    "PoissonToySpec",
    "SampleConfig",
    "__version__",
    "build_equation",
    "get_equation",
    "list_equations",
    "parse_equation",
    "registered_equations",
    "sample_equation",
]
