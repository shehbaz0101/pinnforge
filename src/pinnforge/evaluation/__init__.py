"""Evaluate a PINN against a reference field and its residual.

Importing this package requires torch (the optional ``ml`` extra).
The package root does not import it. :func:`evaluate_model` scores an
in-memory callable. :func:`evaluate_checkpoint` loads a Day 4 checkpoint
first. Burgers has no field reference, so that equation reports residual
metrics only.
"""

from pinnforge.evaluation.api import evaluate_checkpoint, evaluate_model
from pinnforge.evaluation.record import EVAL_FORMAT, EvalResult, ResidualHistogram, write_eval_json
from pinnforge.specs.eval import EvalConfig

__all__ = [
    "EVAL_FORMAT",
    "EvalConfig",
    "EvalResult",
    "ResidualHistogram",
    "evaluate_checkpoint",
    "evaluate_model",
    "write_eval_json",
]
