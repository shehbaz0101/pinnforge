"""Torch-free training and evaluation settings.

:class:`~pinnforge.specs.train.TrainConfig` and
:class:`~pinnforge.specs.eval.EvalConfig` are the pydantic models the
training loop and the evaluator use. They live here so an experiment
file can be validated without the optional ``ml`` extra.
:mod:`pinnforge.training` and :mod:`pinnforge.evaluation` re-export
them, and those packages still import torch.
"""

from pinnforge.specs.eval import EvalConfig
from pinnforge.specs.paths import resolve_inside_cwd
from pinnforge.specs.train import ACTIVATION_NAMES, TrainConfig

__all__ = [
    "ACTIVATION_NAMES",
    "EvalConfig",
    "TrainConfig",
    "resolve_inside_cwd",
]
