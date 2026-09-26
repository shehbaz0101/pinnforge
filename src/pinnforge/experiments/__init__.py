"""Experiment configs for ``pinnforge run``.

Loading a config validates the equation, the training settings, and the
evaluation settings. It does not import torch. The training loop runs
from :mod:`pinnforge.experiments.run`, which needs the ``ml`` extra.
"""

from pinnforge.experiments.config import (
    ExperimentConfig,
    dump_experiment_config,
    load_experiment_config,
    resolve_config_path,
)

__all__ = [
    "ExperimentConfig",
    "dump_experiment_config",
    "load_experiment_config",
    "resolve_config_path",
]
