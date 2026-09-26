"""Re-export of :mod:`pinnforge.specs.train`.

Importing this module loads :mod:`pinnforge.training` first, which needs
torch. Experiment configs import :mod:`pinnforge.specs` instead so a
schema check does not.
"""

from pinnforge.specs.train import ACTIVATION_NAMES, ActivationName, TrainConfig

__all__ = ["ACTIVATION_NAMES", "ActivationName", "TrainConfig"]
