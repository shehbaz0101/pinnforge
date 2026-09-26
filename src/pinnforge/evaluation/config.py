"""Re-export of :mod:`pinnforge.specs.eval`.

Importing this module loads :mod:`pinnforge.evaluation` first, which
needs torch. Experiment configs import :mod:`pinnforge.specs` instead.
"""

from pinnforge.specs.eval import EvalConfig

__all__ = ["EvalConfig"]
