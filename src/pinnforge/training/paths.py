"""Re-export of :mod:`pinnforge.specs.paths`.

Importing this module loads :mod:`pinnforge.training` first, which needs
torch.
"""

from pinnforge.specs.paths import resolve_inside_cwd

__all__ = ["resolve_inside_cwd"]
