"""Guard for modules that need the optional ``ml`` extra.

The package root, equation schemas, and samplers do not import torch.
:mod:`pinnforge.models`, :mod:`pinnforge.residuals`, and
:mod:`pinnforge.losses` call :func:`require_torch` on import.
"""

from __future__ import annotations

from types import ModuleType


class InstallHint(ImportError):
    """Torch is not installed.

    The MLP, residual operators, and condition penalties need the optional
    ``ml`` extra::

        pip install -e ".[ml]"

    Day 4 wires those pieces into a training loop. ``version``,
    ``equations``, and ``sample`` keep working without torch.
    """

    def __init__(self, message: str | None = None) -> None:
        if message is None:
            message = (
                "PINNForge models, residuals, and condition losses need torch "
                'from the optional ml extra. Install it with: pip install -e ".[ml]"'
            )
        super().__init__(message)


def require_torch() -> ModuleType:
    """Return the ``torch`` module, or raise :class:`InstallHint`."""

    try:
        import torch
    except ImportError as exc:
        raise InstallHint() from exc
    return torch
