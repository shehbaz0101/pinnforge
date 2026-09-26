"""Guard for modules that need the optional ``ml`` extra.

The package root, equation schemas, and samplers do not import torch.
:mod:`pinnforge.models`, :mod:`pinnforge.residuals`,
:mod:`pinnforge.losses`, :mod:`pinnforge.training`, and
:mod:`pinnforge.evaluation` call :func:`require_torch` on import.
"""

from __future__ import annotations

from types import ModuleType


class InstallHint(ImportError):
    """Torch is not installed.

    The MLP, residual operators, condition penalties, training loop, and
    evaluation need the optional ``ml`` extra::

        pip install -e ".[ml]"

    ``version``, ``equations``, and ``sample`` keep working without torch.
    """

    def __init__(self, message: str | None = None) -> None:
        if message is None:
            message = (
                "PINNForge models, residuals, condition losses, the training "
                "loop, and evaluation need torch from the optional ml extra. "
                "Install it with: "
                'pip install -e ".[ml]"'
            )
        super().__init__(message)


def require_torch() -> ModuleType:
    """Return the ``torch`` module, or raise :class:`InstallHint`."""

    try:
        import torch
    except ImportError as exc:
        raise InstallHint() from exc
    return torch
