"""In-process equation registry.

Day 1 registers the three spec classes at import. A later day can persist
runs; this module only maps ``equation_id`` to a class.
"""

from __future__ import annotations

import importlib
from collections.abc import Mapping
from typing import TypeVar

from pinnforge.equations.base import EquationSpec

EquationT = TypeVar("EquationT", bound=EquationSpec)

_REGISTRY: dict[str, type[EquationSpec]] = {}
_LOADED = False


def register_equation(model: type[EquationT]) -> type[EquationT]:
    """Register ``model`` under its default ``equation_id``.

    Registering the same class twice is a no-op. A second class with the
    same id is an error.
    """

    field = model.model_fields.get("equation_id")
    equation_id = None if field is None else field.default
    if not isinstance(equation_id, str) or equation_id == "":
        raise TypeError(f"{model.__name__} needs a string equation_id default")
    existing = _REGISTRY.get(equation_id)
    if existing is model:
        return model
    if existing is not None:
        raise ValueError(f"equation_id {equation_id!r} is already registered")
    _REGISTRY[equation_id] = model
    return model


def ensure_registered() -> None:
    """Import the Day 1 specs so their decorators fill the registry."""

    global _LOADED
    if _LOADED:
        return
    # Submodule imports, not package attribute imports, so this stays safe
    # while ``pinnforge.equations`` itself is still initializing.
    importlib.import_module("pinnforge.equations.burgers")
    importlib.import_module("pinnforge.equations.harmonic")
    importlib.import_module("pinnforge.equations.poisson")
    _LOADED = True


def registered_equations() -> tuple[str, ...]:
    """Sorted equation ids."""

    ensure_registered()
    return tuple(sorted(_REGISTRY))


def get_equation(equation_id: str) -> type[EquationSpec]:
    """Return the spec class for ``equation_id``.

    Raises:
        KeyError: ``equation_id`` is not registered.
    """

    ensure_registered()
    try:
        return _REGISTRY[equation_id]
    except KeyError:
        known = ", ".join(sorted(_REGISTRY))
        raise KeyError(f"unknown equation {equation_id!r}; known: {known}") from None


def parse_equation(data: Mapping[str, object]) -> EquationSpec:
    """Validate ``data`` as the spec named by its ``equation_id``."""

    if "equation_id" not in data:
        raise ValueError("equation spec requires equation_id")
    equation_id = data["equation_id"]
    if not isinstance(equation_id, str):
        raise ValueError("equation_id must be a string")
    return get_equation(equation_id).model_validate(data)
