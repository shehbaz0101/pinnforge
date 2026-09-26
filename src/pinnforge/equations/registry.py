"""In-process equation registry.

Day 1 registers the three spec classes at import. This module is the
lookup those classes share: an id or a short alias resolves to the
class, :func:`list_equations` documents the built-ins, and
:func:`build_equation` turns a config selection into a validated spec.

The built-ins are:

- ``harmonic_oscillator`` (alias ``harmonic``). ``u'' + ω² u = 0``.
  Override ``omega``, or ``k`` and ``m``, plus ``time``,
  ``initial_condition``, and ``boundary_conditions``.
- ``burgers_1d`` (alias ``burgers``). ``u_t + u u_x = ν u_xx``.
  Override ``nu``, ``x``, ``t``, ``initial_condition``, and
  ``boundary_conditions``.
- ``poisson_toy`` (alias ``poisson``). ``-Δu = f`` in 1D or 2D.
  Override ``dimensions``, ``source``, ``x``, ``y``, and
  ``boundary_conditions``.

Overrides are merged onto the built-in spec and checked by the Day 1
schema. An inline mapping is that schema in full and is not merged.
"""

from __future__ import annotations

import importlib
from collections.abc import Mapping
from dataclasses import dataclass
from typing import TypeVar

from pydantic import ValidationError

from pinnforge.equations.base import EquationSpec

EquationT = TypeVar("EquationT", bound=EquationSpec)

_REGISTRY: dict[str, type[EquationSpec]] = {}
_LOADED = False


@dataclass(frozen=True, slots=True)
class EquationInfo:
    """One built-in equation.

    ``aliases`` are the short names the CLI and experiment configs
    accept in addition to ``equation_id``. ``summary`` is the residual
    statement. ``parameters`` are the spec fields a config may override;
    the Day 1 schema still decides which combinations are valid.
    """

    equation_id: str
    aliases: tuple[str, ...]
    summary: str
    parameters: tuple[str, ...]


_CATALOG: tuple[EquationInfo, ...] = (
    EquationInfo(
        equation_id="burgers_1d",
        aliases=("burgers",),
        summary="Viscous Burgers: u_t + u u_x = ν u_xx on x and t.",
        parameters=("nu", "x", "t", "initial_condition", "boundary_conditions"),
    ),
    EquationInfo(
        equation_id="harmonic_oscillator",
        aliases=("harmonic",),
        summary="Harmonic oscillator: u'' + ω² u = 0 on a time interval.",
        parameters=("omega", "k", "m", "time", "initial_condition", "boundary_conditions"),
    ),
    EquationInfo(
        equation_id="poisson_toy",
        aliases=("poisson",),
        summary="Poisson toy: -Δu = f on a 1D interval or a 2D rectangle.",
        parameters=("dimensions", "source", "x", "y", "boundary_conditions"),
    ),
)


def _alias_map() -> dict[str, str]:
    mapping: dict[str, str] = {}
    for info in _CATALOG:
        mapping[info.equation_id] = info.equation_id
        for alias in info.aliases:
            if alias in mapping and mapping[alias] != info.equation_id:
                raise RuntimeError(f"duplicate equation alias {alias!r}")
            mapping[alias] = info.equation_id
    return mapping


_ALIAS_TO_ID: dict[str, str] = _alias_map()


def equation_alias_map() -> dict[str, str]:
    """Copy of the alias table, including each canonical id.

    Keys are the names :func:`resolve_equation_id` accepts. Values are
    registry ids.
    """

    return dict(_ALIAS_TO_ID)


def resolve_equation_id(name: str) -> str:
    """Map ``name`` to a registry id.

    ``name`` may be the id (``harmonic_oscillator``) or a short alias
    (``harmonic``).

    Raises:
        ValueError: ``name`` is not a string, or it is not a built-in.
    """

    if not isinstance(name, str):
        raise ValueError("equation_id must be a string")
    try:
        return _ALIAS_TO_ID[name]
    except KeyError:
        known = ", ".join(sorted(_ALIAS_TO_ID))
        raise ValueError(f"unknown equation {name!r}; known: {known}") from None


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


def list_equations() -> tuple[EquationInfo, ...]:
    """Documented built-ins, sorted by equation id.

    The catalog and the registered classes are the same set. A spec that
    is registered without an entry here, or an entry whose class did not
    register, is an error.

    Raises:
        RuntimeError: the catalog and the registry disagree.
    """

    ensure_registered()
    registered = set(_REGISTRY)
    documented = {item.equation_id for item in _CATALOG}
    if registered != documented:
        missing = ", ".join(sorted(registered - documented)) or "none"
        extra = ", ".join(sorted(documented - registered)) or "none"
        raise RuntimeError(
            "equation catalog does not match the registry "
            f"(registered but undocumented: {missing}; documented but not registered: {extra})"
        )
    return tuple(sorted(_CATALOG, key=lambda item: item.equation_id))


def registered_equations() -> tuple[str, ...]:
    """Sorted equation ids. The same ids :func:`list_equations` documents."""

    return tuple(info.equation_id for info in list_equations())


def get_equation(equation_id: str) -> type[EquationSpec]:
    """Return the spec class for ``equation_id`` or a short alias.

    Raises:
        KeyError: ``equation_id`` is not a built-in id or alias.
    """

    ensure_registered()
    canonical = _ALIAS_TO_ID.get(equation_id, equation_id)
    try:
        return _REGISTRY[canonical]
    except KeyError:
        known = ", ".join(sorted(_REGISTRY))
        raise KeyError(f"unknown equation {equation_id!r}; known: {known}") from None


def parse_equation(data: Mapping[str, object]) -> EquationSpec:
    """Validate ``data`` as the spec named by its ``equation_id``.

    ``equation_id`` is the registry id. Short aliases belong in
    :func:`build_equation`, which rewrites them before calling this.
    """

    if "equation_id" not in data:
        raise ValueError("equation spec requires equation_id")
    equation_id = data["equation_id"]
    if not isinstance(equation_id, str):
        raise ValueError("equation_id must be a string")
    canonical = _ALIAS_TO_ID.get(equation_id)
    if canonical is not None and canonical != equation_id:
        data = dict(data)
        data["equation_id"] = canonical
        equation_id = canonical
    return get_equation(equation_id).model_validate(data)


def build_equation(
    equation: str | Mapping[str, object] | EquationSpec,
    params: Mapping[str, object] | None = None,
) -> EquationSpec:
    """Build a Day 1 spec from an id, an alias, or an inline spec.

    A string selects the built-in problem (``harmonic``, ``burgers``,
    ``poisson``, or the registry id). ``params`` are merged onto that
    built-in and validated by the Day 1 schema. Nested mappings merge
    field by field; lists, including boundary conditions, are replaced.

    A mapping is a full inline spec. It cannot be combined with
    ``params``. Its ``equation_id`` may be a short alias.

    Raises:
        ValueError: the selection is not a built-in, ``params`` is
            combined with an inline spec, or the Day 1 schema rejects
            the merged problem.
    """

    if isinstance(equation, EquationSpec):
        if params:
            raise ValueError("equation_params cannot be combined with an inline equation spec")
        if type(equation) is not get_equation(equation.equation_id):
            raise ValueError("equation must be a built-in spec")
        return equation
    if isinstance(equation, str):
        return _from_builtin(equation, params)
    if isinstance(equation, Mapping):
        if params:
            raise ValueError("equation_params cannot be combined with an inline equation spec")
        data = dict(equation)
        equation_id = data.get("equation_id")
        if isinstance(equation_id, str):
            data["equation_id"] = resolve_equation_id(equation_id)
        return _validate_spec(data)
    raise ValueError("equation must be an id or an inline spec mapping")


def _from_builtin(name: str, params: Mapping[str, object] | None) -> EquationSpec:
    from pinnforge.sampling.defaults import default_spec

    equation_id = resolve_equation_id(name)
    merged: dict[str, object] = default_spec(equation_id).model_dump(mode="json")
    if params:
        if isinstance(params, str) or not isinstance(params, Mapping):
            raise ValueError("equation_params must be a mapping")
        override = dict(params)
        if "equation_id" in override:
            raw_id = override.pop("equation_id")
            if not isinstance(raw_id, str) or resolve_equation_id(raw_id) != equation_id:
                raise ValueError("equation_params.equation_id does not match equation")
        merged = _deep_merge(merged, override)
    return _validate_spec(merged)


def _validate_spec(data: Mapping[str, object]) -> EquationSpec:
    try:
        return parse_equation(data)
    except ValidationError as exc:
        raise ValueError(_validation_message(exc)) from exc


def _deep_merge(base: dict[str, object], override: Mapping[str, object]) -> dict[str, object]:
    merged = dict(base)
    for key, value in override.items():
        current = merged.get(key)
        if isinstance(current, dict) and isinstance(value, Mapping) and not isinstance(value, str):
            merged[key] = _deep_merge(current, value)
        else:
            merged[key] = value
    return merged


def _validation_message(exc: ValidationError) -> str:
    parts: list[str] = []
    for error in exc.errors():
        location = ".".join(str(item) for item in error["loc"])
        message = str(error["msg"])
        parts.append(f"{location}: {message}" if location else message)
    return "; ".join(parts) if parts else "invalid equation spec"
