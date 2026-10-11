"""Frozen candidate libraries for the v0.5 term-discovery study."""

from __future__ import annotations

from typing import Any

import numpy as np

from pinnforge.reference.spectral2d.common import Grid2D, fft2, ifft2, spectral_grad
from pinnforge.reference.spectral2d.navier_stokes import kolmogorov_forcing, velocity_from_vorticity

BURGERS_TERMS: tuple[str, ...] = (
    "u_qx",
    "v_qy",
    "q_xx",
    "q_yy",
    "q2",
    "uv",
    "q",
    "q_xy",
    "q_xxx",
    "q_yyy",
    "q_xxxx",
    "v_qx",
    "u_qy",
    "const",
    "force_y",
)

NS_TERMS: tuple[str, ...] = (
    "u_wx",
    "v_wy",
    "w_xx",
    "w_yy",
    "force_y",
    "w2",
    "w",
    "w_xy",
    "w_xxx",
    "w_yyy",
    "w_xxxx",
    "uv",
    "const",
    "force_x",
    "u_wy",
)

BURGERS_TRUE: tuple[str, ...] = ("u_qx", "v_qy", "q_xx", "q_yy")
NS_TRUE_KOLMOGOROV: tuple[str, ...] = ("u_wx", "v_wy", "w_xx", "w_yy", "force_y")
NS_TRUE_DECAYING: tuple[str, ...] = ("u_wx", "v_wy", "w_xx", "w_yy")


def terms_for(problem: str) -> tuple[str, ...]:
    if problem == "burgers2d":
        return BURGERS_TERMS
    if problem == "ns2d":
        return NS_TERMS
    raise ValueError(problem)


def true_support(problem: str, family: str = "all") -> tuple[str, ...]:
    if problem == "burgers2d":
        return BURGERS_TRUE
    if problem == "ns2d":
        if family == "decaying":
            return NS_TRUE_DECAYING
        return NS_TRUE_KOLMOGOROV
    raise ValueError(problem)


def true_coefficients(problem: str, nu: float, family: str = "all") -> dict[str, float]:
    support = true_support(problem, family)
    values: dict[str, float] = {name: 0.0 for name in terms_for(problem)}
    if problem == "burgers2d":
        values.update({"u_qx": -1.0, "v_qy": -1.0, "q_xx": float(nu), "q_yy": float(nu)})
    else:
        values.update({"u_wx": -1.0, "v_wy": -1.0, "w_xx": float(nu), "w_yy": float(nu)})
        if "force_y" in support:
            values["force_y"] = 1.0
    return values


def family_of(problem: str, instance_id: int, forcing: str | None = None) -> str:
    if problem == "burgers2d":
        return "all"
    if forcing is not None:
        return str(forcing)
    return "kolmogorov" if int(instance_id) % 2 == 0 else "decaying"


def primary_family(protocol: dict[str, Any], problem: str) -> str:
    return str(protocol["problems"][problem]["primary_family"])


def is_primary_instance(protocol: dict[str, Any], problem: str, instance_id: int, forcing: str | None) -> bool:
    family = primary_family(protocol, problem)
    if family == "all":
        return True
    return family_of(problem, instance_id, forcing) == family


def spectral_deriv(field: np.ndarray, grid: Grid2D, nx: int, ny: int) -> np.ndarray:
    kx, ky, _ = grid.wavenumbers()
    return ifft2(((1j * kx) ** int(nx)) * ((1j * ky) ** int(ny)) * fft2(field))


def force_x_field(grid: Grid2D) -> np.ndarray:
    xs, _ys = grid.mesh()
    return -4.0 * np.cos(4.0 * xs)


def velocity_on_grid(field: np.ndarray, problem: str, grid: Grid2D) -> tuple[np.ndarray, np.ndarray]:
    """``field`` is one time slice ``(components, n, n)``."""

    values = np.asarray(field, dtype=np.float64)
    if problem == "burgers2d":
        return values[0], values[1]
    velocity = velocity_from_vorticity(values[0], grid)
    return velocity[0], velocity[1]


def scalar_component(field: np.ndarray, problem: str, component: int) -> np.ndarray:
    values = np.asarray(field, dtype=np.float64)
    if problem == "burgers2d":
        return values[component]
    return values[0]


def strong_term(
    name: str,
    *,
    problem: str,
    q: np.ndarray,
    u: np.ndarray,
    v: np.ndarray,
    grid: Grid2D,
) -> np.ndarray:
    """Strong-form field of one library term. Linear derivatives stay on ``q``."""

    if name in {"u_qx", "v_qx"}:
        qx, _qy = spectral_grad(q, grid)
        return (u if name == "u_qx" else v) * qx
    if name in {"v_qy", "u_qy", "u_wy"}:
        _qx, qy = spectral_grad(q, grid)
        if name == "v_qy":
            return v * qy
        return u * qy
    if name in {"u_wx"}:
        qx, _qy = spectral_grad(q, grid)
        return u * qx
    if name in {"v_wy"}:
        _qx, qy = spectral_grad(q, grid)
        return v * qy
    if name in {"q_xx", "w_xx"}:
        return spectral_deriv(q, grid, 2, 0)
    if name in {"q_yy", "w_yy"}:
        return spectral_deriv(q, grid, 0, 2)
    if name in {"q_xy", "w_xy"}:
        return spectral_deriv(q, grid, 1, 1)
    if name in {"q_xxx", "w_xxx"}:
        return spectral_deriv(q, grid, 3, 0)
    if name in {"q_yyy", "w_yyy"}:
        return spectral_deriv(q, grid, 0, 3)
    if name in {"q_xxxx", "w_xxxx"}:
        return spectral_deriv(q, grid, 4, 0)
    if name in {"q2", "w2"}:
        return q * q
    if name == "uv":
        return u * v
    if name in {"q", "w"}:
        return q
    if name == "const":
        return np.ones_like(q)
    if name == "force_y":
        return kolmogorov_forcing(grid)
    if name == "force_x":
        return force_x_field(grid)
    raise KeyError(name)
