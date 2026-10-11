"""Numpy WSINDy: compact test functions and sequential thresholded least squares."""

from __future__ import annotations

from typing import Any

import numpy as np

from pinnforge.operator.weak_form import bump_derivatives as bump_value_first_second
from pinnforge.reference.spectral2d.common import Grid2D, spectral_grad
from pinnforge.reference.spectral2d.navier_stokes import kolmogorov_forcing
from pinnforge.v04.sensors import SensorMask, burst_slices
from pinnforge.v05.library import (
    family_of,
    force_x_field,
    scalar_component,
    terms_for,
    true_support,
    velocity_on_grid,
)

_DEGREE_MIN = 2
_IBP_SIGN = {
    "q_xx": 1.0,
    "w_xx": 1.0,
    "q_yy": 1.0,
    "w_yy": 1.0,
    "q_xy": 1.0,
    "w_xy": 1.0,
    "q_xxx": -1.0,
    "w_xxx": -1.0,
    "q_yyy": -1.0,
    "w_yyy": -1.0,
    "q_xxxx": 1.0,
    "w_xxxx": 1.0,
}


def bump_derivatives(xi: np.ndarray, degree: int) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """Value and the first four ``ξ`` derivatives of ``(1 - ξ²)_+^degree``."""

    if isinstance(degree, bool) or not isinstance(degree, (int, np.integer)) or int(degree) < _DEGREE_MIN:
        raise ValueError(f"degree must be an integer >= {_DEGREE_MIN}")
    order = int(degree)
    value, first, second = bump_value_first_second(xi, order)
    points = np.asarray(xi, dtype=np.float64)
    third = np.zeros(points.shape, dtype=np.float64)
    fourth = np.zeros(points.shape, dtype=np.float64)
    inside = np.abs(points) < 1.0
    if not np.any(inside):
        return value, first, second, third, fourth
    local = points[inside]
    base = 1.0 - local * local
    third[inside] = 4.0 * order * (order - 1) * local * (base ** (order - 3)) * (3.0 - (2.0 * order - 1.0) * local * local)
    fourth[inside] = (
        4.0
        * order
        * (order - 1)
        * (base ** (order - 4))
        * (3.0 + (-12.0 * order + 18.0) * local * local + (2.0 * order - 1.0) * (2.0 * order - 3.0) * local**4)
    )
    return value, first, second, third, fourth


def periodic_bump(nodes: np.ndarray, center: float, radius: float, degree: int, length: float) -> dict[str, np.ndarray]:
    if not np.isfinite(radius) or radius <= 0.0:
        raise ValueError("bump radius must be > 0")
    delta = (nodes - float(center) + 0.5 * length) % length - 0.5 * length
    value, first, second, third, fourth = bump_derivatives(delta / radius, degree)
    scale = float(radius)
    return {
        "value": value,
        "d1": first / scale,
        "d2": second / (scale * scale),
        "d3": third / (scale**3),
        "d4": fourth / (scale**4),
    }


def stlsq(
    design: np.ndarray,
    target: np.ndarray,
    threshold: float,
    *,
    max_iter: int = 10,
    column_normalize: bool = True,
) -> np.ndarray:
    """Sequential thresholded least squares. ``threshold`` applies to normalized coefficients."""

    matrix = np.asarray(design, dtype=np.float64)
    rhs = np.asarray(target, dtype=np.float64).reshape(-1)
    if matrix.ndim != 2 or matrix.shape[0] != rhs.shape[0]:
        raise ValueError("design and target shapes do not match")
    if matrix.size == 0 or rhs.size == 0:
        raise ValueError("STLSQ rows are empty")
    n_terms = matrix.shape[1]
    scales = np.ones(n_terms, dtype=np.float64)
    work = matrix
    if column_normalize:
        scales = np.linalg.norm(matrix, axis=0)
        scales = np.where(scales < 1e-15, 1.0, scales)
        work = matrix / scales
    active = np.ones(n_terms, dtype=bool)
    coef_norm = np.zeros(n_terms, dtype=np.float64)
    for _ in range(int(max_iter)):
        if not np.any(active):
            coef_norm[:] = 0.0
            break
        fit, *_ = np.linalg.lstsq(work[:, active], rhs, rcond=None)
        trial = np.zeros(n_terms, dtype=np.float64)
        trial[active] = fit
        keep = np.abs(trial) >= float(threshold)
        if bool(np.array_equal(keep, active)):
            coef_norm = trial
            break
        active = keep
        coef_norm = trial
    else:
        pass
    return coef_norm / scales


def support_of(coef: np.ndarray, names: tuple[str, ...]) -> list[str]:
    return [name for name, value in zip(names, coef, strict=True) if float(value) != 0.0]


def exact_support(recovered: list[str] | tuple[str, ...], truth: tuple[str, ...] | list[str]) -> bool:
    return set(recovered) == set(truth)


def choose_threshold(
    systems: list[tuple[np.ndarray, np.ndarray]],
    names: tuple[str, ...],
    truth: tuple[str, ...],
    candidates: list[float],
    *,
    max_iter: int,
    column_normalize: bool,
) -> float:
    """Pick the threshold that maximizes exact instance supports. Ties take the largest λ."""

    best_count = -1
    chosen = float(candidates[-1])
    for threshold in candidates:
        count = 0
        for design, target in systems:
            coef = stlsq(
                design,
                target,
                float(threshold),
                max_iter=max_iter,
                column_normalize=column_normalize,
            )
            if exact_support(support_of(coef, names), truth):
                count += 1
        if count > best_count or (count == best_count and float(threshold) >= chosen):
            best_count = count
            chosen = float(threshold)
    return chosen


def refit_support(design: np.ndarray, target: np.ndarray, names: tuple[str, ...], support: list[str]) -> dict[str, float]:
    values = {name: 0.0 for name in names}
    index = [i for i, name in enumerate(names) if name in set(support)]
    if not index:
        return values
    fit, *_ = np.linalg.lstsq(np.asarray(design, dtype=np.float64)[:, index], np.asarray(target, dtype=np.float64), rcond=None)
    for loc, name in zip(index, [names[i] for i in index], strict=True):
        values[name] = float(fit[index.index(loc)])
    return values


def coeff_rel_error(hat: dict[str, float], truth: dict[str, float], support: tuple[str, ...]) -> float:
    errors = []
    for name in support:
        true_value = float(truth[name])
        errors.append(abs(float(hat.get(name, 0.0)) - true_value) / max(abs(true_value), 1e-12))
    return float(np.mean(errors)) if errors else float("nan")


def assemble_weak_system(
    field: np.ndarray,
    *,
    problem: str,
    protocol: dict[str, Any],
    mask: SensorMask,
    length: float,
) -> tuple[np.ndarray, np.ndarray]:
    """One instance. ``field`` is ``(n_times, components, n, n)`` on the grid the method sees.

    A degree-4 bump sampled on a three-frame burst has a zero trapezoidal
    ``η_t``, so the time derivative is the central difference at the middle
    frame and the spatial test function is applied there.
    """

    values = np.asarray(field, dtype=np.float64)
    if values.ndim != 4:
        raise ValueError("field must have shape (n_times, components, n, n)")
    names = terms_for(problem)
    spec = protocol["methods"]["wsindy"]["test_function"]
    degree = int(spec["degree"])
    n_grid = int(values.shape[-1])
    grid = Grid2D(n_grid, float(length))
    axis = grid.axes()
    radius = float(spec["space_radius_fraction"]) * float(length)
    stride = max(1, int(round(float(spec["center_stride_fraction"]) * n_grid)))
    centers = axis[::stride]
    dt = float(protocol["time"]["save_dt"])
    weight = grid.dx * grid.dx
    rows: list[np.ndarray] = []
    targets: list[float] = []
    force_y = kolmogorov_forcing(grid)
    force_x = force_x_field(grid)
    components = range(2) if problem == "burgers2d" else range(1)
    for burst in burst_slices(mask):
        block = values[burst]
        if block.shape[0] < 3:
            continue
        mid = block[1]
        time_deriv = (block[-1] - block[0]) / (2.0 * dt)
        u_mid, v_mid = velocity_on_grid(mid, problem, grid)
        for center_x in centers:
            bump_x = periodic_bump(axis, float(center_x), radius, degree, float(length))
            for center_y in centers:
                bump_y = periodic_bump(axis, float(center_y), radius, degree, float(length))
                test = _separable_test(bump_x, bump_y)
                for component in components:
                    q = scalar_component(mid, problem, component)
                    q_t = scalar_component(time_deriv, problem, component)
                    targets.append(float(np.sum(q_t * test["value"] * weight)))
                    rows.append(
                        _term_row(
                            names,
                            problem=problem,
                            q=q,
                            u=u_mid,
                            v=v_mid,
                            test=test,
                            weight=weight,
                            force_y=force_y,
                            force_x=force_x,
                            grid=grid,
                        )
                    )
    if not rows:
        raise ValueError("weak-form library is empty")
    return np.asarray(rows, dtype=np.float64), np.asarray(targets, dtype=np.float64)


def _separable_test(bump_x: dict[str, np.ndarray], bump_y: dict[str, np.ndarray]) -> dict[str, np.ndarray]:
    vx, vy = bump_x["value"][:, None], bump_y["value"][None, :]
    return {
        "value": vx * vy,
        "x": bump_x["d1"][:, None] * vy,
        "y": vx * bump_y["d1"][None, :],
        "xx": bump_x["d2"][:, None] * vy,
        "yy": vx * bump_y["d2"][None, :],
        "xy": bump_x["d1"][:, None] * bump_y["d1"][None, :],
        "xxx": bump_x["d3"][:, None] * vy,
        "yyy": vx * bump_y["d3"][None, :],
        "xxxx": bump_x["d4"][:, None] * vy,
    }


def _term_row(
    names: tuple[str, ...],
    *,
    problem: str,
    q: np.ndarray,
    u: np.ndarray,
    v: np.ndarray,
    test: dict[str, np.ndarray],
    weight: float,
    force_y: np.ndarray,
    force_x: np.ndarray,
    grid: Grid2D,
) -> np.ndarray:
    qx, qy = spectral_grad(q, grid)
    phi = test["value"]
    row = np.empty(len(names), dtype=np.float64)
    for index, name in enumerate(names):
        if name in _IBP_SIGN:
            deriv = {
                "q_xx": test["xx"],
                "w_xx": test["xx"],
                "q_yy": test["yy"],
                "w_yy": test["yy"],
                "q_xy": test["xy"],
                "w_xy": test["xy"],
                "q_xxx": test["xxx"],
                "w_xxx": test["xxx"],
                "q_yyy": test["yyy"],
                "w_yyy": test["yyy"],
                "q_xxxx": test["xxxx"],
                "w_xxxx": test["xxxx"],
            }[name]
            row[index] = float(_IBP_SIGN[name] * np.sum(q * deriv * weight))
            continue
        if name == "u_qx" or name == "u_wx":
            field = u * qx
        elif name == "v_qy" or name == "v_wy":
            field = v * qy
        elif name == "v_qx":
            field = v * qx
        elif name in {"u_qy", "u_wy"}:
            field = u * qy
        elif name in {"q2", "w2"}:
            field = q * q
        elif name == "uv":
            field = u * v
        elif name in {"q", "w"}:
            field = q
        elif name == "const":
            field = np.ones_like(q)
        elif name == "force_y":
            field = force_y
        elif name == "force_x":
            field = force_x
        else:
            raise KeyError(name)
        row[index] = float(np.sum(field * phi * weight))
    return row


def instance_truth(problem: str, instance_id: int, forcing: str | None) -> tuple[str, ...]:
    return true_support(problem, family_of(problem, instance_id, forcing))
