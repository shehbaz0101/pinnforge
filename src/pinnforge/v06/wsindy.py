"""Weak-form least-squares for nu and kappa, using the v0.5.2 stride."""

from __future__ import annotations

from typing import Any

import numpy as np

from pinnforge.v05.wsindy import periodic_bump
from pinnforge.v06.grid import GridRect, spectral_grad
from pinnforge.v06.masks import ShearMask, burst_slices, read_sensors

# field[..., component, x, y] with components ux, uy, pressure, tracer
UX, UY, P, S = 0, 1, 2, 3


def fit_coefficients(
    field: np.ndarray,
    *,
    protocol: dict[str, Any],
    mask: ShearMask | None,
    full_field: bool = False,
) -> dict[str, float]:
    """Return ``nu`` and ``kappa`` from the known weak forms."""

    values = np.asarray(field, dtype=np.float64)
    if full_field or mask is None:
        used = values
        nx, ny = int(values.shape[-2]), int(values.shape[-1])
        work_mask = ShearMask(
            nx=nx,
            ny=ny,
            n_side_x=nx,
            n_side_y=ny,
            burst_starts=tuple(int(item) for item in protocol["time_window"]["burst_starts"]),
            burst_length=int(protocol["time_window"]["burst_length"]),
        )
    else:
        used = read_sensors(values, mask)
        work_mask = mask
        nx, ny = work_mask.n_side_x, work_mask.n_side_y
    length_x = float(protocol["downsample"]["length_x"])
    length_y = float(protocol["downsample"]["length_y"])
    # The method sees an nx by ny grid whose physical size is the full domain.
    grid = GridRect(nx, ny, length_x, length_y)
    spec = protocol["methods"]["wsindy"]["test_function"]
    degree = int(spec["degree"])
    frac = float(spec["center_stride_fraction"])
    radius_x = float(spec["space_radius_fraction"]) * length_x
    radius_y = float(spec["space_radius_fraction"]) * length_y
    stride_x = max(1, int(round(frac * nx)))
    stride_y = max(1, int(round(frac * ny)))
    axis_x = grid.dx * np.arange(nx, dtype=np.float64)
    axis_y = grid.dy * np.arange(ny, dtype=np.float64)
    dt = float(protocol["time_window"]["dt"])
    weight = grid.dx * grid.dy
    nu_num = 0.0
    nu_den = 0.0
    ka_num = 0.0
    ka_den = 0.0
    for burst in burst_slices(work_mask):
        block = used[burst]
        if block.shape[0] < 3:
            continue
        mid = block[1]
        time_deriv = (block[-1] - block[0]) / (2.0 * dt)
        ux, uy, pressure, tracer = mid[UX], mid[UY], mid[P], mid[S]
        ux_x, ux_y = spectral_grad(ux, grid)
        uy_x, uy_y = spectral_grad(uy, grid)
        p_x, p_y = spectral_grad(pressure, grid)
        s_x, s_y = spectral_grad(tracer, grid)
        adv_u = time_deriv[UX] + ux * ux_x + uy * ux_y + p_x
        adv_v = time_deriv[UY] + ux * uy_x + uy * uy_y + p_y
        adv_s = time_deriv[S] + ux * s_x + uy * s_y
        for cx in axis_x[::stride_x]:
            bump_x = periodic_bump(axis_x, float(cx), radius_x, degree, length_x)
            for cy in axis_y[::stride_y]:
                bump_y = periodic_bump(axis_y, float(cy), radius_y, degree, length_y)
                phi = bump_x["value"][:, None] * bump_y["value"][None, :]
                lap = bump_x["d2"][:, None] * bump_y["value"][None, :] + bump_x["value"][:, None] * bump_y["d2"][None, :]
                nu_num += float(np.sum(adv_u * phi * weight) + np.sum(adv_v * phi * weight))
                nu_den += float(np.sum(ux * lap * weight) + np.sum(uy * lap * weight))
                ka_num += float(np.sum(adv_s * phi * weight))
                ka_den += float(np.sum(tracer * lap * weight))
    return {
        "nu": float(nu_num / nu_den) if abs(nu_den) > 1e-18 else float("nan"),
        "kappa": float(ka_num / ka_den) if abs(ka_den) > 1e-18 else float("nan"),
    }


def relative_error(hat: float, truth: float) -> float:
    return float(abs(float(hat) - float(truth)) / max(abs(float(truth)), 1e-18))
