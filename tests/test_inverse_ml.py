"""Adam check for the sparse viscosity objective. Torch is required."""

from __future__ import annotations

import numpy as np
import pytest

pytestmark = pytest.mark.ml


@pytest.fixture(autouse=True)
def _need_torch() -> None:
    pytest.importorskip("torch")


def test_adam_viscosity_improves_from_a_bad_init() -> None:
    from pinnforge.operator.inverse import ObservationSpec, adam_viscosity, least_squares_viscosity
    from pinnforge.reference.numerical.checks import cole_hopf
    from pinnforge.reference.numerical.solver import grid

    nu = 0.04
    n_space = 32
    n_times = 8
    x = grid(n_space)
    times = 0.01 * np.arange(n_times, dtype=np.float64)
    field = np.stack([cole_hopf(x, time, nu=nu) for time in times], axis=0)
    spec = ObservationSpec(name="burst", n_sensors=16, series=(tuple(range(n_times)),))
    init = 0.8
    baseline = 0.1
    hat, objective_init, objective_final = adam_viscosity(field, spec, init=init, steps=200, lr=0.01)
    closed = least_squares_viscosity(field, spec)
    assert objective_final < objective_init
    assert abs(hat - nu) < abs(init - nu)
    assert abs(hat - closed) < 1e-3
    assert abs(hat - nu) < abs(baseline - nu)
