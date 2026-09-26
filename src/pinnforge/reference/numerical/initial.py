"""Seeded low-frequency initial conditions for periodic Burgers.

Each instance is a trigonometric polynomial on ``x ∈ [-1, 1]``,

    u_raw(x) = Σ_{m=1}^{M} [a_m cos(m π x) + b_m sin(m π x)],

with ``M = 4`` by default. The coefficients are drawn from one
``numpy.random.Generator`` backed by PCG64 and a ``SeedSequence`` whose
entropy is ``(master_seed, instance_id, 0x42555247)``. For each ``m``,
in that order,

    a_m = Uniform(-1, 1) / m,
    b_m = Uniform(-1, 1) / m.

There is no ``m = 0`` term, so the continuous mean is zero and the field
is periodic. The ``1/m`` factor bounds the higher retained modes.

Amplitude normalization is grid-independent. ``u_raw`` is sampled on
``CANONICAL_N = 4096`` nodes and divided by its maximum absolute value,
so the normalized function has maximum norm 1 on that grid. ``M <= 4``
and 4096 nodes put the discrete maximum within a negligible gap of the
true maximum of this trigonometric polynomial. The same coefficients and
scale are then sampled on the solver grid. A draw whose canonical peak
is below ``1e-8`` is discarded and redrawn, up to 8 attempts. Viscosity
is drawn only after a successful field:

    ν ~ Uniform(nu_min, nu_max),

default ``[0.02, 0.10]``. Failed field attempts do not consume the
viscosity draw.
"""

from __future__ import annotations

import math
from dataclasses import dataclass

import numpy as np

from pinnforge.reference.numerical.solver import grid

IC_SALT = 0x42555247
N_MODES = 4
CANONICAL_N = 4096
NU_MIN = 0.02
NU_MAX = 0.10
_PEAK_FLOOR = 1e-8
_MAX_ATTEMPTS = 8


@dataclass(frozen=True)
class InitialCondition:
    """One normalized Fourier initial condition and its viscosity.

    ``a`` and ``b`` are the pre-normalization coefficients. The field on
    any admissible grid is ``scale`` times the trigonometric polynomial
    of those coefficients. ``scale`` is ``1 / max|u_raw|`` on the
    canonical grid.
    """

    a: np.ndarray
    b: np.ndarray
    scale: float
    u0: np.ndarray
    nu: float
    n_modes: int
    instance_id: int
    master_seed: int

    def to_dict(self) -> dict[str, object]:
        return {
            "a": [float(value) for value in self.a],
            "b": [float(value) for value in self.b],
            "scale": self.scale,
            "nu": self.nu,
            "n_modes": self.n_modes,
            "instance_id": self.instance_id,
            "master_seed": self.master_seed,
            "canonical_n": CANONICAL_N,
            "normalization": "max_abs_on_canonical_grid",
            "coefficient_law": "a_m, b_m = Uniform(-1, 1) / m for m = 1..M",
        }


def draw_initial_condition(
    master_seed: int,
    instance_id: int,
    n: int,
    *,
    n_modes: int = N_MODES,
    nu_min: float = NU_MIN,
    nu_max: float = NU_MAX,
) -> InitialCondition:
    """Draw one initial condition and one viscosity for ``instance_id``."""

    _require_seed(master_seed, label="master_seed")
    _require_seed(instance_id, label="instance_id")
    if isinstance(n_modes, bool) or not isinstance(n_modes, (int, np.integer)) or int(n_modes) < 1:
        raise ValueError("n_modes must be an integer >= 1")
    modes = int(n_modes)
    if isinstance(n, bool) or not isinstance(n, (int, np.integer)) or int(n) < 4 or int(n) % 2 != 0:
        raise ValueError("n must be an even integer >= 4")
    if int(n) // 2 <= modes:
        raise ValueError("n/2 must be greater than n_modes so every mode is below Nyquist")
    if not math.isfinite(nu_min) or not math.isfinite(nu_max) or nu_min <= 0 or nu_max < nu_min:
        raise ValueError("nu bounds must be finite, positive, and ordered")
    rng = instance_generator(master_seed, instance_id)
    coefficients: tuple[np.ndarray, np.ndarray, float] | None = None
    for _ in range(1, _MAX_ATTEMPTS + 1):
        raw_a = np.empty(modes, dtype=np.float64)
        raw_b = np.empty(modes, dtype=np.float64)
        for m in range(1, modes + 1):
            raw_a[m - 1] = float(rng.uniform(-1.0, 1.0)) / m
            raw_b[m - 1] = float(rng.uniform(-1.0, 1.0)) / m
        peak = float(np.max(np.abs(trigonometric_field(raw_a, raw_b, CANONICAL_N))))
        if peak >= _PEAK_FLOOR:
            coefficients = (raw_a, raw_b, 1.0 / peak)
            break
    if coefficients is None:
        raise RuntimeError("failed to draw a non-trivial Burgers initial condition")
    a, b, scale = coefficients
    nu = float(rng.uniform(nu_min, nu_max))
    u0 = scale * trigonometric_field(a, b, int(n))
    return InitialCondition(
        a=a,
        b=b,
        scale=scale,
        u0=u0,
        nu=nu,
        n_modes=modes,
        instance_id=int(instance_id),
        master_seed=int(master_seed),
    )


def trigonometric_field(a: np.ndarray, b: np.ndarray, n: int) -> np.ndarray:
    """Evaluate ``Σ (a_m cos(m π x) + b_m sin(m π x))`` on the solver grid."""

    cosine = np.asarray(a, dtype=np.float64)
    sine = np.asarray(b, dtype=np.float64)
    if cosine.ndim != 1 or sine.shape != cosine.shape:
        raise ValueError("a and b must be one-dimensional and the same length")
    if cosine.size == 0 or not np.isfinite(cosine).all() or not np.isfinite(sine).all():
        raise ValueError("coefficients must be non-empty and finite")
    x = grid(n)
    field = np.zeros(n, dtype=np.float64)
    for m, (a_m, b_m) in enumerate(zip(cosine, sine, strict=True), start=1):
        angle = m * math.pi * x
        field += a_m * np.cos(angle) + b_m * np.sin(angle)
    return field


def instance_generator(master_seed: int, instance_id: int) -> np.random.Generator:
    """PCG64 generator for one problem instance. The stream is not split-dependent."""

    entropy = [int(master_seed), int(instance_id), IC_SALT]
    return np.random.Generator(np.random.PCG64(np.random.SeedSequence(entropy)))


def _require_seed(value: int, *, label: str) -> None:
    if isinstance(value, bool) or not isinstance(value, (int, np.integer)) or int(value) < 0:
        raise ValueError(f"{label} must be an integer >= 0")
