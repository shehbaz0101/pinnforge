"""Dealiased Fourier spectral solver for periodic viscous Burgers.

The domain is fixed: ``x ∈ [X_LOWER, X_UPPER]`` with length ``LENGTH = 2``,
periodic, so the last endpoint is identified with the first and is not a
grid node. Grid points are

    x_j = -1 + j * (L / N),    j = 0, ..., N - 1.

Wave numbers use the ordinary derivative on a periodic interval of length
``L``,

    k_m = (2 π / L) * m,    m = 0, 1, ..., N/2 - 1, -N/2, ..., -1,

which is ``k = 2 π * numpy.fft.fftfreq(N, d=L/N)``. The fundamental mode
``m = 1`` therefore has ``k = π``. The Nyquist entry ``m = N/2`` (even
``N``) is set to 0: the corresponding grid mode is not a reliable odd
derivative, and it is removed from the solution after every step.

A derivative of order ``p`` multiplies the unnormalized NumPy FFT by
``(i k)^p``. NumPy's forward transform is the sum

    U_m = Σ_j u_j exp(-2 π i m j / N),

and ``ifft`` divides by ``N``. With that convention, ``i k U`` is the FFT
of ``∂_x u`` with no extra scaling.

The quadratic term ``u u_x`` is dealiased with the 3/2 rule (Orszag).
Spectra are padded to ``M = 3 N / 2`` modes, the Nyquist mode is dropped,
the product is formed on the fine grid, and the result is truncated back
to ``N`` modes. ``N`` must be even so that ``M`` is an integer.

Time stepping is ETDRK4 (Cox and Matthews, 2002), with the Kassam and
Trefethen (2005) coefficients evaluated in closed form. In Fourier space

    ∂_t U = L U + N(U),    L = -ν k²,    N(U) = -FFT(u u_x),

diffusion is the exact exponential factor, and the nonlinear term is
explicit. The practical step restriction is therefore the explicit
advection term, not the viscous eigenvalues. A working rule used by the
pilot is

    dt <= 0.5 / (||u||_∞ * k_max),    k_max = π * (N/2 - 1),

which for ``||u||_∞ <= 1`` and ``N = 256`` allows ``dt <= 1.25e-3``.
This is a stability guide, not an accuracy certificate. Accuracy is the
convergence study.

Fields and coefficients are ``float64``. Spectra are ``complex128``.
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Any

import numpy as np

X_LOWER = -1.0
X_UPPER = 1.0
LENGTH = X_UPPER - X_LOWER
_SERIES_ORDER = 18
_Z_CUT = 1.0


@dataclass(frozen=True)
class SolverConfig:
    """Discrete settings that define one Burgers integration.

    ``dt`` and ``save_dt`` are in the same units as ``t_final``. ``save_dt``
    must be an integer multiple of ``dt``, and ``t_final`` an integer
    multiple of ``save_dt``, up to a roundoff tolerance checked in
    :func:`solve`.
    """

    n: int
    dt: float
    t_final: float = 1.0
    save_dt: float = 0.01
    dealiasing: str = "three_halves"
    integrator: str = "etdrk4"
    precision: str = "float64"

    def to_dict(self) -> dict[str, Any]:
        return {
            "n": self.n,
            "dt": self.dt,
            "t_final": self.t_final,
            "save_dt": self.save_dt,
            "dealiasing": self.dealiasing,
            "integrator": self.integrator,
            "precision": self.precision,
            "domain": [X_LOWER, X_UPPER],
            "length": LENGTH,
            "wavenumber": "k_m = 2*pi/L * m",
            "nyquist": "derivative multiplier and coefficient set to 0",
        }


@dataclass(frozen=True)
class Trajectory:
    """One periodic Burgers trajectory on the solver grid.

    ``u[s, j]`` is the field at time ``t[s]`` and node ``x[j]``. ``t`` is
    ``step * dt`` for the saved steps, which can differ from a decimal
    literal such as ``0.01`` by a unit in the last place.
    """

    x: np.ndarray
    t: np.ndarray
    u: np.ndarray
    nu: float
    dt: float
    config: SolverConfig


def grid(n: int) -> np.ndarray:
    """Return the ``N`` periodic nodes on ``[-1, 1)``."""

    _require_even_n(n)
    dx = LENGTH / n
    return X_LOWER + dx * np.arange(n, dtype=np.float64)


def wavenumbers(n: int) -> np.ndarray:
    """Return ``k_m = 2 π m / L`` in NumPy FFT order, with Nyquist zeroed."""

    _require_even_n(n)
    dx = LENGTH / n
    k = 2.0 * math.pi * np.fft.fftfreq(n, d=dx)
    k = np.asarray(k, dtype=np.float64)
    k[n // 2] = 0.0
    return k


def spectral_derivative(u: np.ndarray, *, order: int = 1) -> np.ndarray:
    """Differentiate a periodic field with the Fourier multiplier ``(i k)^p``."""

    values = _as_field(u)
    if order < 1:
        raise ValueError("order must be >= 1")
    k = wavenumbers(values.shape[-1])
    multiplier = (1j * k) ** order
    derivative = np.fft.ifft(multiplier * np.fft.fft(values, axis=-1), axis=-1).real
    return np.asarray(derivative, dtype=np.float64)


def recommended_dt(n: int, u_max: float, *, cfl: float = 0.5) -> float:
    """Explicit-advection step guide ``cfl / (||u||_∞ k_max)``.

    ``k_max`` is the highest non-Nyquist wave number, ``π (N/2 - 1)``.
    Diffusion does not enter, because ETDRK4 treats ``-ν k²`` exactly.
    """

    _require_even_n(n)
    if not math.isfinite(cfl) or cfl <= 0:
        raise ValueError("cfl must be finite and > 0")
    if not math.isfinite(u_max) or u_max < 0:
        raise ValueError("u_max must be finite and >= 0")
    k_max = math.pi * (n // 2 - 1)
    return cfl / (u_max * k_max + 1e-15)


def solve(
    u0: np.ndarray,
    nu: float,
    *,
    dt: float,
    t_final: float = 1.0,
    save_dt: float | None = None,
) -> Trajectory:
    """Integrate one real initial field to ``t_final``.

    ``u0`` is sampled on :func:`grid`. ``nu`` must be finite and positive.
    The returned field is ``float64``.
    """

    field = _as_field(u0)
    if field.ndim != 1:
        raise ValueError("u0 must be a one-dimensional field")
    viscosity = _require_viscosity(nu)
    config = _config(n=field.shape[0], dt=dt, t_final=t_final, save_dt=save_dt)
    stacked, times = _integrate(field[None, :], np.asarray([viscosity]), config)
    return Trajectory(
        x=grid(field.shape[0]),
        t=times,
        u=stacked[0],
        nu=viscosity,
        dt=config.dt,
        config=config,
    )


def solve_batch(
    u0: np.ndarray,
    nu: np.ndarray,
    *,
    dt: float,
    t_final: float = 1.0,
    save_dt: float | None = None,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, SolverConfig]:
    """Integrate a batch of fields that share ``N``, ``dt``, and ``t_final``.

    ``u0`` has shape ``(batch, N)`` and ``nu`` has shape ``(batch,)``.
    Returns ``u`` with shape ``(batch, n_times, N)``, the time nodes, the
    space nodes, and the shared config. Viscosities may differ.
    """

    fields = _as_field(u0)
    if fields.ndim != 2:
        raise ValueError("u0 must have shape (batch, n)")
    viscosities = np.asarray(nu, dtype=np.float64)
    if viscosities.shape != (fields.shape[0],):
        raise ValueError("nu must have shape (batch,)")
    if not np.isfinite(viscosities).all() or np.any(viscosities <= 0):
        raise ValueError("nu must be finite and > 0")
    config = _config(n=fields.shape[1], dt=dt, t_final=t_final, save_dt=save_dt)
    stacked, times = _integrate(fields, viscosities, config)
    return stacked, times, grid(fields.shape[1]), config


def dealiased_advection_hat(u_hat: np.ndarray, k: np.ndarray) -> np.ndarray:
    """FFT of ``u u_x`` with the 3/2 dealiasing rule.

    ``u_hat`` is a NumPy FFT on the last axis, shape ``(N,)`` or
    ``(batch, N)``. The returned array uses that same unnormalized
    convention.
    """

    spectrum = np.asarray(u_hat)
    if spectrum.dtype != np.complex128:
        spectrum = spectrum.astype(np.complex128, copy=False)
    n = spectrum.shape[-1]
    _require_even_n(n)
    if k.shape != (n,):
        raise ValueError("k must have shape (n,)")
    half = n // 2
    fine = (3 * n) // 2
    ux_hat = (1j * k) * spectrum
    u_pad = _pad_spectrum(spectrum, fine, half)
    ux_pad = _pad_spectrum(ux_hat, fine, half)
    scale_up = fine / n
    u_phys = np.fft.ifft(u_pad, axis=-1).real * scale_up
    ux_phys = np.fft.ifft(ux_pad, axis=-1).real * scale_up
    product_hat = np.fft.fft(u_phys * ux_phys, axis=-1) * (n / fine)
    return _truncate_spectrum(product_hat, n, half)


def etdrk4_coefficients(
    linear: np.ndarray,
    dt: float,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """Return ``E, E2, Q, f1, f2, f3`` for diagonal ``linear = L``.

    For ``|dt L| < 1`` the ``φ`` combinations use a Taylor series about 0.
    Elsewhere they use an ``expm1`` form of the Kassam–Trefethen formulas.
    ``Q``, ``f1``, ``f2``, and ``f3`` already include one factor of ``dt``.
    """

    if not math.isfinite(dt) or dt <= 0:
        raise ValueError("dt must be finite and > 0")
    z = np.asarray(dt * linear, dtype=np.float64)
    q, g1, g2, g3 = _phi_combinations(z)
    step = np.float64(dt)
    return np.exp(z), np.exp(z / 2.0), step * q, step * g1, step * g2, step * g3


def _integrate(
    fields: np.ndarray,
    viscosities: np.ndarray,
    config: SolverConfig,
) -> tuple[np.ndarray, np.ndarray]:
    n = fields.shape[-1]
    k = wavenumbers(n)
    u_hat = np.fft.fft(fields, axis=-1)
    u_hat[..., n // 2] = 0.0
    linear = -viscosities[:, None] * (k * k)[None, :]
    step_count = _multiple(config.t_final, config.dt, label="t_final / dt")
    save_stride = _multiple(config.save_dt, config.dt, label="save_dt / dt")
    if step_count % save_stride != 0:
        raise ValueError("save_dt must divide t_final in an integer number of steps")
    n_save = step_count // save_stride + 1
    stored = np.empty((fields.shape[0], n_save, n), dtype=np.float64)
    stored[:, 0, :] = fields
    coefficients = etdrk4_coefficients(linear, config.dt)
    frame = 1
    for step in range(1, step_count + 1):
        u_hat = _etdrk4_step(u_hat, k, *coefficients)
        u_hat[..., n // 2] = 0.0
        if step % save_stride == 0:
            stored[:, frame, :] = np.fft.ifft(u_hat, axis=-1).real
            if not np.isfinite(stored[:, frame, :]).all():
                raise RuntimeError(f"Burgers solver produced a non-finite field at step {step}")
            frame += 1
    times = (np.arange(n_save, dtype=np.float64) * save_stride) * config.dt
    return stored, times


def _etdrk4_step(
    u_hat: np.ndarray,
    k: np.ndarray,
    exponential: np.ndarray,
    exponential_half: np.ndarray,
    q_coeff: np.ndarray,
    f1: np.ndarray,
    f2: np.ndarray,
    f3: np.ndarray,
) -> np.ndarray:
    nonlinear = -dealiased_advection_hat(u_hat, k)
    stage_a = exponential_half * u_hat + q_coeff * nonlinear
    nonlinear_a = -dealiased_advection_hat(stage_a, k)
    stage_b = exponential_half * u_hat + q_coeff * nonlinear_a
    nonlinear_b = -dealiased_advection_hat(stage_b, k)
    stage_c = exponential_half * stage_a + q_coeff * (2.0 * nonlinear_b - nonlinear)
    nonlinear_c = -dealiased_advection_hat(stage_c, k)
    return (
        exponential * u_hat
        + f1 * nonlinear
        + 2.0 * f2 * (nonlinear_a + nonlinear_b)
        + f3 * nonlinear_c
    )


def _phi_combinations(z: np.ndarray) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """Evaluate ``q, g1, g2, g3`` so the step coefficients are ``dt`` times these."""

    small = np.abs(z) < _Z_CUT
    q_series, g1_series, g2_series, g3_series = _series_phi(z)
    with np.errstate(divide="ignore", invalid="ignore", over="ignore"):
        expm1_z = np.expm1(z)
        expm1_half = np.expm1(z / 2.0)
        q_closed = expm1_half / z
        poly = 4.0 - 3.0 * z + z * z
        g1_closed = ((z * z - 4.0 * z) + expm1_z * poly) / (z**3)
        g2_closed = (2.0 * z + expm1_z * (z - 2.0)) / (z**3)
        g3_closed = ((-z * z - 4.0 * z) + expm1_z * (4.0 - z)) / (z**3)
    return (
        np.where(small, q_series, q_closed),
        np.where(small, g1_series, g1_closed),
        np.where(small, g2_series, g2_closed),
        np.where(small, g3_series, g3_closed),
    )


def _series_phi(z: np.ndarray) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    q, g1, g2, g3 = _phi_series_coefficients()
    return _horner(q, z), _horner(g1, z), _horner(g2, z), _horner(g3, z)


def _horner(coeffs: np.ndarray, z: np.ndarray) -> np.ndarray:
    value = np.full(z.shape, coeffs[-1], dtype=np.float64)
    for coeff in coeffs[-2::-1]:
        value = value * z + coeff
    return value


def _phi_series_coefficients() -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """Taylor coefficients of the ETDRK4 ``φ`` combinations about ``z = 0``.

    The series are built from ``exp`` by polynomial arithmetic so the
    removable singularities at the origin are never evaluated. Coefficient
    ``c[k]`` multiplies ``z^k``. At ``z = 0`` the classical RK4 limits are
    ``q = 1/2`` and ``g1 = g2 = g3 = 1/6``.
    """

    cached = getattr(_phi_series_coefficients, "_cache", None)
    if cached is not None:
        return cached
    order = _SERIES_ORDER
    length = order + 3
    exp_coeff = np.zeros(length, dtype=np.float64)
    exp_half = np.zeros(length, dtype=np.float64)
    exp_coeff[0] = 1.0
    exp_half[0] = 1.0
    term = 1.0
    half_term = 1.0
    for k in range(1, length):
        term /= k
        half_term *= 0.5 / k
        exp_coeff[k] = term
        exp_half[k] = half_term
    expm1 = exp_coeff.copy()
    expm1[0] = 0.0
    # q(z) = (exp(z/2) - 1) / z = Σ_{k>=0} exp_half[k+1] z^k.
    q = exp_half[1 : order + 1].copy()
    g1 = _shift3(
        _add(
            _monomial_quadratic(length, 0.0, -4.0, 1.0),
            _mul(expm1, _poly(length, 4.0, -3.0, 1.0)),
        )
    )
    g2 = _shift3(
        _add(
            _monomial_quadratic(length, 0.0, 2.0, 0.0),
            _mul(expm1, _poly(length, -2.0, 1.0, 0.0)),
        )
    )
    g3 = _shift3(
        _add(
            _monomial_quadratic(length, 0.0, -4.0, -1.0),
            _mul(expm1, _poly(length, 4.0, -1.0, 0.0)),
        )
    )
    cached = (q[:order], g1[:order], g2[:order], g3[:order])
    _phi_series_coefficients._cache = cached  # type: ignore[attr-defined]
    return cached


def _poly(length: int, c0: float, c1: float, c2: float) -> np.ndarray:
    values = np.zeros(length, dtype=np.float64)
    values[0] = c0
    if length > 1:
        values[1] = c1
    if length > 2:
        values[2] = c2
    return values


def _monomial_quadratic(length: int, c0: float, c1: float, c2: float) -> np.ndarray:
    return _poly(length, c0, c1, c2)


def _mul(left: np.ndarray, right: np.ndarray) -> np.ndarray:
    return np.convolve(left, right)[: left.size]


def _add(left: np.ndarray, right: np.ndarray) -> np.ndarray:
    return left + right


def _shift3(numerator: np.ndarray) -> np.ndarray:
    """Divide a series with a root of order 3 at the origin by ``z^3``."""

    if np.max(np.abs(numerator[:3])) > 1e-9:
        raise RuntimeError("ETDRK4 series numerator is not divisible by z^3")
    return numerator[3:]


def _pad_spectrum(spectrum: np.ndarray, fine: int, half: int) -> np.ndarray:
    padded = np.zeros(spectrum.shape[:-1] + (fine,), dtype=np.complex128)
    padded[..., :half] = spectrum[..., :half]
    padded[..., -(half - 1) :] = spectrum[..., half + 1 :]
    return padded


def _truncate_spectrum(spectrum: np.ndarray, n: int, half: int) -> np.ndarray:
    truncated = np.zeros(spectrum.shape[:-1] + (n,), dtype=np.complex128)
    truncated[..., :half] = spectrum[..., :half]
    truncated[..., half + 1 :] = spectrum[..., -(half - 1) :]
    return truncated


def _config(*, n: int, dt: float, t_final: float, save_dt: float | None) -> SolverConfig:
    _require_even_n(n)
    if not math.isfinite(dt) or dt <= 0:
        raise ValueError("dt must be finite and > 0")
    if not math.isfinite(t_final) or t_final <= 0:
        raise ValueError("t_final must be finite and > 0")
    stored = dt if save_dt is None else save_dt
    if not math.isfinite(stored) or stored <= 0:
        raise ValueError("save_dt must be finite and > 0")
    return SolverConfig(n=n, dt=float(dt), t_final=float(t_final), save_dt=float(stored))


def _multiple(total: float, step: float, *, label: str) -> int:
    ratio = total / step
    count = int(round(ratio))
    if count < 1 or abs(count * step - total) > 1e-8 * max(1.0, abs(total)):
        raise ValueError(f"{label} must be an integer >= 1 (got {total} / {step})")
    return count


def _require_even_n(n: int) -> None:
    if isinstance(n, bool) or not isinstance(n, (int, np.integer)) or int(n) < 4 or int(n) % 2 != 0:
        raise ValueError("n must be an even integer >= 4")


def _require_viscosity(nu: float) -> float:
    if isinstance(nu, bool) or not isinstance(nu, (int, float, np.floating)):
        raise ValueError("nu must be a real number")
    value = float(nu)
    if not math.isfinite(value) or value <= 0:
        raise ValueError("nu must be finite and > 0")
    return value


def _as_field(u: np.ndarray) -> np.ndarray:
    values = np.asarray(u, dtype=np.float64)
    if values.size == 0 or not np.isfinite(values).all():
        raise ValueError("initial field must be non-empty and finite")
    return values
