"""Cross-checks and integral diagnostics for periodic Burgers.

The Cole–Hopf formula is an exact solution of the unforced equation, built
from a positive periodic solution of the heat equation. It is independent
of the Fourier time stepper. A second check is a periodic second-order
finite-difference method of lines with classical RK4, which does not share
the spectral spatial operator.

On a periodic domain the unforced equation conserves the spatial mean and
does not increase the energy

    mean = (1 / L) ∫ u dx,
    E = ∫ u² / 2 dx.

On this uniform grid those integrals are the rectangle sums below. The
rectangle rule is exact for Fourier modes that the grid resolves. ``u²``
can contain a Nyquist component, so the discrete energy is the grid energy,
not a claim of exact continuous quadrature.
"""

from __future__ import annotations

import math

import numpy as np

from pinnforge.reference.numerical.solver import (
    LENGTH,
    SolverConfig,
    Trajectory,
    grid,
    solve,
)

_COLE_HOPF_WAVE = 1


def cole_hopf(
    x: np.ndarray,
    t: np.ndarray | float,
    *,
    nu: float,
    amplitude: float = 0.5,
    wave: int = _COLE_HOPF_WAVE,
) -> np.ndarray:
    """Exact viscous Burgers field from a one-mode heat solution.

    ``φ = 1 + a exp(-ν k² t) cos(k x)`` solves the heat equation
    ``φ_t = ν φ_xx`` when ``k = wave * π`` (the ``wave``-th Fourier mode on
    a domain of length 2). Then

        u = -2 ν (ln φ)_x = 2 ν a k exp(-ν k² t) sin(k x) / φ

    solves ``u_t + u u_x = ν u_xx``. ``|a| < 1`` keeps ``φ`` positive for
    every ``t >= 0``. ``x`` and ``t`` broadcast.
    """

    if isinstance(wave, bool) or not isinstance(wave, (int, np.integer)) or int(wave) < 1:
        raise ValueError("wave must be an integer >= 1")
    if isinstance(amplitude, bool) or not isinstance(amplitude, (int, float, np.floating)):
        raise ValueError("amplitude must be a real number")
    if not math.isfinite(amplitude) or not 0.0 < abs(float(amplitude)) < 1.0:
        raise ValueError("amplitude must be finite and satisfy 0 < |a| < 1")
    if isinstance(nu, bool) or not isinstance(nu, (int, float, np.floating)):
        raise ValueError("nu must be a real number")
    viscosity = float(nu)
    if not math.isfinite(viscosity) or viscosity <= 0:
        raise ValueError("nu must be finite and > 0")
    k = float(wave) * math.pi
    xx = np.asarray(x, dtype=np.float64)
    tt = np.asarray(t, dtype=np.float64)
    xx, tt = np.broadcast_arrays(xx, tt)
    decay = np.exp(-viscosity * k * k * tt)
    phi = 1.0 + float(amplitude) * decay * np.cos(k * xx)
    if np.any(phi <= 0.0):
        raise ValueError("Cole-Hopf denominator is not positive")
    return 2.0 * viscosity * float(amplitude) * k * decay * np.sin(k * xx) / phi


def cole_hopf_initial(
    n: int,
    *,
    nu: float,
    amplitude: float = 0.5,
    wave: int = _COLE_HOPF_WAVE,
) -> np.ndarray:
    """Sample the Cole–Hopf field at ``t = 0`` on the solver grid."""

    return np.asarray(
        cole_hopf(grid(n), 0.0, nu=nu, amplitude=amplitude, wave=wave),
        dtype=np.float64,
    )


def finite_difference_solve(
    u0: np.ndarray,
    nu: float,
    *,
    dt: float,
    t_final: float = 1.0,
    save_dt: float | None = None,
) -> Trajectory:
    """Second-order periodic finite differences with classical RK4.

    The spatial operators are the standard centered stencils

        u_x(x_j) ≈ (u_{j+1} - u_{j-1}) / (2 dx),
        u_xx(x_j) ≈ (u_{j+1} - 2 u_j + u_{j-1}) / dx²,

    with indices modulo ``N``. This scheme is independent of the Fourier
    solver. It is second-order in space and fourth-order in time, and it
    is only stable when ``dt`` also resolves the viscous eigenvalues
    ``≈ 4 ν / dx²``. It is a cross-check, not the source of pilot labels.
    """

    field = np.asarray(u0, dtype=np.float64)
    if field.ndim != 1 or field.size < 4 or field.size % 2 != 0:
        raise ValueError("u0 must be an even-length field with n >= 4")
    if not np.isfinite(field).all():
        raise ValueError("initial field must be finite")
    if isinstance(nu, bool) or not isinstance(nu, (int, float, np.floating)):
        raise ValueError("nu must be a real number")
    viscosity = float(nu)
    if not math.isfinite(viscosity) or viscosity <= 0:
        raise ValueError("nu must be finite and > 0")
    if not math.isfinite(dt) or dt <= 0 or not math.isfinite(t_final) or t_final <= 0:
        raise ValueError("dt and t_final must be finite and > 0")
    stored_dt = dt if save_dt is None else save_dt
    if not math.isfinite(stored_dt) or stored_dt <= 0:
        raise ValueError("save_dt must be finite and > 0")
    n = int(field.shape[0])
    dx = LENGTH / n
    step_count = _multiple(t_final, dt, label="t_final / dt")
    save_stride = _multiple(stored_dt, dt, label="save_dt / dt")
    if step_count % save_stride != 0:
        raise ValueError("save_dt must divide t_final in an integer number of steps")
    n_save = step_count // save_stride + 1
    stored = np.empty((n_save, n), dtype=np.float64)
    stored[0] = field
    state = field.copy()
    frame = 1
    for step in range(1, step_count + 1):
        state = _rk4(state, viscosity, dt, dx)
        if step % save_stride == 0:
            if not np.isfinite(state).all():
                raise RuntimeError(f"finite-difference solver produced a non-finite field at step {step}")
            stored[frame] = state
            frame += 1
    config = SolverConfig(
        n=n,
        dt=float(dt),
        t_final=float(t_final),
        save_dt=float(stored_dt),
        dealiasing="none",
        integrator="rk4",
        precision="float64",
    )
    times = (np.arange(n_save, dtype=np.float64) * save_stride) * float(dt)
    return Trajectory(x=grid(n), t=times, u=stored, nu=viscosity, dt=float(dt), config=config)


def spatial_mean(u: np.ndarray) -> np.ndarray:
    """Discrete mean ``(1/N) Σ_j u_j``, equal to ``(1/L) ∫ u dx`` for resolved modes."""

    values = np.asarray(u, dtype=np.float64)
    return np.mean(values, axis=-1)


def energy(u: np.ndarray) -> np.ndarray:
    """Discrete energy ``(dx / 2) Σ_j u_j²``, the rectangle rule for ``∫ u² / 2 dx``."""

    values = np.asarray(u, dtype=np.float64)
    dx = LENGTH / values.shape[-1]
    return 0.5 * dx * np.sum(values * values, axis=-1)


def relative_l2(candidate: np.ndarray, reference: np.ndarray) -> float:
    """``||candidate - reference|| / ||reference||`` on a shared flattened grid.

    The grid weight cancels when both arrays use the same nodes, so this is
    the relative discrete L2 norm.
    """

    got = np.asarray(candidate, dtype=np.float64)
    truth = np.asarray(reference, dtype=np.float64)
    if got.shape != truth.shape:
        raise ValueError(f"shape mismatch: {got.shape} vs {truth.shape}")
    denom = float(np.linalg.norm(truth.ravel()))
    if denom == 0.0 or not math.isfinite(denom):
        raise ValueError("reference norm must be finite and non-zero")
    error = float(np.linalg.norm((got - truth).ravel()) / denom)
    if not math.isfinite(error):
        raise ValueError("relative L2 is not finite")
    return error


def restrict_fourier(u: np.ndarray, n_out: int) -> np.ndarray:
    """Truncate a periodic grid field onto ``n_out`` modes and nodes.

    Coefficients use NumPy's unnormalized FFT, so the coarse coefficients
    are rescaled by ``n_out / n_in``. Modes at or above the coarse Nyquist
    wavenumber are dropped.
    """

    values = np.asarray(u, dtype=np.float64)
    n_in = values.shape[-1]
    if isinstance(n_out, bool) or not isinstance(n_out, (int, np.integer)):
        raise ValueError("n_out must be an even integer")
    n_out = int(n_out)
    if n_out < 4 or n_out % 2 != 0 or n_in % 2 != 0 or n_out > n_in:
        raise ValueError("n_out must be even, >= 4, and <= the input length")
    spectrum = np.fft.fft(values, axis=-1)
    half = n_out // 2
    coarse = np.zeros(values.shape[:-1] + (n_out,), dtype=np.complex128)
    coarse[..., :half] = spectrum[..., :half]
    coarse[..., half + 1 :] = spectrum[..., -(half - 1) :]
    coarse *= n_out / n_in
    return np.fft.ifft(coarse, axis=-1).real


def spectral_error_against_cole_hopf(
    *,
    n: int,
    nu: float,
    dt: float,
    t_final: float,
    amplitude: float = 0.5,
    save_dt: float | None = None,
) -> tuple[Trajectory, float, float]:
    """Integrate the Cole–Hopf initial field and return final and space-time errors."""

    initial = cole_hopf_initial(n, nu=nu, amplitude=amplitude)
    trajectory = solve(initial, nu, dt=dt, t_final=t_final, save_dt=save_dt or dt)
    exact = cole_hopf(trajectory.x[None, :], trajectory.t[:, None], nu=nu, amplitude=amplitude)
    final = relative_l2(trajectory.u[-1], exact[-1])
    spacetime = relative_l2(trajectory.u, exact)
    return trajectory, final, spacetime


def _rk4(u: np.ndarray, nu: float, dt: float, dx: float) -> np.ndarray:
    k1 = _burgers_fd(u, nu, dx)
    k2 = _burgers_fd(u + 0.5 * dt * k1, nu, dx)
    k3 = _burgers_fd(u + 0.5 * dt * k2, nu, dx)
    k4 = _burgers_fd(u + dt * k3, nu, dx)
    return u + (dt / 6.0) * (k1 + 2.0 * k2 + 2.0 * k3 + k4)


def _burgers_fd(u: np.ndarray, nu: float, dx: float) -> np.ndarray:
    up = np.roll(u, -1)
    um = np.roll(u, 1)
    ux = (up - um) / (2.0 * dx)
    uxx = (up - 2.0 * u + um) / (dx * dx)
    return -u * ux + nu * uxx


def _multiple(total: float, step: float, *, label: str) -> int:
    ratio = total / step
    count = int(round(ratio))
    if count < 1 or abs(count * step - total) > 1e-8 * max(1.0, abs(total)):
        raise ValueError(f"{label} must be an integer >= 1 (got {total} / {step})")
    return count
