"""Harder periodic Burgers pilot: richer initial data and lower viscosity.

The Stage 2 pilot (:mod:`pinnforge.reference.numerical.dataset`) draws
modes ``1..4`` with amplitudes ``Uniform(-1, 1) / m`` and viscosities in
``[0.02, 0.10]``. This module is a separate family and a separate
manifest. It does not change those defaults, those seeds, or that
manifest.

The initial-condition family is ``tanh_bandlimited``. For modes
``m = 1 .. 8``,

    a_m = Uniform(-1, 1) / sqrt(m),
    b_m = Uniform(-1, 1) / sqrt(m),

in that order, from PCG64 and ``SeedSequence([master_seed, instance_id,
0x48415244])``. Let ``p`` be that trigonometric polynomial. On the
canonical grid of 8192 nodes,

    p_hat = (p - mean(p)) / max|p - mean(p)|,
    s = tanh(3 p_hat).

``s`` is projected onto Fourier modes ``|m| <= 48``, the mean mode is
removed, and the result is divided by its maximum absolute value on the
canonical grid. A resolved solver grid needs ``N/2 > 48``. The
projection makes the stored field an exact trigonometric polynomial:
both the canonical grid and a finer solver grid sample that same
polynomial. Viscosity is drawn only after a successful field:

    ν ~ Uniform(0.005, 0.10).

``tanh(3 p_hat)`` steepens the Stage 2 profile, and keeping 48 modes
leaves a much wider band than modes ``1..4``. The lower viscosity keeps
the evolved front thinner. The solver is the same dealiased Fourier
ETDRK4 scheme. This module does not train an operator.
"""

from __future__ import annotations

import hashlib
import json
import math
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np

from pinnforge.reference.numerical.checks import energy, spatial_mean
from pinnforge.reference.numerical.dataset import (
    SPLIT_SALT,
    RunningStats,
    _field_sha256,
    _frame_count,
    assign_splits,
    sha256_json,
)
from pinnforge.reference.numerical.initial import NU_MIN as STAGE2_NU_MIN
from pinnforge.reference.numerical.initial import trigonometric_field
from pinnforge.reference.numerical.solver import SolverConfig, solve_batch, spectral_derivative

FORMAT = "pinnforge.burgers_hard_pilot.v1"
IC_FAMILY = "tanh_bandlimited"
# ASCII "HARD". Distinct from the Stage 2 salt 0x42555247 so the streams
# are not a prefix of each other.
HARD_IC_SALT = 0x48415244
N_MODES = 8
M_KEEP = 48
BETA = 3.0
AMPLITUDE_DECAY = 0.5
CANONICAL_N = 8192
NU_MIN = 0.005
NU_MAX = 0.10
# Label grid. These match the Stage A convergence study: the pilot step
# stays inside the explicit-advection guide at ||u||_∞ = 1, and the
# comparison against REFERENCE_N / REFERENCE_DT meets the gate below at
# ν = NU_MIN. Update both together if the study is rerun.
HARD_N = 1024
HARD_DT = 2.5e-4
HARD_T_FINAL = 1.0
HARD_SAVE_DT = 0.01
HARD_MASTER_SEED = 20260929
HARD_SPLIT_SEED = 20260929
REFERENCE_N = 2048
REFERENCE_DT = 1.25e-4
# Tighter than the Stage 2 gate of 1e-8. The Stage A study measures about
# 4e-11 in space-time relative L2 at ν = 0.005 on the steep initial data.
LABEL_REL_L2_MAX = 1e-9
LABEL_MEAN_DRIFT_MAX = 1e-12
LABEL_ENERGY_INCREASE_MAX = 1e-10
PROTOCOL_FORMAT = "pinnforge.burgers_hard_pilot_protocol.v1"
HARD_OOD_NAME = "hard_ood"
HARD_OOD_QUANTILE = 0.25
HARD_OOD_QUANTILE_METHOD = "linear"
_PEAK_FLOOR = 1e-8
_MAX_ATTEMPTS = 8
_CODE_SOURCES = ("solver.py", "initial.py", "dataset.py", "checks.py", "harder.py")


@dataclass(frozen=True)
class HardPilotConfig:
    """Sizes and discrete settings for one harder-pilot generation.

    Counts are problem instances, not windows. ``nu_min``, ``n_modes``,
    ``m_keep``, ``beta``, and ``amplitude_decay`` define the family.
    They are not Stage 2 defaults.
    """

    n_train: int = 512
    n_val: int = 128
    n_test: int = 128
    n: int = HARD_N
    dt: float = HARD_DT
    t_final: float = HARD_T_FINAL
    save_dt: float = HARD_SAVE_DT
    master_seed: int = HARD_MASTER_SEED
    split_seed: int = HARD_SPLIT_SEED
    n_modes: int = N_MODES
    m_keep: int = M_KEEP
    beta: float = BETA
    amplitude_decay: float = AMPLITUDE_DECAY
    batch_size: int = 8
    nu_min: float = NU_MIN
    nu_max: float = NU_MAX

    def total(self) -> int:
        return self.n_train + self.n_val + self.n_test

    def to_dict(self) -> dict[str, Any]:
        return {
            "profile": "harder_burgers_v1",
            "ic_family": IC_FAMILY,
            "n_train": self.n_train,
            "n_val": self.n_val,
            "n_test": self.n_test,
            "n": self.n,
            "dt": self.dt,
            "t_final": self.t_final,
            "save_dt": self.save_dt,
            "master_seed": self.master_seed,
            "split_seed": self.split_seed,
            "n_modes": self.n_modes,
            "m_keep": self.m_keep,
            "beta": self.beta,
            "amplitude_decay": self.amplitude_decay,
            "batch_size": self.batch_size,
            "nu_min": self.nu_min,
            "nu_max": self.nu_max,
            "ic_salt": HARD_IC_SALT,
            "split_salt": SPLIT_SALT,
            "canonical_n": CANONICAL_N,
            "bit_generator": "PCG64",
            "windowing": "none",
            "split_unit": "problem_instance",
            "coefficient_law": (
                "a_m, b_m = Uniform(-1, 1) / m^decay; "
                "s = tanh(beta * p_hat) on the canonical grid; "
                "keep modes |m| <= m_keep, drop the mean, scale by 1/max|S|"
            ),
        }


@dataclass(frozen=True)
class HardInitialCondition:
    """One normalized bandlimited initial condition and its viscosity.

    ``a`` and ``b`` are the pre-sharpening trigonometric coefficients.
    On a grid that resolves modes through ``m_keep``, ``u0`` is the
    canonical projection described in the module docstring.
    """

    a: np.ndarray
    b: np.ndarray
    scale: float
    beta: float
    amplitude_decay: float
    u0: np.ndarray
    nu: float
    n_modes: int
    m_keep: int
    instance_id: int
    master_seed: int


def draw_hard_initial_condition(
    master_seed: int,
    instance_id: int,
    n: int,
    *,
    n_modes: int = N_MODES,
    m_keep: int = M_KEEP,
    beta: float = BETA,
    amplitude_decay: float = AMPLITUDE_DECAY,
    nu_min: float = NU_MIN,
    nu_max: float = NU_MAX,
) -> HardInitialCondition:
    """Draw one harder initial condition and one viscosity for ``instance_id``."""

    _require_seed(master_seed, label="master_seed")
    _require_seed(instance_id, label="instance_id")
    modes = _require_modes(n_modes)
    kept = _require_keep(m_keep)
    _require_resolved_grid(n, kept)
    _require_decay(amplitude_decay)
    _require_beta(beta)
    if not math.isfinite(nu_min) or not math.isfinite(nu_max) or nu_min <= 0 or nu_max < nu_min:
        raise ValueError("nu bounds must be finite, positive, and ordered")
    if kept < modes:
        raise ValueError("m_keep must be >= n_modes")
    rng = instance_generator(master_seed, instance_id)
    drawn: tuple[np.ndarray, np.ndarray, float, np.ndarray] | None = None
    for _ in range(1, _MAX_ATTEMPTS + 1):
        raw_a = np.empty(modes, dtype=np.float64)
        raw_b = np.empty(modes, dtype=np.float64)
        for m in range(1, modes + 1):
            weight = float(m) ** (-float(amplitude_decay))
            raw_a[m - 1] = float(rng.uniform(-1.0, 1.0)) * weight
            raw_b[m - 1] = float(rng.uniform(-1.0, 1.0)) * weight
        projected = _project(raw_a, raw_b, beta=float(beta), m_keep=kept)
        if projected is not None:
            scale, field = projected
            drawn = (raw_a, raw_b, scale, field)
            break
    if drawn is None:
        raise RuntimeError("failed to draw a non-trivial harder Burgers initial condition")
    a, b, scale, canonical = drawn
    nu = float(rng.uniform(nu_min, nu_max))
    u0 = resample_bandlimited(canonical, int(n))
    return HardInitialCondition(
        a=a,
        b=b,
        scale=scale,
        beta=float(beta),
        amplitude_decay=float(amplitude_decay),
        u0=u0,
        nu=nu,
        n_modes=modes,
        m_keep=kept,
        instance_id=int(instance_id),
        master_seed=int(master_seed),
    )


def resample_bandlimited(canonical: np.ndarray, n: int) -> np.ndarray:
    """Sample a canonical bandlimited field on ``n`` periodic nodes.

    ``canonical`` is the scaled projection on :data:`CANONICAL_N` nodes.
    Modes at or above the output Nyquist wavenumber are dropped, which
    is exact when ``n`` resolves every retained mode.
    """

    values = np.asarray(canonical, dtype=np.float64)
    if values.shape != (CANONICAL_N,):
        raise ValueError(f"canonical field must have shape ({CANONICAL_N},)")
    if isinstance(n, bool) or not isinstance(n, (int, np.integer)) or int(n) < 4 or int(n) % 2 != 0:
        raise ValueError("n must be an even integer >= 4")
    if int(n) > CANONICAL_N:
        raise ValueError("n must be <= the canonical grid")
    spectrum = np.fft.fft(values)
    n_out = int(n)
    half = n_out // 2
    coarse = np.zeros(n_out, dtype=np.complex128)
    coarse[:half] = spectrum[:half]
    coarse[half + 1 :] = spectrum[-(half - 1) :]
    coarse *= n_out / CANONICAL_N
    resampled = np.fft.ifft(coarse).real
    return np.asarray(resampled, dtype=np.float64)


def instance_generator(master_seed: int, instance_id: int) -> np.random.Generator:
    """PCG64 generator for one harder-pilot instance. The stream is not split-dependent."""

    entropy = [int(master_seed), int(instance_id), HARD_IC_SALT]
    return np.random.Generator(np.random.PCG64(np.random.SeedSequence(entropy)))


def high_mode_energy_fraction(field: np.ndarray, *, cutoff: int = 4) -> float:
    """Fraction of Fourier energy in modes ``|m| > cutoff``.

    Stage 2 initial data is exactly modes ``1..4``, so this fraction is
    numerical noise there. The bandlimited tanh family keeps modes
    through ``M_KEEP``.
    """

    values = np.asarray(field, dtype=np.float64)
    if values.ndim != 1 or values.size < 4:
        raise ValueError("field must be a one-dimensional grid with at least 4 nodes")
    if isinstance(cutoff, bool) or not isinstance(cutoff, (int, np.integer)) or int(cutoff) < 0:
        raise ValueError("cutoff must be an integer >= 0")
    spectrum = np.fft.fft(values)
    energy_density = np.abs(spectrum) ** 2
    total = float(np.sum(energy_density))
    if total == 0.0 or not math.isfinite(total):
        raise ValueError("field energy must be finite and non-zero")
    mode_index = np.rint(np.fft.fftfreq(values.size) * values.size).astype(int)
    high = np.abs(mode_index) > int(cutoff)
    return float(np.sum(energy_density[high]) / total)


def spectral_slope_max(field: np.ndarray) -> float:
    """Maximum absolute spectral derivative, the discrete ``||u_x||_∞``."""

    return float(np.max(np.abs(spectral_derivative(field))))


def estimate_hard_storage(config: HardPilotConfig) -> dict[str, float | int]:
    """Uncompressed ``float64`` field bytes, excluding JSON and coordinates."""

    _validate_config(config)
    n_times = _frame_count(config.t_final, config.save_dt)
    per_trajectory = n_times * config.n * 8
    total = config.total() * per_trajectory
    return {
        "n_instances": config.total(),
        "n_times": n_times,
        "n": config.n,
        "bytes_per_trajectory": per_trajectory,
        "field_bytes": total,
        "field_mebibytes": total / (1024 * 1024),
    }


def generate_hard_pilot(output: Path, config: HardPilotConfig) -> dict[str, Any]:
    """Write a harder-pilot directory and return the manifest.

    ``output`` is created and must not already exist. Each instance is
    ``{split}/instance_{id:06d}.npz``. Invariant summaries are computed
    from the saved frames while they are still in memory.
    """

    _validate_config(config)
    destination = Path(output)
    if destination.exists():
        raise FileExistsError(f"hard pilot output already exists: {destination}")
    splits = assign_splits(config.n_train, config.n_val, config.n_test, config.split_seed)
    split_of = {instance_id: name for name, ids in splits.items() for instance_id in ids}
    destination.mkdir(parents=True)
    for name in ("train", "val", "test"):
        (destination / name).mkdir()
    stats = RunningStats()
    records: list[dict[str, Any]] = []
    viscosities_by_split: dict[str, list[float]] = {"train": [], "val": [], "test": []}
    solver_config: SolverConfig | None = None
    worst_drift = 0.0
    worst_drift_id = 0
    worst_energy = 0.0
    worst_energy_id = 0
    batch = config.batch_size
    for start in range(0, config.total(), batch):
        ids = list(range(start, min(start + batch, config.total())))
        drawn = [
            draw_hard_initial_condition(
                config.master_seed,
                instance_id,
                config.n,
                n_modes=config.n_modes,
                m_keep=config.m_keep,
                beta=config.beta,
                amplitude_decay=config.amplitude_decay,
                nu_min=config.nu_min,
                nu_max=config.nu_max,
            )
            for instance_id in ids
        ]
        fields = np.stack([item.u0 for item in drawn], axis=0)
        viscosities = np.asarray([item.nu for item in drawn], dtype=np.float64)
        stacked, times, nodes, solver_config = solve_batch(
            fields,
            viscosities,
            dt=config.dt,
            t_final=config.t_final,
            save_dt=config.save_dt,
        )
        print(f"integrated hard instances {ids[0]}..{ids[-1]}", flush=True)
        for local, item in enumerate(drawn):
            split = split_of[item.instance_id]
            drift, energy_increase = _saved_invariants(stacked[local])
            if drift > worst_drift:
                worst_drift = drift
                worst_drift_id = item.instance_id
            if energy_increase > worst_energy:
                worst_energy = energy_increase
                worst_energy_id = item.instance_id
            relative = Path(split) / f"instance_{item.instance_id:06d}.npz"
            path = destination / relative
            payload = {
                "u": stacked[local],
                "x": nodes,
                "t": times,
                "nu": np.float64(item.nu),
                "instance_id": np.int64(item.instance_id),
                "master_seed": np.int64(item.master_seed),
                "coefficients_a": item.a,
                "coefficients_b": item.b,
                "amplitude_scale": np.float64(item.scale),
                "beta": np.float64(item.beta),
                "amplitude_decay": np.float64(item.amplitude_decay),
                "n_modes": np.int64(item.n_modes),
                "m_keep": np.int64(item.m_keep),
            }
            np.savez_compressed(path, **payload)
            digest = hashlib.sha256(path.read_bytes()).hexdigest()
            if split == "train":
                stats.update(stacked[local])
            viscosities_by_split[split].append(item.nu)
            records.append(
                {
                    "instance_id": item.instance_id,
                    "split": split,
                    "path": relative.as_posix(),
                    "sha256": digest,
                    "field_sha256": _field_sha256(stacked[local]),
                    "nu": item.nu,
                    "amplitude_scale": item.scale,
                    "max_abs_mean_drift": drift,
                    "max_energy_increase": energy_increase,
                    "n": config.n,
                    "n_times": int(times.shape[0]),
                }
            )
    if solver_config is None:
        raise RuntimeError("hard pilot generation produced no trajectories")
    all_nu = [float(record["nu"]) for record in records]
    manifest = {
        "format": FORMAT,
        "ic_family": IC_FAMILY,
        "pilot": config.to_dict(),
        "solver": solver_config.to_dict(),
        "solver_config_sha256": sha256_json(solver_config.to_dict()),
        "pilot_config_sha256": sha256_json(config.to_dict()),
        "code_sha256": source_hashes(),
        "numpy": np.__version__,
        "splits": {name: sorted(ids) for name, ids in splits.items()},
        "split_order_note": (
            "splits lists are sorted for display. Assignment used Fisher-Yates "
            "on instance ids before integration. No windows were extracted."
        ),
        "normalization": {
            "fit_on": "train",
            "applied_to_files": False,
            "u_mean": stats.mean,
            "u_std": stats.population_std(),
            "n_values": stats.count,
            "definition": "population mean and std (ddof=0) of raw training u(x, t)",
        },
        "nu_stats": {
            "all": _nu_summary(all_nu),
            "train": _nu_summary(viscosities_by_split["train"]),
            "val": _nu_summary(viscosities_by_split["val"]),
            "test": _nu_summary(viscosities_by_split["test"]),
        },
        "invariants": {
            "definition": "rectangle-rule spatial mean and energy on saved frames",
            "max_abs_mean_drift": worst_drift,
            "max_abs_mean_drift_instance_id": worst_drift_id,
            "max_energy_increase": worst_energy,
            "max_energy_increase_instance_id": worst_energy_id,
            "mean_drift_gate": LABEL_MEAN_DRIFT_MAX,
            "energy_increase_gate": LABEL_ENERGY_INCREASE_MAX,
        },
        "label_gate": label_gate_document(),
        "protocol": build_protocol(
            config,
            viscosities_by_split["train"],
            sha256_json(solver_config.to_dict()),
        ),
        "instances": records,
    }
    manifest_path = destination / "manifest.json"
    manifest_path.write_text(
        json.dumps(manifest, indent=2, sort_keys=True, allow_nan=False) + "\n",
        encoding="utf-8",
    )
    return manifest


def hard_ood_threshold(train_viscosities: np.ndarray | list[float]) -> float:
    """Training-split quantile that defines the ``hard_ood`` slice.

    The quantile and the method are fixed. The returned number is that
    quantile of ``train_viscosities`` only. Validation and test viscosities
    do not enter.
    """

    values = np.asarray(train_viscosities, dtype=np.float64)
    if values.ndim != 1 or values.size < 1 or not np.isfinite(values).all():
        raise ValueError("hard_ood threshold needs a non-empty finite training sample")
    if np.any(values <= 0.0):
        raise ValueError("training viscosities must be > 0")
    return float(np.quantile(values, HARD_OOD_QUANTILE, method=HARD_OOD_QUANTILE_METHOD))


def build_protocol(
    config: HardPilotConfig,
    train_viscosities: np.ndarray | list[float],
    solver_config_sha256: str,
) -> dict[str, Any]:
    """Preregistered harder-pilot protocol.

    ``label_gate`` is the Stage A gate at ``N = 1024`` and ``dt = 2.5e-4``,
    including when ``config`` is a smaller diagnostic run. ``hard_ood`` is
    the lowest training-split quartile of viscosity. Later stages score that
    slice on its own. They do not use it, or the test split, to choose a model.
    """

    _validate_config(config)
    if not isinstance(solver_config_sha256, str) or len(solver_config_sha256) != 64:
        raise ValueError("solver_config_sha256 must be a 64-character hex digest")
    try:
        int(solver_config_sha256, 16)
    except ValueError as exc:
        raise ValueError("solver_config_sha256 must be hexadecimal") from exc
    threshold = hard_ood_threshold(train_viscosities)
    train = np.asarray(train_viscosities, dtype=np.float64)
    return {
        "format": PROTOCOL_FORMAT,
        "frozen_before_operator_training": True,
        "retuned_after_inverse_measurement": False,
        "ic_family": IC_FAMILY,
        "pilot": config.to_dict(),
        "pilot_config_sha256": sha256_json(config.to_dict()),
        "solver_config_sha256": solver_config_sha256,
        "seeds": {
            "master_seed": config.master_seed,
            "split_seed": config.split_seed,
            "ic_salt": HARD_IC_SALT,
            "split_salt": SPLIT_SALT,
            "bit_generator": "PCG64",
        },
        "split_sizes": {
            "train": config.n_train,
            "val": config.n_val,
            "test": config.n_test,
            "unit": "problem_instance",
        },
        "nu_range": {
            "distribution": "uniform",
            "min": config.nu_min,
            "max": config.nu_max,
            "drawn_after": "successful initial field",
        },
        "time_horizon": {
            "t_final": config.t_final,
            "domain": "x in [-1, 1], periodic",
        },
        "label_gate": label_gate_document(),
        "hard_ood": {
            "name": HARD_OOD_NAME,
            "rule": (
                "An instance is hard_ood iff nu <= threshold_nu. "
                "threshold_nu is the training-split quantile. Fit on train only. "
                "Later stages must score this slice separately and must not use "
                "hard_ood, validation, or the test split to choose a model or "
                "to refit the quantile."
            ),
            "quantile": HARD_OOD_QUANTILE,
            "method": HARD_OOD_QUANTILE_METHOD,
            "fit_on": "train",
            "comparison": "nu <= threshold_nu",
            "threshold_nu": threshold,
            "n_train_at_or_below": int(np.count_nonzero(train <= threshold)),
        },
        "stage2_nu_min": STAGE2_NU_MIN,
        "inverse_reference": {
            "estimator": "sensors32_bursts",
            "symbol": "pinnforge.operator.inverse.PREREGISTERED_OBSERVATION",
            "role": (
                "Closed-form residual least squares from Stage 5, scored as a "
                "ceiling/floor reference. The observation pattern is not retuned."
            ),
        },
    }


def label_gate_document() -> dict[str, Any]:
    """Preregistered acceptance bounds for a harder-pilot label.

    The measured comparison is the convergence study. These numbers are
    the gate, not a claim that every instance was integrated twice.
    """

    return {
        "relative_l2_final_max": LABEL_REL_L2_MAX,
        "relative_l2_spacetime_max": LABEL_REL_L2_MAX,
        "max_abs_mean_drift_max": LABEL_MEAN_DRIFT_MAX,
        "max_energy_increase_max": LABEL_ENERGY_INCREASE_MAX,
        "pilot_n": HARD_N,
        "pilot_dt": HARD_DT,
        "reference_n": REFERENCE_N,
        "reference_dt": REFERENCE_DT,
        "nu_min": NU_MIN,
        "nu_max": NU_MAX,
        "study": "docs/stage_a/convergence.json",
    }


def source_hashes() -> dict[str, str]:
    directory = Path(__file__).resolve().parent
    return {
        name: hashlib.sha256((directory / name).read_bytes()).hexdigest() for name in _CODE_SOURCES
    }


def _project(
    a: np.ndarray,
    b: np.ndarray,
    *,
    beta: float,
    m_keep: int,
) -> tuple[float, np.ndarray] | None:
    """Return ``(scale, canonical field)`` or ``None`` if the draw is trivial."""

    polynomial = trigonometric_field(a, b, CANONICAL_N)
    centered = polynomial - float(np.mean(polynomial))
    peak_p = float(np.max(np.abs(centered)))
    if peak_p < _PEAK_FLOOR:
        return None
    shaped = np.tanh(float(beta) * centered / peak_p)
    spectrum = np.fft.fft(shaped)
    mode_index = np.rint(np.fft.fftfreq(CANONICAL_N) * CANONICAL_N).astype(np.int64)
    spectrum = np.array(spectrum, dtype=np.complex128, copy=True)
    spectrum[np.abs(mode_index) > m_keep] = 0.0
    spectrum[0] = 0.0
    projected = np.fft.ifft(spectrum)
    if float(np.max(np.abs(projected.imag))) > 1e-8:
        raise RuntimeError("bandlimited projection left a non-real field")
    canonical = np.asarray(projected.real, dtype=np.float64)
    peak = float(np.max(np.abs(canonical)))
    if peak < _PEAK_FLOOR or not math.isfinite(peak):
        return None
    return 1.0 / peak, canonical / peak


def _saved_invariants(field: np.ndarray) -> tuple[float, float]:
    means = spatial_mean(field)
    energies = energy(field)
    drift = float(np.max(np.abs(means - means[0])))
    increase = float(np.max(energies - energies[0]))
    return drift, increase


def _nu_summary(values: list[float]) -> dict[str, float | int]:
    array = np.asarray(values, dtype=np.float64)
    if array.size == 0 or not np.isfinite(array).all():
        raise ValueError("viscosity summary needs a non-empty finite sample")
    return {
        "count": int(array.size),
        "min": float(np.min(array)),
        "max": float(np.max(array)),
        "mean": float(np.mean(array)),
        "std": float(np.std(array)),
    }


def _validate_config(config: HardPilotConfig) -> None:
    _require_count(config.n_train, label="n_train")
    _require_count(config.n_val, label="n_val")
    _require_count(config.n_test, label="n_test")
    _require_count(config.batch_size, label="batch_size")
    _require_count(config.master_seed, label="master_seed", allow_zero=True)
    _require_count(config.split_seed, label="split_seed", allow_zero=True)
    modes = _require_modes(config.n_modes)
    kept = _require_keep(config.m_keep)
    if kept < modes:
        raise ValueError("m_keep must be >= n_modes")
    _require_resolved_grid(config.n, kept)
    _require_decay(config.amplitude_decay)
    _require_beta(config.beta)
    for label, value in (
        ("dt", config.dt),
        ("t_final", config.t_final),
        ("save_dt", config.save_dt),
        ("nu_min", config.nu_min),
        ("nu_max", config.nu_max),
    ):
        if (
            isinstance(value, bool)
            or not isinstance(value, (int, float))
            or not math.isfinite(value)
        ):
            raise ValueError(f"{label} must be finite")
    if config.dt <= 0 or config.t_final <= 0 or config.save_dt <= 0:
        raise ValueError("dt, t_final, and save_dt must be > 0")
    if config.nu_min <= 0 or config.nu_max < config.nu_min:
        raise ValueError("viscosity bounds must satisfy 0 < nu_min <= nu_max")
    if config.nu_min > NU_MIN + 1e-15 or config.nu_max < NU_MAX - 1e-15:
        raise ValueError(f"harder pilot viscosity interval must cover [{NU_MIN}, {NU_MAX}]")


def _require_modes(n_modes: int) -> int:
    if isinstance(n_modes, bool) or not isinstance(n_modes, (int, np.integer)) or int(n_modes) < 1:
        raise ValueError("n_modes must be an integer >= 1")
    return int(n_modes)


def _require_keep(m_keep: int) -> int:
    if isinstance(m_keep, bool) or not isinstance(m_keep, (int, np.integer)) or int(m_keep) < 1:
        raise ValueError("m_keep must be an integer >= 1")
    return int(m_keep)


def _require_resolved_grid(n: int, m_keep: int) -> None:
    if isinstance(n, bool) or not isinstance(n, (int, np.integer)) or int(n) < 4 or int(n) % 2 != 0:
        raise ValueError("n must be an even integer >= 4")
    if int(n) // 2 <= int(m_keep):
        raise ValueError("n/2 must be greater than m_keep so every retained mode is below Nyquist")
    if int(n) > CANONICAL_N:
        raise ValueError("n must be <= the canonical grid")


def _require_decay(amplitude_decay: float) -> None:
    if (
        isinstance(amplitude_decay, bool)
        or not isinstance(amplitude_decay, (int, float, np.floating))
        or not math.isfinite(float(amplitude_decay))
        or float(amplitude_decay) <= 0
    ):
        raise ValueError("amplitude_decay must be finite and > 0")


def _require_beta(beta: float) -> None:
    if (
        isinstance(beta, bool)
        or not isinstance(beta, (int, float, np.floating))
        or not math.isfinite(float(beta))
        or float(beta) <= 0
    ):
        raise ValueError("beta must be finite and > 0")


def _require_seed(value: int, *, label: str) -> None:
    if isinstance(value, bool) or not isinstance(value, (int, np.integer)) or int(value) < 0:
        raise ValueError(f"{label} must be an integer >= 0")


def _require_count(value: int, *, label: str, allow_zero: bool = False) -> None:
    if isinstance(value, bool) or not isinstance(value, (int, np.integer)):
        raise ValueError(f"{label} must be an integer")
    if int(value) < 0 or (int(value) == 0 and not allow_zero):
        raise ValueError(f"{label} must be >= {0 if allow_zero else 1}")
