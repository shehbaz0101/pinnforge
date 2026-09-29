"""Stage D operator-conditioned viscosity inverse.

The contract is ``docs/v02/stage_d_inverse_protocol.json``. It is loaded
and checked here. Test scores do not live in that file.

The sparse estimator reads ``sensors32_bursts`` only. Frames 0 through 4
are lifted from the 32 sensors onto the trajectory grid and extended to
the eight-frame FNO window by a quadratic extrapolation that does not
depend on ``ν``. ``ν`` enters as the normalized input channel of a
trained Stage B or Stage C operator. An autoregressive rollout is
compared with the later sensor bursts. A uniform grid on the pilot
viscosity range supplies the minimizer. Ties keep the smaller grid value.

A combined objective can add the Stage 5 residual mean square. Its
weight is chosen on the validation ``hard_ood`` slice, per arm. ``λ = 0``
is always scored. The oracle one-step ceiling uses the true field as
input and is not a sparse estimator.

This module does not import torch. The rollout lives in
:mod:`pinnforge.operator.stage_d_fit`.
"""

from __future__ import annotations

import hashlib
import json
import math
from collections.abc import Mapping, Sequence
from pathlib import Path
from typing import Any

import numpy as np

from pinnforge.operator.inverse import (
    FAILURE_RELATIVE_ERROR,
    PREREGISTERED_OBSERVATION,
    InstanceRecovery,
    RecoveryScore,
    dense_reference_observation,
    observed_residual_terms,
    recovery_from_rows,
    score_trajectories,
)
from pinnforge.operator.slices import HARD_OOD_COMPARISON, hard_ood_threshold, load_data_protocol
from pinnforge.operator.windows import Trajectory, WindowSpec

STAGE_D_FORMAT = "pinnforge.stage_d_inverse_protocol.v1"
RUN_FORMAT = "pinnforge.stage_d_inverse_run.v1"
SELECTION_FORMAT = "pinnforge.stage_d_objective_selection.v1"
SCORES_FORMAT = "pinnforge.stage_d_scores.v1"
ARM_NAMES = ("data_only", "hybrid_1e-2")
SLICE_NAMES = ("full_test", "hard_ood", "complement", "below_stage2_floor")
# Degree-2 Newton step from frames 2, 3, 4 to frames 5, 6, 7.
NEWTON_AHEAD_OF_FRAME_4 = {
    3: (1.0, -3.0, 3.0),
    4: (3.0, -8.0, 6.0),
    5: (6.0, -15.0, 10.0),
}
_FORBIDDEN = frozenset({"results", "scores", "metrics", "test_mean_rel_error", "nu_hat"})
_SCORE_FIELDS = (
    "mean_abs_error",
    "median_abs_error",
    "max_abs_error",
    "mean_rel_error",
    "median_rel_error",
    "max_rel_error",
    "n_failures",
    "n_nonpositive",
    "n_worse_than_baseline",
    "correlation",
    "mean_abs_residual",
    "mean_abs_residual_at_truth",
    "mean_square_residual",
    "mean_square_residual_at_truth",
)
_PUBLISHED_BASELINES = {
    ("sensors32_bursts", "full_test"): "sensors32_bursts",
    ("sensors32_bursts", "hard_ood"): "hard_ood_sensors32_bursts",
    ("sensors32_bursts", "complement"): "complement_sensors32_bursts",
    ("sensors32_bursts", "below_stage2_floor"): "below_stage2_floor_sensors32_bursts",
    ("dense_reference", "full_test"): "dense_reference",
    ("dense_reference", "hard_ood"): "hard_ood_dense_reference",
}


def load_stage_d_protocol(path: Path) -> dict[str, Any]:
    """Read the Stage D contract and reject a file that already holds scores."""

    payload = json.loads(Path(path).read_text(encoding="utf-8"))
    if not isinstance(payload, dict):
        raise ValueError("stage D protocol must be a JSON object")
    if payload.get("format") != STAGE_D_FORMAT:
        raise ValueError(f"stage D protocol format must be {STAGE_D_FORMAT}")
    leaked = _FORBIDDEN & set(payload)
    if leaked:
        names = ", ".join(sorted(leaked))
        raise ValueError(f"stage D protocol must not contain test scores ({names})")
    if payload.get("frozen_before_test_evaluation") is not True:
        raise ValueError("stage D protocol must be frozen before test evaluation")
    if payload.get("test_used_for_selection") is not False:
        raise ValueError("stage D protocol must not select on the test split")
    if payload.get("hard_ood_refit") is not False or payload.get("mask_retuned") is not False:
        raise ValueError("stage D protocol must not retune hard_ood or the mask")
    _require_mask(payload)
    _require_search(payload)
    _require_estimator(payload)
    _require_objective(payload)
    _require_ceiling(payload)
    _require_arms(payload)
    _require_slices(payload)
    if payload.get("failure_relative_error") != FAILURE_RELATIVE_ERROR:
        raise ValueError("failure relative error must stay 0.5")
    if payload.get("failure_rule") != "nu_hat <= 0 or rel_error > 0.5":
        raise ValueError("failure rule must stay the Stage 5 rule")
    if int(payload.get("inference_batch_size", 0)) != 32:
        raise ValueError("inference batch size must stay 32")
    if payload.get("device") != "cpu":
        raise ValueError("stage D device must stay cpu")
    window = payload.get("window")
    if not isinstance(window, dict):
        raise ValueError("stage D protocol is missing the window")
    expected_window = {"input_frames": 8, "output_frames": 8, "stride": 8}
    if {key: int(window[key]) for key in expected_window} != expected_window:
        raise ValueError("stage D window must stay 8/8/8")
    data_protocol = load_data_protocol(Path(payload["data_protocol"]))
    if hard_ood_threshold(payload) != hard_ood_threshold(data_protocol):
        raise ValueError("stage D hard_ood threshold does not match the pilot protocol")
    if payload.get("hard_ood", {}).get("comparison") != HARD_OOD_COMPARISON:
        raise ValueError(f"hard_ood comparison must stay {HARD_OOD_COMPARISON!r}")
    _require_sha256(Path(payload["data_protocol"]), str(payload["data_protocol_sha256"]))
    _require_sha256(Path(payload["baseline_record"]), str(payload["baseline_record_sha256"]))
    return payload


def protocol_sha256(path: Path) -> str:
    """SHA-256 of the protocol file bytes."""

    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def viscosity_grid(protocol: Mapping[str, Any]) -> np.ndarray:
    """Inclusive linear grid on the preregistered viscosity bounds."""

    search = protocol["nu_search"]
    grid = np.linspace(float(search["min"]), float(search["max"]), int(search["n_grid"]))
    if grid[0] != float(search["min"]) or grid[-1] != float(search["max"]):
        raise ValueError("viscosity grid does not include both endpoints")
    return np.asarray(grid, dtype=np.float64)


def window_from_protocol(protocol: Mapping[str, Any]) -> WindowSpec:
    window = protocol["window"]
    return WindowSpec(
        input_frames=int(window["input_frames"]),
        output_frames=int(window["output_frames"]),
        stride=int(window["stride"]),
    )


def lift_sensors(samples: np.ndarray, n_space: int) -> np.ndarray:
    """Trigonometric interpolation from equispaced sensors onto ``n_space``.

    The coarse Nyquist coefficient is halved before zero-padding, which is
    the real-FFT resample that does not double-count that mode. Samples
    may have leading dimensions. The last axis is the sensor axis.
    """

    values = np.asarray(samples, dtype=np.float64)
    if values.ndim < 1:
        raise ValueError("sensor samples must have a sensor axis")
    n_sensors = int(values.shape[-1])
    if n_sensors < 4 or n_sensors % 2 != 0:
        raise ValueError("n_sensors must be an even integer >= 4")
    if isinstance(n_space, bool) or not isinstance(n_space, (int, np.integer)):
        raise ValueError("n_space must be an even integer >= n_sensors")
    n_out = int(n_space)
    if n_out < n_sensors or n_out % 2 != 0 or n_out % n_sensors != 0:
        raise ValueError("n_space must be an even multiple of n_sensors")
    if not np.isfinite(values).all():
        raise ValueError("sensor samples must be finite")
    spectrum = np.fft.rfft(values, axis=-1)
    padded = np.zeros(values.shape[:-1] + (n_out // 2 + 1,), dtype=np.complex128)
    n_copy = int(spectrum.shape[-1])
    padded[..., :n_copy] = spectrum
    if n_out > n_sensors:
        padded[..., n_sensors // 2] *= 0.5
    padded *= n_out / n_sensors
    lifted = np.fft.irfft(padded, n=n_out, axis=-1)
    return np.asarray(lifted, dtype=np.float64)


def extrapolate_initial_window(frames: np.ndarray) -> np.ndarray:
    """Extend five lifted frames to eight with the locked quadratic rule.

    ``frames`` has shape ``(..., 5, n_space)`` and is frames 0 through 4.
    Frames 5, 6, and 7 are a degree-2 Newton step from frames 2, 3, and 4.
    The map does not depend on viscosity.
    """

    values = np.asarray(frames, dtype=np.float64)
    if values.ndim < 2 or values.shape[-2] != 5:
        raise ValueError("extrapolation expects five frames on the second-to-last axis")
    if not np.isfinite(values).all():
        raise ValueError("frames must be finite")
    earlier = values[..., 2, :]
    middle = values[..., 3, :]
    latest = values[..., 4, :]
    filled = []
    for step in (3, 4, 5):
        first, second, third = NEWTON_AHEAD_OF_FRAME_4[step]
        filled.append(first * earlier + second * middle + third * latest)
    extra = np.stack(filled, axis=-2)
    return np.concatenate((values, extra), axis=-2)


def read_mask(field: np.ndarray, spec: Any = PREREGISTERED_OBSERVATION) -> np.ndarray:
    """Sensor values on each burst. Shape ``(n_series, burst_length, n_sensors)``.

    Columns that are not sensor nodes are not read. Frames outside the
    bursts are not read.
    """

    values = np.asarray(field, dtype=np.float64)
    if values.ndim != 2:
        raise ValueError("field must have shape (n_times, n_space)")
    n_sensors = spec.check_grid(int(values.shape[0]), int(values.shape[1]))
    stride = int(values.shape[1]) // n_sensors
    blocks = [np.ascontiguousarray(values[list(indexes), ::stride]) for indexes in spec.series]
    return np.stack(blocks, axis=0)


def initial_window_from_mask(mask: np.ndarray, n_space: int) -> np.ndarray:
    """Eight-frame physical window built only from the opening burst."""

    values = np.asarray(mask, dtype=np.float64)
    if values.ndim != 3 or values.shape[0] < 1:
        raise ValueError("mask must have shape (n_series, burst_length, n_sensors)")
    lifted = lift_sensors(values[0], n_space)
    return extrapolate_initial_window(lifted)


def target_slots(spec: Any, window: WindowSpec) -> tuple[tuple[int, int, int, int], ...]:
    """Locate each scored burst inside one autoregressive block.

    Each slot is ``(series_index, step, local_start, length)``. Step 0 is
    the first predicted block. The opening burst is the conditioning
    window and is not a target.
    """

    if not isinstance(window, WindowSpec):
        raise TypeError("window must be a WindowSpec")
    if window.stride != window.output_frames:
        raise ValueError("autoregressive rollout requires stride == output_frames")
    opening = tuple(range(window.input_frames - 3))
    if spec.series[0] != opening:
        raise ValueError(f"the opening burst must be frames {opening[0]}..{opening[-1]}")
    slots: list[tuple[int, int, int, int]] = []
    for series_index, indexes in enumerate(spec.series):
        if series_index == 0:
            continue
        start_frame = int(indexes[0])
        if start_frame < window.input_frames:
            raise ValueError("a target burst overlaps the initial window")
        offset = start_frame - window.input_frames
        if offset % window.output_frames != 0:
            raise ValueError("a target burst does not start on a predicted block")
        step = offset // window.output_frames
        block_start = window.input_frames + step * window.output_frames
        local = start_frame - block_start
        if int(indexes[-1]) >= block_start + window.output_frames:
            raise ValueError("a target burst does not fit in one predicted block")
        if list(indexes) != list(range(start_frame, start_frame + len(indexes))):
            raise ValueError("a target burst must be contiguous")
        slots.append((series_index, step, local, len(indexes)))
    if not slots:
        raise ValueError("the mask has no target bursts")
    return tuple(slots)


def prepare_sparse_batch(
    trajectories: Sequence[Trajectory],
    spec: Any,
    window: WindowSpec,
) -> dict[str, Any]:
    """Windows and targets built from the mask. Off-mask samples are not read."""

    if len(trajectories) < 1:
        raise ValueError("at least one trajectory is required")
    masks: list[np.ndarray] = []
    windows: list[np.ndarray] = []
    n_space: int | None = None
    for item in trajectories:
        if not isinstance(item, Trajectory):
            raise TypeError("trajectories must be Trajectory values")
        mask = read_mask(item.u, spec)
        width = int(item.u.shape[1])
        if n_space is None:
            n_space = width
        elif width != n_space:
            raise ValueError("trajectories must share a spatial grid")
        masks.append(mask)
        windows.append(initial_window_from_mask(mask, width))
    raw_slots = target_slots(spec, window)
    slots = tuple(
        (index, step, local, length) for index, (_, step, local, length) in enumerate(raw_slots)
    )
    targets = np.stack(
        [
            np.stack([mask[series_index] for series_index, _, _, _ in raw_slots], axis=0)
            for mask in masks
        ],
        axis=0,
    )
    return {
        "masks": masks,
        "windows": np.stack(windows, axis=0),
        "targets": targets,
        "slots": slots,
    }


def prepare_oracle_windows(
    trajectories: Sequence[Trajectory],
    spec: Any,
    window: WindowSpec,
) -> np.ndarray:
    """True input blocks for the one-step ceiling. This is not the sparse estimator."""

    slots = target_slots(spec, window)
    batch = []
    for item in trajectories:
        if not isinstance(item, Trajectory):
            raise TypeError("trajectories must be Trajectory values")
        blocks = []
        for _, step, _, _ in slots:
            start = step * window.stride
            blocks.append(np.asarray(item.u[start : start + window.input_frames], dtype=np.float64))
        batch.append(np.stack(blocks, axis=0))
    return np.stack(batch, axis=0)


def sensor_field(mask: np.ndarray, spec: Any, n_times: int) -> np.ndarray:
    """Place mask bursts on a sensor-only grid. Unobserved frames stay 0.

    The residual stencil reads only the bursts, so the zeros are not nodes.
    """

    values = np.asarray(mask, dtype=np.float64)
    if spec.n_sensors is None:
        raise ValueError("sensor field requires a finite sensor count")
    field = np.zeros((int(n_times), int(spec.n_sensors)), dtype=np.float64)
    for block, indexes in zip(values, spec.series, strict=True):
        field[list(indexes)] = block
    return field


def objective_curve(
    sensor_mse: np.ndarray,
    residual_mse: np.ndarray,
    sensor_anchor: float,
    residual_anchor: float,
    lam: float,
) -> np.ndarray:
    """Combined grid objective. ``λ = 0`` is the sensor mean square alone."""

    sensor = np.asarray(sensor_mse, dtype=np.float64)
    weight = float(lam)
    if weight < 0.0 or not math.isfinite(weight):
        raise ValueError("lambda must be finite and >= 0")
    if weight == 0.0:
        if not np.isfinite(sensor).all():
            raise ValueError("sensor objective is not finite")
        return sensor
    anchor_sensor = float(sensor_anchor)
    anchor_residual = float(residual_anchor)
    if anchor_sensor <= 0.0 or anchor_residual <= 0.0:
        raise ValueError("objective anchors must be > 0")
    if not math.isfinite(anchor_sensor) or not math.isfinite(anchor_residual):
        raise ValueError("objective anchors must be finite")
    residual = np.asarray(residual_mse, dtype=np.float64)
    if sensor.shape != residual.shape or not np.isfinite(residual).all() or not np.isfinite(sensor).all():
        raise ValueError("sensor and residual curves must be finite and the same shape")
    return sensor / anchor_sensor + weight * residual / anchor_residual


def nu_from_objective(curve: np.ndarray, grid: np.ndarray) -> float:
    """Smallest grid viscosity among the minimizing nodes."""

    values = np.asarray(curve, dtype=np.float64)
    nodes = np.asarray(grid, dtype=np.float64)
    if values.shape != nodes.shape or values.ndim != 1 or values.size < 1:
        raise ValueError("objective and grid must be the same non-empty vector")
    if not np.isfinite(values).all() or not np.isfinite(nodes).all():
        raise ValueError("objective and grid must be finite")
    return float(nodes[int(np.argmin(values))])


def choose_nu(item: Mapping[str, Any], grid: np.ndarray, lam: float, *, oracle: bool = False) -> float:
    """Viscosity minimizer for one stored instance curve."""

    if oracle:
        if float(lam) != 0.0:
            raise ValueError("the oracle ceiling is scored only at lambda 0")
        return nu_from_objective(np.asarray(item["oracle_sensor_mse"], dtype=np.float64), grid)
    curve = objective_curve(
        np.asarray(item["sensor_mse"], dtype=np.float64),
        np.asarray(item["residual_mse"], dtype=np.float64),
        float(item["sensor_mse_at_baseline"]),
        float(item["residual_mse_at_baseline"]),
        lam,
    )
    return nu_from_objective(curve, grid)


def slice_masks(viscosities: np.ndarray, threshold: float, floor: float) -> dict[str, np.ndarray]:
    """Boolean masks for the four preregistered slices. They may overlap."""

    nu = np.asarray(viscosities, dtype=np.float64)
    masks = {
        "full_test": np.ones(nu.shape[0], dtype=bool),
        "hard_ood": nu <= float(threshold),
        "complement": nu > float(threshold),
        "below_stage2_floor": nu < float(floor),
    }
    if not np.any(masks["hard_ood"]) or not np.any(masks["complement"]) or not np.any(masks["below_stage2_floor"]):
        raise ValueError("a required slice is empty")
    return masks


def mean_std(values: Sequence[float]) -> tuple[float, float]:
    """Arithmetic mean and sample standard deviation (``ddof = 1``)."""

    array = np.asarray(list(values), dtype=np.float64)
    if array.size < 2:
        raise ValueError("sample standard deviation needs at least two values")
    if not np.isfinite(array).all():
        raise ValueError("aggregate values must be finite")
    return float(np.mean(array)), float(np.std(array, ddof=1))


def residual_moments(advection: np.ndarray, diffusion: np.ndarray, nu: float) -> tuple[float, float]:
    """Mean absolute residual and mean square residual at one viscosity."""

    residual = np.asarray(advection, dtype=np.float64) - float(nu) * np.asarray(diffusion, dtype=np.float64)
    if not np.isfinite(residual).all():
        raise ValueError("observed residual is not finite")
    return float(np.mean(np.abs(residual))), float(np.mean(residual * residual))


def rows_from_estimates(
    items: Sequence[Mapping[str, Any]],
    nu_hat: Sequence[float],
    baseline_nu: float,
) -> tuple[InstanceRecovery, ...]:
    """Build recovery rows. ``nu_hat`` follows ``items`` and is not refit."""

    if len(items) != len(nu_hat):
        raise ValueError("estimates and instances must have the same length")
    baseline = float(baseline_nu)
    rows: list[InstanceRecovery] = []
    for item, hat in zip(items, nu_hat, strict=True):
        nu = float(item["nu"])
        estimate = float(hat)
        if not math.isfinite(estimate):
            raise ValueError(f"instance {item['instance_id']} estimate is not finite")
        advection = np.asarray(item["advection"], dtype=np.float64)
        diffusion = np.asarray(item["diffusion"], dtype=np.float64)
        absolute = abs(estimate - nu)
        relative = absolute / nu
        baseline_absolute = abs(baseline - nu)
        residual_abs, residual_square = residual_moments(advection, diffusion, estimate)
        truth_abs, truth_square = residual_moments(advection, diffusion, nu)
        rows.append(
            InstanceRecovery(
                instance_id=int(item["instance_id"]),
                nu=nu,
                nu_hat=estimate,
                abs_error=absolute,
                rel_error=relative,
                baseline_abs_error=baseline_absolute,
                baseline_rel_error=baseline_absolute / nu,
                mean_abs_residual=residual_abs,
                mean_abs_residual_at_truth=truth_abs,
                mean_square_residual=residual_square,
                mean_square_residual_at_truth=truth_square,
                failure=estimate <= 0.0 or relative > FAILURE_RELATIVE_ERROR,
            )
        )
    return tuple(rows)


def score_pattern(
    items: Sequence[Mapping[str, Any]],
    nu_hat: Sequence[float],
    baseline_nu: float,
    pattern: str,
) -> RecoveryScore:
    return recovery_from_rows(pattern, baseline_nu, rows_from_estimates(items, nu_hat, baseline_nu))


def load_run(path: Path) -> dict[str, Any]:
    """Read one operator-inverse curve file."""

    payload = json.loads(Path(path).read_text(encoding="utf-8"))
    if not isinstance(payload, dict) or payload.get("format") != RUN_FORMAT:
        raise ValueError(f"{path} is not a stage D inverse run")
    if payload.get("split") not in ("val", "test"):
        raise ValueError(f"{path} has an unknown split")
    if payload.get("arm") not in ARM_NAMES:
        raise ValueError(f"{path} has an unknown arm")
    return payload


def select_objective(run_paths: Sequence[Path], protocol_path: Path) -> dict[str, Any]:
    """Choose ``λ`` on validation ``hard_ood`` only, separately for each arm.

    Every frozen seed must be present for both arms. A test run is rejected.
    """

    source = Path(protocol_path)
    protocol = load_stage_d_protocol(source)
    digest = protocol_sha256(source)
    runs = [_require_val_run(load_run(path), digest) for path in run_paths]
    seeds = [int(seed) for seed in protocol["arms"]["data_only"]["seeds"]]
    if len(runs) != len(ARM_NAMES) * len(seeds):
        raise ValueError(f"expected {len(ARM_NAMES) * len(seeds)} validation runs, got {len(runs)}")
    grouped: dict[str, dict[int, dict[str, Any]]] = {arm: {} for arm in ARM_NAMES}
    for run in runs:
        arm = str(run["arm"])
        seed = int(run["seed"])
        if seed in grouped[arm]:
            raise ValueError(f"duplicate validation run for {arm} seed {seed}")
        grouped[arm][seed] = run
    arms: dict[str, Any] = {}
    lambdas = [float(value) for value in protocol["combined_objective"]["lambda_grid"]]
    threshold = hard_ood_threshold(protocol)
    for arm in ARM_NAMES:
        rows = grouped[arm]
        if set(rows) != set(seeds):
            raise ValueError(f"{arm} is missing a frozen validation seed")
        candidates = []
        for lam in lambdas:
            per_seed = [_hard_ood_mean_rel(rows[seed], lam, threshold) for seed in seeds]
            center, spread = mean_std(per_seed)
            candidates.append(
                {
                    "lambda": lam,
                    "seeds": seeds,
                    "hard_ood_mean_rel_error": per_seed,
                    "mean_hard_ood_mean_rel_error": center,
                    "std_hard_ood_mean_rel_error": spread,
                }
            )
        chosen = min(candidates, key=lambda row: (float(row["mean_hard_ood_mean_rel_error"]), float(row["lambda"])))
        arms[arm] = {
            "candidates": candidates,
            "selected_lambda": chosen["lambda"],
            "selected_mean_hard_ood_mean_rel_error": chosen["mean_hard_ood_mean_rel_error"],
        }
    return {
        "format": SELECTION_FORMAT,
        "frozen_before_test_evaluation": True,
        "test_used_for_selection": False,
        "test_splits_read": False,
        "protocol_path": source.as_posix(),
        "protocol_sha256": digest,
        "selection_split": "val",
        "selection_slice": "hard_ood",
        "metric": "arithmetic mean across seeds of mean relative viscosity error",
        "tie_break": "smallest lambda",
        "arms": arms,
    }


def load_objective_selection(
    path: Path,
    protocol: Mapping[str, Any],
    digest: str,
) -> dict[str, Any]:
    """Read a selection file and re-derive each arm's winner from its table."""

    payload = json.loads(Path(path).read_text(encoding="utf-8"))
    if not isinstance(payload, dict) or payload.get("format") != SELECTION_FORMAT:
        raise ValueError("unsupported stage D selection format")
    leaked = _FORBIDDEN & set(payload)
    if leaked or "slices" in payload:
        raise ValueError("objective selection must not contain test scores")
    if payload.get("frozen_before_test_evaluation") is not True:
        raise ValueError("objective selection must be frozen before test evaluation")
    if payload.get("test_used_for_selection") is not False or payload.get("test_splits_read") is not False:
        raise ValueError("objective selection must not use the test split")
    if payload.get("protocol_sha256") != digest:
        raise ValueError("objective selection protocol hash does not match")
    lambdas = [float(value) for value in protocol["combined_objective"]["lambda_grid"]]
    seeds = [int(seed) for seed in protocol["arms"]["data_only"]["seeds"]]
    arms = payload.get("arms")
    if not isinstance(arms, dict) or set(arms) != set(ARM_NAMES):
        raise ValueError("objective selection must contain both operator arms")
    for arm in ARM_NAMES:
        block = arms[arm]
        candidates = block.get("candidates")
        if not isinstance(candidates, list) or len(candidates) != len(lambdas):
            raise ValueError(f"{arm} selection candidates must be the preregistered lambda grid")
        checked = []
        seen: list[float] = []
        for row in candidates:
            lam = float(row["lambda"])
            if lam not in lambdas or lam in seen:
                raise ValueError(f"{arm} selection lambda {lam} is not the preregistered grid")
            seen.append(lam)
            values = [float(value) for value in row["hard_ood_mean_rel_error"]]
            if [int(seed) for seed in row["seeds"]] != seeds or len(values) != len(seeds):
                raise ValueError(f"{arm} selection values must follow the frozen seeds")
            center, spread = mean_std(values)
            if abs(center - float(row["mean_hard_ood_mean_rel_error"])) > 1e-12:
                raise ValueError(f"{arm} selection mean was not recomputed from the values")
            if abs(spread - float(row["std_hard_ood_mean_rel_error"])) > 1e-12:
                raise ValueError(f"{arm} selection std was not recomputed from the values")
            checked.append({**row, "lambda": lam, "mean_hard_ood_mean_rel_error": center})
        chosen = min(checked, key=lambda row: (float(row["mean_hard_ood_mean_rel_error"]), float(row["lambda"])))
        if float(block["selected_lambda"]) != float(chosen["lambda"]):
            raise ValueError(f"{arm} selected lambda does not match the validation table")
    return payload


def baseline_tables(
    trajectories: Sequence[Trajectory],
    baseline_nu: float,
    protocol: Mapping[str, Any],
) -> dict[str, dict[str, RecoveryScore]]:
    """Closed-form residual least squares and the training-mean constant.

    The patterns are the Stage 5 mask and the dense stencil. Slices are cut
    after the estimate, so the normal equation on a slice is the same one
    used on the full split.
    """

    sparse = score_trajectories(list(trajectories), PREREGISTERED_OBSERVATION, baseline_nu)
    moments_at_baseline = {
        item.instance_id: residual_moments(*_residual_terms(item), baseline_nu) for item in trajectories
    }
    n_times = int(trajectories[0].u.shape[0])
    dense = score_trajectories(
        list(trajectories),
        dense_reference_observation(n_times),
        baseline_nu,
    )
    threshold = hard_ood_threshold(protocol)
    floor = float(protocol["below_stage2_floor"]["threshold_nu"])
    masks = slice_masks(np.asarray([row.nu for row in sparse.instances]), threshold, floor)
    by_id = {row.instance_id: index for index, row in enumerate(sparse.instances)}
    order = [by_id[row.instance_id] for row in sparse.instances]
    if order != list(range(len(order))):
        raise ValueError("baseline rows are not in instance-id order")
    tables: dict[str, dict[str, RecoveryScore]] = {
        "sensors32_bursts": {},
        "dense_reference": {},
        "training_mean": {},
    }
    nu = np.asarray([row.nu for row in sparse.instances], dtype=np.float64)
    for name, mask in masks.items():
        tables["sensors32_bursts"][name] = _subset_score(sparse, mask, "sensors32_bursts")
        tables["dense_reference"][name] = _subset_score(dense, _mask_for_ids(dense, sparse, mask), "dense_reference")
        tables["training_mean"][name] = _constant_score(sparse, mask, baseline_nu, moments_at_baseline)
    if not np.array_equal(nu <= threshold, masks["hard_ood"]):
        raise ValueError("hard_ood mask drifted")
    return tables


def assert_published_baselines(tables: Mapping[str, Mapping[str, RecoveryScore]], stress_path: Path) -> None:
    """Refuse to continue when a recomputed baseline leaves the Stage A record."""

    published = json.loads(Path(stress_path).read_text(encoding="utf-8"))
    for (pattern, slice_name), key in _PUBLISHED_BASELINES.items():
        got = tables[pattern][slice_name]
        reference = published[key]
        if int(reference["n_instances"]) != got.n_instances:
            raise ValueError(f"{key} instance count does not match the recomputation")
        for field in (
            "mean_abs_error",
            "mean_rel_error",
            "median_rel_error",
            "max_rel_error",
            "n_failures",
            "n_worse_than_baseline",
            "correlation",
        ):
            left = getattr(got, field)
            right = reference[field]
            if isinstance(left, float) and isinstance(right, float):
                if abs(left - right) > 1e-12:
                    raise ValueError(f"{key} {field} changed relative to docs/v02/inverse_stress.json")
            elif left != right:
                raise ValueError(f"{key} {field} changed relative to docs/v02/inverse_stress.json")


def assemble_scores(
    test_runs: Sequence[Mapping[str, Any]],
    selection: Mapping[str, Any],
    protocol: Mapping[str, Any],
    baselines: Mapping[str, Mapping[str, RecoveryScore]],
) -> dict[str, Any]:
    """Reduce test curves with the validation-selected ``λ`` and with ``λ = 0``."""

    seeds = [int(seed) for seed in protocol["arms"]["data_only"]["seeds"]]
    if len(test_runs) != len(ARM_NAMES) * len(seeds):
        raise ValueError(f"expected {len(ARM_NAMES) * len(seeds)} test runs, got {len(test_runs)}")
    grouped: dict[str, dict[int, Mapping[str, Any]]] = {arm: {} for arm in ARM_NAMES}
    for run in test_runs:
        if run.get("split") != "test":
            raise ValueError("stage D scores read the test split only")
        arm = str(run["arm"])
        seed = int(run["seed"])
        if arm not in grouped or seed in grouped[arm]:
            raise ValueError(f"duplicate or unknown test run {arm} seed {seed}")
        grouped[arm][seed] = run
    threshold = hard_ood_threshold(protocol)
    floor = float(protocol["below_stage2_floor"]["threshold_nu"])
    reference_items = _reference_items(grouped["data_only"][seeds[0]])
    order = np.argsort([int(item["instance_id"]) for item in reference_items], kind="mergesort")
    sorted_items = [reference_items[int(index)] for index in order]
    masks = slice_masks(np.asarray([float(item["nu"]) for item in sorted_items]), threshold, floor)
    ls_full = baselines["sensors32_bursts"]["full_test"]
    arms: dict[str, Any] = {}
    for arm in ARM_NAMES:
        if set(grouped[arm]) != set(seeds):
            raise ValueError(f"{arm} is missing a frozen test seed")
        selected = float(selection["arms"][arm]["selected_lambda"])
        lambdas = [0.0] if selected == 0.0 else [0.0, selected]
        arm_block: dict[str, Any] = {"selected_lambda": selected, "objectives": {}}
        for lam in lambdas:
            arm_block["objectives"][str(lam)] = _objective_block(
                [grouped[arm][seed] for seed in seeds],
                lam,
                masks,
                ls_full,
                oracle=False,
            )
        arm_block["oracle_onestep"] = _objective_block(
            [grouped[arm][seed] for seed in seeds],
            0.0,
            masks,
            ls_full,
            oracle=True,
        )
        arms[arm] = arm_block
    return {
        "format": SCORES_FORMAT,
        "threshold_nu": threshold,
        "comparison": HARD_OOD_COMPARISON,
        "below_stage2_floor": floor,
        "failure_rule": "nu_hat <= 0 or rel_error > 0.5",
        "lambda_zero_always_scored": True,
        "oracle_used_for_selection": False,
        "arms": arms,
        "baselines": {
            pattern: {name: _score_dict(score) for name, score in slices.items()}
            for pattern, slices in baselines.items()
        },
        "instances": _instance_table(grouped, seeds, selection, ls_full),
    }


def write_stage_d_json(payload: dict[str, Any], path: Path) -> Path:
    destination = Path(path)
    destination.parent.mkdir(parents=True, exist_ok=True)
    destination.write_text(json.dumps(_jsonify(payload), indent=2) + "\n", encoding="utf-8")
    return destination


def _objective_block(
    runs: Sequence[Mapping[str, Any]],
    lam: float,
    masks: Mapping[str, np.ndarray],
    ls_full: RecoveryScore,
    *,
    oracle: bool,
) -> dict[str, Any]:
    per_seed_scores: dict[str, list[RecoveryScore]] = {name: [] for name in SLICE_NAMES}
    paired: list[dict[str, Any]] = []
    ident: list[float] = []
    alias_rel: list[float] = []
    alias_delta: list[float] = []
    ls_by_id = {row.instance_id: row for row in ls_full.instances}
    pattern = "oracle_onestep" if oracle else "sparse_rollout"
    for run in runs:
        grid = np.asarray(run["grid"], dtype=np.float64)
        items = run["instances"]
        estimates = [choose_nu(item, grid, lam, oracle=oracle) for item in items]
        full = score_pattern(items, estimates, float(run["baseline_nu"]), pattern)
        for name, mask in masks.items():
            per_seed_scores[name].append(_subset_score(full, mask, pattern))
        hard_rows = [row for row, keep in zip(full.instances, masks["hard_ood"], strict=True) if keep]
        paired.append(_paired(hard_rows, ls_by_id, int(run["seed"])))
        ident.append(_median_identifiability(items, masks["hard_ood"], oracle=oracle))
        alias_rel.append(_alias_pearson(items, hard_rows, "rel_error"))
        alias_delta.append(_alias_pearson(items, hard_rows, "delta", ls_by_id=ls_by_id))
    slices = {name: _aggregate_scores(per_seed_scores[name]) for name in SLICE_NAMES}
    hard_ids = {row.instance_id for row in per_seed_scores["hard_ood"][0].instances}
    return {
        "lambda": float(lam),
        "pattern": pattern,
        "slices": slices,
        "paired_hard_ood_vs_sensors32": _aggregate_paired(paired),
        "hard_ood_identifiability": _aggregate_float(ident),
        "hard_ood_alias_vs_rel_error": _aggregate_float(alias_rel),
        "hard_ood_alias_vs_paired_delta": _aggregate_float(alias_delta),
        "failure_examples": _failure_examples(runs, lam, oracle, ls_by_id, hard_ids),
    }


def _aggregate_scores(scores: Sequence[RecoveryScore]) -> dict[str, Any]:
    payload: dict[str, Any] = {"n_instances": scores[0].n_instances, "seeds": []}
    for field in _SCORE_FIELDS:
        values = []
        for score in scores:
            value = getattr(score, field)
            if value is None:
                values.append(None)
            else:
                values.append(float(value) if not isinstance(value, (int, np.integer)) else int(value))
        if any(value is None for value in values):
            payload[field] = {"mean": None, "std": None, "per_seed": values}
            continue
        numeric = [float(value) for value in values]
        center, spread = mean_std(numeric)
        payload[field] = {
            "mean": center,
            "std": spread,
            "per_seed": [int(value) if field.startswith("n_") else value for value in numeric],
        }
    if len({score.n_instances for score in scores}) != 1:
        raise ValueError("seeds disagree on the slice size")
    return payload


def _paired(
    rows: Sequence[InstanceRecovery],
    ls_by_id: Mapping[int, InstanceRecovery],
    seed: int,
) -> dict[str, Any]:
    lower = higher = tie = 0
    rel_delta = []
    abs_delta = []
    for row in rows:
        other = ls_by_id[row.instance_id]
        rel_delta.append(row.rel_error - other.rel_error)
        abs_delta.append(row.abs_error - other.abs_error)
        if row.rel_error < other.rel_error:
            lower += 1
        elif row.rel_error > other.rel_error:
            higher += 1
        else:
            tie += 1
    return {
        "seed": seed,
        "n_lower_rel_error": lower,
        "n_higher_rel_error": higher,
        "n_tie_rel_error": tie,
        "mean_rel_error_minus_ls": float(np.mean(rel_delta)),
        "mean_abs_error_minus_ls": float(np.mean(abs_delta)),
        "n_failures": sum(1 for row in rows if row.failure),
    }


def _aggregate_paired(rows: Sequence[Mapping[str, Any]]) -> dict[str, Any]:
    payload: dict[str, Any] = {"per_seed": list(rows)}
    for field in (
        "n_lower_rel_error",
        "n_higher_rel_error",
        "n_tie_rel_error",
        "mean_rel_error_minus_ls",
        "mean_abs_error_minus_ls",
        "n_failures",
    ):
        center, spread = mean_std([float(row[field]) for row in rows])
        payload[field] = {"mean": center, "std": spread}
    return payload


def _aggregate_float(values: Sequence[float]) -> dict[str, Any]:
    if any(not math.isfinite(float(value)) for value in values):
        return {"mean": None, "std": None, "per_seed": list(values)}
    center, spread = mean_std([float(value) for value in values])
    return {"mean": center, "std": spread, "per_seed": [float(value) for value in values]}


def _median_identifiability(
    items: Sequence[Mapping[str, Any]],
    mask: np.ndarray,
    *,
    oracle: bool,
) -> float:
    ratios = []
    key = "oracle_sensor_mse" if oracle else "sensor_mse"
    for item, keep in zip(items, mask, strict=True):
        if not keep:
            continue
        curve = np.asarray(item[key], dtype=np.float64)
        floor = max(float(np.min(curve)), 1e-30)
        ratios.append(float((np.max(curve) - np.min(curve)) / floor))
    return float(np.median(ratios))


def _alias_pearson(
    items: Sequence[Mapping[str, Any]],
    rows: Sequence[InstanceRecovery],
    kind: str,
    ls_by_id: Mapping[int, InstanceRecovery] | None = None,
) -> float:
    alias_by_id = {int(item["instance_id"]): float(item["alias_rel_l2"]) for item in items}
    alias = np.asarray([alias_by_id[row.instance_id] for row in rows], dtype=np.float64)
    if kind == "rel_error":
        target = np.asarray([row.rel_error for row in rows], dtype=np.float64)
    else:
        if ls_by_id is None:
            raise ValueError("paired alias correlation needs the least-squares rows")
        target = np.asarray(
            [row.rel_error - ls_by_id[row.instance_id].rel_error for row in rows],
            dtype=np.float64,
        )
    value = _pearson(alias, target)
    if value is None:
        raise ValueError("alias correlation is undefined")
    return value


def _failure_examples(
    runs: Sequence[Mapping[str, Any]],
    lam: float,
    oracle: bool,
    ls_by_id: Mapping[int, InstanceRecovery],
    hard_ids: set[int],
) -> list[dict[str, Any]]:
    """Worst ``hard_ood`` instances by mean relative error across seeds."""

    items = runs[0]["instances"]
    per_instance: dict[int, dict[str, Any]] = {}
    for item in items:
        if int(item["instance_id"]) not in hard_ids:
            continue
        per_instance[int(item["instance_id"])] = {
            "instance_id": int(item["instance_id"]),
            "nu": float(item["nu"]),
            "alias_rel_l2": float(item["alias_rel_l2"]),
            "ls_nu_hat": ls_by_id[int(item["instance_id"])].nu_hat,
            "ls_rel_error": ls_by_id[int(item["instance_id"])].rel_error,
            "ls_failure": ls_by_id[int(item["instance_id"])].failure,
            "rel_error": [],
            "nu_hat": [],
            "failure": [],
        }
    for run in runs:
        grid = np.asarray(run["grid"], dtype=np.float64)
        estimates = [choose_nu(item, grid, lam, oracle=oracle) for item in run["instances"]]
        for item, hat in zip(run["instances"], estimates, strict=True):
            if int(item["instance_id"]) not in hard_ids:
                continue
            nu = float(item["nu"])
            relative = abs(float(hat) - nu) / nu
            record = per_instance[int(item["instance_id"])]
            record["rel_error"].append(relative)
            record["nu_hat"].append(float(hat))
            record["failure"].append(bool(float(hat) <= 0.0 or relative > FAILURE_RELATIVE_ERROR))
    rows = []
    for record in per_instance.values():
        record["mean_rel_error"] = float(np.mean(record["rel_error"]))
        record["n_failures"] = int(sum(record["failure"]))
        rows.append(record)
    rows.sort(key=lambda row: (-float(row["mean_rel_error"]), int(row["instance_id"])))
    worst = rows[:5]
    failing = [row for row in rows if int(row["n_failures"]) >= 3 and row not in worst]
    failing.sort(key=lambda row: (-int(row["n_failures"]), -float(row["mean_rel_error"]), int(row["instance_id"])))
    return worst + failing[:10]


def _instance_table(
    grouped: Mapping[str, Mapping[int, Mapping[str, Any]]],
    seeds: Sequence[int],
    selection: Mapping[str, Any],
    ls_full: RecoveryScore,
) -> list[dict[str, Any]]:
    data_items = grouped["data_only"][seeds[0]]["instances"]
    ls_by_id = {row.instance_id: row for row in ls_full.instances}
    table = []
    for index, item in enumerate(data_items):
        instance_id = int(item["instance_id"])
        other = ls_by_id[instance_id]
        row: dict[str, Any] = {
            "instance_id": instance_id,
            "nu": float(item["nu"]),
            "alias_rel_l2": float(item["alias_rel_l2"]),
            "ls_nu_hat": other.nu_hat,
            "ls_rel_error": other.rel_error,
            "ls_failure": other.failure,
        }
        for arm in ARM_NAMES:
            grid = np.asarray(grouped[arm][seeds[0]]["grid"], dtype=np.float64)
            selected = float(selection["arms"][arm]["selected_lambda"])
            hats = []
            oracle_hats = []
            combined = []
            for seed in seeds:
                instance = grouped[arm][seed]["instances"][index]
                if int(instance["instance_id"]) != instance_id:
                    raise ValueError("test runs are not aligned by instance id")
                hats.append(choose_nu(instance, grid, 0.0, oracle=False))
                oracle_hats.append(choose_nu(instance, grid, 0.0, oracle=True))
                if selected != 0.0:
                    combined.append(choose_nu(instance, grid, selected, oracle=False))
            row[arm] = {
                "lambda0_nu_hat": hats,
                "oracle_nu_hat": oracle_hats,
                "selected_lambda": selected,
                "selected_nu_hat": combined if selected != 0.0 else hats,
            }
        table.append(row)
    return table


def _reference_items(run: Mapping[str, Any]) -> list[Mapping[str, Any]]:
    items = run["instances"]
    if not isinstance(items, list) or len(items) < 1:
        raise ValueError("test run has no instances")
    return items


def _hard_ood_mean_rel(run: Mapping[str, Any], lam: float, threshold: float) -> float:
    grid = np.asarray(run["grid"], dtype=np.float64)
    relative = []
    for item in run["instances"]:
        nu = float(item["nu"])
        if nu > threshold:
            continue
        hat = choose_nu(item, grid, lam, oracle=False)
        relative.append(abs(hat - nu) / nu)
    if not relative:
        raise ValueError("validation hard_ood slice is empty")
    return float(np.mean(relative))


def _require_val_run(run: Mapping[str, Any], digest: str) -> dict[str, Any]:
    if run.get("split") != "val":
        raise ValueError("objective selection reads validation runs only")
    if run.get("protocol_sha256") != digest:
        raise ValueError("validation run protocol hash does not match")
    if run.get("test_used_for_curves") is not False:
        raise ValueError("validation run must not record test curves")
    return dict(run)


def _subset_score(score: RecoveryScore, mask: np.ndarray, pattern: str) -> RecoveryScore:
    chosen = [row for row, keep in zip(score.instances, mask, strict=True) if bool(keep)]
    if not chosen:
        raise ValueError(f"{pattern} slice is empty")
    return recovery_from_rows(pattern, score.baseline_nu, chosen)


def _mask_for_ids(target: RecoveryScore, reference: RecoveryScore, mask: np.ndarray) -> np.ndarray:
    keep = {row.instance_id for row, bit in zip(reference.instances, mask, strict=True) if bit}
    return np.asarray([row.instance_id in keep for row in target.instances], dtype=bool)


def _constant_score(
    sparse: RecoveryScore,
    mask: np.ndarray,
    baseline_nu: float,
    moments_at_baseline: Mapping[int, tuple[float, float]],
) -> RecoveryScore:
    rows = []
    for row, keep in zip(sparse.instances, mask, strict=True):
        if not keep:
            continue
        residual_abs, residual_square = moments_at_baseline[row.instance_id]
        rows.append(
            InstanceRecovery(
                instance_id=row.instance_id,
                nu=row.nu,
                nu_hat=float(baseline_nu),
                abs_error=row.baseline_abs_error,
                rel_error=row.baseline_rel_error,
                baseline_abs_error=row.baseline_abs_error,
                baseline_rel_error=row.baseline_rel_error,
                mean_abs_residual=residual_abs,
                mean_abs_residual_at_truth=row.mean_abs_residual_at_truth,
                mean_square_residual=residual_square,
                mean_square_residual_at_truth=row.mean_square_residual_at_truth,
                failure=float(baseline_nu) <= 0.0 or row.baseline_rel_error > FAILURE_RELATIVE_ERROR,
            )
        )
    return recovery_from_rows("training_mean", baseline_nu, rows)


def _residual_terms(item: Trajectory) -> tuple[np.ndarray, np.ndarray]:
    mask = read_mask(item.u)
    return observed_residual_terms(
        sensor_field(mask, PREREGISTERED_OBSERVATION, int(item.u.shape[0])),
        PREREGISTERED_OBSERVATION,
    )


def _score_dict(score: RecoveryScore) -> dict[str, Any]:
    payload = score.to_dict(include_instances=False)
    payload["n_instances"] = score.n_instances
    return payload


def _pearson(left: np.ndarray, right: np.ndarray) -> float | None:
    if left.size < 2:
        return None
    left_centered = left - float(np.mean(left))
    right_centered = right - float(np.mean(right))
    denom = math.sqrt(float(np.dot(left_centered, left_centered) * np.dot(right_centered, right_centered)))
    if denom == 0.0:
        return None
    value = float(np.dot(left_centered, right_centered) / denom)
    if not math.isfinite(value):
        return None
    return value


def _jsonify(value: Any) -> Any:
    if isinstance(value, dict):
        return {str(key): _jsonify(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [_jsonify(item) for item in value]
    if isinstance(value, np.ndarray):
        return _jsonify(value.tolist())
    if isinstance(value, np.floating):
        number = float(value)
        if not math.isfinite(number):
            raise ValueError("stage D record has a non-finite number")
        return number
    if isinstance(value, np.integer):
        return int(value)
    if isinstance(value, float):
        if not math.isfinite(value):
            raise ValueError("stage D record has a non-finite number")
        return value
    if isinstance(value, (str, int, bool)) or value is None:
        return value
    raise TypeError(f"cannot serialize {type(value).__name__}")


def _require_mask(payload: Mapping[str, Any]) -> None:
    mask = payload.get("mask")
    if not isinstance(mask, dict) or mask.get("unchanged") is not True:
        raise ValueError("stage D mask must stay sensors32_bursts")
    spec = PREREGISTERED_OBSERVATION
    if mask.get("name") != spec.name or mask.get("symbol") != "pinnforge.operator.inverse.PREREGISTERED_OBSERVATION":
        raise ValueError("stage D mask must stay sensors32_bursts")
    if int(mask.get("n_sensors", 0)) != spec.n_sensors or float(mask.get("frame_dt", 0.0)) != spec.frame_dt:
        raise ValueError("stage D sensor count and frame spacing must stay the Stage 5 values")
    starts = [int(value) for value in mask.get("burst_starts", [])]
    if starts != [indexes[0] for indexes in spec.series] or int(mask.get("burst_length", 0)) != len(spec.series[0]):
        raise ValueError("stage D bursts must stay the Stage 5 bursts")


def _require_search(payload: Mapping[str, Any]) -> None:
    search = payload.get("nu_search")
    if not isinstance(search, dict):
        raise ValueError("stage D protocol is missing nu_search")
    if float(search.get("min", -1)) != 0.005 or float(search.get("max", -1)) != 0.1:
        raise ValueError("viscosity search bounds must stay the pilot range [0.005, 0.1]")
    if search.get("optimizer") != "uniform_grid_argmin" or int(search.get("n_grid", 0)) != 191:
        raise ValueError("viscosity search must stay a 191-point uniform grid")
    if search.get("tie_break") != "smallest nu on the grid":
        raise ValueError("viscosity ties must keep the smallest grid value")


def _require_estimator(payload: Mapping[str, Any]) -> None:
    estimator = payload.get("estimator")
    if not isinstance(estimator, dict) or estimator.get("name") != "sparse_rollout":
        raise ValueError("primary estimator must stay sparse_rollout")
    if estimator.get("reads_unmasked_field") is not False:
        raise ValueError("sparse estimator must not read the field off the mask")
    window = estimator.get("initial_window")
    if not isinstance(window, dict) or window.get("nu_dependent") is not False:
        raise ValueError("the initial window must not depend on nu")
    coefficients = window.get("coefficients_ahead_of_frame_4")
    if not isinstance(coefficients, dict):
        raise ValueError("quadratic coefficients are missing")
    for step, expected in NEWTON_AHEAD_OF_FRAME_4.items():
        got = coefficients.get(str(step))
        if [float(value) for value in got] != list(expected):
            raise ValueError("quadratic extrapolation coefficients changed")


def _require_objective(payload: Mapping[str, Any]) -> None:
    objective = payload.get("combined_objective")
    if not isinstance(objective, dict):
        raise ValueError("stage D protocol is missing combined_objective")
    if [float(value) for value in objective.get("lambda_grid", [])] != [0.0, 1.0, 10.0]:
        raise ValueError("lambda grid must stay [0, 1, 10]")
    if objective.get("lambda_zero_always_scored") is not True:
        raise ValueError("lambda 0 must always be scored")
    if objective.get("selection_split") != "val" or objective.get("selection_slice") != "hard_ood":
        raise ValueError("lambda selection must stay on validation hard_ood")
    if objective.get("selection_per_arm") is not True:
        raise ValueError("lambda selection is per arm")
    if objective.get("nonselected_positive_lambda_scored_on_test") is not False:
        raise ValueError("a positive lambda that loses validation is not scored on test")
    if objective.get("tie_break") != "smallest lambda":
        raise ValueError("lambda ties must keep the smallest weight")


def _require_ceiling(payload: Mapping[str, Any]) -> None:
    ceiling = payload.get("analysis_ceiling")
    if not isinstance(ceiling, dict) or ceiling.get("name") != "oracle_onestep":
        raise ValueError("analysis ceiling must stay oracle_onestep")
    if ceiling.get("used_for_selection") is not False:
        raise ValueError("the oracle ceiling must not choose lambda")


def _require_arms(payload: Mapping[str, Any]) -> None:
    arms = payload.get("arms")
    if not isinstance(arms, dict) or set(arms) != set(ARM_NAMES):
        raise ValueError("stage D arms must be data_only and hybrid_1e-2")
    data = arms["data_only"]
    hybrid = arms["hybrid_1e-2"]
    if data.get("loss") != "data" or data.get("residual_weight") is not None:
        raise ValueError("data_only must stay the Stage B data loss")
    if hybrid.get("loss") != "hybrid" or float(hybrid.get("residual_weight", 0.0)) != 0.01:
        raise ValueError("hybrid arm must stay residual weight 1e-2")
    if [int(seed) for seed in data["seeds"]] != [0, 1, 2, 3, 4]:
        raise ValueError("seeds must stay 0 through 4")
    if [int(seed) for seed in hybrid["seeds"]] != [0, 1, 2, 3, 4]:
        raise ValueError("hybrid seeds must stay 0 through 4")
    _require_sha256(Path(data["train_protocol"]), str(data["train_protocol_sha256"]))
    _require_sha256(Path(hybrid["train_protocol"]), str(hybrid["train_protocol_sha256"]))
    _require_sha256(Path(hybrid["weight_source"]), str(hybrid["weight_source_sha256"]))
    weight = json.loads(Path(hybrid["weight_source"]).read_text(encoding="utf-8"))
    if float(weight.get("selected_residual_weight", -1)) != 0.01:
        raise ValueError("hybrid arm must use the Stage C validation-selected weight 1e-2")


def _require_slices(payload: Mapping[str, Any]) -> None:
    if list(payload.get("slices", [])) != list(SLICE_NAMES):
        raise ValueError("stage D slices must stay full_test, hard_ood, complement, below_stage2_floor")
    floor = payload.get("below_stage2_floor")
    if not isinstance(floor, dict) or float(floor.get("threshold_nu", 0.0)) != 0.02:
        raise ValueError("the nu < 0.02 slice must stay the Stage 2 floor")
    hard = payload.get("hard_ood")
    if not isinstance(hard, dict) or hard.get("refit") is not False:
        raise ValueError("hard_ood.refit must be false")


def _require_sha256(path: Path, digest: str) -> None:
    if not path.is_file():
        raise ValueError(f"missing protocol dependency {path}")
    actual = hashlib.sha256(path.read_bytes()).hexdigest()
    if actual != digest:
        raise ValueError(f"{path} hash does not match the stage D protocol")
