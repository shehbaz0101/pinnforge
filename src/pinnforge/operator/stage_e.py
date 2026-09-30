"""Stage E noise and sparsity stress for the viscosity inverse.

The contract is ``docs/v02/stage_e_stress_protocol.json``. It is loaded
and checked here. Test scores do not live in that file.

Noise is additive Gaussian, scaled by a fraction of the clean-field
standard deviation, and drawn once per instance and noise seed. Every
mask reads the same epsilon. Fraction 0 adds nothing, so the reference
cell is the Stage 5 ``sensors32_bursts`` pattern on the clean field.

The operator is the Stage D sparse rollout at ``λ = 0``. A single
opening burst has no target, and that cell is recorded as inapplicable.
This module does not import torch.
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
    PREREGISTERED_BURST_LENGTH,
    PREREGISTERED_BURST_STARTS,
    PREREGISTERED_FRAME_DT,
    PREREGISTERED_N_SENSORS,
    PREREGISTERED_OBSERVATION,
    InstanceRecovery,
    ObservationSpec,
    RecoveryScore,
    dense_reference_observation,
    least_squares_viscosity,
    recovery_from_rows,
    score_trajectories,
)
from pinnforge.operator.slices import HARD_OOD_COMPARISON, hard_ood_threshold, load_data_protocol
from pinnforge.operator.stage_d import (
    ARM_NAMES,
    mean_std,
    slice_masks,
    viscosity_grid,
    window_from_protocol,
)
from pinnforge.operator.windows import Trajectory, WindowSpec

STAGE_E_FORMAT = "pinnforge.stage_e_stress_protocol.v1"
SCORES_FORMAT = "pinnforge.stage_e_scores.v1"
CONDITION_FORMAT = "pinnforge.stage_e_condition.v1"
SLICE_NAMES = ("full_test", "hard_ood", "complement", "below_stage2_floor")
AXES = ("noise", "sensors", "bursts")
_FORBIDDEN = frozenset(
    {
        "results",
        "scores",
        "metrics",
        "test_mean_rel_error",
        "nu_hat",
        "breakdowns",
        "mean_rel_error",
        "n_failures",
    }
)
_SCORE_KEYS = (
    "n_instances",
    "mean_abs_error",
    "median_abs_error",
    "max_abs_error",
    "mean_rel_error",
    "median_rel_error",
    "max_rel_error",
    "correlation",
    "n_failures",
    "n_nonpositive",
    "n_worse_than_baseline",
)
_INV_PHI = (math.sqrt(5) - 1.0) / 2.0
CERTIFICATE_STRIDE = 16


def coarse_indexes(n_grid: int, stride: int) -> np.ndarray:
    """Indexes for a uniform subsample, including both endpoints."""

    if isinstance(n_grid, bool) or not isinstance(n_grid, (int, np.integer)):
        raise ValueError("n_grid must be an integer")
    n_points = int(n_grid)
    step = int(stride)
    if n_points < 2 or step < 1:
        raise ValueError("grid search needs at least two nodes and a positive stride")
    indexes = np.arange(0, n_points, step, dtype=int)
    if int(indexes[-1]) != n_points - 1:
        indexes = np.concatenate([indexes, np.array([n_points - 1], dtype=int)])
    return indexes


def local_minimum_count(curve: np.ndarray) -> int:
    """Count valleys. A flat bottom counts once, on its right edge."""

    finite = np.asarray(curve, dtype=np.float64)
    if finite.ndim != 1 or finite.size < 1 or not np.isfinite(finite).all():
        raise ValueError("local-minimum count needs a finite vector")
    count = 0
    last = int(finite.size) - 1
    for index in range(int(finite.size)):
        left = float(finite[index - 1]) if index > 0 else math.inf
        right = float(finite[index + 1]) if index < last else math.inf
        if float(finite[index]) <= left and float(finite[index]) < right:
            count += 1
    return count


def golden_probe_indexes(left: int, right: int) -> tuple[int, int] | None:
    """Two interior probes, or ``None`` when the bracket should be scanned."""

    span = int(right) - int(left)
    if span <= 3:
        return None
    i1 = int(left) + int(math.floor(span * (1.0 - _INV_PHI)))
    i2 = int(left) + int(math.ceil(span * _INV_PHI))
    if i1 <= int(left):
        i1 = int(left) + 1
    if i2 >= int(right):
        i2 = int(right) - 1
    if i1 >= i2:
        third = max(1, span // 3)
        i1 = int(left) + third
        i2 = int(right) - third
        if i1 >= i2:
            return None
    return i1, i2


def golden_shrink(left: int, right: int, value_left: float, value_right: float) -> tuple[int, int]:
    """Shrink a unimodal bracket. The smaller probe keeps the side that holds it."""

    probes = golden_probe_indexes(int(left), int(right))
    if probes is None:
        return int(left), int(right)
    i1, i2 = probes
    if float(value_left) <= float(value_right):
        updated = (int(left), i2)
    else:
        updated = (i1, int(right))
    if updated == (int(left), int(right)):
        return int(left), int(right)
    return updated


def unimodal_cache_argmin(cache_row: np.ndarray) -> int:
    """Leftmost finite entry. Unevaluated nodes are ignored."""

    values = np.asarray(cache_row, dtype=np.float64)
    if values.ndim != 1:
        raise ValueError("cache row must be one-dimensional")
    finite = np.isfinite(values)
    if not bool(finite.any()):
        raise ValueError("grid search cache is empty")
    masked = np.where(finite, values, np.inf)
    return int(np.argmin(masked))


def unimodal_grid_index(curve: np.ndarray) -> int:
    """Leftmost minimizer of a unimodal curve, without reading every node.

    The bracket is the discrete golden cut. A plateau keeps the smaller
    index because a tie shrinks toward the left and ``argmin`` breaks ties
    toward the first finite entry. A curve with several valleys is not
    covered; the caller falls back to the full grid in that case.
    """

    values = np.asarray(curve, dtype=np.float64)
    if values.ndim != 1 or values.size < 2 or not np.isfinite(values).all():
        raise ValueError("unimodal search needs a finite curve")
    n_points = int(values.size)
    cache = np.full(n_points, np.nan, dtype=np.float64)
    left = 0
    right = n_points - 1
    for _ in range(n_points):
        probes = golden_probe_indexes(left, right)
        if probes is None:
            cache[left : right + 1] = values[left : right + 1]
            break
        i1, i2 = probes
        cache[i1] = values[i1]
        cache[i2] = values[i2]
        updated = golden_shrink(left, right, float(cache[i1]), float(cache[i2]))
        if updated == (left, right):
            cache[left : right + 1] = values[left : right + 1]
            break
        left, right = updated
    else:
        raise ValueError("grid search did not finish")
    return unimodal_cache_argmin(cache)


def load_stage_e_protocol(path: Path) -> dict[str, Any]:
    """Read the Stage E contract and reject a file that already holds scores."""

    payload = json.loads(Path(path).read_text(encoding="utf-8"))
    if not isinstance(payload, dict):
        raise ValueError("stage E protocol must be a JSON object")
    if payload.get("format") != STAGE_E_FORMAT:
        raise ValueError(f"stage E protocol format must be {STAGE_E_FORMAT}")
    leaked = _FORBIDDEN & set(payload)
    if leaked:
        names = ", ".join(sorted(leaked))
        raise ValueError(f"stage E protocol must not contain test scores ({names})")
    if payload.get("frozen_before_test_evaluation") is not True:
        raise ValueError("stage E protocol must be frozen before test evaluation")
    if payload.get("test_used_for_selection") is not False:
        raise ValueError("stage E protocol must not select on the test split")
    if payload.get("hard_ood_refit") is not False or payload.get("lambda_retuned") is not False:
        raise ValueError("stage E must not retune hard_ood or lambda")
    if payload.get("prior_protocols_edited") is not False:
        raise ValueError("stage E must not edit prior protocols")
    if payload.get("failure_relative_error") != FAILURE_RELATIVE_ERROR:
        raise ValueError("failure relative error must stay 0.5")
    if payload.get("failure_rule") != "nu_hat <= 0 or rel_error > 0.5":
        raise ValueError("failure rule must stay the Stage 5 rule")
    if list(payload.get("slices", [])) != list(SLICE_NAMES):
        raise ValueError("stage E slices must stay full_test, hard_ood, complement, below_stage2_floor")
    if payload.get("split") != "test":
        raise ValueError("stage E scores the test split")
    data_protocol = load_data_protocol(Path(payload["data_protocol"]))
    if hard_ood_threshold(payload) != hard_ood_threshold(data_protocol):
        raise ValueError("stage E hard_ood threshold does not match the pilot protocol")
    if payload.get("hard_ood", {}).get("comparison") != HARD_OOD_COMPARISON:
        raise ValueError(f"hard_ood comparison must stay {HARD_OOD_COMPARISON!r}")
    _require_sha256(Path(payload["data_protocol"]), str(payload["data_protocol_sha256"]))
    _require_sha256(Path(payload["baseline_record"]), str(payload["baseline_record_sha256"]))
    _require_sha256(Path(payload["stage_d_protocol"]), str(payload["stage_d_protocol_sha256"]))
    _require_objective(payload)
    _require_noise(payload)
    _require_schedule(payload)
    _require_conditions(payload)
    _require_search(payload)
    _require_arms(payload)
    _require_breakdown(payload)
    window = payload.get("window")
    if not isinstance(window, dict):
        raise ValueError("stage E protocol is missing the window")
    expected = {"input_frames": 8, "output_frames": 8, "stride": 8}
    if {key: int(window[key]) for key in expected} != expected:
        raise ValueError("stage E window must stay 8/8/8")
    if int(payload.get("n_space", 0)) != 1024:
        raise ValueError("stage E spatial grid must stay 1024")
    if payload.get("operator_requires_target_burst") is not True:
        raise ValueError("the operator must declare that it needs a target burst")
    if payload.get("zero_noise_reference_must_match_stage_a") is not True:
        raise ValueError("the clean reference cell must be locked to the Stage A record")
    return payload


def stage_e_protocol_sha256(path: Path) -> str:
    """SHA-256 of the protocol file bytes."""

    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def observation_spec(protocol: Mapping[str, Any], n_sensors: int, n_bursts: int) -> ObservationSpec:
    """Equispaced sensors and the locked burst starts for one condition."""

    schedule = protocol["burst_schedule"]
    starts = [int(item) for item in schedule["starts"][str(int(n_bursts))]]
    length = int(schedule["length"])
    name = f"sensors{int(n_sensors)}_bursts{int(n_bursts)}"
    if int(n_sensors) == PREREGISTERED_N_SENSORS and starts == list(PREREGISTERED_BURST_STARTS):
        name = PREREGISTERED_OBSERVATION.name
    spec = ObservationSpec(
        name=name,
        n_sensors=int(n_sensors),
        series=tuple(tuple(range(start, start + length)) for start in starts),
        frame_dt=float(schedule["frame_dt"]),
    )
    if name == PREREGISTERED_OBSERVATION.name and spec != PREREGISTERED_OBSERVATION:
        raise ValueError("the 32-sensor four-burst mask must stay sensors32_bursts")
    return spec


def operator_applicable(spec: ObservationSpec, window: WindowSpec) -> bool:
    """True when the mask has a target burst after the opening window."""

    from pinnforge.operator.stage_d import target_slots

    try:
        target_slots(spec, window)
    except ValueError:
        return False
    return True


def field_std(field: np.ndarray) -> float:
    """Population standard deviation of one clean saved trajectory."""

    values = np.asarray(field, dtype=np.float64)
    if values.ndim != 2 or not np.isfinite(values).all():
        raise ValueError("field must be a finite array of shape (n_times, n_space)")
    std = float(np.std(values, ddof=0))
    if not math.isfinite(std) or std <= 0.0:
        raise ValueError("field standard deviation must be > 0")
    return std


def standard_normal_field(
    master_seed: int,
    noise_seed: int,
    instance_id: int,
    shape: tuple[int, ...],
) -> np.ndarray:
    """One epsilon field. The same seed always returns the same array."""

    entropy = [int(master_seed), int(noise_seed), int(instance_id)]
    generator = np.random.Generator(np.random.PCG64(np.random.SeedSequence(entropy)))
    draw = generator.standard_normal(shape)
    return np.asarray(draw, dtype=np.float64)


def apply_noise(clean: np.ndarray, epsilon: np.ndarray, fraction: float) -> np.ndarray:
    """Return a copy. Fraction 0 is the clean field and does not read epsilon."""

    values = np.asarray(clean, dtype=np.float64)
    if float(fraction) == 0.0:
        return np.array(values, copy=True)
    weight = float(fraction)
    if not math.isfinite(weight) or weight < 0.0:
        raise ValueError("noise fraction must be finite and >= 0")
    noise = np.asarray(epsilon, dtype=np.float64)
    if noise.shape != values.shape or not np.isfinite(noise).all():
        raise ValueError("epsilon must be a finite field with the same shape as u")
    return values + weight * field_std(values) * noise


def noisy_trajectories(
    clean: Sequence[Trajectory],
    epsilon_by_id: Mapping[int, np.ndarray],
    fraction: float,
) -> list[Trajectory]:
    """Same instances, with noise on the stored field. ``nu`` is unchanged."""

    noisy: list[Trajectory] = []
    for item in clean:
        if not isinstance(item, Trajectory):
            raise TypeError("trajectories must be Trajectory values")
        field = apply_noise(item.u, epsilon_by_id[item.instance_id], fraction)
        noisy.append(Trajectory(instance_id=item.instance_id, split=item.split, nu=item.nu, u=field))
    return noisy


def is_failure(nu_hat: float, nu: float) -> bool:
    """Stage 5 rule: nonpositive estimate, or relative error above 0.5."""

    if not math.isfinite(nu_hat) or not math.isfinite(nu) or nu == 0.0:
        return True
    return nu_hat <= 0.0 or abs(nu_hat - nu) / nu > FAILURE_RELATIVE_ERROR


def score_closed_form(
    trajectories: Sequence[Trajectory],
    spec: ObservationSpec,
    baseline_nu: float,
) -> RecoveryScore:
    """Residual least squares on one mask. The true viscosity is only an error column."""

    return score_trajectories(list(trajectories), spec, baseline_nu)


def recovery_from_operator_records(
    records: Sequence[Mapping[str, Any]],
    baseline_nu: float,
    pattern: str,
) -> RecoveryScore:
    """Rebuild rows from stored estimates. The viscosity is not refit."""

    rows: list[InstanceRecovery] = []
    baseline = float(baseline_nu)
    for record in records:
        nu = float(record["nu"])
        hat = float(record["nu_hat"])
        absolute = abs(hat - nu)
        relative = absolute / nu
        baseline_absolute = abs(baseline - nu)
        rows.append(
            InstanceRecovery(
                instance_id=int(record["instance_id"]),
                nu=nu,
                nu_hat=hat,
                abs_error=absolute,
                rel_error=relative,
                baseline_abs_error=baseline_absolute,
                baseline_rel_error=baseline_absolute / nu,
                mean_abs_residual=float(record["mean_abs_residual"]),
                mean_abs_residual_at_truth=float(record["mean_abs_residual_at_truth"]),
                mean_square_residual=float(record["mean_square_residual"]),
                mean_square_residual_at_truth=float(record["mean_square_residual_at_truth"]),
                failure=is_failure(hat, nu),
            )
        )
    return recovery_from_rows(pattern, baseline, rows)


def slice_summaries(
    score: RecoveryScore,
    protocol: Mapping[str, Any],
    *,
    grid: np.ndarray | None = None,
) -> dict[str, Any]:
    """Per-slice reductions. Endpoint counts need the viscosity grid."""

    viscosities = np.asarray([row.nu for row in score.instances], dtype=np.float64)
    floor = float(protocol["below_stage2_floor"]["threshold_nu"])
    masks = slice_masks(viscosities, hard_ood_threshold(protocol), floor)
    payload: dict[str, Any] = {}
    for name, mask in masks.items():
        part = _subset_score(score, mask, score.pattern)
        block = _score_dict(part)
        if grid is not None:
            hats = np.asarray([row.nu_hat for row, keep in zip(score.instances, mask, strict=True) if keep])
            block["n_at_grid_min"] = int(np.sum(np.isclose(hats, float(grid[0]), rtol=0.0, atol=0.0)))
            block["n_at_grid_max"] = int(np.sum(np.isclose(hats, float(grid[-1]), rtol=0.0, atol=0.0)))
        payload[name] = block
    return payload


def aggregate_seed_summaries(per_seed: Sequence[Mapping[str, Any]]) -> dict[str, Any]:
    """Mean and sample standard deviation across seeds, one slice at a time."""

    if len(per_seed) < 2:
        raise ValueError("operator aggregation needs at least two seeds")
    names = list(per_seed[0])
    combined: dict[str, Any] = {}
    for name in names:
        block: dict[str, Any] = {"n_instances": int(per_seed[0][name]["n_instances"]), "per_seed": []}
        for seed_summary in per_seed:
            if int(seed_summary[name]["n_instances"]) != block["n_instances"]:
                raise ValueError(f"{name} instance count changed across seeds")
            block["per_seed"].append(
                {
                    "seed": int(seed_summary[name]["seed"]),
                    "mean_rel_error": float(seed_summary[name]["mean_rel_error"]),
                    "n_failures": int(seed_summary[name]["n_failures"]),
                    "mean_abs_error": float(seed_summary[name]["mean_abs_error"]),
                    "max_rel_error": float(seed_summary[name]["max_rel_error"]),
                }
            )
        for field in ("mean_rel_error", "mean_abs_error", "max_rel_error", "n_failures", "n_nonpositive"):
            center, spread = mean_std([float(item[name][field]) for item in per_seed])
            block[field] = center
            block[f"{field}_std"] = spread
        block["median_rel_error"] = float(np.mean([float(item[name]["median_rel_error"]) for item in per_seed]))
        block["correlation"] = _mean_optional([item[name]["correlation"] for item in per_seed])
        if "n_at_grid_min" in per_seed[0][name]:
            block["n_at_grid_min"] = float(np.mean([int(item[name]["n_at_grid_min"]) for item in per_seed]))
            block["n_at_grid_max"] = float(np.mean([int(item[name]["n_at_grid_max"]) for item in per_seed]))
        combined[name] = block
    return combined


def axis_conditions(protocol: Mapping[str, Any], axis: str) -> list[dict[str, Any]]:
    """Conditions on one degradation axis, in the preregistered stress order."""

    if axis not in AXES:
        raise ValueError(f"unknown axis {axis!r}")
    chosen = [dict(item) for item in protocol["conditions"] if axis in item["axes"]]
    if axis == "noise":
        chosen.sort(key=lambda item: float(item["noise_fraction"]))
    elif axis == "sensors":
        chosen.sort(key=lambda item: -int(item["n_sensors"]))
    else:
        chosen.sort(key=lambda item: -int(item["n_bursts"]))
    return chosen


def reference_examples(
    records: Sequence[Mapping[str, Any]],
    *,
    limit: int = 5,
) -> dict[str, Any]:
    """Worst relative errors, and the failures, for one method on one condition."""

    ranked = sorted(records, key=lambda item: (-float(item["rel_error"]), int(item["instance_id"])))
    failures = [item for item in ranked if bool(item["failure"])]
    kept_failures = failures if len(failures) <= 12 else failures[:8]
    return {
        "n_failures": len(failures),
        "worst": [_example(item) for item in ranked[:limit]],
        "failures": [_example(item) for item in kept_failures],
        "failures_truncated": len(failures) > len(kept_failures),
    }


def write_stress_chart(
    path: Path,
    series: Sequence[Mapping[str, Any]],
    *,
    xlabel: str,
    ylabel: str,
    title: str,
    x_labels: Sequence[str],
    hline: float | None = None,
    ylog: bool = False,
) -> None:
    """Categorical SVG chart. Each series has ``name``, ``x`` indexes, and ``y``."""

    if len(x_labels) < 2:
        raise ValueError("a stress chart needs at least two categories")
    prepared = []
    for item in series:
        xs = [int(value) for value in item["x"]]
        ys = [float(value) for value in item["y"]]
        if len(xs) != len(ys) or len(xs) < 1:
            raise ValueError(f"series {item['name']} has mismatched coordinates")
        if any(index < 0 or index >= len(x_labels) for index in xs):
            raise ValueError(f"series {item['name']} leaves the category axis")
        if ylog and any(value <= 0.0 for value in ys):
            raise ValueError(f"series {item['name']} is not positive")
        prepared.append({"name": str(item["name"]), "x": xs, "y": ys})
    width, height = 760, 460
    left, right, top, bottom = 84, 28, 48, 64
    plot_w = width - left - right
    plot_h = height - top - bottom
    y_values = [math.log10(value) if ylog else value for item in prepared for value in item["y"]]
    if hline is not None:
        y_values.append(math.log10(hline) if ylog else float(hline))
    y_min, y_max = min(y_values), max(y_values)
    if y_max == y_min:
        y_max = y_min + 1.0
    pad = 0.08 * (y_max - y_min)
    y_min -= pad
    y_max += pad

    def sx(index: int) -> float:
        return left + index / (len(x_labels) - 1) * plot_w

    def sy(value: float) -> float:
        mapped = math.log10(value) if ylog else value
        return top + (y_max - mapped) / (y_max - y_min) * plot_h

    colors = ("#b85c38", "#1f4e79", "#2f6b4f", "#6b4c9a", "#5c5c5c")
    parts = [
        f'<svg xmlns="http://www.w3.org/2000/svg" width="{width}" height="{height}" viewBox="0 0 {width} {height}">',
        '<rect width="100%" height="100%" fill="#ffffff"/>',
        f'<text x="{left}" y="28" font-family="sans-serif" font-size="16">{_escape(title)}</text>',
    ]
    for tick in range(6):
        fraction = tick / 5
        y = top + fraction * plot_h
        mapped = y_max - fraction * (y_max - y_min)
        label = _format_tick(10**mapped if ylog else mapped)
        parts.append(f'<line x1="{left}" y1="{y:.2f}" x2="{left + plot_w}" y2="{y:.2f}" stroke="#e6e6e6"/>')
        parts.append(
            f'<text x="{left - 8}" y="{y + 4:.2f}" text-anchor="end" font-family="sans-serif" font-size="11">{label}</text>'
        )
    for index, label in enumerate(x_labels):
        x = sx(index)
        parts.append(f'<line x1="{x:.2f}" y1="{top}" x2="{x:.2f}" y2="{top + plot_h}" stroke="#f3f3f3"/>')
        parts.append(
            f'<text x="{x:.2f}" y="{top + plot_h + 18}" text-anchor="middle" font-family="sans-serif" font-size="11">{_escape(label)}</text>'
        )
    parts.append(f'<rect x="{left}" y="{top}" width="{plot_w}" height="{plot_h}" fill="none" stroke="#222222"/>')
    if hline is not None:
        y = sy(float(hline))
        parts.append(
            f'<line x1="{left}" y1="{y:.2f}" x2="{left + plot_w}" y2="{y:.2f}" stroke="#888888" stroke-dasharray="5 4"/>'
        )
    for index, item in enumerate(prepared):
        color = colors[index % len(colors)]
        points = " ".join(f"{sx(x):.2f},{sy(y):.2f}" for x, y in zip(item["x"], item["y"], strict=True))
        parts.append(f'<polyline fill="none" stroke="{color}" stroke-width="2" points="{points}"/>')
        for x, y in zip(item["x"], item["y"], strict=True):
            parts.append(f'<circle cx="{sx(x):.2f}" cy="{sy(y):.2f}" r="3.5" fill="{color}"/>')
        legend_y = top + 16 + index * 16
        parts.append(
            f'<line x1="{left + 12}" y1="{legend_y}" x2="{left + 36}" y2="{legend_y}" stroke="{color}" stroke-width="2"/>'
        )
        parts.append(
            f'<text x="{left + 42}" y="{legend_y + 4}" font-family="sans-serif" font-size="12">{_escape(item["name"])}</text>'
        )
    parts.append(
        f'<text x="{left + plot_w / 2:.2f}" y="{height - 16}" text-anchor="middle" font-family="sans-serif" font-size="13">{_escape(xlabel)}</text>'
    )
    parts.append(
        f'<text x="18" y="{top + plot_h / 2:.2f}" text-anchor="middle" transform="rotate(-90 18 {top + plot_h / 2:.2f})" font-family="sans-serif" font-size="13">{_escape(ylabel)}</text>'
    )
    parts.append("</svg>")
    destination = Path(path)
    destination.parent.mkdir(parents=True, exist_ok=True)
    destination.write_text("\n".join(parts) + "\n", encoding="utf-8")


def write_stage_e_figures(scores: Mapping[str, Any], protocol: Mapping[str, Any], directory: Path) -> list[Path]:
    """Degradation curves for mean relative error and failure counts."""

    root = Path(directory)
    written: list[Path] = []
    slice_title = {
        "full_test": "full test",
        "hard_ood": "hard_ood",
        "complement": "complement",
        "below_stage2_floor": "nu < 0.02",
    }
    for axis in AXES:
        ordered = axis_conditions(protocol, axis)
        labels = [_axis_label(axis, item) for item in ordered]
        indexes = list(range(len(ordered)))
        for slice_name in SLICE_NAMES:
            for metric, ylabel, hline, ylog in (
                ("mean_rel_error", "mean relative error", 0.1, True),
                ("n_failures", "failure count", None, False),
            ):
                series = _chart_series(scores, ordered, indexes, slice_name, metric)
                path = root / f"{metric}_vs_{axis}_{slice_name}.svg"
                write_stress_chart(
                    path,
                    series,
                    xlabel=_axis_xlabel(axis),
                    ylabel=ylabel,
                    title=f"{slice_title[slice_name]}: {ylabel} vs {_axis_xlabel(axis)}",
                    x_labels=labels,
                    hline=hline,
                    ylog=ylog,
                )
                written.append(path)
    return written


def write_stage_e_json(payload: dict[str, Any], path: Path) -> Path:
    destination = Path(path)
    destination.parent.mkdir(parents=True, exist_ok=True)
    destination.write_text(json.dumps(_jsonify(payload), indent=2) + "\n", encoding="utf-8")
    return destination


def grid_from_protocol(protocol: Mapping[str, Any]) -> np.ndarray:
    return viscosity_grid(protocol)


def window_spec(protocol: Mapping[str, Any]) -> WindowSpec:
    return window_from_protocol(protocol)


def dense_spec(n_times: int) -> ObservationSpec:
    return dense_reference_observation(n_times, frame_dt=PREREGISTERED_FRAME_DT)


def _subset_score(score: RecoveryScore, mask: np.ndarray, pattern: str) -> RecoveryScore:
    chosen = [row for row, keep in zip(score.instances, mask, strict=True) if bool(keep)]
    if not chosen:
        raise ValueError(f"{pattern} slice is empty")
    return recovery_from_rows(pattern, score.baseline_nu, chosen)


def _score_dict(score: RecoveryScore) -> dict[str, Any]:
    payload = {key: getattr(score, key) for key in _SCORE_KEYS}
    payload["pattern"] = score.pattern
    return payload


def _method_onsets_per_slice(
    ordered: Sequence[Mapping[str, Any]],
    conditions: Mapping[str, Any],
    method: str,
    threshold: float,
) -> dict[str, Any]:
    per_slice: dict[str, Any] = {}
    for slice_name in SLICE_NAMES:
        rel_onset = None
        failure_onset = None
        inapplicable_onset = None
        for condition in ordered:
            block = conditions[str(condition["id"])][method]
            if block.get("applicable") is False:
                if inapplicable_onset is None:
                    inapplicable_onset = {
                        "condition_id": str(condition["id"]),
                        "noise_fraction": float(condition["noise_fraction"]),
                        "n_sensors": int(condition["n_sensors"]),
                        "n_bursts": int(condition["n_bursts"]),
                        "reason": block.get("reason", "inapplicable"),
                    }
                continue
            stats = block["slices"][slice_name]
            rel_value = float(stats["mean_rel_error"])
            failures = float(stats["n_failures"])
            if rel_onset is None and rel_value > threshold:
                rel_onset = _slice_onset(condition, rel_value, failures)
            if failure_onset is None and failures > 0.0:
                failure_onset = _slice_onset(condition, rel_value, failures)
        reference_id = str(ordered[0]["id"])
        per_slice[slice_name] = {
            "rel_onset": rel_onset,
            "failure_onset": failure_onset,
            "inapplicable_onset": inapplicable_onset,
            "rel_exceeds_at_reference": rel_onset is not None and rel_onset["condition_id"] == reference_id,
            "failures_at_reference": failure_onset is not None and failure_onset["condition_id"] == reference_id,
        }
    return per_slice


def _slice_onset(condition: Mapping[str, Any], rel_value: float, failures: float) -> dict[str, Any]:
    return {
        "condition_id": str(condition["id"]),
        "noise_fraction": float(condition["noise_fraction"]),
        "n_sensors": int(condition["n_sensors"]),
        "n_bursts": int(condition["n_bursts"]),
        "mean_rel_error": rel_value,
        "n_failures": failures,
    }


def breakdown_table(protocol: Mapping[str, Any], conditions: Mapping[str, Any]) -> dict[str, Any]:
    """Where each method first crosses 10% mean relative error or first fails.

    Onsets are computed separately on each slice. An inapplicable operator
    cell is not a 0.5-rule failure.
    """

    threshold = float(protocol["breakdown"]["mean_rel_error_threshold"])
    table: dict[str, Any] = {}
    for axis in AXES:
        ordered = axis_conditions(protocol, axis)
        table[axis] = {
            method: _method_onsets_per_slice(ordered, conditions, method, threshold)
            for method in ("closed_form_ls", "operator_data_only", "operator_hybrid_1e-2", "training_mean")
        }
    return table


def _chart_series(
    scores: Mapping[str, Any],
    ordered: Sequence[Mapping[str, Any]],
    indexes: Sequence[int],
    slice_name: str,
    metric: str,
) -> list[dict[str, Any]]:
    methods = (
        ("closed-form LS", "closed_form_ls"),
        ("data-only FNO", "operator_data_only"),
        ("hybrid FNO", "operator_hybrid_1e-2"),
        ("training mean", "training_mean"),
    )
    series = []
    for label, method in methods:
        xs: list[int] = []
        ys: list[float] = []
        for index, condition in zip(indexes, ordered, strict=True):
            block = scores["conditions"][str(condition["id"])][method]
            if block.get("applicable") is False:
                continue
            value = float(block["slices"][slice_name][metric])
            if metric == "mean_rel_error" and value <= 0.0:
                continue
            xs.append(int(index))
            ys.append(value)
        if xs:
            series.append({"name": label, "x": xs, "y": ys})
    dense = scores.get("dense_ls_clean", {}).get(slice_name)
    if dense is not None and metric in dense:
        value = float(dense[metric])
        if metric != "mean_rel_error" or value > 0.0:
            series.append({"name": "dense LS", "x": list(indexes), "y": [value for _ in indexes]})
    return series


def _axis_label(axis: str, condition: Mapping[str, Any]) -> str:
    if axis == "noise":
        percent = float(condition["noise_fraction"]) * 100.0
        if percent == 0.0:
            return "0"
        text = f"{percent:.3f}".rstrip("0").rstrip(".")
        return f"{text}%"
    if axis == "sensors":
        return str(int(condition["n_sensors"]))
    return str(int(condition["n_bursts"]))


def _axis_xlabel(axis: str) -> str:
    if axis == "noise":
        return "noise, fraction of field std"
    if axis == "sensors":
        return "equispaced sensors"
    return "observation bursts"


def _example(item: Mapping[str, Any]) -> dict[str, Any]:
    payload = {
        "instance_id": int(item["instance_id"]),
        "nu": float(item["nu"]),
        "nu_hat": float(item["nu_hat"]),
        "rel_error": float(item["rel_error"]),
        "failure": bool(item["failure"]),
        "nonpositive": bool(item["nonpositive"]),
    }
    if "alias_rel_l2" in item:
        payload["alias_rel_l2"] = float(item["alias_rel_l2"])
    if "nu_hat_per_seed" in item:
        payload["nu_hat_per_seed"] = [float(value) for value in item["nu_hat_per_seed"]]
    return payload


def _mean_optional(values: Sequence[Any]) -> float | None:
    numbers = [float(item) for item in values if item is not None]
    if len(numbers) != len(values):
        return None
    return float(np.mean(numbers))


def _require_objective(payload: Mapping[str, Any]) -> None:
    objective = payload.get("objective")
    if not isinstance(objective, dict):
        raise ValueError("stage E protocol is missing the objective")
    if float(objective.get("lambda", -1)) != 0.0 or objective.get("retuned") is not False:
        raise ValueError("stage E lambda must stay 0 and must not be retuned")
    source = Path(str(objective["source"]))
    _require_sha256(source, str(objective["source_sha256"]))
    selection = json.loads(source.read_text(encoding="utf-8"))
    if selection.get("test_used_for_selection") is not False:
        raise ValueError("stage D selection must not have used the test split")
    for arm in ARM_NAMES:
        if float(selection["arms"][arm]["selected_lambda"]) != 0.0:
            raise ValueError(f"stage D {arm} did not select lambda 0")


def _require_noise(payload: Mapping[str, Any]) -> None:
    noise = payload.get("noise")
    if not isinstance(noise, dict):
        raise ValueError("stage E protocol is missing the noise model")
    if noise.get("model") != "additive_gaussian":
        raise ValueError("stage E noise must stay additive Gaussian")
    if int(noise.get("ddof", -1)) != 0:
        raise ValueError("field std must stay the population standard deviation")
    if noise.get("fraction_zero_adds_nothing") is not True:
        raise ValueError("fraction 0 must add no noise")
    if noise.get("estimator_reads_unobserved_nodes") is not False:
        raise ValueError("the estimator must not read unobserved nodes")
    if noise.get("estimator_reads_clean_field") is not False:
        raise ValueError("the estimator must not read the clean field")
    fractions = [float(item) for item in noise["fractions"]]
    if fractions[0] != 0.0 or any(item < 0.0 for item in fractions):
        raise ValueError("noise fractions must start at 0 and stay non-negative")
    if [float(item) for item in fractions] != [0.0, 0.001, 0.005, 0.01, 0.02, 0.05]:
        raise ValueError("noise fractions must stay 0, 0.1%, 0.5%, 1%, 2%, 5%")
    seeds = [int(item) for item in noise["noise_seeds"]]
    if int(noise["primary_noise_seed"]) not in seeds:
        raise ValueError("the primary noise seed must be one of the noise seeds")
    if [int(item) for item in noise["operator_noise_seeds"]] != [int(noise["primary_noise_seed"])]:
        raise ValueError("the operator is scored on the primary noise seed")
    extra = [int(item) for item in noise["closed_form_extra_seeds_on_noise_axis"]]
    if not extra or any(item not in seeds or item == int(noise["primary_noise_seed"]) for item in extra):
        raise ValueError("closed-form extra seeds must be other listed noise seeds")
    if int(noise["master_seed"]) != 20260930:
        raise ValueError("noise master seed must stay 20260930")
    if noise.get("bit_generator") != "PCG64":
        raise ValueError("noise draws must stay PCG64")


def _require_schedule(payload: Mapping[str, Any]) -> None:
    schedule = payload.get("burst_schedule")
    sensors = payload.get("sensors")
    if not isinstance(schedule, dict) or not isinstance(sensors, dict):
        raise ValueError("stage E protocol is missing the observation schedule")
    if int(schedule["length"]) != PREREGISTERED_BURST_LENGTH:
        raise ValueError("burst length must stay 5")
    if float(schedule["frame_dt"]) != PREREGISTERED_FRAME_DT:
        raise ValueError("frame dt must stay 0.01")
    starts = schedule["starts"]
    if [int(item) for item in starts["4"]] != list(PREREGISTERED_BURST_STARTS):
        raise ValueError("four bursts must stay the Stage 5 starts")
    if [int(item) for item in starts["2"]] != [0, 48]:
        raise ValueError("two bursts must stay frames 0 and 48")
    if [int(item) for item in starts["1"]] != [0]:
        raise ValueError("one burst must stay the opening burst")
    if [int(item) for item in sensors["counts"]] != [32, 16, 8]:
        raise ValueError("sensor counts must stay 32, 16, 8")
    if schedule.get("reference_matches_stage_5") is not True:
        raise ValueError("the reference mask must stay the Stage 5 pattern")


def _require_conditions(payload: Mapping[str, Any]) -> None:
    conditions = payload.get("conditions")
    if not isinstance(conditions, list) or len(conditions) < 1:
        raise ValueError("stage E protocol is missing the condition grid")
    seen: set[str] = set()
    fractions = {float(item) for item in payload["noise"]["fractions"]}
    counts = {int(item) for item in payload["sensors"]["counts"]}
    bursts = {int(item) for item in payload["burst_schedule"]["starts"]}
    has_reference = False
    for condition in conditions:
        if not isinstance(condition, dict):
            raise ValueError("each condition must be an object")
        leaked = _FORBIDDEN & set(condition)
        if leaked:
            raise ValueError("a condition must not carry scores")
        identifier = str(condition["id"])
        if identifier in seen:
            raise ValueError(f"duplicate condition {identifier}")
        seen.add(identifier)
        if float(condition["noise_fraction"]) not in fractions:
            raise ValueError(f"{identifier} has a noise fraction outside the locked list")
        if int(condition["n_sensors"]) not in counts or int(condition["n_bursts"]) not in bursts:
            raise ValueError(f"{identifier} leaves the locked sensor or burst list")
        axes = list(condition["axes"])
        if any(axis not in (*AXES, "interaction") for axis in axes):
            raise ValueError(f"{identifier} has an unknown axis")
        if (
            float(condition["noise_fraction"]) == 0.0
            and int(condition["n_sensors"]) == 32
            and int(condition["n_bursts"]) == 4
        ):
            has_reference = True
            if set(axes) != set(AXES):
                raise ValueError("the reference cell must sit on the noise, sensor, and burst axes")
    if not has_reference:
        raise ValueError("the grid is missing the clean sensors32_bursts cell")
    for axis, held in (
        ("noise", {"n_sensors": 32, "n_bursts": 4}),
        ("sensors", {"noise_fraction": 0.0, "n_bursts": 4}),
        ("bursts", {"noise_fraction": 0.0, "n_sensors": 32}),
    ):
        for condition in axis_conditions(payload, axis):
            for key, expected in held.items():
                got = condition[key]
                if float(got) != float(expected):
                    raise ValueError(f"{axis} axis does not hold {key} fixed")


def _require_search(payload: Mapping[str, Any]) -> None:
    search = payload.get("nu_search")
    if not isinstance(search, dict):
        raise ValueError("stage E protocol is missing the viscosity grid")
    if float(search["min"]) != 0.005 or float(search["max"]) != 0.1 or int(search["n_grid"]) != 191:
        raise ValueError("the viscosity grid must stay the Stage D 191-point grid")
    stage_d = json.loads(Path(payload["stage_d_protocol"]).read_text(encoding="utf-8"))
    for key in ("min", "max", "n_grid", "tie_break"):
        if search.get(key) != stage_d["nu_search"].get(key):
            raise ValueError(f"viscosity search {key} does not match Stage D")
    grid = viscosity_grid(payload)
    if grid.size != 191 or float(grid[0]) != 0.005 or float(grid[-1]) != 0.1:
        raise ValueError("viscosity grid endpoints drifted")


def _require_arms(payload: Mapping[str, Any]) -> None:
    operator = payload.get("operator")
    if not isinstance(operator, dict):
        raise ValueError("stage E protocol is missing the operator block")
    if operator.get("reads_unmasked_field") is not False or operator.get("oracle_onestep") is not False:
        raise ValueError("stage E operator must stay on the sparse mask and must not score the oracle")
    if float(operator["arms"]["hybrid_1e-2"]["residual_weight"]) != 0.01:
        raise ValueError("hybrid residual weight must stay 1e-2")
    for arm in ARM_NAMES:
        if [int(seed) for seed in operator["arms"][arm]["seeds"]] != [0, 1, 2, 3, 4]:
            raise ValueError(f"{arm} seeds must stay 0 through 4")
        _require_sha256(
            Path(operator["arms"][arm]["train_protocol"]),
            str(operator["arms"][arm]["train_protocol_sha256"]),
        )
    _require_sha256(
        Path(operator["arms"]["hybrid_1e-2"]["weight_source"]),
        str(operator["arms"]["hybrid_1e-2"]["weight_source_sha256"]),
    )
    if payload.get("aggregation", {}).get("ddof") != 1:
        raise ValueError("aggregation must stay the sample standard deviation")
    if int(payload.get("aggregation", {}).get("minimum_seeds", 0)) != 5:
        raise ValueError("aggregation must keep all five seeds")


def _require_breakdown(payload: Mapping[str, Any]) -> None:
    block = payload.get("breakdown")
    if not isinstance(block, dict):
        raise ValueError("stage E protocol is missing the breakdown rule")
    if float(block["mean_rel_error_threshold"]) != 0.1:
        raise ValueError("the relative-error breakdown must stay 10%")
    if int(block["primary_noise_seed"]) != int(payload["noise"]["primary_noise_seed"]):
        raise ValueError("breakdown seed does not match the primary noise seed")
    for axis in AXES:
        if axis not in block["axes"]:
            raise ValueError(f"breakdown is missing the {axis} axis")


def _require_sha256(path: Path, expected: str) -> None:
    digest = hashlib.sha256(Path(path).read_bytes()).hexdigest()
    if digest != expected:
        raise ValueError(f"{path} sha256 does not match the stage E protocol")


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
            raise ValueError("stage E record has a non-finite number")
        return number
    if isinstance(value, np.integer):
        return int(value)
    if isinstance(value, float):
        if not math.isfinite(value):
            raise ValueError("stage E record has a non-finite number")
        return value
    return value


def _format_tick(value: float) -> str:
    if abs(value) >= 100 or (abs(value) > 0 and abs(value) < 0.01):
        return f"{value:.2e}"
    text = f"{value:.3f}".rstrip("0").rstrip(".")
    return text or "0"


def _escape(text: str) -> str:
    return text.replace("&", "&amp;").replace("<", "&lt;").replace(">", "&gt;")


def closed_form_matches(score: RecoveryScore, reference: Mapping[str, Any]) -> bool:
    """True when a recomputed clean mask agrees with a published Stage A row."""

    if int(reference["n_instances"]) != score.n_instances:
        return False
    if int(reference["n_failures"]) != score.n_failures:
        return False
    for field in ("mean_abs_error", "mean_rel_error", "median_rel_error", "max_rel_error"):
        if abs(float(getattr(score, field)) - float(reference[field])) > 1e-12:
            return False
    return True


def sensor_columns(n_space: int, n_sensors: int) -> np.ndarray:
    """Indexes of the equispaced sensors, starting at 0."""

    if int(n_space) % int(n_sensors) != 0:
        raise ValueError("n_space must be divisible by n_sensors")
    stride = int(n_space) // int(n_sensors)
    return np.arange(0, int(n_space), stride, dtype=np.int64)


def condition_by_id(protocol: Mapping[str, Any], identifier: str) -> dict[str, Any]:
    for condition in protocol["conditions"]:
        if str(condition["id"]) == identifier:
            return dict(condition)
    raise ValueError(f"unknown condition {identifier}")


def noise_seeds_for(protocol: Mapping[str, Any], condition: Mapping[str, Any], method: str) -> list[int]:
    """Primary seed for every method. Closed form also repeats the noise axis."""

    primary = int(protocol["noise"]["primary_noise_seed"])
    if method != "closed_form_ls":
        return [primary]
    if "noise" not in condition["axes"] or float(condition["noise_fraction"]) == 0.0:
        return [primary]
    extra = [int(item) for item in protocol["noise"]["closed_form_extra_seeds_on_noise_axis"]]
    return [primary, *extra]


def least_squares_or_none(field: np.ndarray, spec: ObservationSpec) -> float | None:
    """Closed-form viscosity, or ``None`` when ``u_xx`` vanishes on the mask."""

    try:
        return least_squares_viscosity(field, spec)
    except ValueError:
        return None
