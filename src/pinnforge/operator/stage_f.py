"""Stage F contract: train above the frozen viscosity cut, score below it.

Stages B–D trained on the full harder-pilot range, so ``hard_ood`` was
inside the training support. This stage keeps that threshold and does not
refit it. Training and validation instances with ``nu <= threshold_nu``
are left out of the loss, out of the checkpoint selection, and out of the
``u`` and ``nu`` normalization. The hybrid weight stays the Stage C
choice ``1e-2``. It is not chosen again on out-of-distribution data.

The viscosity search used by the inverse is the Stage D grid. It is not
clipped to the restricted training range, so a minimizer can land below
every training viscosity. This module does not import torch.
"""

from __future__ import annotations

import hashlib
import json
import math
from collections.abc import Mapping, Sequence
from pathlib import Path

import numpy as np

from pinnforge.operator.metrics import (
    mean_relative_l2,
    median_relative_l2,
    normalized_mse,
    per_window_relative_l2,
    persistence_prediction,
    physical_targets,
    pooled_relative_l2,
)
from pinnforge.operator.slices import hard_ood_threshold, load_data_protocol
from pinnforge.operator.stage_d import nu_from_objective
from pinnforge.operator.windows import (
    FieldNorm,
    WindowDataset,
    WindowSpec,
    load_pilot_manifest,
    split_sets,
    window_starts,
)

STAGE_F_PROTOCOL_FORMAT = "pinnforge.stage_f_ood_protocol.v1"
SLICE_EVAL_FORMAT = "pinnforge.stage_f_slice_eval.v1"
SLICE_AGGREGATE_FORMAT = "pinnforge.stage_f_slice_aggregate.v1"
INVERSE_RUN_FORMAT = "pinnforge.stage_f_inverse_run.v1"
SCORES_FORMAT = "pinnforge.stage_f_scores.v1"
SLICE_NAMES = ("full_test", "in_range", "ood", "below_0_02")
ARM_NAMES = ("data_only", "hybrid_1e-2")
HYBRID_WEIGHT = 0.01
FAILURE_RELATIVE_ERROR = 0.5
IN_RANGE_RULE = "nu > threshold_nu"
OOD_RULE = "nu <= threshold_nu"
_NORM_ABS = 1e-12
_FORBIDDEN_PROTOCOL_KEYS = frozenset(
    {
        "results",
        "scores",
        "metrics",
        "selected_epoch",
        "val_relative_l2",
        "test_mean_relative_l2",
        "nu_hat",
    }
)
_FORWARD_SLICE_IN_PUBLISHED = {"ood": "hard_ood", "in_range": "complement", "full_test": "full_test"}
_INVERSE_SLICE_IN_PUBLISHED = {
    "ood": "hard_ood",
    "in_range": "complement",
    "full_test": "full_test",
    "below_0_02": "below_stage2_floor",
}
_LS_KEYS = {
    "full_test": "sensors32_bursts",
    "ood": "hard_ood_sensors32_bursts",
    "in_range": "complement_sensors32_bursts",
    "below_0_02": "below_stage2_floor_sensors32_bursts",
}
_DENSE_KEYS = {
    "full_test": "dense_reference",
    "ood": "hard_ood_dense_reference",
}


def load_stage_f_protocol(path: Path) -> dict[str, object]:
    """Read the Stage F contract and reject a file that already holds scores."""

    payload = _read_object(path)
    if payload.get("format") != STAGE_F_PROTOCOL_FORMAT:
        raise ValueError(f"unsupported training protocol format {payload.get('format')!r}")
    leaked = _FORBIDDEN_PROTOCOL_KEYS & set(payload)
    if leaked:
        names = ", ".join(sorted(leaked))
        raise ValueError(f"training protocol must not contain test scores ({names})")
    if payload.get("frozen_before_test_evaluation") is not True:
        raise ValueError("training protocol must be frozen before test evaluation")
    if payload.get("test_used_for_training") is not False or payload.get("test_used_for_selection") is not False:
        raise ValueError("training protocol must keep the test split out of training and selection")
    if payload.get("ood_used_for_selection") is not False:
        raise ValueError("training protocol must not select on out-of-distribution instances")
    if payload.get("hard_ood_refit") is not False:
        raise ValueError("training protocol must not refit hard_ood")
    if payload.get("hybrid_weight_reselected") is not False:
        raise ValueError("training protocol must not reselect the hybrid weight")
    if payload.get("lambda_reselected") is not False:
        raise ValueError("training protocol must not reselect the inverse lambda")
    if payload.get("early_stopping") is not False:
        raise ValueError("training protocol must set early_stopping to false")
    _require_seeds(payload.get("seeds"))
    _require_architecture(payload)
    _require_subset_rules(payload)
    _require_normalization_block(payload)
    _require_inverse_block(payload)
    _require_reference_hashes(payload)
    _require_manifest_locks(payload)
    return payload


def assert_stage_f_train_call(
    protocol: Mapping[str, object],
    *,
    pilot: Path,
    manifest: Path,
    epochs: int,
    batch_size: int,
    lr: float,
    width: int,
    modes: int,
    layers: int,
    input_frames: int,
    output_frames: int,
    stride: int,
    seed: int,
    loss: str,
    residual_weight: float | None,
    residual_scope: str,
    residual_space: str,
    dt: float,
) -> str:
    """Raise when a train invocation leaves the frozen Stage F contract.

    Returns the arm name, ``data_only`` or ``hybrid_1e-2``.
    """

    if loss == "data":
        if residual_weight is not None:
            raise ValueError("data-only Stage F training rejects --residual-weight")
        arm = "data_only"
    elif loss == "hybrid":
        if residual_weight is None or abs(float(residual_weight) - HYBRID_WEIGHT) > 1e-12:
            raise ValueError("Stage F hybrid training requires residual weight 1e-2 and does not reselect it")
        arm = "hybrid_1e-2"
    else:
        raise ValueError("Stage F train loss must be data or hybrid")
    if residual_scope != protocol["residual_scope"] or residual_space != protocol["residual_space"]:
        raise ValueError("residual scope and space must match the Stage F protocol")
    if float(dt) != float(protocol["dt"]):  # type: ignore[arg-type]
        raise ValueError("residual dt does not match the Stage F protocol")
    seeds = protocol["seeds"]
    if not isinstance(seeds, list) or seed not in seeds:
        raise ValueError(f"seed {seed} is not in the frozen training protocol")
    pairs = (
        ("pilot", Path(pilot), Path(str(protocol["pilot"]))),
        ("manifest", Path(manifest), Path(str(protocol["manifest"]))),
    )
    for label, got, expected in pairs:
        if not _same_path(got, expected):
            raise ValueError(f"{label} {got} does not match the frozen training protocol ({expected})")
    integers = {
        "epochs": epochs,
        "batch_size": batch_size,
        "width": width,
        "modes": modes,
        "layers": layers,
        "input_frames": input_frames,
        "output_frames": output_frames,
        "stride": stride,
    }
    for label, got in integers.items():
        if int(got) != int(protocol[label]):  # type: ignore[arg-type]
            raise ValueError(f"{label} {got} does not match the frozen training protocol ({protocol[label]})")
    if float(lr) != float(protocol["lr"]):  # type: ignore[arg-type]
        raise ValueError(f"lr {lr} does not match the frozen training protocol ({protocol['lr']})")
    return arm


def in_range_ids(manifest: Mapping[str, object], split: str, threshold: float) -> list[int]:
    """Instance ids in ``split`` with ``nu > threshold``, sorted."""

    if split not in {"train", "val", "test"}:
        raise ValueError(f"unknown split {split!r}")
    splits = split_sets(manifest)
    nu = _nu_by_id(manifest)
    chosen = [instance_id for instance_id in sorted(splits[split]) if nu[instance_id] > float(threshold)]
    if not chosen:
        raise ValueError(f"{split} has no instances with nu > threshold")
    return chosen


def population_field_norm(fields: Sequence[np.ndarray], viscosities: Sequence[float]) -> FieldNorm:
    """Population mean and standard deviation of raw ``u`` and of ``nu``.

    ``fields`` must already be in the order the protocol locks (ascending
    instance id). Validation and test fields do not belong in this call.
    ``ddof`` is 0.
    """

    if len(fields) < 2 or len(fields) != len(viscosities):
        raise ValueError("normalization needs at least two fields and one viscosity each")
    chunks = [np.asarray(field, dtype=np.float64).ravel() for field in fields]
    values = np.concatenate(chunks)
    nu = np.asarray(list(viscosities), dtype=np.float64)
    if values.size < 2 or not np.isfinite(values).all() or not np.isfinite(nu).all():
        raise ValueError("normalization inputs must be finite")
    if np.any(nu <= 0.0):
        raise ValueError("normalization viscosities must be > 0")
    u_std = float(np.std(values, ddof=0))
    nu_std = float(np.std(nu, ddof=0))
    if u_std <= 0.0 or nu_std <= 0.0:
        raise ValueError("population standard deviations must be > 0")
    return FieldNorm(
        u_mean=float(np.mean(values)),
        u_std=u_std,
        nu_mean=float(np.mean(nu)),
        nu_std=nu_std,
    )


def assert_norm_matches_protocol(norm: FieldNorm, protocol: Mapping[str, object]) -> None:
    """Raise when a checkpoint norm leaves the locked restricted-train stats."""

    if not isinstance(norm, FieldNorm):
        raise TypeError("norm must be a FieldNorm")
    block = protocol.get("normalization")
    if not isinstance(block, dict):
        raise ValueError("protocol is missing normalization")
    for label in ("u_mean", "u_std", "nu_mean", "nu_std"):
        expected = float(block[label])
        got = float(getattr(norm, label))
        if abs(got - expected) > _NORM_ABS:
            raise ValueError(f"{label} {got} does not match the Stage F protocol ({expected})")


def viscosity_grid(protocol: Mapping[str, object]) -> np.ndarray:
    """Inclusive linear grid. Endpoints stay on the Stage D bounds."""

    search = protocol.get("nu_search")
    if not isinstance(search, dict):
        raise ValueError("protocol is missing nu_search")
    grid = np.linspace(float(search["min"]), float(search["max"]), int(search["n_grid"]))
    if abs(float(grid[0]) - float(search["min"])) > 0.0 or abs(float(grid[-1]) - float(search["max"])) > 0.0:
        raise ValueError("viscosity grid does not include both endpoints")
    return np.asarray(grid, dtype=np.float64)


def search_support(protocol: Mapping[str, object]) -> dict[str, object]:
    """Whether the inverse grid can return a viscosity outside the training range."""

    grid = viscosity_grid(protocol)
    train = _train_subset(protocol)
    nu_min = float(train["nu_min"])
    nu_max = float(train["nu_max"])
    below = int(np.count_nonzero(grid < nu_min))
    above = int(np.count_nonzero(grid > nu_max))
    search = protocol["nu_search"]
    if not isinstance(search, dict):
        raise ValueError("protocol is missing nu_search")
    return {
        "min": float(search["min"]),
        "max": float(search["max"]),
        "n_grid": int(grid.size),
        "step": float(grid[1] - grid[0]),
        "clipped_to_training_support": False,
        "train_nu_min": nu_min,
        "train_nu_max": nu_max,
        "n_grid_below_train_min": below,
        "n_grid_above_train_max": above,
        "can_return_values_outside_training_range": below > 0 or above > 0,
    }


def ood_masks(viscosities: np.ndarray, threshold: float, floor: float) -> dict[str, np.ndarray]:
    """Boolean masks. ``below_0_02`` is a subset of ``ood`` when ``floor <= threshold``."""

    nu = np.asarray(viscosities, dtype=np.float64)
    if nu.ndim != 1 or nu.size < 1 or not np.isfinite(nu).all():
        raise ValueError("viscosities must be a non-empty finite vector")
    threshold_nu = float(threshold)
    floor_nu = float(floor)
    masks = {
        "full_test": np.ones(nu.shape[0], dtype=bool),
        "in_range": nu > threshold_nu,
        "ood": nu <= threshold_nu,
        "below_0_02": nu < floor_nu,
    }
    if np.any(masks["in_range"] & masks["ood"]):
        raise ValueError("in_range and ood overlap")
    if not np.array_equal(masks["in_range"] | masks["ood"], masks["full_test"]):
        raise ValueError("in_range and ood do not partition the split")
    if floor_nu <= threshold_nu and np.any(masks["below_0_02"] & ~masks["ood"]):
        raise ValueError("below_0_02 must sit inside ood")
    for name in SLICE_NAMES:
        if not np.any(masks[name]):
            raise ValueError(f"{name} has no instances")
    return masks


def score_ood_windows(
    dataset: WindowDataset,
    prediction_normalized: np.ndarray,
    nu_by_id: Mapping[int, float],
    threshold: float,
    floor: float,
) -> dict[str, dict[str, object]]:
    """One-step scores on the Stage F slices. The cut is not refit."""

    if dataset.split != "test":
        raise ValueError("Stage F slice scoring is defined on the test split")
    prediction = np.asarray(prediction_normalized, dtype=np.float64)
    if prediction.shape != dataset.targets.shape:
        raise ValueError("prediction shape does not match targets")
    if not np.isfinite(prediction).all():
        raise ValueError("prediction must be finite")
    predicted = dataset.norm.denormalize_u(prediction)
    reference = physical_targets(dataset)
    persistence = persistence_prediction(dataset)
    window_l2 = per_window_relative_l2(predicted, reference)
    persist_l2 = per_window_relative_l2(persistence, reference)
    nu = np.asarray([float(nu_by_id[int(instance_id)]) for instance_id in dataset.instance_ids], dtype=np.float64)
    masks = ood_masks(nu, threshold, floor)
    scores: dict[str, dict[str, object]] = {}
    for name in SLICE_NAMES:
        mask = masks[name]
        pred_slice = predicted[mask]
        ref_slice = reference[mask]
        persist_slice = persistence[mask]
        ids = dataset.instance_ids[mask]
        scores[name] = {
            "n_windows": int(np.count_nonzero(mask)),
            "n_instances": len({int(value) for value in ids}),
            "mean_relative_l2": mean_relative_l2(pred_slice, ref_slice),
            "median_relative_l2": median_relative_l2(pred_slice, ref_slice),
            "pooled_relative_l2": pooled_relative_l2(pred_slice, ref_slice),
            "normalized_mse": normalized_mse(prediction[mask], dataset.targets[mask]),
            "persistence_mean_relative_l2": mean_relative_l2(persist_slice, ref_slice),
            "persistence_median_relative_l2": median_relative_l2(persist_slice, ref_slice),
            "persistence_pooled_relative_l2": pooled_relative_l2(persist_slice, ref_slice),
            "instances": _per_instance(ids, nu[mask], window_l2[mask], persist_l2[mask]),
        }
    return scores


def summarize_ood_rollout(
    rollout: Mapping[str, np.ndarray],
    threshold: float,
    floor: float,
) -> dict[str, dict[str, object]]:
    """Reduce an autoregressive rollout onto the Stage F slices."""

    nu = np.asarray(rollout["nu"], dtype=np.float64)
    masks = ood_masks(nu, threshold, floor)
    instance_scores = np.asarray(rollout["instance_relative_l2"], dtype=np.float64)
    persistence = np.asarray(rollout["persistence_instance_relative_l2"], dtype=np.float64)
    steps = np.asarray(rollout["step_relative_l2"], dtype=np.float64)
    ids = np.asarray(rollout["instance_ids"], dtype=np.int64)
    if steps.ndim != 2 or steps.shape[0] != nu.shape[0]:
        raise ValueError("rollout step scores must have shape (n_instances, n_steps)")
    summary: dict[str, dict[str, object]] = {}
    for name in SLICE_NAMES:
        mask = masks[name]
        chosen = instance_scores[mask]
        summary[name] = {
            "n_instances": int(np.count_nonzero(mask)),
            "n_steps": int(steps.shape[1]),
            "mean_instance_relative_l2": float(np.mean(chosen)),
            "median_instance_relative_l2": float(np.median(chosen)),
            "persistence_mean_instance_relative_l2": float(np.mean(persistence[mask])),
            "per_step_mean_relative_l2": [float(value) for value in np.mean(steps[mask], axis=0)],
            "instances": [
                {
                    "instance_id": int(ids[index]),
                    "nu": float(nu[index]),
                    "relative_l2": float(instance_scores[index]),
                    "persistence_relative_l2": float(persistence[index]),
                }
                for index in np.flatnonzero(mask)
            ],
        }
    return summary


def aggregate_ood_records(
    records: Sequence[Mapping[str, object]],
    protocol: Mapping[str, object],
    *,
    arm: str,
) -> dict[str, object]:
    """Mean and sample standard deviation of one scalar per seed."""

    if arm not in ARM_NAMES:
        raise ValueError(f"unknown arm {arm!r}")
    expected = protocol.get("seeds")
    if not isinstance(expected, list) or len(records) != len(expected):
        raise ValueError("aggregation requires one record per frozen seed")
    ordered = sorted(records, key=lambda row: int(row["seed"]))
    seeds = [int(row["seed"]) for row in ordered]
    if seeds != sorted(int(seed) for seed in expected):
        raise ValueError(f"seed records {seeds} do not match the frozen seeds")
    threshold = float(protocol["threshold_nu"])  # type: ignore[arg-type]
    floor = float(protocol["below_0_02_threshold"])  # type: ignore[arg-type]
    for row in ordered:
        if row.get("format") != SLICE_EVAL_FORMAT:
            raise ValueError(f"unsupported slice record format {row.get('format')!r}")
        if row.get("arm") != arm:
            raise ValueError(f"slice record arm {row.get('arm')!r} is not {arm}")
        if float(row["threshold_nu"]) != threshold or float(row["below_0_02_threshold"]) != floor:
            raise ValueError("slice record cuts do not match the Stage F protocol")
        if row.get("ood_used_for_selection") is not False or row.get("rollout_used_for_selection") is not False:
            raise ValueError("slice record claims a selection use that the protocol forbids")
    counts = _test_counts(protocol)
    payload: dict[str, object] = {
        "format": SLICE_AGGREGATE_FORMAT,
        "arm": arm,
        "seeds": seeds,
        "n_seeds": len(seeds),
        "ddof": 1,
        "threshold_nu": threshold,
        "below_0_02_threshold": floor,
        "in_range_rule": IN_RANGE_RULE,
        "ood_rule": OOD_RULE,
        "primary_metric": "mean_relative_l2",
        "aggregation": (
            "arithmetic mean and sample standard deviation of one scalar per seed; "
            "windows are not pooled across seeds"
        ),
        "loss_mode": "data" if arm == "data_only" else "hybrid",
        "residual_weight": None if arm == "data_only" else HYBRID_WEIGHT,
        "selected_epoch": _summarize([float(row["selected_epoch"]) for row in ordered]),
        "val_relative_l2": _summarize([float(row["val_relative_l2"]) for row in ordered]),
        "slices": {},
        "rollout": {},
    }
    window_metrics = (
        "mean_relative_l2",
        "median_relative_l2",
        "pooled_relative_l2",
        "normalized_mse",
        "persistence_mean_relative_l2",
    )
    rollout_metrics = (
        "mean_instance_relative_l2",
        "median_instance_relative_l2",
        "persistence_mean_instance_relative_l2",
    )
    slice_block: dict[str, object] = {}
    rollout_block: dict[str, object] = {}
    for name in SLICE_NAMES:
        slice_block[name] = _aggregate_block(ordered, "slices", name, window_metrics, counts[name])
        rollout_block[name] = _aggregate_block(ordered, "rollout", name, rollout_metrics, counts[name])
        rollout_block[name]["per_step_mean_relative_l2"] = _aggregate_steps(ordered, name)
    payload["slices"] = slice_block
    payload["rollout"] = rollout_block
    one_step_gap = _ratio_gap(ordered, "slices", "mean_relative_l2")
    rollout_gap = _ratio_gap(ordered, "rollout", "mean_instance_relative_l2")
    payload["ood_gap"] = {"one_step": one_step_gap, "rollout": rollout_gap}
    payload["worst_ood_one_step"] = _worst_across_seeds(ordered, "slices", "ood")
    payload["worst_ood_rollout"] = _worst_across_seeds(ordered, "rollout", "ood")
    return payload


def aggregate_inverse_runs(
    runs: Sequence[Mapping[str, object]],
    protocol: Mapping[str, object],
) -> dict[str, object]:
    """Reduce lambda-0 inverse runs. Curves are re-minimized here."""

    seeds = [int(seed) for seed in protocol["seeds"]]  # type: ignore[union-attr]
    if len(runs) != len(ARM_NAMES) * len(seeds):
        raise ValueError(f"expected {len(ARM_NAMES) * len(seeds)} inverse runs, got {len(runs)}")
    grid = viscosity_grid(protocol)
    support = search_support(protocol)
    grouped: dict[str, dict[int, Mapping[str, object]]] = {arm: {} for arm in ARM_NAMES}
    for run in runs:
        if run.get("format") != INVERSE_RUN_FORMAT:
            raise ValueError(f"unsupported inverse run format {run.get('format')!r}")
        if run.get("split") != "test":
            raise ValueError("inverse aggregation requires the test split")
        if run.get("lambda") != 0.0 or run.get("lambda_reselected") is not False:
            raise ValueError("Stage F inverse runs must stay at lambda 0 without reselection")
        arm = str(run.get("arm"))
        seed = int(run["seed"])
        if arm not in grouped or seed in grouped[arm]:
            raise ValueError(f"unexpected or duplicate inverse run {arm} seed {seed}")
        if seed not in seeds:
            raise ValueError(f"inverse seed {seed} is not in the protocol")
        grouped[arm][seed] = run
    threshold = float(protocol["threshold_nu"])  # type: ignore[arg-type]
    floor = float(protocol["below_0_02_threshold"])  # type: ignore[arg-type]
    train_min = float(support["train_nu_min"])
    train_max = float(support["train_nu_max"])
    arms: dict[str, object] = {}
    prepared: dict[str, dict[int, list[dict[str, object]]]] = {}
    for arm in ARM_NAMES:
        if set(grouped[arm]) != set(seeds):
            raise ValueError(f"{arm} is missing a frozen seed")
        by_seed: dict[int, list[dict[str, object]]] = {}
        for seed in seeds:
            by_seed[seed] = _rows_from_run(grouped[arm][seed], grid, train_min, train_max)
        prepared[arm] = by_seed
        arms[arm] = _arm_inverse_summary(by_seed, seeds, threshold, floor)
    return {
        "lambda": 0.0,
        "lambda_reselected": False,
        "failure_rule": "nu_hat <= 0 or rel_error > 0.5",
        "nu_search": support,
        "arms": arms,
        "rows": prepared,
    }


def assemble_stage_f_scores(
    protocol_path: Path,
    data_forward: Mapping[str, object],
    hybrid_forward: Mapping[str, object],
    inverse_runs: Sequence[Mapping[str, object]],
) -> dict[str, object]:
    """Join the restricted-operator scores to the published full-range tables.

    The published files are hashed and copied. Their numbers are not
    recomputed and not edited.
    """

    protocol = load_stage_f_protocol(protocol_path)
    if data_forward.get("arm") != "data_only" or hybrid_forward.get("arm") != "hybrid_1e-2":
        raise ValueError("forward aggregates must be the data-only and hybrid 1e-2 arms")
    if data_forward.get("format") != SLICE_AGGREGATE_FORMAT or hybrid_forward.get("format") != SLICE_AGGREGATE_FORMAT:
        raise ValueError("forward aggregates have the wrong format")
    inverse = aggregate_inverse_runs(inverse_runs, protocol)
    published_forward = _published_forward(protocol)
    published_inverse = _published_inverse(protocol)
    rows = inverse.pop("rows")
    if not isinstance(rows, dict):
        raise ValueError("inverse rows were not prepared")
    paired = _paired_forward(data_forward, hybrid_forward)
    inverse_paired = _paired_inverse(rows, protocol)
    examples = _inverse_examples(rows, protocol)
    support = search_support(protocol)
    return {
        "format": SCORES_FORMAT,
        "protocol_path": Path(protocol_path).as_posix(),
        "protocol_sha256": hashlib.sha256(Path(protocol_path).read_bytes()).hexdigest(),
        "threshold_nu": protocol["threshold_nu"],
        "threshold_refit": False,
        "hybrid_weight": HYBRID_WEIGHT,
        "hybrid_weight_reselected": False,
        "lambda": 0.0,
        "lambda_reselected": False,
        "ood_used_for_selection": False,
        "test_used_for_selection": False,
        "normalization": protocol["normalization"],
        "train_subset": protocol["train_subset"],
        "validation_subset": protocol["validation_subset"],
        "nu_search": support,
        "forward": {
            "data_only": _forward_view(data_forward),
            "hybrid_1e-2": _forward_view(hybrid_forward),
            "hybrid_minus_data_only": paired,
            "full_range": published_forward,
        },
        "inverse": {
            "lambda": 0.0,
            "arms": inverse["arms"],
            "hybrid_minus_data_only": inverse_paired,
            "closed_form_sensors32": published_inverse["closed_form"],
            "dense_reference": published_inverse["dense"],
            "full_range_operators": published_inverse["operators"],
            "failure_examples": examples,
        },
    }


def _forward_view(aggregate: Mapping[str, object]) -> dict[str, object]:
    return {
        "seeds": aggregate["seeds"],
        "selected_epoch": aggregate["selected_epoch"],
        "val_relative_l2": aggregate["val_relative_l2"],
        "slices": aggregate["slices"],
        "rollout": aggregate["rollout"],
        "ood_gap": aggregate["ood_gap"],
        "worst_ood_one_step": aggregate["worst_ood_one_step"],
        "worst_ood_rollout": aggregate["worst_ood_rollout"],
    }


def _paired_forward(
    data_forward: Mapping[str, object],
    hybrid_forward: Mapping[str, object],
) -> dict[str, object]:
    paired: dict[str, object] = {}
    for name in ("ood", "in_range", "below_0_02", "full_test"):
        data_values = _metric_values(data_forward, "slices", name, "mean_relative_l2")
        hybrid_values = _metric_values(hybrid_forward, "slices", name, "mean_relative_l2")
        data_roll = _metric_values(data_forward, "rollout", name, "mean_instance_relative_l2")
        hybrid_roll = _metric_values(hybrid_forward, "rollout", name, "mean_instance_relative_l2")
        paired[name] = {
            "one_step_hybrid_minus_data": _difference(hybrid_values, data_values),
            "rollout_hybrid_minus_data": _difference(hybrid_roll, data_roll),
        }
    return paired


def _paired_inverse(
    rows: Mapping[str, Mapping[int, Sequence[Mapping[str, object]]]],
    protocol: Mapping[str, object],
) -> dict[str, object]:
    threshold = float(protocol["threshold_nu"])  # type: ignore[arg-type]
    floor = float(protocol["below_0_02_threshold"])  # type: ignore[arg-type]
    seeds = [int(seed) for seed in protocol["seeds"]]  # type: ignore[union-attr]
    paired: dict[str, object] = {}
    for name in SLICE_NAMES:
        hybrid_means = []
        data_means = []
        versus_ls = []
        for seed in seeds:
            data_slice = _slice_rows(rows["data_only"][seed], name, threshold, floor)
            hybrid_slice = _slice_rows(rows["hybrid_1e-2"][seed], name, threshold, floor)
            data_means.append(float(np.mean([float(row["rel_error"]) for row in data_slice])))
            hybrid_means.append(float(np.mean([float(row["rel_error"]) for row in hybrid_slice])))
            versus_ls.append(
                float(
                    np.mean(
                        [float(row["rel_error"]) - float(row["ls_rel_error"]) for row in hybrid_slice]
                    )
                )
            )
        data_vs_ls = []
        for seed in seeds:
            data_slice = _slice_rows(rows["data_only"][seed], name, threshold, floor)
            data_vs_ls.append(
                float(np.mean([float(row["rel_error"]) - float(row["ls_rel_error"]) for row in data_slice]))
            )
        paired[name] = {
            "hybrid_minus_data": _difference(hybrid_means, data_means),
            "data_only_minus_ls": _summarize(data_vs_ls),
            "hybrid_minus_ls": _summarize(versus_ls),
        }
    return paired


def _inverse_examples(
    rows: Mapping[str, Mapping[int, Sequence[Mapping[str, object]]]],
    protocol: Mapping[str, object],
) -> dict[str, object]:
    threshold = float(protocol["threshold_nu"])  # type: ignore[arg-type]
    floor = float(protocol["below_0_02_threshold"])  # type: ignore[arg-type]
    seeds = [int(seed) for seed in protocol["seeds"]]  # type: ignore[union-attr]
    examples: dict[str, object] = {}
    for arm in ARM_NAMES:
        pooled: dict[int, list[Mapping[str, object]]] = {}
        for seed in seeds:
            for row in _slice_rows(rows[arm][seed], "ood", threshold, floor):
                pooled.setdefault(int(row["instance_id"]), []).append(row)
        ranked = []
        for instance_id, items in pooled.items():
            if len(items) != len(seeds):
                raise ValueError(f"instance {instance_id} is missing a seed")
            ranked.append(
                {
                    "instance_id": instance_id,
                    "nu": float(items[0]["nu"]),
                    "mean_rel_error": float(np.mean([float(item["rel_error"]) for item in items])),
                    "nu_hat": [float(item["nu_hat"]) for item in items],
                    "ls_nu_hat": float(items[0]["ls_nu_hat"]),
                    "ls_rel_error": float(items[0]["ls_rel_error"]),
                    "n_outside_training_range": int(sum(bool(item["outside_training_range"]) for item in items)),
                    "n_failures": int(sum(bool(item["failure"]) for item in items)),
                }
            )
        ranked.sort(key=lambda row: (-float(row["mean_rel_error"]), int(row["instance_id"])))
        examples[arm] = ranked[:5]
    return examples


def _rows_from_run(
    run: Mapping[str, object],
    grid: np.ndarray,
    train_min: float,
    train_max: float,
) -> list[dict[str, object]]:
    raw = run.get("instances")
    if not isinstance(raw, list) or len(raw) < 1:
        raise ValueError("inverse run is missing instances")
    rows: list[dict[str, object]] = []
    for item in raw:
        if not isinstance(item, dict):
            raise ValueError("inverse instance must be an object")
        sensor = np.asarray(item["sensor_mse"], dtype=np.float64)
        oracle = np.asarray(item["oracle_sensor_mse"], dtype=np.float64)
        hat = nu_from_objective(sensor, grid)
        oracle_hat = nu_from_objective(oracle, grid)
        nu = float(item["nu"])
        if nu <= 0.0:
            raise ValueError("instance viscosity must be > 0")
        relative = abs(hat - nu) / nu
        outside = bool(hat < train_min or hat > train_max)
        rows.append(
            {
                "instance_id": int(item["instance_id"]),
                "nu": nu,
                "nu_hat": hat,
                "rel_error": relative,
                "abs_error": abs(hat - nu),
                "failure": bool(hat <= 0.0 or relative > FAILURE_RELATIVE_ERROR),
                "outside_training_range": outside,
                "oracle_nu_hat": oracle_hat,
                "oracle_rel_error": abs(oracle_hat - nu) / nu,
                "ls_nu_hat": float(item["ls_nu_hat"]),
                "ls_rel_error": float(item["ls_rel_error"]),
            }
        )
    rows.sort(key=lambda row: int(row["instance_id"]))
    return rows


def _arm_inverse_summary(
    by_seed: Mapping[int, Sequence[Mapping[str, object]]],
    seeds: Sequence[int],
    threshold: float,
    floor: float,
) -> dict[str, object]:
    slices: dict[str, object] = {}
    for name in SLICE_NAMES:
        means = []
        failures = []
        outside = []
        oracle_means = []
        for seed in seeds:
            chosen = _slice_rows(by_seed[seed], name, threshold, floor)
            means.append(float(np.mean([float(row["rel_error"]) for row in chosen])))
            failures.append(float(sum(bool(row["failure"]) for row in chosen)))
            outside.append(float(sum(bool(row["outside_training_range"]) for row in chosen)))
            oracle_means.append(float(np.mean([float(row["oracle_rel_error"]) for row in chosen])))
        slices[name] = {
            "n_instances": len(_slice_rows(by_seed[seeds[0]], name, threshold, floor)),
            "mean_rel_error": _summarize(means),
            "n_failures": _summarize(failures),
            "n_outside_training_range": _summarize(outside),
            "oracle_mean_rel_error": _summarize(oracle_means),
        }
    return {"slices": slices}


def _slice_rows(
    rows: Sequence[Mapping[str, object]],
    name: str,
    threshold: float,
    floor: float,
) -> list[Mapping[str, object]]:
    nu = np.asarray([float(row["nu"]) for row in rows], dtype=np.float64)
    mask = ood_masks(nu, threshold, floor)[name]
    return [row for row, keep in zip(rows, mask, strict=True) if bool(keep)]


def _published_forward(protocol: Mapping[str, object]) -> dict[str, object]:
    references = _references(protocol)
    data_path = Path(str(references["stage_b_scores"]))
    hybrid_path = Path(str(references["stage_c_scores"]))
    _require_sha256(data_path, str(references["stage_b_scores_sha256"]))
    _require_sha256(hybrid_path, str(references["stage_c_scores_sha256"]))
    data = _read_object(data_path)
    stage_c = _read_object(hybrid_path)
    hybrid = stage_c.get("hybrid")
    copied = stage_c.get("data_only")
    if not isinstance(hybrid, dict) or not isinstance(copied, dict):
        raise ValueError("Stage C scores are missing the hybrid or data-only block")
    _require_same_published_mean(data, copied, "hard_ood")
    _require_same_published_mean(data, copied, "complement")
    return {
        "note": (
            "Full-range operators from Stages B and C. hard_ood was inside their "
            "training support. These numbers are copied. They were not recomputed."
        ),
        "ood_name_in_source": "hard_ood",
        "in_range_name_in_source": "complement",
        "below_0_02_published": False,
        "data_only": _copy_forward_arm(data, str(references["stage_b_scores"]), str(references["stage_b_scores_sha256"])),
        "hybrid_1e-2": _copy_forward_arm(
            hybrid,
            str(references["stage_c_scores"]),
            str(references["stage_c_scores_sha256"]),
        ),
    }


def _copy_forward_arm(payload: Mapping[str, object], source: str, digest: str) -> dict[str, object]:
    copied: dict[str, object] = {"source": source, "sha256": digest, "slices": {}}
    slices: dict[str, object] = {}
    for name, published in _FORWARD_SLICE_IN_PUBLISHED.items():
        block = _published_metric(payload, "slices", published, "mean_relative_l2")
        rollout = _published_metric(payload, "rollout", published, "mean_instance_relative_l2")
        count = _slice_container(payload, "slices", published)
        slices[name] = {
            "source_slice": published,
            "n_instances": int(count["n_instances"]),
            "one_step_mean_relative_l2": {"mean": float(block["mean"]), "std": float(block["std"])},
            "rollout_mean_instance_relative_l2": {"mean": float(rollout["mean"]), "std": float(rollout["std"])},
        }
    copied["slices"] = slices
    return copied


def _published_inverse(protocol: Mapping[str, object]) -> dict[str, object]:
    references = _references(protocol)
    stress_path = Path(str(references["inverse_stress"]))
    stage_d_path = Path(str(references["stage_d_scores"]))
    _require_sha256(stress_path, str(references["inverse_stress_sha256"]))
    _require_sha256(stage_d_path, str(references["stage_d_scores_sha256"]))
    stress = _read_object(stress_path)
    stage_d = _read_object(stage_d_path)
    closed: dict[str, object] = {"source": str(references["inverse_stress"]), "slices": {}}
    dense: dict[str, object] = {"source": str(references["inverse_stress"]), "slices": {}}
    closed_slices: dict[str, object] = {}
    dense_slices: dict[str, object] = {}
    for name, key in _LS_KEYS.items():
        block = stress[key]
        if not isinstance(block, dict):
            raise ValueError(f"inverse stress is missing {key}")
        closed_slices[name] = {
            "source_key": key,
            "n_instances": int(block["n_instances"]),
            "mean_rel_error": float(block["mean_rel_error"]),
            "n_failures": int(block["n_failures"]),
        }
    for name, key in _DENSE_KEYS.items():
        block = stress[key]
        if not isinstance(block, dict):
            raise ValueError(f"inverse stress is missing {key}")
        dense_slices[name] = {
            "source_key": key,
            "n_instances": int(block["n_instances"]),
            "mean_rel_error": float(block["mean_rel_error"]),
            "n_failures": int(block["n_failures"]),
        }
    closed["slices"] = closed_slices
    dense["slices"] = dense_slices
    operators: dict[str, object] = {
        "source": str(references["stage_d_scores"]),
        "sha256": str(references["stage_d_scores_sha256"]),
        "note": "Full-range Stage D operators at lambda 0. Copied, not rerun.",
        "arms": {},
    }
    arms = stage_d.get("arms")
    if not isinstance(arms, dict):
        raise ValueError("Stage D scores are missing arms")
    copied_arms: dict[str, object] = {}
    for arm in ARM_NAMES:
        block = arms.get(arm)
        if not isinstance(block, dict):
            raise ValueError(f"Stage D scores are missing {arm}")
        objective = block.get("objectives")
        if not isinstance(objective, dict) or "0.0" not in objective:
            raise ValueError(f"Stage D {arm} is missing lambda 0")
        lam = objective["0.0"]
        if not isinstance(lam, dict):
            raise ValueError(f"Stage D {arm} lambda 0 is not an object")
        published_slices = lam.get("slices")
        if not isinstance(published_slices, dict):
            raise ValueError(f"Stage D {arm} lambda 0 is missing slices")
        view: dict[str, object] = {}
        for name, published in _INVERSE_SLICE_IN_PUBLISHED.items():
            section = published_slices.get(published)
            if not isinstance(section, dict):
                raise ValueError(f"Stage D {arm} is missing {published}")
            metric = section.get("mean_rel_error")
            failures = section.get("n_failures")
            if not isinstance(metric, dict) or not isinstance(failures, dict):
                raise ValueError(f"Stage D {arm} {published} is missing the mean")
            view[name] = {
                "source_slice": published,
                "n_instances": int(section["n_instances"]),
                "mean_rel_error": {"mean": float(metric["mean"]), "std": float(metric["std"])},
                "n_failures": {"mean": float(failures["mean"]), "std": float(failures["std"])},
            }
        copied_arms[arm] = view
    operators["arms"] = copied_arms
    return {"closed_form": closed, "dense": dense, "operators": operators}


def _require_same_published_mean(
    stage_b: Mapping[str, object],
    stage_c_data: Mapping[str, object],
    slice_name: str,
) -> None:
    left = _published_metric(stage_b, "slices", slice_name, "mean_relative_l2")
    right = _published_metric(stage_c_data, "slices", slice_name, "mean_relative_l2")
    if abs(float(left["mean"]) - float(right["mean"])) > 0.0 or abs(float(left["std"]) - float(right["std"])) > 0.0:
        raise ValueError(f"Stage C data-only {slice_name} does not match the Stage B file")


def _published_metric(
    payload: Mapping[str, object],
    block: str,
    slice_name: str,
    metric: str,
) -> Mapping[str, object]:
    section = _slice_container(payload, block, slice_name)
    values = section.get(metric)
    if not isinstance(values, dict) or "mean" not in values or "std" not in values:
        raise ValueError(f"published {block}.{slice_name}.{metric} is missing mean and std")
    return values


def _slice_container(payload: Mapping[str, object], block: str, slice_name: str) -> Mapping[str, object]:
    container = payload.get(block)
    if not isinstance(container, dict) or slice_name not in container:
        raise ValueError(f"published record is missing {block}.{slice_name}")
    section = container[slice_name]
    if not isinstance(section, dict):
        raise ValueError(f"published {block}.{slice_name} must be an object")
    return section


def _aggregate_block(
    records: Sequence[Mapping[str, object]],
    block: str,
    name: str,
    metrics: Sequence[str],
    expected_count: int,
) -> dict[str, object]:
    first = _record_section(records[0], block, name)
    if int(first["n_instances"]) != int(expected_count):
        raise ValueError(f"{block}.{name} has {first['n_instances']} instances, protocol locked {expected_count}")
    summary: dict[str, object] = {
        "n_instances": int(first["n_instances"]),
    }
    if "n_windows" in first:
        summary["n_windows"] = int(first["n_windows"])
    for row in records[1:]:
        other = _record_section(row, block, name)
        if int(other["n_instances"]) != summary["n_instances"]:
            raise ValueError(f"{name} instance counts differ across seeds")
    for metric in metrics:
        summary[metric] = _summarize([float(_record_section(row, block, name)[metric]) for row in records])
    return summary


def _aggregate_steps(records: Sequence[Mapping[str, object]], name: str) -> dict[str, object]:
    steps = [
        np.asarray(_record_section(row, "rollout", name)["per_step_mean_relative_l2"], dtype=np.float64)
        for row in records
    ]
    if any(step.shape != steps[0].shape or step.ndim != 1 for step in steps):
        raise ValueError(f"{name} rollout step counts differ across seeds")
    stacked = np.stack(steps, axis=0)
    if not np.isfinite(stacked).all():
        raise ValueError(f"{name} rollout step scores must be finite")
    return {
        "n": int(stacked.shape[0]),
        "ddof": 1,
        "mean": [float(value) for value in np.mean(stacked, axis=0)],
        "std": [float(value) for value in np.std(stacked, axis=0, ddof=1)],
    }


def _ratio_gap(records: Sequence[Mapping[str, object]], block: str, metric: str) -> dict[str, object]:
    ood = [float(_record_section(row, block, "ood")[metric]) for row in records]
    in_range = [float(_record_section(row, block, "in_range")[metric]) for row in records]
    if any(value <= 0.0 for value in in_range):
        raise ValueError("in-range score must be > 0 to form the OOD ratio")
    ratios = [left / right for left, right in zip(ood, in_range, strict=True)]
    return {
        "definition": "ood divided by in_range, one ratio per seed, plus the ratio of the two means",
        "per_seed": _summarize(ratios),
        "ratio_of_means": float(np.mean(ood) / np.mean(in_range)),
    }


def _worst_across_seeds(
    records: Sequence[Mapping[str, object]],
    block: str,
    name: str,
) -> list[dict[str, object]]:
    metric = "mean_relative_l2" if block == "slices" else "relative_l2"
    pooled: dict[int, list[Mapping[str, object]]] = {}
    nu_of: dict[int, float] = {}
    for row in records:
        section = _record_section(row, block, name)
        instances = section.get("instances")
        if not isinstance(instances, list):
            raise ValueError(f"{block}.{name} is missing per-instance rows")
        for item in instances:
            if not isinstance(item, dict):
                raise ValueError("per-instance row must be an object")
            instance_id = int(item["instance_id"])
            pooled.setdefault(instance_id, []).append(item)
            nu_of[instance_id] = float(item["nu"])
    ranked = []
    n_seeds = len(records)
    for instance_id, items in pooled.items():
        if len(items) != n_seeds:
            raise ValueError(f"instance {instance_id} is missing a seed in {block}.{name}")
        values = [float(item[metric]) for item in items]
        ranked.append(
            {
                "instance_id": instance_id,
                "nu": nu_of[instance_id],
                "mean": float(np.mean(values)),
                "std": float(np.std(values, ddof=1)),
                "values": values,
            }
        )
    ranked.sort(key=lambda row: (-float(row["mean"]), int(row["instance_id"])))
    return ranked[:5]


def _per_instance(
    instance_ids: np.ndarray,
    nu: np.ndarray,
    scores: np.ndarray,
    persistence: np.ndarray,
) -> list[dict[str, object]]:
    grouped: dict[int, list[int]] = {}
    for index, instance_id in enumerate(instance_ids):
        grouped.setdefault(int(instance_id), []).append(index)
    rows = []
    for instance_id, indexes in sorted(grouped.items()):
        chosen = np.asarray(indexes, dtype=np.int64)
        rows.append(
            {
                "instance_id": instance_id,
                "nu": float(nu[chosen[0]]),
                "mean_relative_l2": float(np.mean(scores[chosen])),
                "persistence_mean_relative_l2": float(np.mean(persistence[chosen])),
            }
        )
    return rows


def _difference(left: Sequence[float], right: Sequence[float]) -> dict[str, object]:
    if len(left) != len(right):
        raise ValueError("paired scores must share the seed count")
    return _summarize([float(a) - float(b) for a, b in zip(left, right, strict=True)])


def _metric_values(payload: Mapping[str, object], block: str, name: str, metric: str) -> list[float]:
    section = _record_section(payload, block, name)
    values = section.get(metric)
    if not isinstance(values, dict) or not isinstance(values.get("values"), list):
        raise ValueError(f"{block}.{name}.{metric} is missing per-seed values")
    return [float(value) for value in values["values"]]


def _record_section(record: Mapping[str, object], block: str, name: str) -> Mapping[str, object]:
    container = record.get(block)
    if not isinstance(container, dict) or name not in container:
        raise ValueError(f"record is missing {block}.{name}")
    section = container[name]
    if not isinstance(section, dict):
        raise ValueError(f"record {block}.{name} must be an object")
    return section


def _summarize(values: Sequence[float]) -> dict[str, object]:
    array = np.asarray(list(values), dtype=np.float64)
    if array.ndim != 1 or array.size < 2 or not np.isfinite(array).all():
        raise ValueError("aggregation needs at least two finite scalars")
    return {
        "n": int(array.size),
        "ddof": 1,
        "mean": float(np.mean(array)),
        "std": float(np.std(array, ddof=1)),
        "min": float(np.min(array)),
        "max": float(np.max(array)),
        "values": [float(value) for value in array],
    }


def _require_manifest_locks(payload: Mapping[str, object]) -> None:
    manifest_path = Path(str(payload["manifest"]))
    manifest = load_pilot_manifest(manifest_path)
    threshold = float(payload["threshold_nu"])  # type: ignore[arg-type]
    data = load_data_protocol(Path(str(payload["data_protocol"])))
    if threshold != hard_ood_threshold(data):
        raise ValueError("Stage F threshold does not match the Stage A data protocol")
    if float(payload["below_0_02_threshold"]) != float(data["stage2_nu_min"]):  # type: ignore[arg-type]
        raise ValueError("below_0_02 threshold must stay the Stage 2 viscosity floor")
    nu = _nu_by_id(manifest)
    splits = split_sets(manifest)
    for split, block_name, excluded_name in (
        ("train", "train_subset", "n_excluded"),
        ("val", "validation_subset", "n_excluded"),
    ):
        block = payload.get(block_name)
        if not isinstance(block, dict):
            raise ValueError(f"protocol is missing {block_name}")
        ids = in_range_ids(manifest, split, threshold)
        if ids != in_range_ids(manifest, split, threshold):
            raise ValueError("in-range ids are not stable")
        if int(block["n_instances"]) != len(ids):
            raise ValueError(f"{block_name} instance count does not match the manifest")
        excluded = len(splits[split]) - len(ids)
        if int(block[excluded_name]) != excluded:
            raise ValueError(f"{block_name} excluded count does not match the manifest")
        values = np.asarray([nu[instance_id] for instance_id in ids], dtype=np.float64)
        if abs(float(np.min(values)) - float(block["nu_min"])) > 0.0:
            raise ValueError(f"{block_name} nu_min does not match the manifest")
        if abs(float(np.max(values)) - float(block["nu_max"])) > 0.0:
            raise ValueError(f"{block_name} nu_max does not match the manifest")
        if split == "train":
            mean = float(np.mean(values))
            std = float(np.std(values, ddof=0))
            if abs(mean - float(block["nu_mean"])) > _NORM_ABS:
                raise ValueError("train nu_mean does not match the manifest")
            if abs(std - float(block["nu_std"])) > _NORM_ABS:
                raise ValueError("train nu_std does not match the manifest")
            norm = payload.get("normalization")
            if not isinstance(norm, dict):
                raise ValueError("protocol is missing normalization")
            if abs(mean - float(norm["nu_mean"])) > _NORM_ABS or abs(std - float(norm["nu_std"])) > _NORM_ABS:
                raise ValueError("normalization nu statistics must match the restricted training viscosities")
    counts = _test_counts(payload)
    test_nu = np.asarray([nu[instance_id] for instance_id in sorted(splits["test"])], dtype=np.float64)
    masks = ood_masks(test_nu, threshold, float(payload["below_0_02_threshold"]))  # type: ignore[arg-type]
    for name in SLICE_NAMES:
        if int(counts[name]) != int(np.count_nonzero(masks[name])):
            raise ValueError(f"test {name} count does not match the manifest")
    spec = WindowSpec(
        input_frames=int(payload["input_frames"]),  # type: ignore[arg-type]
        output_frames=int(payload["output_frames"]),  # type: ignore[arg-type]
        stride=int(payload["stride"]),  # type: ignore[arg-type]
    )
    n_times = {int(item["n_times"]) for item in _instance_list(manifest)}
    if n_times != {101}:
        raise ValueError("harder pilot trajectories must have 101 saved frames")
    n_windows = len(window_starts(101, spec))
    train = _train_subset(payload)
    if int(train["n_windows"]) != int(train["n_instances"]) * n_windows:
        raise ValueError("train window count does not match the window spec")
    validation = payload["validation_subset"]
    if not isinstance(validation, dict) or int(validation["n_windows"]) != int(validation["n_instances"]) * n_windows:
        raise ValueError("validation window count does not match the window spec")


def _require_reference_hashes(payload: Mapping[str, object]) -> None:
    references = _references(payload)
    pairs = (
        ("data_protocol", "data_protocol_sha256"),
        ("stage_b_scores", "stage_b_scores_sha256"),
        ("stage_c_scores", "stage_c_scores_sha256"),
        ("stage_c_weight_selection", "stage_c_weight_selection_sha256"),
        ("stage_d_inverse_protocol", "stage_d_inverse_protocol_sha256"),
        ("stage_d_scores", "stage_d_scores_sha256"),
        ("inverse_stress", "inverse_stress_sha256"),
    )
    for path_key, digest_key in pairs:
        _require_sha256(Path(str(references[path_key])), str(references[digest_key]))
    selection = _read_object(Path(str(references["stage_c_weight_selection"])))
    if abs(float(selection["selected_residual_weight"]) - HYBRID_WEIGHT) > 0.0:
        raise ValueError("Stage C weight selection is not 1e-2; Stage F will not choose another weight")
    data_path = payload.get("data_protocol")
    digest = payload.get("data_protocol_sha256")
    if data_path != references["data_protocol"] or digest != references["data_protocol_sha256"]:
        raise ValueError("data protocol path and hash must match the reference block")


def _require_inverse_block(payload: Mapping[str, object]) -> None:
    search = payload.get("nu_search")
    if not isinstance(search, dict):
        raise ValueError("protocol is missing nu_search")
    if search.get("clipped_to_training_support") is not False:
        raise ValueError("the viscosity search must not be clipped to the training support")
    if float(search.get("min")) != 0.005 or float(search.get("max")) != 0.1 or int(search.get("n_grid")) != 191:
        raise ValueError("the viscosity search must stay the Stage D grid")
    if payload.get("inverse_lambda") != 0.0:
        raise ValueError("Stage F inverse lambda must stay 0")
    if payload.get("mask_name") != "sensors32_bursts" or payload.get("mask_retuned") is not False:
        raise ValueError("the sensor mask must stay sensors32_bursts")
    support = search_support(payload)
    if support["can_return_values_outside_training_range"] is not True:
        raise ValueError("the locked grid must be able to return a viscosity outside the training range")


def _require_normalization_block(payload: Mapping[str, object]) -> None:
    block = payload.get("normalization")
    if not isinstance(block, dict):
        raise ValueError("protocol is missing normalization")
    if block.get("fit_on") != "restricted_train":
        raise ValueError("normalization must be fit on the restricted training instances")
    if block.get("ddof") != 0:
        raise ValueError("normalization ddof must be 0")
    if block.get("includes_ood") is not False or block.get("includes_validation_or_test") is not False:
        raise ValueError("normalization must exclude OOD, validation, and test")
    for label in ("u_mean", "u_std", "nu_mean", "nu_std"):
        value = block.get(label)
        if isinstance(value, bool) or not isinstance(value, (int, float)) or not math.isfinite(float(value)):
            raise ValueError(f"normalization {label} must be finite")
    if float(block["u_std"]) <= 0.0 or float(block["nu_std"]) <= 0.0:
        raise ValueError("normalization standard deviations must be > 0")


def _require_subset_rules(payload: Mapping[str, object]) -> None:
    hard = payload.get("hard_ood")
    if not isinstance(hard, dict) or hard.get("refit") is not False:
        raise ValueError("hard_ood.refit must be false")
    if hard.get("comparison") != OOD_RULE:
        raise ValueError(f"hard_ood comparison must stay {OOD_RULE!r}")
    threshold = hard.get("threshold_nu")
    if threshold != payload.get("threshold_nu"):
        raise ValueError("threshold_nu must match hard_ood.threshold_nu")
    for name in ("train_subset", "validation_subset"):
        block = payload.get(name)
        if not isinstance(block, dict) or block.get("rule") != IN_RANGE_RULE:
            raise ValueError(f"{name} rule must be {IN_RANGE_RULE!r}")
    if payload.get("slices") != list(SLICE_NAMES):
        raise ValueError(f"slices must be {list(SLICE_NAMES)}")


def _require_architecture(payload: Mapping[str, object]) -> None:
    expected = {
        "epochs": 30,
        "batch_size": 32,
        "width": 32,
        "modes": 16,
        "layers": 4,
        "input_frames": 8,
        "output_frames": 8,
        "stride": 8,
    }
    for label, value in expected.items():
        if payload.get(label) != value:
            raise ValueError(f"Stage F {label} must stay {value}")
    if abs(float(payload.get("lr", 0.0)) - 0.001) > 0.0:  # type: ignore[arg-type]
        raise ValueError("Stage F lr must stay 0.001")
    if payload.get("optimizer") != "Adam" or payload.get("schedule") != "constant" or payload.get("device") != "cpu":
        raise ValueError("Stage F optimizer, schedule, and device must stay Adam, constant, cpu")
    if payload.get("residual_scope") != "with_input" or payload.get("residual_space") != "physical":
        raise ValueError("Stage F residual stencil must stay with_input and physical")
    if payload.get("dt") != 0.01:
        raise ValueError("Stage F residual dt must stay 0.01")
    if payload.get("loss_weight_grid_searched") is not False:
        raise ValueError("Stage F must not search the hybrid weight grid")


def _require_seeds(seeds: object) -> None:
    if not isinstance(seeds, list) or seeds != [0, 1, 2, 3, 4]:
        raise ValueError("Stage F seeds must be [0, 1, 2, 3, 4]")


def _train_subset(protocol: Mapping[str, object]) -> Mapping[str, object]:
    block = protocol.get("train_subset")
    if not isinstance(block, dict):
        raise ValueError("protocol is missing train_subset")
    return block


def _test_counts(protocol: Mapping[str, object]) -> Mapping[str, int]:
    block = protocol.get("test_counts")
    if not isinstance(block, dict):
        raise ValueError("protocol is missing test_counts")
    return {name: int(block[name]) for name in SLICE_NAMES}


def _references(protocol: Mapping[str, object]) -> Mapping[str, object]:
    block = protocol.get("published_references")
    if not isinstance(block, dict):
        raise ValueError("protocol is missing published_references")
    return block


def _nu_by_id(manifest: Mapping[str, object]) -> dict[int, float]:
    values: dict[int, float] = {}
    for item in _instance_list(manifest):
        instance_id = int(item["instance_id"])
        nu = float(item["nu"])
        if instance_id in values:
            raise ValueError(f"duplicate instance record {instance_id}")
        values[instance_id] = nu
    return values


def _instance_list(manifest: Mapping[str, object]) -> list[Mapping[str, object]]:
    raw = manifest.get("instances")
    if not isinstance(raw, list) or len(raw) < 1:
        raise ValueError("pilot manifest is missing instances")
    rows = []
    for item in raw:
        if not isinstance(item, dict):
            raise ValueError("pilot instance records must be objects")
        rows.append(item)
    return rows


def _require_sha256(path: Path, digest: str) -> None:
    if len(digest) != 64:
        raise ValueError(f"sha256 for {path} must be 64 hex characters")
    actual = hashlib.sha256(path.read_bytes()).hexdigest()
    if actual != digest:
        raise ValueError(f"{path} bytes do not match the locked sha256")


def _same_path(left: Path, right: Path) -> bool:
    if left == right:
        return True
    try:
        return left.resolve() == right.resolve()
    except OSError:
        return False


def _read_object(path: Path) -> dict[str, object]:
    source = Path(path)
    if not source.is_file():
        raise ValueError(f"JSON file does not exist: {source}")
    payload = json.loads(source.read_text(encoding="utf-8"))
    if not isinstance(payload, dict):
        raise ValueError(f"{source} must be a JSON object")
    return payload


__all__ = [
    "ARM_NAMES",
    "HYBRID_WEIGHT",
    "INVERSE_RUN_FORMAT",
    "SCORES_FORMAT",
    "SLICE_AGGREGATE_FORMAT",
    "SLICE_EVAL_FORMAT",
    "SLICE_NAMES",
    "STAGE_F_PROTOCOL_FORMAT",
    "aggregate_inverse_runs",
    "aggregate_ood_records",
    "assert_norm_matches_protocol",
    "assert_stage_f_train_call",
    "assemble_stage_f_scores",
    "in_range_ids",
    "load_stage_f_protocol",
    "ood_masks",
    "population_field_norm",
    "score_ood_windows",
    "search_support",
    "summarize_ood_rollout",
    "viscosity_grid",
]
