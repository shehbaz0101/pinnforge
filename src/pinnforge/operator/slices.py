"""Slice scores for the harder Burgers pilot.

``hard_ood`` is the rule frozen in ``docs/v02/pilot_protocol.json``:
an instance is in the slice when its viscosity is at most
``threshold_nu``. This module reads that number. It does not refit the
training quartile, and it does not read a test table to choose the cut.

Window metrics are the Stage 3 reductions: mean, median, and pooled
relative L2 on denormalized ``u``, plus normalized MSE and the
persistence baseline on the same windows. Aggregation across seeds is
the arithmetic mean and the sample standard deviation of those
per-seed scalars. Windows are not pooled across seeds before the
reduction.
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
from pinnforge.operator.windows import WindowDataset

DATA_PROTOCOL_FORMAT = "pinnforge.burgers_hard_pilot_protocol.v1"
TRAIN_PROTOCOL_FORMAT = "pinnforge.stage_b_train_protocol.v1"
SLICE_EVAL_FORMAT = "pinnforge.fno_slice_eval.v1"
SLICE_AGGREGATE_FORMAT = "pinnforge.fno_slice_aggregate.v1"
SLICE_NAMES = ("full_test", "hard_ood", "complement")
HARD_OOD_COMPARISON = "nu <= threshold_nu"
_WORST_COUNT = 5
_FORBIDDEN_TRAIN_KEYS = frozenset(
    {
        "test_mean_relative_l2",
        "results",
        "scores",
        "metrics",
        "selected_epoch",
        "val_relative_l2",
    }
)
WINDOW_SCALAR_METRICS = (
    "mean_relative_l2",
    "median_relative_l2",
    "pooled_relative_l2",
    "normalized_mse",
    "persistence_mean_relative_l2",
    "persistence_median_relative_l2",
    "persistence_pooled_relative_l2",
)
ROLLOUT_SCALAR_METRICS = (
    "mean_instance_relative_l2",
    "median_instance_relative_l2",
    "persistence_mean_instance_relative_l2",
)


def load_data_protocol(path: Path) -> dict[str, object]:
    """Read the Stage A pilot protocol and keep its ``hard_ood`` rule."""

    payload = _read_object(path)
    if payload.get("format") != DATA_PROTOCOL_FORMAT:
        raise ValueError(f"unsupported data protocol format {payload.get('format')!r}")
    hard = payload.get("hard_ood")
    if not isinstance(hard, dict):
        raise ValueError("data protocol is missing hard_ood")
    if hard.get("comparison") != HARD_OOD_COMPARISON:
        raise ValueError(f"hard_ood comparison must stay {HARD_OOD_COMPARISON!r}")
    if hard.get("fit_on") != "train":
        raise ValueError("hard_ood threshold must stay the training-split fit")
    _require_threshold(hard.get("threshold_nu"))
    return payload


def hard_ood_threshold(protocol: Mapping[str, object]) -> float:
    """Return the frozen threshold. The quartile is not recomputed."""

    hard = protocol.get("hard_ood")
    if not isinstance(hard, dict):
        raise ValueError("protocol is missing hard_ood")
    return _require_threshold(hard.get("threshold_nu"))


def load_train_protocol(path: Path) -> dict[str, object]:
    """Read the Stage B training protocol and reject a file that already holds scores."""

    source = Path(path)
    payload = _read_object(source)
    if payload.get("format") != TRAIN_PROTOCOL_FORMAT:
        raise ValueError(f"unsupported training protocol format {payload.get('format')!r}")
    leaked = _FORBIDDEN_TRAIN_KEYS & set(payload)
    if leaked:
        names = ", ".join(sorted(leaked))
        raise ValueError(f"training protocol must not contain test scores ({names})")
    if payload.get("frozen_before_test_evaluation") is not True:
        raise ValueError("training protocol must be frozen before test evaluation")
    if payload.get("loss") != "data":
        raise ValueError("training protocol loss must be data")
    if payload.get("early_stopping") is not False:
        raise ValueError("training protocol must set early_stopping to false")
    if payload.get("hard_ood_refit") is not False:
        raise ValueError("training protocol must not refit hard_ood")
    seeds = payload.get("seeds")
    if not isinstance(seeds, list) or len(seeds) < 3:
        raise ValueError("training protocol needs at least 3 seeds")
    if any(isinstance(seed, bool) or not isinstance(seed, int) or seed < 0 for seed in seeds):
        raise ValueError("training protocol seeds must be non-negative integers")
    if len(set(seeds)) != len(seeds):
        raise ValueError("training protocol seeds must be unique")
    for label in (
        "epochs",
        "batch_size",
        "width",
        "modes",
        "layers",
        "input_frames",
        "output_frames",
        "stride",
    ):
        value = payload.get(label)
        if isinstance(value, bool) or not isinstance(value, int) or value < 1:
            raise ValueError(f"training protocol {label} must be an integer >= 1")
    lr = payload.get("lr")
    if isinstance(lr, bool) or not isinstance(lr, (int, float)) or not math.isfinite(float(lr)) or float(lr) <= 0.0:
        raise ValueError("training protocol lr must be a finite number > 0")
    if payload.get("residual_weight") is not None:
        raise ValueError("training protocol residual_weight must be null for a data-only loss")
    data_path = payload.get("data_protocol")
    digest = payload.get("data_protocol_sha256")
    if not isinstance(data_path, str) or data_path == "":
        raise ValueError("training protocol data_protocol must be a path")
    if not isinstance(digest, str) or len(digest) != 64:
        raise ValueError("training protocol data_protocol_sha256 must be 64 hex characters")
    data_file = Path(data_path)
    if not data_file.is_file():
        raise ValueError(f"data protocol does not exist: {data_file}")
    actual = hashlib.sha256(data_file.read_bytes()).hexdigest()
    if actual != digest:
        raise ValueError("data protocol bytes do not match data_protocol_sha256")
    data_protocol = load_data_protocol(data_file)
    hard = payload.get("hard_ood")
    if not isinstance(hard, dict) or hard.get("refit") is not False:
        raise ValueError("training protocol hard_ood.refit must be false")
    if hard.get("comparison") != HARD_OOD_COMPARISON:
        raise ValueError(f"training protocol hard_ood comparison must stay {HARD_OOD_COMPARISON!r}")
    if hard_ood_threshold(payload) != hard_ood_threshold(data_protocol):
        raise ValueError("training protocol threshold does not match the Stage A data protocol")
    counts = payload.get("instance_counts_from_manifest")
    if not isinstance(counts, dict):
        raise ValueError("training protocol is missing instance_counts_from_manifest")
    return payload


def assert_train_call_matches_protocol(
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
) -> None:
    """Raise when a train invocation leaves the frozen hyperparameters."""

    if loss != "data":
        raise ValueError("frozen training protocol requires --loss data")
    if residual_weight is not None:
        raise ValueError("frozen training protocol rejects --residual-weight")
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
            raise ValueError(
                f"{label} {got} does not match the frozen training protocol ({protocol[label]})"
            )
    if float(lr) != float(protocol["lr"]):  # type: ignore[arg-type]
        raise ValueError(f"lr {lr} does not match the frozen training protocol ({protocol['lr']})")


def assert_checkpoint_matches_train_protocol(
    summary: Mapping[str, object],
    protocol: Mapping[str, object],
) -> None:
    """Raise when a checkpoint was not trained under the frozen protocol."""

    if summary.get("loss_mode") != "data":
        raise ValueError("slice scoring requires a data-only checkpoint")
    seeds = protocol.get("seeds")
    seed = summary.get("seed")
    if not isinstance(seeds, list) or seed not in seeds:
        raise ValueError(f"checkpoint seed {seed} is not in the frozen training protocol")
    for label in ("width", "modes", "layers", "input_frames", "output_frames", "stride"):
        if int(summary[label]) != int(protocol[label]):  # type: ignore[arg-type]
            raise ValueError(f"checkpoint {label} does not match the frozen training protocol")


def assert_manifest_slice_counts(
    slice_scores: Mapping[str, Mapping[str, object]],
    protocol: Mapping[str, object],
    split: str,
) -> None:
    """Raise when instance counts leave the counts locked from the manifest."""

    block = protocol.get("instance_counts_from_manifest")
    if not isinstance(block, dict):
        raise ValueError("training protocol is missing instance_counts_from_manifest")
    expected = block.get(split)
    if not isinstance(expected, dict):
        raise ValueError(f"training protocol has no instance counts for split {split!r}")
    for name, count in expected.items():
        if name not in slice_scores:
            raise ValueError(f"slice scores are missing {name}")
        got = int(slice_scores[name]["n_instances"])
        if got != int(count):
            raise ValueError(f"{split} {name} has {got} instances, protocol locked {int(count)}")


def physical_nu_by_instance(manifest: Mapping[str, object]) -> dict[int, float]:
    """Physical viscosity of every instance record."""

    raw = manifest.get("instances")
    if not isinstance(raw, list) or len(raw) < 1:
        raise ValueError("pilot manifest is missing instances")
    values: dict[int, float] = {}
    for item in raw:
        if not isinstance(item, dict):
            raise ValueError("pilot instance records must be objects")
        instance_id = item.get("instance_id")
        nu = item.get("nu")
        if isinstance(instance_id, bool) or not isinstance(instance_id, int):
            raise ValueError("instance_id must be an integer")
        if isinstance(nu, bool) or not isinstance(nu, (int, float)) or not math.isfinite(float(nu)):
            raise ValueError(f"instance {instance_id} nu must be finite")
        if instance_id in values:
            raise ValueError(f"duplicate instance record {instance_id}")
        values[int(instance_id)] = float(nu)
    return values


def score_prediction_slices(
    dataset: WindowDataset,
    prediction_normalized: np.ndarray,
    nu_by_id: Mapping[int, float],
    threshold: float,
    *,
    worst_count: int = _WORST_COUNT,
) -> dict[str, dict[str, object]]:
    """Score full test, ``hard_ood``, and the complement from one prediction tensor.

    ``prediction_normalized`` matches ``dataset.targets``. The viscosity
    used for the cut is the physical value in ``nu_by_id``, not the
    normalized channel stored on the dataset. ``threshold`` is the frozen
    protocol number.
    """

    if dataset.split not in {"train", "val", "test"}:
        raise ValueError("dataset split must be train, val, or test")
    _require_threshold(threshold)
    prediction = np.asarray(prediction_normalized, dtype=np.float64)
    if prediction.shape != dataset.targets.shape:
        raise ValueError(
            f"prediction shape {prediction.shape} does not match targets {dataset.targets.shape}"
        )
    if not np.isfinite(prediction).all():
        raise ValueError("prediction must be finite")
    predicted = dataset.norm.denormalize_u(prediction)
    reference = physical_targets(dataset)
    persistence = persistence_prediction(dataset)
    window_l2 = per_window_relative_l2(predicted, reference)
    persist_l2 = per_window_relative_l2(persistence, reference)
    nu = np.asarray(
        [_lookup_nu(nu_by_id, int(instance_id)) for instance_id in dataset.instance_ids],
        dtype=np.float64,
    )
    masks = {
        "full_test": np.ones(nu.shape[0], dtype=bool),
        "hard_ood": nu <= threshold,
        "complement": nu > threshold,
    }
    if np.any(masks["hard_ood"] & masks["complement"]):
        raise ValueError("hard_ood and complement overlap")
    if not np.array_equal(masks["hard_ood"] | masks["complement"], masks["full_test"]):
        raise ValueError("hard_ood and complement do not partition the split")
    scores: dict[str, dict[str, object]] = {}
    for name in SLICE_NAMES:
        mask = masks[name]
        if not np.any(mask):
            raise ValueError(f"{name} has no windows")
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
            "worst": _worst_windows(
                ids,
                nu[mask],
                window_l2[mask],
                persist_l2[mask],
                worst_count,
            ),
        }
    return scores


def aggregate_slice_records(
    records: Sequence[Mapping[str, object]],
    protocol: Mapping[str, object],
) -> dict[str, object]:
    """Mean and sample standard deviation across the protocol seeds.

    ``records`` must be one slice-eval object per protocol seed. A missing
    seed, an extra seed, or a duplicate is rejected. The spread is
    ``ddof = 1``.
    """

    expected = protocol.get("seeds")
    if not isinstance(expected, list) or len(expected) < 3:
        raise ValueError("aggregation requires the frozen seed list")
    if len(records) != len(expected):
        raise ValueError(f"expected {len(expected)} seed records, got {len(records)}")
    ordered = sorted(records, key=lambda row: int(row["seed"]))
    seeds = [int(row["seed"]) for row in ordered]
    if seeds != sorted(int(seed) for seed in expected):
        raise ValueError(f"seed records {seeds} do not match the frozen seeds {sorted(expected)}")
    threshold = hard_ood_threshold(protocol)
    for row in ordered:
        if row.get("format") != SLICE_EVAL_FORMAT:
            raise ValueError(f"unsupported slice record format {row.get('format')!r}")
        if row.get("loss_mode") != "data":
            raise ValueError("slice record is not a data-only run")
        if float(row["threshold_nu"]) != threshold:
            raise ValueError("slice record threshold does not match the training protocol")
        assert_checkpoint_matches_train_protocol(_summary_from_record(row), protocol)
    aggregation = protocol.get("aggregation")
    if not isinstance(aggregation, dict) or int(aggregation.get("ddof", 1)) != 1:
        raise ValueError("training protocol aggregation ddof must be 1")
    payload: dict[str, object] = {
        "format": SLICE_AGGREGATE_FORMAT,
        "seeds": seeds,
        "n_seeds": len(seeds),
        "ddof": 1,
        "threshold_nu": threshold,
        "comparison": HARD_OOD_COMPARISON,
        "primary_metric": "mean_relative_l2",
        "aggregation": (
            "arithmetic mean and sample standard deviation of one scalar per seed; "
            "windows are not pooled across seeds"
        ),
        "loss_mode": "data",
        "slices": {},
        "rollout": {},
        "selected_epoch": _summarize_scalars([float(row["selected_epoch"]) for row in ordered]),
        "val_relative_l2": _summarize_scalars([float(row["val_relative_l2"]) for row in ordered]),
    }
    slice_block: dict[str, object] = {}
    rollout_block: dict[str, object] = {}
    for name in SLICE_NAMES:
        slice_block[name] = _aggregate_slice(ordered, name, "slices", WINDOW_SCALAR_METRICS)
        rollout_block[name] = _aggregate_rollout(ordered, name)
    payload["slices"] = slice_block
    payload["rollout"] = rollout_block
    payload["persistence_seed_max_abs_diff"] = _persistence_disagreement(ordered)
    return payload


def write_json(payload: Mapping[str, object], path: Path) -> Path:
    """Write one JSON object. Parent directories are created."""

    destination = Path(path)
    destination.parent.mkdir(parents=True, exist_ok=True)
    text = json.dumps(payload, indent=2, allow_nan=False) + "\n"
    destination.write_text(text, encoding="utf-8")
    return destination


def _aggregate_slice(
    records: Sequence[Mapping[str, object]],
    name: str,
    block: str,
    metrics: Sequence[str],
) -> dict[str, object]:
    first = _slice_block(records[0], block, name)
    summary: dict[str, object] = {
        "n_instances": int(first["n_instances"]),
        "n_windows": int(first["n_windows"]),
    }
    for row in records[1:]:
        other = _slice_block(row, block, name)
        if int(other["n_instances"]) != summary["n_instances"] or int(other["n_windows"]) != summary["n_windows"]:
            raise ValueError(f"{name} instance or window counts differ across seeds")
    for metric in metrics:
        summary[metric] = _summarize_scalars([float(_slice_block(row, block, name)[metric]) for row in records])
    return summary


def _aggregate_rollout(records: Sequence[Mapping[str, object]], name: str) -> dict[str, object]:
    summary = _aggregate_slice(records, name, "rollout", ROLLOUT_SCALAR_METRICS)
    steps = [
        np.asarray(_slice_block(row, "rollout", name)["per_step_mean_relative_l2"], dtype=np.float64)
        for row in records
    ]
    width = steps[0].shape
    if any(step.shape != width or step.ndim != 1 for step in steps):
        raise ValueError(f"{name} rollout step counts differ across seeds")
    stacked = np.stack(steps, axis=0)
    if not np.isfinite(stacked).all():
        raise ValueError(f"{name} rollout step scores must be finite")
    summary["per_step_mean_relative_l2"] = {
        "n": int(stacked.shape[0]),
        "ddof": 1,
        "mean": [float(value) for value in np.mean(stacked, axis=0)],
        "std": [float(value) for value in np.std(stacked, axis=0, ddof=1)],
    }
    return summary


def _slice_block(record: Mapping[str, object], block: str, name: str) -> Mapping[str, object]:
    container = record.get(block)
    if not isinstance(container, dict) or name not in container:
        raise ValueError(f"slice record is missing {block}.{name}")
    section = container[name]
    if not isinstance(section, dict):
        raise ValueError(f"slice record {block}.{name} must be an object")
    return section


def _summary_from_record(record: Mapping[str, object]) -> dict[str, object]:
    model = record.get("model")
    window = record.get("window")
    if not isinstance(model, dict) or not isinstance(window, dict):
        raise ValueError("slice record must include model and window")
    return {
        "loss_mode": record.get("loss_mode"),
        "seed": record.get("seed"),
        "width": model.get("width"),
        "modes": model.get("modes"),
        "layers": model.get("n_layers"),
        "input_frames": window.get("input_frames"),
        "output_frames": window.get("output_frames"),
        "stride": window.get("stride"),
    }


def _persistence_disagreement(records: Sequence[Mapping[str, object]]) -> float:
    """Largest absolute gap in a seed-independent persistence scalar."""

    gap = 0.0
    for name in SLICE_NAMES:
        for metric in ("persistence_mean_relative_l2",):
            values = [float(_slice_block(row, "slices", name)[metric]) for row in records]
            gap = max(gap, max(values) - min(values))
        rollout_values = [
            float(_slice_block(row, "rollout", name)["persistence_mean_instance_relative_l2"]) for row in records
        ]
        gap = max(gap, max(rollout_values) - min(rollout_values))
    if gap > 1e-12:
        raise ValueError(f"persistence scores differ across seeds by {gap}")
    return gap


def _summarize_scalars(values: Sequence[float]) -> dict[str, object]:
    array = np.asarray(values, dtype=np.float64)
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


def _worst_windows(
    instance_ids: np.ndarray,
    nu: np.ndarray,
    scores: np.ndarray,
    persistence: np.ndarray,
    worst_count: int,
) -> list[dict[str, object]]:
    if isinstance(worst_count, bool) or not isinstance(worst_count, int) or worst_count < 1:
        raise ValueError("worst_count must be an integer >= 1")
    grouped: dict[int, list[int]] = {}
    for index, instance_id in enumerate(instance_ids):
        grouped.setdefault(int(instance_id), []).append(index)
    rows: list[dict[str, object]] = []
    for instance_id, indexes in grouped.items():
        chosen = np.asarray(indexes, dtype=np.int64)
        rows.append(
            {
                "instance_id": instance_id,
                "nu": float(nu[chosen[0]]),
                "n_windows": int(chosen.size),
                "mean_relative_l2": float(np.mean(scores[chosen])),
                "persistence_mean_relative_l2": float(np.mean(persistence[chosen])),
            }
        )
    rows.sort(key=lambda row: (-float(row["mean_relative_l2"]), int(row["instance_id"])))
    return rows[:worst_count]


def _lookup_nu(nu_by_id: Mapping[int, float], instance_id: int) -> float:
    if instance_id not in nu_by_id:
        raise ValueError(f"instance {instance_id} has no viscosity in the manifest")
    return float(nu_by_id[instance_id])


def _require_threshold(value: object) -> float:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise ValueError("threshold_nu must be a finite number > 0")
    number = float(value)
    if not math.isfinite(number) or number <= 0.0:
        raise ValueError("threshold_nu must be a finite number > 0")
    return number


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
        raise ValueError(f"protocol does not exist: {source}")
    payload = json.loads(source.read_text(encoding="utf-8"))
    if not isinstance(payload, dict):
        raise ValueError("protocol must be a JSON object")
    return payload
