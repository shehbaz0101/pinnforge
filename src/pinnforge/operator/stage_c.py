"""Stage C training contract for the harder Burgers pilot.

The data-only arm is the Stage B aggregate. This module does not retrain
it and does not read a Stage C test table while choosing the hybrid
weight. Residual and hybrid runs use the Stage 4 stencil
(``with_input``, physical ``u``, ``Δt = 0.01``) and the Stage 4 weight
grid. The weight is the one with the lowest mean, across the frozen
seeds, of the selected-epoch validation mean relative L2. Ties keep the
smaller weight. This module does not import torch.
"""

from __future__ import annotations

import hashlib
import json
import math
from collections.abc import Mapping, Sequence
from pathlib import Path

import numpy as np

from pinnforge.operator.defaults import PREREGISTERED_HYBRID_WEIGHTS
from pinnforge.operator.residual import DEFAULT_FRAME_DT
from pinnforge.operator.slices import (
    HARD_OOD_COMPARISON,
    SLICE_EVAL_FORMAT,
    SLICE_NAMES,
    WINDOW_SCALAR_METRICS,
    _aggregate_rollout,
    _aggregate_slice,
    _persistence_disagreement,
    _summarize_scalars,
    hard_ood_threshold,
    load_data_protocol,
    write_json,
)

STAGE_C_PROTOCOL_FORMAT = "pinnforge.stage_c_train_protocol.v1"
STAGE_B_PROTOCOL_FORMAT = "pinnforge.stage_b_train_protocol.v1"
STAGE_C_SELECTION_FORMAT = "pinnforge.stage_c_weight_selection.v1"
STAGE_C_AGGREGATE_FORMAT = "pinnforge.stage_c_slice_aggregate.v1"
STAGE_C_SCORES_FORMAT = "pinnforge.stage_c_scores.v1"
DATA_REFERENCE_FORMAT = "pinnforge.fno_slice_aggregate.v1"
RESIDUAL_WINDOW_METRICS = (
    "mean_abs_residual",
    "residual_mse",
    "target_mean_abs_residual",
)
_STAGE_C_METRICS = WINDOW_SCALAR_METRICS + RESIDUAL_WINDOW_METRICS
_FORBIDDEN_PROTOCOL_KEYS = frozenset(
    {
        "test_mean_relative_l2",
        "results",
        "scores",
        "metrics",
        "selected_epoch",
        "val_relative_l2",
    }
)
_FORBIDDEN_MANIFEST_KEYS = frozenset(
    {
        "slices",
        "rollout",
        "scores",
        "results",
        "test_mean_relative_l2",
    }
)
_WEIGHT_ATOL = 1e-18


def protocol_format(path: Path) -> str:
    """Return the ``format`` string of a protocol file."""

    payload = _read_object(path)
    fmt = payload.get("format")
    if not isinstance(fmt, str) or fmt == "":
        raise ValueError(f"protocol is missing a format: {path}")
    return fmt


def load_stage_c_protocol(path: Path) -> dict[str, object]:
    """Read the Stage C contract and reject a file that already holds scores."""

    payload = _read_object(path)
    if payload.get("format") != STAGE_C_PROTOCOL_FORMAT:
        raise ValueError(f"unsupported training protocol format {payload.get('format')!r}")
    leaked = _FORBIDDEN_PROTOCOL_KEYS & set(payload)
    if leaked:
        names = ", ".join(sorted(leaked))
        raise ValueError(f"training protocol must not contain test scores ({names})")
    if payload.get("frozen_before_test_evaluation") is not True:
        raise ValueError("training protocol must be frozen before test evaluation")
    if payload.get("early_stopping") is not False:
        raise ValueError("training protocol must set early_stopping to false")
    if payload.get("test_used_for_training") is not False or payload.get("test_used_for_selection") is not False:
        raise ValueError("training protocol must keep the test split out of training and selection")
    if payload.get("hard_ood_refit") is not False:
        raise ValueError("training protocol must not refit hard_ood")
    _require_seeds(payload.get("seeds"))
    _require_architecture(payload)
    if payload.get("residual_scope") != "with_input":
        raise ValueError("Stage C residual scope must stay with_input")
    if payload.get("residual_space") != "physical":
        raise ValueError("Stage C residual space must stay physical")
    dt = payload.get("dt")
    if isinstance(dt, bool) or not isinstance(dt, (int, float)) or float(dt) != float(DEFAULT_FRAME_DT):
        raise ValueError("Stage C residual dt must stay 0.01")
    _require_stage_c_losses(payload)
    _require_hard_ood_lock(payload)
    inverse = payload.get("inverse")
    if not isinstance(inverse, dict) or inverse.get("learned_inverse") is not False or inverse.get("retrain") is not False:
        raise ValueError("Stage C must not train a learned inverse")
    counts = payload.get("instance_counts_from_manifest")
    if not isinstance(counts, dict):
        raise ValueError("training protocol is missing instance_counts_from_manifest")
    return payload


def assert_stage_c_train_call(
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
) -> None:
    """Raise when a train invocation leaves the frozen Stage C contract."""

    if loss == "data":
        reference = _losses(protocol)["data"]["reference"]
        raise ValueError(
            "Stage C does not retrain the data-only arm; "
            f"the baseline is the Stage B aggregate {reference}"
        )
    if loss not in {"residual", "hybrid"}:
        raise ValueError("Stage C train loss must be residual or hybrid")
    if residual_scope != protocol["residual_scope"] or residual_space != protocol["residual_space"]:
        raise ValueError("residual scope and space must match the Stage C protocol")
    if float(dt) != float(protocol["dt"]):  # type: ignore[arg-type]
        raise ValueError("residual dt does not match the Stage C protocol")
    if loss == "residual":
        if residual_weight is not None:
            raise ValueError("residual loss rejects --residual-weight")
    else:
        if residual_weight is None:
            raise ValueError("hybrid loss requires --residual-weight")
        _match_weight(float(residual_weight), _hybrid_weights(protocol))
    seeds = protocol["seeds"]
    if not isinstance(seeds, list) or seed not in seeds:
        raise ValueError(f"seed {seed} is not in the frozen training protocol")
    _require_same_paths(protocol, pilot=pilot, manifest=manifest)
    _require_same_integers(
        protocol,
        epochs=epochs,
        batch_size=batch_size,
        width=width,
        modes=modes,
        layers=layers,
        input_frames=input_frames,
        output_frames=output_frames,
        stride=stride,
    )
    if float(lr) != float(protocol["lr"]):  # type: ignore[arg-type]
        raise ValueError(f"lr {lr} does not match the frozen training protocol ({protocol['lr']})")


def assert_checkpoint_matches_stage_c(
    summary: Mapping[str, object],
    protocol: Mapping[str, object],
    selection: Mapping[str, object],
) -> None:
    """Raise when a checkpoint is not a Stage C test arm.

    Data-only checkpoints are rejected so the Stage B numbers stay the
    baseline. A hybrid checkpoint must use the weight already chosen on
    validation. Residual checkpoints are the other test arm.
    """

    mode = summary.get("loss_mode")
    if mode == "data":
        raise ValueError("Stage C does not rescore the data-only arm; use the Stage B aggregate")
    if mode not in {"residual", "hybrid"}:
        raise ValueError(f"unsupported Stage C loss mode {mode!r}")
    _require_summary_stencil(summary, protocol)
    weight = float(summary["residual_weight"])  # type: ignore[arg-type]
    if mode == "residual":
        if weight != 0.0:
            raise ValueError("residual checkpoint must have residual_weight 0")
    else:
        selected = float(selection["selected_residual_weight"])  # type: ignore[arg-type]
        if not _weights_equal(weight, selected):
            raise ValueError(
                "hybrid slice scoring requires the validation-selected weight "
                f"{selected}, got {weight}"
            )
    seeds = protocol.get("seeds")
    seed = summary.get("seed")
    if not isinstance(seeds, list) or seed not in seeds:
        raise ValueError(f"checkpoint seed {seed} is not in the frozen training protocol")
    for label in ("width", "modes", "layers", "input_frames", "output_frames", "stride"):
        if int(summary[label]) != int(protocol[label]):  # type: ignore[arg-type]
            raise ValueError(f"checkpoint {label} does not match the frozen training protocol")


def select_hybrid_weight(manifest_paths: Sequence[Path], protocol_path: Path) -> dict[str, object]:
    """Choose the hybrid weight from validation manifests.

    Every preregistered weight must be present for every frozen seed.
    The test split is not read. The returned object records the validation
    table and the weight the table selects.
    """

    source = Path(protocol_path)
    protocol = load_stage_c_protocol(source)
    digest = hashlib.sha256(source.read_bytes()).hexdigest()
    weights = _hybrid_weights(protocol)
    seeds = [int(seed) for seed in protocol["seeds"]]  # type: ignore[union-attr]
    grouped: dict[float, dict[int, Mapping[str, object]]] = {weight: {} for weight in weights}
    if len(manifest_paths) != len(weights) * len(seeds):
        raise ValueError(
            f"expected {len(weights) * len(seeds)} hybrid validation manifests, got {len(manifest_paths)}"
        )
    for path in manifest_paths:
        manifest = _load_validation_manifest(path)
        _require_manifest_matches_protocol(manifest, protocol, digest)
        weight = _match_weight(float(manifest["loss_config"]["residual_weight"]), weights)  # type: ignore[index]
        seed = int(manifest["seed"])
        if seed in grouped[weight]:
            raise ValueError(f"duplicate hybrid manifest for weight {weight} seed {seed}")
        grouped[weight][seed] = manifest
    candidates: list[dict[str, object]] = []
    for weight in weights:
        rows = grouped[weight]
        if set(rows) != set(seeds):
            raise ValueError(f"hybrid weight {weight} is missing a frozen seed")
        values = [float(rows[seed]["selected_val_relative_l2"]) for seed in seeds]
        epochs = [int(rows[seed]["selected_epoch"]) for seed in seeds]
        array = np.asarray(values, dtype=np.float64)
        candidates.append(
            {
                "residual_weight": weight,
                "seeds": seeds,
                "selected_epoch": epochs,
                "val_relative_l2": values,
                "mean_val_relative_l2": float(np.mean(array)),
                "std_val_relative_l2": float(np.std(array, ddof=1)),
            }
        )
    chosen = min(candidates, key=lambda row: (float(row["mean_val_relative_l2"]), float(row["residual_weight"])))
    return {
        "format": STAGE_C_SELECTION_FORMAT,
        "frozen_before_test_evaluation": True,
        "test_used_for_selection": False,
        "test_splits_read": False,
        "protocol_path": source.as_posix(),
        "protocol_sha256": digest,
        "metric": "selected_epoch validation mean relative L2",
        "aggregation": "arithmetic mean across the frozen seeds",
        "tie_break": "smallest residual_weight",
        "candidates": candidates,
        "selected_residual_weight": chosen["residual_weight"],
        "selected_mean_val_relative_l2": chosen["mean_val_relative_l2"],
    }


def load_weight_selection(
    path: Path,
    protocol: Mapping[str, object],
    protocol_sha256: str,
) -> dict[str, object]:
    """Read a selection file and re-derive the winner from its validation table."""

    payload = _read_object(path)
    if payload.get("format") != STAGE_C_SELECTION_FORMAT:
        raise ValueError(f"unsupported weight selection format {payload.get('format')!r}")
    leaked = _FORBIDDEN_PROTOCOL_KEYS & set(payload)
    if leaked or "slices" in payload or "rollout" in payload:
        raise ValueError("weight selection must not contain test scores")
    if payload.get("frozen_before_test_evaluation") is not True:
        raise ValueError("weight selection must be frozen before test evaluation")
    if payload.get("test_used_for_selection") is not False or payload.get("test_splits_read") is not False:
        raise ValueError("weight selection must not use the test split")
    if payload.get("protocol_sha256") != protocol_sha256:
        raise ValueError("weight selection protocol hash does not match the training protocol")
    candidates = payload.get("candidates")
    if not isinstance(candidates, list) or len(candidates) != len(_hybrid_weights(protocol)):
        raise ValueError("weight selection candidates must be the preregistered grid")
    seeds = [int(seed) for seed in protocol["seeds"]]  # type: ignore[union-attr]
    checked: list[dict[str, object]] = []
    seen: list[float] = []
    for row in candidates:
        if not isinstance(row, dict):
            raise ValueError("weight selection candidate must be an object")
        weight = _match_weight(float(row["residual_weight"]), _hybrid_weights(protocol))
        if weight in seen:
            raise ValueError(f"duplicate candidate weight {weight}")
        seen.append(weight)
        values = row.get("val_relative_l2")
        if not isinstance(values, list) or [int(seed) for seed in row.get("seeds", [])] != seeds:
            raise ValueError("weight selection values must follow the frozen seeds")
        array = np.asarray([float(value) for value in values], dtype=np.float64)
        if array.size != len(seeds) or not np.isfinite(array).all() or np.any(array < 0.0):
            raise ValueError("validation relative L2 values must be finite and >= 0")
        mean = float(np.mean(array))
        if abs(mean - float(row["mean_val_relative_l2"])) > 1e-12:
            raise ValueError("selection mean does not match its validation values")
        checked.append({"residual_weight": weight, "mean_val_relative_l2": mean})
    if set(seen) != set(_hybrid_weights(protocol)):
        raise ValueError("weight selection is missing a preregistered weight")
    winner = min(checked, key=lambda row: (float(row["mean_val_relative_l2"]), float(row["residual_weight"])))
    selected = float(payload["selected_residual_weight"])  # type: ignore[arg-type]
    if not _weights_equal(selected, float(winner["residual_weight"])):
        raise ValueError("selected weight does not match the validation table")
    return payload


def aggregate_stage_c_records(
    records: Sequence[Mapping[str, object]],
    protocol: Mapping[str, object],
    selection: Mapping[str, object],
) -> dict[str, object]:
    """Mean and sample standard deviation for one Stage C test arm.

    ``records`` are slice-eval objects for every frozen seed of one loss.
    Hybrid records must use the validation-selected weight. Data-only
    records are rejected.
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
    modes = {row.get("loss_mode") for row in ordered}
    if len(modes) != 1:
        raise ValueError("Stage C aggregate records must share one loss mode")
    mode = modes.pop()
    if mode == "data":
        raise ValueError("Stage C does not rescore the data-only arm; use the Stage B aggregate")
    if mode not in {"residual", "hybrid"}:
        raise ValueError(f"unsupported Stage C loss mode {mode!r}")
    weights = [float(row["residual_weight"]) for row in ordered]
    if mode == "residual":
        if any(weight != 0.0 for weight in weights):
            raise ValueError("residual slice records must have residual_weight 0")
        residual_weight: float | None = 0.0
    else:
        selected = float(selection["selected_residual_weight"])  # type: ignore[arg-type]
        if any(not _weights_equal(weight, selected) for weight in weights):
            raise ValueError("hybrid slice records must use the validation-selected weight")
        residual_weight = selected
    threshold = hard_ood_threshold(protocol)
    for row in ordered:
        if row.get("format") != SLICE_EVAL_FORMAT:
            raise ValueError(f"unsupported slice record format {row.get('format')!r}")
        if float(row["threshold_nu"]) != threshold:
            raise ValueError("slice record threshold does not match the training protocol")
        _require_summary_stencil(row, protocol)
        for name in SLICE_NAMES:
            section = row["slices"][name]  # type: ignore[index]
            if not isinstance(section, dict):
                raise ValueError(f"slice record is missing slices.{name}")
            missing = [metric for metric in RESIDUAL_WINDOW_METRICS if metric not in section]
            if missing:
                raise ValueError(f"slice record is missing residual metrics ({', '.join(missing)})")
        assert_checkpoint_matches_stage_c(_summary_from_record(row), protocol, selection)
    aggregation = protocol.get("aggregation")
    if not isinstance(aggregation, dict) or int(aggregation.get("ddof", 1)) != 1:
        raise ValueError("training protocol aggregation ddof must be 1")
    payload: dict[str, object] = {
        "format": STAGE_C_AGGREGATE_FORMAT,
        "seeds": seeds,
        "n_seeds": len(seeds),
        "ddof": 1,
        "threshold_nu": threshold,
        "comparison": HARD_OOD_COMPARISON,
        "threshold_refit": False,
        "primary_metric": "mean_relative_l2",
        "aggregation": (
            "arithmetic mean and sample standard deviation of one scalar per seed; "
            "windows are not pooled across seeds"
        ),
        "loss_mode": mode,
        "residual_weight": residual_weight,
        "residual_scope": protocol["residual_scope"],
        "residual_space": protocol["residual_space"],
        "residual_dt": protocol["dt"],
        "rollout_used_for_selection": False,
        "slices": {},
        "rollout": {},
        "selected_epoch": _summarize_scalars([float(row["selected_epoch"]) for row in ordered]),
        "val_relative_l2": _summarize_scalars([float(row["val_relative_l2"]) for row in ordered]),
    }
    slice_block: dict[str, object] = {}
    rollout_block: dict[str, object] = {}
    for name in SLICE_NAMES:
        slice_block[name] = _aggregate_slice(ordered, name, "slices", _STAGE_C_METRICS)
        rollout_block[name] = _aggregate_rollout(ordered, name)
    payload["slices"] = slice_block
    payload["rollout"] = rollout_block
    payload["persistence_seed_max_abs_diff"] = _persistence_disagreement(ordered)
    return payload


def assemble_stage_c_scores(
    protocol_path: Path,
    selection_path: Path,
    residual_aggregate: Mapping[str, object],
    hybrid_aggregate: Mapping[str, object],
) -> dict[str, object]:
    """Pin the Stage B reference beside the residual and selected hybrid arms.

    The hybrid arm must be the validation-selected weight. The comparison
    flag is the sign of the difference of means. It is not a second
    selection step.
    """

    source = Path(protocol_path)
    protocol = load_stage_c_protocol(source)
    digest = hashlib.sha256(source.read_bytes()).hexdigest()
    selection = load_weight_selection(Path(selection_path), protocol, digest)
    if residual_aggregate.get("format") != STAGE_C_AGGREGATE_FORMAT:
        raise ValueError("residual aggregate has the wrong format")
    if hybrid_aggregate.get("format") != STAGE_C_AGGREGATE_FORMAT:
        raise ValueError("hybrid aggregate has the wrong format")
    if residual_aggregate.get("loss_mode") != "residual":
        raise ValueError("residual aggregate is not the residual arm")
    if hybrid_aggregate.get("loss_mode") != "hybrid":
        raise ValueError("hybrid aggregate is not the hybrid arm")
    selected = float(selection["selected_residual_weight"])  # type: ignore[arg-type]
    if not _weights_equal(float(hybrid_aggregate["residual_weight"]), selected):  # type: ignore[arg-type]
        raise ValueError("hybrid aggregate weight does not match the validation selection")
    if residual_aggregate.get("seeds") != protocol["seeds"] or hybrid_aggregate.get("seeds") != protocol["seeds"]:
        raise ValueError("aggregate seeds do not match the training protocol")
    data = _load_data_reference(protocol)
    if data.get("seeds") != protocol["seeds"]:
        raise ValueError("Stage B reference seeds do not match the Stage C protocol")
    threshold = hard_ood_threshold(protocol)
    if float(data["threshold_nu"]) != threshold:
        raise ValueError("Stage B reference threshold does not match the Stage C protocol")
    if float(residual_aggregate["threshold_nu"]) != threshold or float(hybrid_aggregate["threshold_nu"]) != threshold:
        raise ValueError("aggregate threshold does not match the Stage C protocol")
    comparison = {
        name: {
            "metric": "mean_relative_l2",
            "residual": _paired(data, residual_aggregate, name),
            "hybrid": _paired(data, hybrid_aggregate, name),
        }
        for name in SLICE_NAMES
    }
    return {
        "format": STAGE_C_SCORES_FORMAT,
        "protocol_path": source.as_posix(),
        "protocol_sha256": digest,
        "weight_selection_path": Path(selection_path).as_posix(),
        "threshold_nu": threshold,
        "comparison": HARD_OOD_COMPARISON,
        "threshold_refit": False,
        "test_used_for_weight_selection": False,
        "selected_residual_weight": selected,
        "data_only_retrained": False,
        "learned_inverse": False,
        "data_only": _data_reference_view(protocol, data),
        "residual": dict(residual_aggregate),
        "hybrid": dict(hybrid_aggregate),
        "versus_stage_b_data_only": comparison,
        "residual_metric_note": (
            "Test mean |R| is reported for the residual and hybrid arms. "
            "The Stage B data-only aggregate has no test residual, and this stage does not rescore that arm."
        ),
    }


def write_selection(payload: Mapping[str, object], path: Path) -> Path:
    """Write the validation selection. This is not a test table."""

    return write_json(payload, path)


def _paired(
    data: Mapping[str, object],
    arm: Mapping[str, object],
    slice_name: str,
) -> dict[str, object]:
    data_metric = _slice_metric(data, slice_name)
    arm_metric = _slice_metric(arm, slice_name)
    data_values = [float(value) for value in data_metric["values"]]
    arm_values = [float(value) for value in arm_metric["values"]]
    if len(data_values) != len(arm_values):
        raise ValueError(f"{slice_name} seed counts differ between the arm and the Stage B reference")
    paired = [arm_value - data_value for arm_value, data_value in zip(arm_values, data_values, strict=True)]
    data_mean = float(data_metric["mean"])
    arm_mean = float(arm_metric["mean"])
    return {
        "data_only_mean": data_mean,
        "data_only_std": float(data_metric["std"]),
        "arm_mean": arm_mean,
        "arm_std": float(arm_metric["std"]),
        "arm_minus_data_only": arm_mean - data_mean,
        "arm_mean_is_lower": arm_mean < data_mean,
        "paired_arm_minus_data_only": paired,
        "n_seeds_arm_lower": sum(1 for value in paired if value < 0.0),
        "n_seeds": len(paired),
    }


def _slice_metric(payload: Mapping[str, object], slice_name: str) -> Mapping[str, object]:
    slices = payload.get("slices")
    if not isinstance(slices, dict) or slice_name not in slices:
        raise ValueError(f"missing slice {slice_name}")
    section = slices[slice_name]
    if not isinstance(section, dict) or "mean_relative_l2" not in section:
        raise ValueError(f"missing mean_relative_l2 for {slice_name}")
    metric = section["mean_relative_l2"]
    if not isinstance(metric, dict):
        raise ValueError(f"{slice_name} mean_relative_l2 must be an object")
    return metric


def _data_reference_view(protocol: Mapping[str, object], data: Mapping[str, object]) -> dict[str, object]:
    losses = _losses(protocol)
    copied: dict[str, object] = {}
    for name in SLICE_NAMES:
        section = data["slices"][name]  # type: ignore[index]
        if not isinstance(section, dict):
            raise ValueError(f"Stage B reference is missing slices.{name}")
        copied[name] = {
            "n_instances": section["n_instances"],
            "n_windows": section["n_windows"],
            "mean_relative_l2": section["mean_relative_l2"],
            "median_relative_l2": section["median_relative_l2"],
            "pooled_relative_l2": section["pooled_relative_l2"],
            "normalized_mse": section["normalized_mse"],
            "persistence_mean_relative_l2": section["persistence_mean_relative_l2"],
        }
    return {
        "retrained": False,
        "source": losses["data"]["reference"],
        "sha256": losses["data"]["reference_sha256"],
        "loss_mode": "data",
        "note": "Copied from the Stage B aggregate. This stage did not retrain or rescore that arm.",
        "slices": copied,
    }


def _load_data_reference(protocol: Mapping[str, object]) -> dict[str, object]:
    losses = _losses(protocol)
    path = Path(str(losses["data"]["reference"]))
    raw = path.read_bytes()
    if hashlib.sha256(raw).hexdigest() != losses["data"]["reference_sha256"]:
        raise ValueError("Stage B reference bytes do not match reference_sha256")
    payload = json.loads(raw)
    if not isinstance(payload, dict):
        raise ValueError("Stage B reference must be a JSON object")
    if payload.get("format") != DATA_REFERENCE_FORMAT or payload.get("loss_mode") != "data":
        raise ValueError("Stage B reference must be the data-only slice aggregate")
    return payload


def _load_validation_manifest(path: Path) -> dict[str, object]:
    payload = _read_object(path)
    leaked = _FORBIDDEN_MANIFEST_KEYS & set(payload)
    if leaked:
        names = ", ".join(sorted(leaked))
        raise ValueError(f"{path} contains {names}; selection reads validation manifests only")
    if payload.get("test_used_for_selection") is not False or payload.get("test_used_for_training") is not False:
        raise ValueError(f"{path} records that the test split was used")
    loss = payload.get("loss_config")
    if not isinstance(loss, dict) or loss.get("mode") != "hybrid":
        raise ValueError(f"{path} is not a hybrid validation manifest")
    value = payload.get("selected_val_relative_l2")
    if isinstance(value, bool) or not isinstance(value, (int, float)) or not math.isfinite(float(value)):
        raise ValueError(f"{path} selected_val_relative_l2 must be finite")
    if float(value) < 0.0:
        raise ValueError(f"{path} selected_val_relative_l2 must be >= 0")
    return payload


def _require_manifest_matches_protocol(
    manifest: Mapping[str, object],
    protocol: Mapping[str, object],
    protocol_sha256: str,
) -> None:
    recorded = manifest.get("training_protocol")
    if not isinstance(recorded, dict) or recorded.get("sha256") != protocol_sha256:
        raise ValueError("validation manifest protocol hash does not match the Stage C protocol")
    if recorded.get("test_metrics_not_read") is not True:
        raise ValueError("validation manifest must record that test metrics were not read")
    if int(manifest["epochs_requested"]) != int(protocol["epochs"]):  # type: ignore[arg-type]
        raise ValueError("validation manifest epochs do not match the Stage C protocol")
    if int(manifest["batch_size"]) != int(protocol["batch_size"]):  # type: ignore[arg-type]
        raise ValueError("validation manifest batch_size does not match the Stage C protocol")
    if float(manifest["lr"]) != float(protocol["lr"]):  # type: ignore[arg-type]
        raise ValueError("validation manifest lr does not match the Stage C protocol")
    epoch = manifest.get("selected_epoch")
    if isinstance(epoch, bool) or not isinstance(epoch, int) or epoch < 0 or epoch > int(protocol["epochs"]):  # type: ignore[arg-type]
        raise ValueError("selected_epoch is outside the Stage C epoch budget")
    loss = manifest["loss_config"]
    if not isinstance(loss, dict):
        raise ValueError("loss_config must be an object")
    if loss.get("residual_scope") != protocol["residual_scope"] or loss.get("residual_space") != protocol["residual_space"]:
        raise ValueError("validation manifest residual stencil does not match the Stage C protocol")
    if float(loss["dt"]) != float(protocol["dt"]):  # type: ignore[arg-type]
        raise ValueError("validation manifest residual dt does not match the Stage C protocol")
    model = manifest.get("model")
    window = manifest.get("window")
    if not isinstance(model, dict) or not isinstance(window, dict):
        raise ValueError("validation manifest must include model and window")
    pairs = (
        ("width", model.get("width"), protocol["width"]),
        ("modes", model.get("modes"), protocol["modes"]),
        ("layers", model.get("n_layers"), protocol["layers"]),
        ("input_frames", window.get("input_frames"), protocol["input_frames"]),
        ("output_frames", window.get("output_frames"), protocol["output_frames"]),
        ("stride", window.get("stride"), protocol["stride"]),
    )
    for label, got, expected in pairs:
        if int(got) != int(expected):  # type: ignore[arg-type]
            raise ValueError(f"validation manifest {label} does not match the Stage C protocol")
    seed = manifest.get("seed")
    seeds = protocol["seeds"]
    if not isinstance(seeds, list) or seed not in seeds:
        raise ValueError(f"validation manifest seed {seed} is not in the frozen training protocol")


def _summary_from_record(record: Mapping[str, object]) -> dict[str, object]:
    model = record.get("model")
    window = record.get("window")
    if not isinstance(model, dict) or not isinstance(window, dict):
        raise ValueError("slice record must include model and window")
    return {
        "loss_mode": record.get("loss_mode"),
        "residual_weight": record.get("residual_weight"),
        "residual_scope": record.get("residual_scope"),
        "residual_space": record.get("residual_space"),
        "residual_dt": record.get("residual_dt"),
        "seed": record.get("seed"),
        "width": model.get("width"),
        "modes": model.get("modes"),
        "layers": model.get("n_layers"),
        "input_frames": window.get("input_frames"),
        "output_frames": window.get("output_frames"),
        "stride": window.get("stride"),
    }


def _require_summary_stencil(summary: Mapping[str, object], protocol: Mapping[str, object]) -> None:
    if summary.get("residual_scope") != protocol["residual_scope"]:
        raise ValueError("slice record residual scope does not match the Stage C protocol")
    if summary.get("residual_space") != protocol["residual_space"]:
        raise ValueError("slice record residual space does not match the Stage C protocol")
    if float(summary["residual_dt"]) != float(protocol["dt"]):  # type: ignore[arg-type]
        raise ValueError("slice record residual dt does not match the Stage C protocol")


def _require_stage_c_losses(payload: Mapping[str, object]) -> None:
    losses = payload.get("losses")
    if not isinstance(losses, dict):
        raise ValueError("training protocol is missing losses")
    data = losses.get("data")
    residual = losses.get("residual")
    hybrid = losses.get("hybrid")
    if not isinstance(data, dict) or data.get("retrain") is not False:
        raise ValueError("Stage C data-only arm must not be retrained")
    reference = data.get("reference")
    digest = data.get("reference_sha256")
    if not isinstance(reference, str) or reference == "":
        raise ValueError("Stage C data reference must be a path")
    if not isinstance(digest, str) or len(digest) != 64:
        raise ValueError("Stage C data reference_sha256 must be 64 hex characters")
    reference_path = Path(reference)
    if not reference_path.is_file():
        raise ValueError(f"Stage B reference does not exist: {reference_path}")
    if hashlib.sha256(reference_path.read_bytes()).hexdigest() != digest:
        raise ValueError("Stage B reference bytes do not match reference_sha256")
    if not isinstance(residual, dict) or residual.get("mode") != "residual":
        raise ValueError("Stage C residual arm must use loss mode residual")
    if not isinstance(hybrid, dict) or hybrid.get("mode") != "hybrid":
        raise ValueError("Stage C hybrid arm must use loss mode hybrid")
    if hybrid.get("nonselected_weights_scored_on_test") is not False:
        raise ValueError("non-selected hybrid weights must not be scored on test")
    if hybrid.get("selection_split") != "val":
        raise ValueError("hybrid weight selection split must stay val")
    weights = hybrid.get("weights")
    if not isinstance(weights, list):
        raise ValueError("hybrid weights must be a list")
    parsed = [_match_weight(float(weight), PREREGISTERED_HYBRID_WEIGHTS) for weight in weights]
    if parsed != [float(weight) for weight in PREREGISTERED_HYBRID_WEIGHTS]:
        raise ValueError("hybrid weights must stay the Stage 4 preregistered grid, in that order")
    _load_data_reference(payload)


def _require_hard_ood_lock(payload: Mapping[str, object]) -> None:
    data_path = payload.get("data_protocol")
    digest = payload.get("data_protocol_sha256")
    if not isinstance(data_path, str) or data_path == "":
        raise ValueError("training protocol data_protocol must be a path")
    if not isinstance(digest, str) or len(digest) != 64:
        raise ValueError("training protocol data_protocol_sha256 must be 64 hex characters")
    data_file = Path(data_path)
    if not data_file.is_file():
        raise ValueError(f"data protocol does not exist: {data_file}")
    if hashlib.sha256(data_file.read_bytes()).hexdigest() != digest:
        raise ValueError("data protocol bytes do not match data_protocol_sha256")
    data_protocol = load_data_protocol(data_file)
    hard = payload.get("hard_ood")
    if not isinstance(hard, dict) or hard.get("refit") is not False:
        raise ValueError("training protocol hard_ood.refit must be false")
    if hard.get("comparison") != HARD_OOD_COMPARISON:
        raise ValueError(f"training protocol hard_ood comparison must stay {HARD_OOD_COMPARISON!r}")
    if hard_ood_threshold(payload) != hard_ood_threshold(data_protocol):
        raise ValueError("training protocol threshold does not match the Stage A data protocol")


def _require_seeds(seeds: object) -> None:
    if not isinstance(seeds, list) or len(seeds) < 3:
        raise ValueError("training protocol needs at least 3 seeds")
    if any(isinstance(seed, bool) or not isinstance(seed, int) or seed < 0 for seed in seeds):
        raise ValueError("training protocol seeds must be non-negative integers")
    if len(set(seeds)) != len(seeds):
        raise ValueError("training protocol seeds must be unique")


def _require_architecture(payload: Mapping[str, object]) -> None:
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


def _require_same_paths(protocol: Mapping[str, object], *, pilot: Path, manifest: Path) -> None:
    pairs = (
        ("pilot", Path(pilot), Path(str(protocol["pilot"]))),
        ("manifest", Path(manifest), Path(str(protocol["manifest"]))),
    )
    for label, got, expected in pairs:
        if not _same_path(got, expected):
            raise ValueError(f"{label} {got} does not match the frozen training protocol ({expected})")


def _require_same_integers(protocol: Mapping[str, object], **values: int) -> None:
    for label, got in values.items():
        if int(got) != int(protocol[label]):  # type: ignore[arg-type]
            raise ValueError(f"{label} {got} does not match the frozen training protocol ({protocol[label]})")


def _losses(protocol: Mapping[str, object]) -> dict[str, dict[str, object]]:
    losses = protocol.get("losses")
    if not isinstance(losses, dict):
        raise ValueError("training protocol is missing losses")
    return losses  # type: ignore[return-value]


def _hybrid_weights(protocol: Mapping[str, object]) -> list[float]:
    hybrid = _losses(protocol)["hybrid"]
    weights = hybrid.get("weights")
    if not isinstance(weights, list):
        raise ValueError("hybrid weights must be a list")
    return [float(weight) for weight in weights]


def _match_weight(weight: float, grid: Sequence[float]) -> float:
    if isinstance(weight, bool) or not math.isfinite(weight):
        raise ValueError("residual_weight must be a finite number")
    for candidate in grid:
        if _weights_equal(weight, float(candidate)):
            return float(candidate)
    rendered = ", ".join(str(item) for item in grid)
    raise ValueError(f"residual_weight {weight} is not in the preregistered hybrid grid ({rendered})")


def _weights_equal(left: float, right: float) -> bool:
    return math.isclose(left, right, rel_tol=0.0, abs_tol=_WEIGHT_ATOL)


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
