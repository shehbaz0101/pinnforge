"""Train and score the Stage F restricted-viscosity operators.

Torch is imported by the training and checkpoint modules. The protocol
loader in :mod:`pinnforge.operator.stage_f` does not import torch, and
this module is imported only when a command trains or scores.
"""

from __future__ import annotations

import hashlib
import json
from pathlib import Path
from typing import Any

from pinnforge.operator.checkpoint import load_fno_checkpoint, save_fno_checkpoint
from pinnforge.operator.data import load_split_trajectories
from pinnforge.operator.evaluate import (
    _physical_predictor,
    _require_rollout_matches_first_window,
    predict_normalized,
)
from pinnforge.operator.inverse import PREREGISTERED_OBSERVATION, least_squares_viscosity
from pinnforge.operator.residual import LossConfig
from pinnforge.operator.rollout import rollout_trajectories
from pinnforge.operator.slices import physical_nu_by_instance, write_json
from pinnforge.operator.stage_d import prepare_oracle_windows, prepare_sparse_batch
from pinnforge.operator.stage_d_fit import evaluate_curves
from pinnforge.operator.stage_f import (
    HYBRID_WEIGHT,
    INVERSE_RUN_FORMAT,
    SLICE_EVAL_FORMAT,
    assert_norm_matches_protocol,
    assert_stage_f_train_call,
    in_range_ids,
    load_stage_f_protocol,
    population_field_norm,
    score_ood_windows,
    search_support,
    summarize_ood_rollout,
    viscosity_grid,
)
from pinnforge.operator.train import fit_fno
from pinnforge.operator.windows import WindowSpec, build_window_datasets, load_pilot_manifest
from pinnforge.training.manifest import environment

RUN_MANIFEST_FORMAT = "pinnforge.stage_f_run_manifest.v1"


def train_stage_f(
    *,
    pilot_dir: Path,
    manifest_path: Path,
    output_dir: Path,
    protocol_path: Path,
    epochs: int,
    batch_size: int,
    lr: float,
    width: int,
    modes: int,
    n_layers: int,
    input_frames: int,
    output_frames: int,
    stride: int,
    seed: int,
    loss: LossConfig,
    log: Any = None,
) -> dict[str, Any]:
    """Train on viscosities above the frozen cut. Test files are not read."""

    protocol = load_stage_f_protocol(protocol_path)
    arm = assert_stage_f_train_call(
        protocol,
        pilot=pilot_dir,
        manifest=manifest_path,
        epochs=epochs,
        batch_size=batch_size,
        lr=lr,
        width=width,
        modes=modes,
        layers=n_layers,
        input_frames=input_frames,
        output_frames=output_frames,
        stride=stride,
        seed=seed,
        loss=loss.mode,
        residual_weight=None if loss.mode == "data" else loss.residual_weight,
        residual_scope=loss.residual_scope,
        residual_space=loss.residual_space,
        dt=loss.dt,
    )
    manifest = load_pilot_manifest(manifest_path)
    threshold = float(protocol["threshold_nu"])  # type: ignore[arg-type]
    train_ids = set(in_range_ids(manifest, "train", threshold))
    val_ids = set(in_range_ids(manifest, "val", threshold))
    trajectories = load_split_trajectories(
        pilot_dir,
        manifest_path,
        ("train", "val"),
        check_field_hash=True,
        keep_ids=train_ids | val_ids,
    )
    train_items = sorted((item for item in trajectories if item.split == "train"), key=lambda item: item.instance_id)
    val_items = sorted((item for item in trajectories if item.split == "val"), key=lambda item: item.instance_id)
    if {item.instance_id for item in train_items} != train_ids:
        raise ValueError("restricted training load did not match the in-range ids")
    if {item.instance_id for item in val_items} != val_ids:
        raise ValueError("restricted validation load did not match the in-range ids")
    if any(item.nu <= threshold for item in train_items + val_items):
        raise ValueError("an out-of-distribution viscosity entered the training load")
    train_block = protocol["train_subset"]
    val_block = protocol["validation_subset"]
    if not isinstance(train_block, dict) or not isinstance(val_block, dict):
        raise ValueError("protocol is missing the restricted subsets")
    norm = population_field_norm([item.u for item in train_items], [item.nu for item in train_items])
    assert_norm_matches_protocol(norm, protocol)
    spec = WindowSpec(input_frames=input_frames, output_frames=output_frames, stride=stride)
    datasets = build_window_datasets([*train_items, *val_items], spec, norm)
    if datasets["train"].n_instances() != int(train_block["n_instances"]):
        raise ValueError("training instance count does not match the Stage F protocol")
    if datasets["train"].n_windows() != int(train_block["n_windows"]):
        raise ValueError("training window count does not match the Stage F protocol")
    if datasets["val"].n_instances() != int(val_block["n_instances"]):
        raise ValueError("validation instance count does not match the Stage F protocol")
    if datasets["val"].n_windows() != int(val_block["n_windows"]):
        raise ValueError("validation window count does not match the Stage F protocol")
    import time

    started = time.perf_counter()
    model, history = fit_fno(
        datasets["train"],
        datasets["val"],
        width=width,
        modes=modes,
        n_layers=n_layers,
        epochs=epochs,
        batch_size=batch_size,
        lr=lr,
        seed=seed,
        loss=loss,
        log=log,
    )
    elapsed = time.perf_counter() - started
    selected = min(history, key=lambda row: (row.val_relative_l2, row.epoch))
    destination = Path(output_dir)
    destination.mkdir(parents=True, exist_ok=True)
    checkpoint = save_fno_checkpoint(
        destination / "checkpoint.pt",
        model,
        epoch=selected.epoch,
        spec=spec,
        norm=norm,
        seed=seed,
        val_relative_l2=selected.val_relative_l2,
        loss=loss,
    )
    manifest_file = destination / "manifest.json"
    payload = {
        "format": RUN_MANIFEST_FORMAT,
        "arm": arm,
        "loss": loss.to_dict(),
        "seed": seed,
        "selected_epoch": selected.epoch,
        "selected_val_relative_l2": selected.val_relative_l2,
        "selected_val_mse": selected.val_mse,
        "selected_train_mse": selected.train_mse,
        "wall_clock_seconds": elapsed,
        "parameter_count": model.parameter_count(),
        "model": model.config_dict(),
        "normalization": norm.to_dict(),
        "n_train_instances": datasets["train"].n_instances(),
        "n_val_instances": datasets["val"].n_instances(),
        "n_train_windows": datasets["train"].n_windows(),
        "n_val_windows": datasets["val"].n_windows(),
        "train_instance_ids": sorted(datasets["train"].instance_id_set()),
        "val_instance_ids": sorted(datasets["val"].instance_id_set()),
        "train_nu_min": min(item.nu for item in train_items),
        "train_nu_max": max(item.nu for item in train_items),
        "ood_instances_loaded": False,
        "test_used_for_training": False,
        "test_used_for_selection": False,
        "ood_used_for_selection": False,
        "selection": "minimum validation mean relative L2 on the in-range subset, ties take the earliest epoch",
        "training_protocol": {
            "path": Path(protocol_path).as_posix(),
            "sha256": hashlib.sha256(Path(protocol_path).read_bytes()).hexdigest(),
        },
        "environment": environment(),
        "history": [row.as_dict() for row in history],
    }
    write_json(payload, manifest_file)
    metrics_path = destination / "metrics.jsonl"
    metrics_path.write_text(
        "\n".join(json.dumps(row.as_dict(), allow_nan=False) for row in history) + "\n",
        encoding="utf-8",
    )
    return {
        "checkpoint": checkpoint,
        "manifest_path": manifest_file,
        "metrics_path": metrics_path,
        "selected_epoch": selected.epoch,
        "val_relative_l2": selected.val_relative_l2,
        "val_mse": selected.val_mse,
        "train_mse": selected.train_mse,
        "parameter_count": model.parameter_count(),
        "wall_clock_seconds": elapsed,
        "arm": arm,
    }


def evaluate_stage_f_slices(
    checkpoint: Path,
    pilot_dir: Path,
    manifest_path: Path,
    protocol_path: Path,
    *,
    batch_size: int = 32,
) -> dict[str, Any]:
    """Score one restricted operator on the existing harder test split."""

    protocol = load_stage_f_protocol(protocol_path)
    loaded = load_fno_checkpoint(checkpoint)
    arm = _require_checkpoint(loaded, protocol)
    manifest = load_pilot_manifest(manifest_path)
    trajectories = load_split_trajectories(
        pilot_dir,
        manifest_path,
        ("test",),
        check_field_hash=True,
    )
    dataset = build_window_datasets(trajectories, loaded.spec, loaded.norm)["test"]
    prediction = predict_normalized(loaded.model, dataset, batch_size=batch_size).numpy()
    threshold = float(protocol["threshold_nu"])  # type: ignore[arg-type]
    floor = float(protocol["below_0_02_threshold"])  # type: ignore[arg-type]
    nu_by_id = physical_nu_by_instance(manifest)
    slices = score_ood_windows(dataset, prediction, nu_by_id, threshold, floor)
    rollout = rollout_trajectories(
        [item.u for item in trajectories],
        [item.instance_id for item in trajectories],
        [item.nu for item in trajectories],
        loaded.spec,
        _physical_predictor(loaded.model, loaded.norm),
        batch_size=batch_size,
    )
    _require_rollout_matches_first_window(dataset, prediction, rollout, loaded.norm)
    _require_counts(slices, protocol)
    return {
        "format": SLICE_EVAL_FORMAT,
        "arm": arm,
        "split": "test",
        "seed": loaded.seed,
        "selected_epoch": loaded.epoch,
        "val_relative_l2": loaded.val_relative_l2,
        "checkpoint": Path(checkpoint).as_posix(),
        "loss_mode": loaded.loss.mode,
        "residual_weight": loaded.loss.residual_weight,
        "threshold_nu": threshold,
        "below_0_02_threshold": floor,
        "threshold_refit": False,
        "ood_used_for_selection": False,
        "rollout_used_for_selection": False,
        "normalization": loaded.norm.to_dict(),
        "model": loaded.model.config_dict(),
        "window": {
            "input_frames": loaded.spec.input_frames,
            "output_frames": loaded.spec.output_frames,
            "stride": loaded.spec.stride,
        },
        "slices": slices,
        "rollout": summarize_ood_rollout(rollout, threshold, floor),
        "environment": environment(),
    }


def run_stage_f_inverse(
    *,
    protocol_path: Path,
    pilot_dir: Path,
    manifest_path: Path,
    checkpoint: Path,
    split: str,
) -> dict[str, Any]:
    """Sensor-MSE viscosity fit on the Stage D grid. Lambda stays 0."""

    if split != "test":
        raise ValueError("Stage F inverse scoring is defined on the test split")
    protocol = load_stage_f_protocol(protocol_path)
    loaded = load_fno_checkpoint(checkpoint)
    arm = _require_checkpoint(loaded, protocol)
    trajectories = load_split_trajectories(
        pilot_dir,
        manifest_path,
        ("test",),
        check_field_hash=True,
    )
    window = WindowSpec(input_frames=8, output_frames=8, stride=8)
    prepared = prepare_sparse_batch(trajectories, PREREGISTERED_OBSERVATION, window)
    oracle = prepare_oracle_windows(trajectories, PREREGISTERED_OBSERVATION, window)
    grid = viscosity_grid(protocol)
    baseline = float(loaded.norm.nu_mean)
    curves = evaluate_curves(
        loaded.model,
        loaded.norm,
        prepared["windows"],
        prepared["targets"],
        prepared["slots"],
        grid,
        baseline,
        oracle,
        batch_size=int(protocol["batch_size"]),  # type: ignore[arg-type]
        n_sensors=int(PREREGISTERED_OBSERVATION.n_sensors or 0),
    )
    support = search_support(protocol)
    instances = []
    for index, item in enumerate(trajectories):
        ls_hat = least_squares_viscosity(item.u, PREREGISTERED_OBSERVATION)
        instances.append(
            {
                "instance_id": item.instance_id,
                "nu": item.nu,
                "sensor_mse": [float(value) for value in curves["sensor_mse"][index]],
                "oracle_sensor_mse": [float(value) for value in curves["oracle_sensor_mse"][index]],
                "ls_nu_hat": ls_hat,
                "ls_rel_error": abs(ls_hat - item.nu) / item.nu,
            }
        )
    return {
        "format": INVERSE_RUN_FORMAT,
        "split": split,
        "arm": arm,
        "seed": loaded.seed,
        "selected_epoch": loaded.epoch,
        "lambda": 0.0,
        "lambda_reselected": False,
        "mask": "sensors32_bursts",
        "reads_unmasked_field_for_sparse_objective": False,
        "nu_search": support,
        "train_nu_min": support["train_nu_min"],
        "train_nu_max": support["train_nu_max"],
        "grid": [float(value) for value in grid],
        "instances": instances,
    }


def _require_checkpoint(loaded: Any, protocol: dict[str, Any]) -> str:
    seeds = protocol["seeds"]
    if int(loaded.seed) not in seeds:
        raise ValueError(f"checkpoint seed {loaded.seed} is not in the Stage F protocol")
    expected = WindowSpec(input_frames=8, output_frames=8, stride=8)
    if loaded.spec != expected:
        raise ValueError("checkpoint window does not match the Stage F protocol")
    if (
        int(loaded.model.width) != 32
        or int(loaded.model.modes) != 16
        or int(loaded.model.n_layers) != 4
    ):
        raise ValueError("checkpoint architecture does not match the Stage F protocol")
    assert_norm_matches_protocol(loaded.norm, protocol)
    if loaded.loss.mode == "data":
        return "data_only"
    if loaded.loss.mode == "hybrid" and abs(float(loaded.loss.residual_weight) - HYBRID_WEIGHT) <= 1e-12:
        return "hybrid_1e-2"
    raise ValueError("Stage F checkpoint must be data-only or hybrid with weight 1e-2")


def _require_counts(slices: dict[str, dict[str, object]], protocol: dict[str, Any]) -> None:
    counts = protocol["test_counts"]
    for name, expected in counts.items():
        got = int(slices[name]["n_instances"])
        if got != int(expected):
            raise ValueError(f"test {name} has {got} instances, protocol locked {int(expected)}")
