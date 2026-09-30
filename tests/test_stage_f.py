"""Stage F protocol locks. Scoring uses synthetic windows. No torch."""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pytest

from pinnforge.operator.data import load_split_trajectories
from pinnforge.operator.stage_f import (
    HYBRID_WEIGHT,
    INVERSE_RUN_FORMAT,
    SLICE_AGGREGATE_FORMAT,
    SLICE_EVAL_FORMAT,
    aggregate_inverse_runs,
    aggregate_ood_records,
    assemble_stage_f_scores,
    in_range_ids,
    load_stage_f_protocol,
    ood_masks,
    population_field_norm,
    score_ood_windows,
    search_support,
    viscosity_grid,
)
from pinnforge.operator.windows import HARD_PILOT_FORMAT, FieldNorm, WindowDataset, WindowSpec
from pinnforge.reference.numerical.dataset import _field_sha256

ROOT = Path(__file__).resolve().parents[1]
PROTOCOL_PATH = ROOT / "docs" / "v02" / "stage_f_ood_protocol.json"
STAGE_B_SCORES = ROOT / "docs" / "v02" / "stage_b_scores.json"
STAGE_D_SCORES = ROOT / "docs" / "v02" / "stage_d_scores.json"
THRESHOLD = 0.027028120493367818


def test_committed_protocol_is_frozen_and_does_not_reselect() -> None:
    protocol = load_stage_f_protocol(PROTOCOL_PATH)
    support = search_support(protocol)
    assert protocol["frozen_before_test_evaluation"] is True
    assert protocol["test_used_for_selection"] is False
    assert protocol["ood_used_for_selection"] is False
    assert protocol["hard_ood_refit"] is False
    assert protocol["hybrid_weight_reselected"] is False
    assert protocol["lambda_reselected"] is False
    assert protocol["loss_weight_grid_searched"] is False
    assert protocol["seeds"] == [0, 1, 2, 3, 4]
    assert protocol["epochs"] == 30
    assert protocol["width"] == 32
    assert protocol["modes"] == 16
    assert protocol["layers"] == 4
    assert protocol["inverse_lambda"] == 0.0
    assert protocol["mask_name"] == "sensors32_bursts"
    assert protocol["mask_retuned"] is False
    assert protocol["nu_search"]["clipped_to_training_support"] is False
    assert protocol["train_subset"]["n_instances"] == 384
    assert protocol["train_subset"]["n_excluded"] == 128
    assert protocol["validation_subset"]["n_instances"] == 101
    assert protocol["test_counts"] == {"full_test": 128, "in_range": 98, "ood": 30, "below_0_02": 20}
    assert protocol["threshold_nu"] == pytest.approx(THRESHOLD)
    assert support["can_return_values_outside_training_range"] is True
    assert support["n_grid_below_train_min"] == 45
    assert support["n_grid_above_train_max"] == 1
    assert float(support["min"]) < float(protocol["train_subset"]["nu_min"])
    assert "results" not in protocol
    assert "scores" not in protocol
    weight = json.loads((ROOT / "docs" / "v02" / "stage_c_weight_selection.json").read_text(encoding="utf-8"))
    assert weight["selected_residual_weight"] == HYBRID_WEIGHT


def test_protocol_rejects_a_file_that_already_holds_scores(tmp_path: Path) -> None:
    payload = json.loads(PROTOCOL_PATH.read_text(encoding="utf-8"))
    payload["results"] = {"ood": 0.0}
    path = tmp_path / "protocol.json"
    path.write_text(json.dumps(payload), encoding="utf-8")
    with pytest.raises(ValueError, match="test scores"):
        load_stage_f_protocol(path)


def test_train_call_rejects_a_reselected_hybrid_weight() -> None:
    from pinnforge.operator.stage_f import assert_stage_f_train_call

    protocol = load_stage_f_protocol(PROTOCOL_PATH)
    kwargs = dict(
        pilot=Path("artifacts/burgers_hard_pilot"),
        manifest=Path("docs/stage_a/pilot_manifest.json"),
        epochs=30,
        batch_size=32,
        lr=0.001,
        width=32,
        modes=16,
        layers=4,
        input_frames=8,
        output_frames=8,
        stride=8,
        seed=0,
        residual_scope="with_input",
        residual_space="physical",
        dt=0.01,
    )
    assert assert_stage_f_train_call(protocol, loss="data", residual_weight=None, **kwargs) == "data_only"
    assert (
        assert_stage_f_train_call(protocol, loss="hybrid", residual_weight=1e-2, **kwargs) == "hybrid_1e-2"
    )
    with pytest.raises(ValueError, match="1e-2"):
        assert_stage_f_train_call(protocol, loss="hybrid", residual_weight=1e-4, **kwargs)
    with pytest.raises(ValueError, match="data or hybrid"):
        assert_stage_f_train_call(protocol, loss="residual", residual_weight=None, **kwargs)


def test_threshold_equality_is_excluded_from_the_training_subset() -> None:
    manifest = {
        "splits": {"train": [1, 2], "val": [3], "test": [4]},
        "instances": [
            {"instance_id": 1, "nu": 0.04},
            {"instance_id": 2, "nu": THRESHOLD},
            {"instance_id": 3, "nu": 0.08},
            {"instance_id": 4, "nu": 0.01},
        ],
    }
    assert in_range_ids(manifest, "train", THRESHOLD) == [1]
    nu = np.asarray([0.04, THRESHOLD, 0.01], dtype=np.float64)
    masks = ood_masks(nu, THRESHOLD, 0.02)
    assert masks["in_range"].tolist() == [True, False, False]
    assert masks["ood"].tolist() == [False, True, True]
    assert masks["below_0_02"].tolist() == [False, False, True]


def test_one_step_slices_partition_the_test_windows() -> None:
    spec = WindowSpec(2, 2, 2)
    norm = FieldNorm(u_mean=0.0, u_std=1.0, nu_mean=0.05, nu_std=0.02)
    dataset = WindowDataset(
        spec=spec,
        norm=norm,
        split="test",
        inputs=np.zeros((3, 2, 4)),
        targets=np.ones((3, 2, 4)),
        nu=norm.normalize_nu(np.asarray([0.05, 0.025, 0.01])),
        instance_ids=np.asarray([1, 2, 3]),
        starts=np.zeros(3, dtype=np.int64),
    )
    scores = score_ood_windows(
        dataset,
        dataset.targets.copy(),
        {1: 0.05, 2: 0.025, 3: 0.01},
        THRESHOLD,
        0.02,
    )
    assert scores["full_test"]["n_instances"] == 3
    assert scores["in_range"]["n_instances"] == 1
    assert scores["ood"]["n_instances"] == 2
    assert scores["below_0_02"]["n_instances"] == 1
    assert scores["full_test"]["mean_relative_l2"] == pytest.approx(0.0)
    assert scores["ood"]["persistence_mean_relative_l2"] > 0.0
    assert {row["instance_id"] for row in scores["below_0_02"]["instances"]} == {3}


def test_aggregate_reports_the_ood_ratio() -> None:
    protocol = {
        "seeds": [0, 1],
        "threshold_nu": THRESHOLD,
        "below_0_02_threshold": 0.02,
        "test_counts": {"full_test": 2, "in_range": 1, "ood": 1, "below_0_02": 1},
    }
    summary = aggregate_ood_records([_eval_record(0, 0.2, 0.4), _eval_record(1, 0.4, 0.8)], protocol, arm="data_only")
    metric = summary["slices"]["ood"]["mean_relative_l2"]
    assert metric["mean"] == pytest.approx(0.6)
    assert metric["std"] == pytest.approx(float(np.std([0.4, 0.8], ddof=1)))
    assert summary["ood_gap"]["one_step"]["ratio_of_means"] == pytest.approx(0.6 / 0.3)
    assert summary["worst_ood_one_step"][0]["instance_id"] == 2
    with pytest.raises(ValueError, match="seed"):
        aggregate_ood_records([_eval_record(0, 0.2, 0.4)], protocol, arm="data_only")


def test_inverse_grid_can_return_a_viscosity_below_the_training_range() -> None:
    protocol = load_stage_f_protocol(PROTOCOL_PATH)
    grid = viscosity_grid(protocol)
    runs = []
    for arm in ("data_only", "hybrid_1e-2"):
        for seed in range(5):
            runs.append(_inverse_run(arm, seed, grid))
    summary = aggregate_inverse_runs(runs, protocol)
    ood = summary["arms"]["data_only"]["slices"]["ood"]
    assert ood["n_instances"] == 2
    assert ood["n_outside_training_range"]["mean"] == pytest.approx(2.0)
    assert ood["n_failures"]["mean"] == pytest.approx(1.0)
    in_range = summary["arms"]["hybrid_1e-2"]["slices"]["in_range"]
    assert in_range["n_outside_training_range"]["mean"] == pytest.approx(0.0)
    assert summary["nu_search"]["can_return_values_outside_training_range"] is True
    assert summary["lambda_reselected"] is False


def test_assembled_scores_copy_the_published_full_range_numbers() -> None:
    protocol = load_stage_f_protocol(PROTOCOL_PATH)
    grid = viscosity_grid(protocol)
    runs = [_inverse_run(arm, seed, grid) for arm in ("data_only", "hybrid_1e-2") for seed in range(5)]
    payload = assemble_stage_f_scores(
        PROTOCOL_PATH,
        _forward_aggregate("data_only", 0.02),
        _forward_aggregate("hybrid_1e-2", 0.01),
        runs,
    )
    stage_b = json.loads(STAGE_B_SCORES.read_text(encoding="utf-8"))
    stage_d = json.loads(STAGE_D_SCORES.read_text(encoding="utf-8"))
    copied = payload["forward"]["full_range"]["data_only"]["slices"]["ood"]["one_step_mean_relative_l2"]["mean"]
    published = stage_b["slices"]["hard_ood"]["mean_relative_l2"]["mean"]
    assert copied == published
    assert published == pytest.approx(0.007049252763443842)
    inverse = payload["inverse"]["full_range_operators"]["arms"]["data_only"]["ood"]["mean_rel_error"]["mean"]
    assert inverse == stage_d["arms"]["data_only"]["objectives"]["0.0"]["slices"]["hard_ood"]["mean_rel_error"]["mean"]
    closed = payload["inverse"]["closed_form_sensors32"]["slices"]["ood"]
    assert closed["mean_rel_error"] == pytest.approx(0.3494591802034973)
    assert closed["n_failures"] == 9
    paired = payload["forward"]["hybrid_minus_data_only"]["ood"]["one_step_hybrid_minus_data"]["mean"]
    assert paired == pytest.approx(-0.01)
    assert payload["hybrid_weight_reselected"] is False
    assert payload["nu_search"]["clipped_to_training_support"] is False


def test_keep_ids_does_not_read_the_excluded_file(tmp_path: Path) -> None:
    pilot = tmp_path / "pilot"
    pilot.mkdir()
    kept = _field(1)
    relative = Path("train") / "instance_000001.npz"
    path = pilot / relative
    path.parent.mkdir()
    np.savez_compressed(path, u=kept, nu=np.float64(0.08), instance_id=np.int64(1))
    manifest = {
        "format": HARD_PILOT_FORMAT,
        "normalization": {"fit_on": "train", "applied_to_files": False, "u_mean": 0.0, "u_std": 1.0},
        "splits": {"train": [1, 2], "val": [3], "test": [4]},
        "instances": [
            {
                "instance_id": 1,
                "split": "train",
                "nu": 0.08,
                "path": relative.as_posix(),
                "n": 4,
                "n_times": 8,
                "field_sha256": _field_sha256(kept),
            },
            {
                "instance_id": 2,
                "split": "train",
                "nu": 0.01,
                "path": "train/instance_000002.npz",
                "n": 4,
                "n_times": 8,
                "field_sha256": "unused",
            },
            {"instance_id": 3, "split": "val", "nu": 0.05, "path": "val/missing.npz", "n": 4, "n_times": 8},
            {"instance_id": 4, "split": "test", "nu": 0.05, "path": "test/missing.npz", "n": 4, "n_times": 8},
        ],
    }
    manifest_path = tmp_path / "manifest.json"
    manifest_path.write_text(json.dumps(manifest), encoding="utf-8")
    loaded = load_split_trajectories(pilot, manifest_path, ("train",), keep_ids={1})
    assert [item.instance_id for item in loaded] == [1]
    assert loaded[0].nu == pytest.approx(0.08)


def test_population_norm_uses_only_the_arrays_it_is_given() -> None:
    norm = population_field_norm(
        [np.asarray([[0.0, 2.0]]), np.asarray([[0.0, 0.0]])],
        [0.04, 0.08],
    )
    assert norm.u_mean == pytest.approx(0.5)
    assert norm.nu_mean == pytest.approx(0.06)
    assert norm.nu_std == pytest.approx(float(np.std([0.04, 0.08], ddof=0)))


def test_train_cli_rejects_stage_f_weight_reselection(capsys: pytest.CaptureFixture[str]) -> None:
    from pinnforge.operator.__main__ import main

    with pytest.raises(SystemExit) as caught:
        main(
            [
                "train",
                "--protocol",
                str(PROTOCOL_PATH),
                "--pilot",
                "artifacts/burgers_hard_pilot",
                "--manifest",
                "docs/stage_a/pilot_manifest.json",
                "--loss",
                "hybrid",
                "--residual-weight",
                "1e-6",
                "--seed",
                "0",
            ]
        )
    assert caught.value.code == 2
    assert "1e-2" in capsys.readouterr().err


def _field(instance_id: int) -> np.ndarray:
    return np.arange(8 * 4, dtype=np.float64).reshape(8, 4) + instance_id


def _eval_record(seed: int, in_range: float, ood: float) -> dict[str, object]:
    def block(value: float, instance_id: int, nu: float) -> dict[str, object]:
        return {
            "n_instances": 1,
            "n_windows": 2,
            "mean_relative_l2": value,
            "median_relative_l2": value,
            "pooled_relative_l2": value,
            "normalized_mse": value,
            "persistence_mean_relative_l2": 0.5,
            "instances": [
                {
                    "instance_id": instance_id,
                    "nu": nu,
                    "mean_relative_l2": value,
                    "persistence_mean_relative_l2": 0.5,
                }
            ],
        }

    def rollout(value: float, instance_id: int, nu: float) -> dict[str, object]:
        return {
            "n_instances": 1,
            "n_steps": 2,
            "mean_instance_relative_l2": value,
            "median_instance_relative_l2": value,
            "persistence_mean_instance_relative_l2": 0.9,
            "per_step_mean_relative_l2": [value, value],
            "instances": [
                {
                    "instance_id": instance_id,
                    "nu": nu,
                    "relative_l2": value,
                    "persistence_relative_l2": 0.9,
                }
            ],
        }

    return {
        "format": SLICE_EVAL_FORMAT,
        "arm": "data_only",
        "seed": seed,
        "selected_epoch": 1,
        "val_relative_l2": 0.1,
        "threshold_nu": THRESHOLD,
        "below_0_02_threshold": 0.02,
        "ood_used_for_selection": False,
        "rollout_used_for_selection": False,
        "slices": {
            "full_test": {**block(0.5 * (in_range + ood), 1, 0.05), "n_instances": 2},
            "in_range": block(in_range, 1, 0.05),
            "ood": block(ood, 2, 0.01),
            "below_0_02": block(ood, 2, 0.01),
        },
        "rollout": {
            "full_test": {**rollout(0.5 * (in_range + ood), 1, 0.05), "n_instances": 2},
            "in_range": rollout(in_range, 1, 0.05),
            "ood": rollout(ood, 2, 0.01),
            "below_0_02": rollout(ood, 2, 0.01),
        },
    }


def _forward_aggregate(arm: str, value: float) -> dict[str, object]:
    metric = {
        "n": 5,
        "ddof": 1,
        "mean": value,
        "std": 0.0,
        "min": value,
        "max": value,
        "values": [value, value, value, value, value],
    }
    counts = {"full_test": 128, "in_range": 98, "ood": 30, "below_0_02": 20}
    return {
        "format": SLICE_AGGREGATE_FORMAT,
        "arm": arm,
        "seeds": [0, 1, 2, 3, 4],
        "selected_epoch": metric,
        "val_relative_l2": metric,
        "slices": {name: {"n_instances": count, "mean_relative_l2": metric} for name, count in counts.items()},
        "rollout": {
            name: {"n_instances": count, "mean_instance_relative_l2": metric} for name, count in counts.items()
        },
        "ood_gap": {"one_step": {"ratio_of_means": 1.0}, "rollout": {"ratio_of_means": 1.0}},
        "worst_ood_one_step": [],
        "worst_ood_rollout": [],
    }


def _inverse_run(arm: str, seed: int, grid: np.ndarray) -> dict[str, object]:
    inside = int(np.argmin(np.abs(grid - 0.05)))
    outside = np.full(grid.shape, 1.0)
    outside[0] = 0.01
    matched = np.full(grid.shape, 1.0)
    matched[inside] = 0.01

    def instance(instance_id: int, nu: float, curve: np.ndarray) -> dict[str, object]:
        return {
            "instance_id": instance_id,
            "nu": nu,
            "sensor_mse": [float(value) for value in curve],
            "oracle_sensor_mse": [float(value) for value in curve],
            "ls_nu_hat": nu,
            "ls_rel_error": 0.0,
        }

    return {
        "format": INVERSE_RUN_FORMAT,
        "split": "test",
        "arm": arm,
        "seed": seed,
        "lambda": 0.0,
        "lambda_reselected": False,
        "instances": [
            instance(1, 0.05, matched),
            instance(2, 0.025, outside),
            instance(3, 0.01, outside),
        ],
    }
