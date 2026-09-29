"""Stage B protocol, slice assignment, and the harder-pilot loader. No torch except where marked."""

from __future__ import annotations

import json
import math
from pathlib import Path

import numpy as np
import pytest

from pinnforge.operator.rollout import rollout_trajectories, summarize_rollout
from pinnforge.operator.slices import (
    aggregate_slice_records,
    assert_checkpoint_matches_train_protocol,
    assert_train_call_matches_protocol,
    hard_ood_threshold,
    load_data_protocol,
    load_train_protocol,
    physical_nu_by_instance,
    score_prediction_slices,
)
from pinnforge.operator.windows import (
    HARD_PILOT_FORMAT,
    FieldNorm,
    Trajectory,
    WindowSpec,
    build_window_datasets,
    load_pilot_manifest,
    split_sets,
)
from pinnforge.reference.numerical.dataset import _field_sha256

ROOT = Path(__file__).resolve().parents[1]
TRAIN_PROTOCOL = ROOT / "docs" / "v02" / "stage_b_train_protocol.json"
DATA_PROTOCOL = ROOT / "docs" / "v02" / "pilot_protocol.json"
HARD_MANIFEST = ROOT / "docs" / "stage_a" / "pilot_manifest.json"
NORM = FieldNorm(u_mean=0.0, u_std=1.0, nu_mean=0.05, nu_std=0.02)


def test_committed_training_protocol_is_data_only_and_frozen() -> None:
    protocol = load_train_protocol(TRAIN_PROTOCOL)
    data = load_data_protocol(DATA_PROTOCOL)
    assert protocol["loss"] == "data"
    assert protocol["early_stopping"] is False
    assert protocol["hard_ood_refit"] is False
    assert protocol["residual_weight"] is None
    assert protocol["seeds"] == [0, 1, 2, 3, 4]
    assert protocol["epochs"] == 30
    assert protocol["width"] == 32
    assert protocol["modes"] == 16
    assert protocol["layers"] == 4
    assert protocol["input_frames"] == 8
    assert protocol["output_frames"] == 8
    assert protocol["stride"] == 8
    assert protocol["aggregation"]["ddof"] == 1
    assert protocol["aggregation"]["pool_windows_across_seeds"] is False
    assert protocol["rollout"]["used_for_selection"] is False
    assert protocol["inverse"]["retrain"] is False
    assert hard_ood_threshold(protocol) == hard_ood_threshold(data)
    assert hard_ood_threshold(protocol) == pytest.approx(0.027028120493367818)
    assert "results" not in protocol
    assert "scores" not in protocol


def test_training_protocol_rejects_a_file_that_already_holds_scores(tmp_path: Path) -> None:
    payload = json.loads(TRAIN_PROTOCOL.read_text(encoding="utf-8"))
    payload["results"] = {"full_test": 0.0}
    path = tmp_path / "protocol.json"
    path.write_text(json.dumps(payload), encoding="utf-8")
    with pytest.raises(ValueError, match="test scores"):
        load_train_protocol(path)


def test_train_call_must_match_the_frozen_hyperparameters() -> None:
    protocol = load_train_protocol(TRAIN_PROTOCOL)
    assert_train_call_matches_protocol(
        protocol,
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
        seed=2,
        loss="data",
        residual_weight=None,
    )
    with pytest.raises(ValueError, match="data"):
        assert_train_call_matches_protocol(
            protocol,
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
            loss="hybrid",
            residual_weight=1e-4,
        )
    with pytest.raises(ValueError, match="width"):
        assert_checkpoint_matches_train_protocol(
            {
                "loss_mode": "data",
                "seed": 0,
                "width": 64,
                "modes": 16,
                "layers": 4,
                "input_frames": 8,
                "output_frames": 8,
                "stride": 8,
            },
            protocol,
        )


def test_harder_manifest_loads_and_slice_counts_match_the_protocol() -> None:
    manifest = load_pilot_manifest(HARD_MANIFEST)
    protocol = load_train_protocol(TRAIN_PROTOCOL)
    assert manifest["format"] == HARD_PILOT_FORMAT
    norm_block = manifest["normalization"]
    assert norm_block["u_mean"] == pytest.approx(-4.623541922284834e-19)
    assert norm_block["u_std"] == pytest.approx(0.37254954772040344)
    assert norm_block["fit_on"] == "train"
    threshold = hard_ood_threshold(protocol)
    nu = physical_nu_by_instance(manifest)
    splits = split_sets(manifest)
    counts = protocol["instance_counts_from_manifest"]
    for split in ("train", "val", "test"):
        ids = splits[split]
        hard = {instance_id for instance_id in ids if nu[instance_id] <= threshold}
        complement = ids - hard
        expected = counts[split]
        assert expected["hard_ood"] == len(hard)
        assert expected["complement"] == len(complement)
        if split == "test":
            assert expected["full_test"] == len(ids)
            assert len(hard) == 30
            assert len(complement) == 98


def test_unknown_pilot_format_is_still_rejected(tmp_path: Path) -> None:
    path = tmp_path / "manifest.json"
    path.write_text(json.dumps({"format": "pinnforge.other.v1"}), encoding="utf-8")
    with pytest.raises(ValueError, match="unsupported pilot manifest format"):
        load_pilot_manifest(path)


def test_missing_hard_pilot_file_names_the_hard_generator(tmp_path: Path) -> None:
    from pinnforge.operator.data import load_split_windows

    pilot = tmp_path / "pilot"
    pilot.mkdir()
    low = _field(1)
    high = _field(4)
    manifest = _manifest(
        [
            (1, "train", 0.01, low),
            (4, "train", 0.08, high),
            (2, "val", 0.02, low),
            (3, "test", 0.04, high),
        ],
        write_files=False,
        root=pilot,
    )
    (pilot / "manifest.json").write_text(json.dumps(manifest), encoding="utf-8")
    with pytest.raises(ValueError, match="hard-pilot"):
        load_split_windows(pilot, pilot / "manifest.json", WindowSpec(4, 4, 4), ("train",))


def test_threshold_equality_is_hard_ood_and_is_not_refit() -> None:
    dataset = _dataset()
    prediction = dataset.targets.copy()
    nu_by_id = {1: 0.03, 2: 0.0300001}
    scores = score_prediction_slices(dataset, prediction, nu_by_id, threshold=0.03)
    assert scores["hard_ood"]["n_instances"] == 1
    assert scores["complement"]["n_instances"] == 1
    assert scores["full_test"]["n_instances"] == 2
    assert scores["full_test"]["mean_relative_l2"] == pytest.approx(0.0)
    assert scores["hard_ood"]["persistence_mean_relative_l2"] > 0.0
    with pytest.raises(ValueError, match="no windows"):
        score_prediction_slices(dataset, prediction, {1: 0.01, 2: 0.02}, threshold=0.03)


def test_rollout_perfect_predictor_is_zero_and_stride_is_required() -> None:
    spec = WindowSpec(4, 4, 4)
    fields = [_ramp(1), _ramp(2)]
    starts = (0, 4, 8, 12, 16)

    def predict(windows: np.ndarray, nu: np.ndarray) -> np.ndarray:
        output = np.zeros((windows.shape[0], spec.output_frames, windows.shape[2]), dtype=np.float64)
        del nu
        for index in range(windows.shape[0]):
            matched = False
            for field in fields:
                for start in starts:
                    truth_in = field[start : start + spec.input_frames]
                    if np.allclose(windows[index], truth_in):
                        output[index] = field[start + spec.input_frames : start + spec.span()]
                        matched = True
                        break
                if matched:
                    break
        return output

    rolled = rollout_trajectories(fields, [2, 1], [0.01, 0.2], spec, predict, batch_size=1)
    assert rolled["instance_ids"].tolist() == [1, 2]
    assert np.allclose(rolled["instance_relative_l2"], 0.0)
    summary = summarize_rollout(rolled, 0.05)
    assert summary["hard_ood"]["n_instances"] == 1
    assert summary["hard_ood"]["mean_instance_relative_l2"] == pytest.approx(0.0)
    assert summary["complement"]["n_instances"] == 1
    with pytest.raises(ValueError, match="stride"):
        rollout_trajectories(fields, [1, 2], [0.01, 0.2], WindowSpec(4, 4, 2), predict, batch_size=1)


def test_open_loop_persistence_matches_a_hold_last_frame_predictor() -> None:
    spec = WindowSpec(4, 4, 4)
    fields = [_ramp(3)]

    def predict(windows: np.ndarray, nu: np.ndarray) -> np.ndarray:
        del nu
        last = windows[:, -1:, :]
        return np.repeat(last, spec.output_frames, axis=1)

    rolled = rollout_trajectories(fields, [3], [0.01], spec, predict, batch_size=2)
    assert rolled["instance_relative_l2"][0] == pytest.approx(rolled["persistence_instance_relative_l2"][0])
    assert rolled["instance_relative_l2"][0] > 0.0


def test_aggregate_reports_sample_std_and_rejects_a_missing_seed() -> None:
    protocol = {
        "seeds": [0, 1, 2],
        "hard_ood": {"threshold_nu": 0.03, "comparison": "nu <= threshold_nu"},
        "aggregation": {"ddof": 1},
        "width": 4,
        "modes": 2,
        "layers": 1,
        "input_frames": 4,
        "output_frames": 4,
        "stride": 4,
    }
    records = [_record(seed, error) for seed, error in ((0, 0.2), (1, 0.4), (2, 0.6))]
    summary = aggregate_slice_records(records, protocol)
    metric = summary["slices"]["hard_ood"]["mean_relative_l2"]
    assert metric["mean"] == pytest.approx(0.4)
    assert metric["std"] == pytest.approx(float(np.std([0.2, 0.4, 0.6], ddof=1)))
    assert summary["seeds"] == [0, 1, 2]
    assert summary["persistence_seed_max_abs_diff"] == 0.0
    with pytest.raises(ValueError, match="seed records"):
        aggregate_slice_records(records[:2], protocol)


def test_fno_help_lists_slices_and_aggregate(capsys: pytest.CaptureFixture[str]) -> None:
    from pinnforge.cli import main

    with pytest.raises(SystemExit) as caught:
        main(["fno", "--help"])
    assert caught.value.code == 0
    text = capsys.readouterr().out
    assert "slices" in text
    assert "aggregate" in text
    assert "harder" in text


def test_train_cli_rejects_a_protocol_mismatch(capsys: pytest.CaptureFixture[str]) -> None:
    from pinnforge.operator.__main__ import main

    with pytest.raises(SystemExit) as caught:
        main(
            [
                "train",
                "--protocol",
                str(TRAIN_PROTOCOL),
                "--pilot",
                "artifacts/burgers_hard_pilot",
                "--manifest",
                "docs/stage_a/pilot_manifest.json",
                "--loss",
                "residual",
                "--seed",
                "0",
            ]
        )
    assert caught.value.code == 2
    assert "data" in capsys.readouterr().err


@pytest.mark.ml
def test_slices_cli_on_a_tiny_hard_pilot(tmp_path: Path) -> None:
    pytest.importorskip("torch")
    from pinnforge.cli import main

    pilot = tmp_path / "pilot"
    pilot.mkdir()
    rows = [
        (1, "train", 0.01),
        (4, "train", 0.08),
        (2, "val", 0.02),
        (3, "test", 0.01),
        (5, "test", 0.09),
    ]
    instances = []
    for instance_id, split, nu in rows:
        field = _field(instance_id)
        relative = Path(split) / f"instance_{instance_id:06d}.npz"
        path = pilot / relative
        path.parent.mkdir(parents=True, exist_ok=True)
        np.savez_compressed(path, u=field, nu=np.float64(nu), instance_id=np.int64(instance_id))
        instances.append(
            {
                "instance_id": instance_id,
                "split": split,
                "nu": nu,
                "path": relative.as_posix(),
                "n": int(field.shape[1]),
                "n_times": int(field.shape[0]),
                "field_sha256": _field_sha256(field),
            }
        )
    manifest = {
        "format": HARD_PILOT_FORMAT,
        "normalization": {"fit_on": "train", "applied_to_files": False, "u_mean": 0.0, "u_std": 1.0},
        "splits": {"train": [1, 4], "val": [2], "test": [3, 5]},
        "instances": instances,
    }
    manifest_path = pilot / "manifest.json"
    manifest_path.write_text(json.dumps(manifest), encoding="utf-8")
    protocol_path = tmp_path / "protocol.json"
    protocol_path.write_text(
        json.dumps(
            {
                "format": "pinnforge.burgers_hard_pilot_protocol.v1",
                "hard_ood": {
                    "comparison": "nu <= threshold_nu",
                    "fit_on": "train",
                    "threshold_nu": 0.03,
                },
            }
        ),
        encoding="utf-8",
    )
    output = tmp_path / "run"
    assert (
        main(
            [
                "fno",
                "train",
                "--pilot",
                str(pilot),
                "--manifest",
                str(manifest_path),
                "--output",
                str(output),
                "--epochs",
                "1",
                "--width",
                "4",
                "--modes",
                "2",
                "--layers",
                "1",
                "--input-frames",
                "4",
                "--output-frames",
                "4",
                "--stride",
                "4",
                "--batch-size",
                "2",
                "--seed",
                "0",
            ]
        )
        == 0
    )
    slices_path = output / "slices.json"
    assert (
        main(
            [
                "fno",
                "slices",
                "--pilot",
                str(pilot),
                "--manifest",
                str(manifest_path),
                "--protocol",
                str(protocol_path),
                "--checkpoint",
                str(output / "checkpoint.pt"),
                "--batch-size",
                "2",
                "--output",
                str(slices_path),
            ]
        )
        == 0
    )
    record = json.loads(slices_path.read_text(encoding="utf-8"))
    assert record["format"] == "pinnforge.fno_slice_eval.v1"
    assert record["loss_mode"] == "data"
    assert record["threshold_refit"] is False
    assert record["rollout_used_for_selection"] is False
    assert record["slices"]["hard_ood"]["n_instances"] == 1
    assert record["slices"]["complement"]["n_instances"] == 1
    assert record["slices"]["full_test"]["n_instances"] == 2
    assert record["slices"]["hard_ood"]["worst"][0]["instance_id"] == 3
    assert record["rollout"]["hard_ood"]["n_steps"] == record["slices"]["hard_ood"]["n_windows"]
    assert math.isfinite(record["rollout"]["complement"]["mean_instance_relative_l2"])
    assert math.isfinite(record["slices"]["full_test"]["mean_abs_residual"])
    assert math.isfinite(record["slices"]["hard_ood"]["target_mean_abs_residual"])


def _field(instance_id: int, n_times: int = 24, n_space: int = 8) -> np.ndarray:
    x = np.linspace(-1.0, 1.0, n_space, endpoint=False)
    time = np.linspace(0.0, 1.0, n_times)
    return np.sin(math.pi * x)[None, :] * np.cos(time)[:, None] + 0.01 * instance_id


def _ramp(instance_id: int) -> np.ndarray:
    x = np.linspace(-1.0, 1.0, 8, endpoint=False)
    time = np.arange(24, dtype=np.float64)[:, None]
    return np.sin(math.pi * x)[None, :] * (time + instance_id)


def _dataset():
    spec = WindowSpec(4, 4, 4)
    trajectories = [
        Trajectory(1, "test", 0.03, _field(1)),
        Trajectory(2, "test", 0.04, _field(2)),
    ]
    return build_window_datasets(trajectories, spec, NORM)["test"]


def _manifest(rows: list[tuple[int, str, float, np.ndarray]], *, write_files: bool, root: Path) -> dict[str, object]:
    instances = []
    splits: dict[str, list[int]] = {"train": [], "val": [], "test": []}
    for instance_id, split, nu, field in rows:
        relative = Path(split) / f"instance_{instance_id:06d}.npz"
        if write_files:
            path = root / relative
            path.parent.mkdir(parents=True, exist_ok=True)
            np.savez_compressed(path, u=field, nu=np.float64(nu), instance_id=np.int64(instance_id))
        splits[split].append(instance_id)
        instances.append(
            {
                "instance_id": instance_id,
                "split": split,
                "nu": nu,
                "path": relative.as_posix(),
                "n": int(field.shape[1]),
                "n_times": int(field.shape[0]),
                "field_sha256": _field_sha256(field),
            }
        )
    return {
        "format": HARD_PILOT_FORMAT,
        "normalization": {"fit_on": "train", "applied_to_files": False, "u_mean": 0.0, "u_std": 1.0},
        "splits": splits,
        "instances": instances,
    }


def _record(seed: int, error: float) -> dict[str, object]:
    def section(value: float, *, rollout: bool) -> dict[str, object]:
        if rollout:
            return {
                "n_instances": 2,
                "n_windows": 4,
                "n_steps": 2,
                "mean_instance_relative_l2": value,
                "median_instance_relative_l2": value / 2,
                "persistence_mean_instance_relative_l2": 0.5,
                "per_step_mean_relative_l2": [value, value / 2],
                "worst": [],
            }
        return {
            "n_instances": 2,
            "n_windows": 4,
            "mean_relative_l2": value,
            "median_relative_l2": value / 2,
            "pooled_relative_l2": value / 3,
            "normalized_mse": value / 10,
            "persistence_mean_relative_l2": 0.7,
            "persistence_median_relative_l2": 0.6,
            "persistence_pooled_relative_l2": 0.65,
            "worst": [],
        }

    return {
        "format": "pinnforge.fno_slice_eval.v1",
        "seed": seed,
        "selected_epoch": seed + 1,
        "val_relative_l2": error / 5,
        "loss_mode": "data",
        "threshold_nu": 0.03,
        "model": {"width": 4, "modes": 2, "n_layers": 1},
        "window": {"input_frames": 4, "output_frames": 4, "stride": 4},
        "slices": {name: section(error if name == "hard_ood" else error / 2, rollout=False) for name in ("full_test", "hard_ood", "complement")},
        "rollout": {name: section(error if name == "hard_ood" else error / 2, rollout=True) for name in ("full_test", "hard_ood", "complement")},
    }
