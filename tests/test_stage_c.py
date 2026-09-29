"""Stage C protocol, validation weight selection, and slice aggregation. No torch except where marked."""

from __future__ import annotations

import hashlib
import json
import math
from pathlib import Path

import numpy as np
import pytest

from pinnforge.operator.defaults import PREREGISTERED_HYBRID_WEIGHTS
from pinnforge.operator.slices import (
    hard_ood_threshold,
    load_data_protocol,
    score_prediction_slices,
)
from pinnforge.operator.stage_c import (
    aggregate_stage_c_records,
    assemble_stage_c_scores,
    assert_checkpoint_matches_stage_c,
    assert_stage_c_train_call,
    load_stage_c_protocol,
    load_weight_selection,
    select_hybrid_weight,
)
from pinnforge.operator.windows import FieldNorm, Trajectory, WindowSpec, build_window_datasets

ROOT = Path(__file__).resolve().parents[1]
PROTOCOL = ROOT / "docs" / "v02" / "stage_c_train_protocol.json"
STAGE_B_PROTOCOL = ROOT / "docs" / "v02" / "stage_b_train_protocol.json"
DATA_PROTOCOL = ROOT / "docs" / "v02" / "pilot_protocol.json"
NORM = FieldNorm(u_mean=0.0, u_std=1.0, nu_mean=0.05, nu_std=0.02)


def test_committed_stage_c_protocol_matches_stage_b_and_the_stage_4_grid() -> None:
    protocol = load_stage_c_protocol(PROTOCOL)
    stage_b = json.loads(STAGE_B_PROTOCOL.read_text(encoding="utf-8"))
    data = load_data_protocol(DATA_PROTOCOL)
    assert protocol["seeds"] == [0, 1, 2, 3, 4]
    assert protocol["seeds"] == stage_b["seeds"]
    for label in ("epochs", "batch_size", "lr", "width", "modes", "layers", "input_frames", "output_frames", "stride"):
        assert protocol[label] == stage_b[label]
    assert protocol["residual_scope"] == "with_input"
    assert protocol["residual_space"] == "physical"
    assert protocol["dt"] == pytest.approx(0.01)
    assert protocol["losses"]["data"]["retrain"] is False
    assert protocol["losses"]["hybrid"]["weights"] == list(PREREGISTERED_HYBRID_WEIGHTS)
    assert protocol["losses"]["hybrid"]["nonselected_weights_scored_on_test"] is False
    assert protocol["losses"]["hybrid"]["selection_split"] == "val"
    assert protocol["inverse"]["learned_inverse"] is False
    assert protocol["inverse"]["retrain"] is False
    assert protocol["early_stopping"] is False
    assert protocol["hard_ood_refit"] is False
    assert hard_ood_threshold(protocol) == hard_ood_threshold(data)
    assert hard_ood_threshold(protocol) == pytest.approx(0.027028120493367818)
    assert "results" not in protocol
    assert "scores" not in protocol
    reference = json.loads(Path(protocol["losses"]["data"]["reference"]).read_text(encoding="utf-8"))
    assert reference["loss_mode"] == "data"
    assert reference["seeds"] == protocol["seeds"]


def test_stage_c_protocol_rejects_a_file_that_already_holds_scores(tmp_path: Path) -> None:
    payload = json.loads(PROTOCOL.read_text(encoding="utf-8"))
    payload["results"] = {"hard_ood": 0.0}
    path = tmp_path / "protocol.json"
    path.write_text(json.dumps(payload), encoding="utf-8")
    with pytest.raises(ValueError, match="test scores"):
        load_stage_c_protocol(path)


def test_stage_c_train_call_rejects_data_only_and_an_unregistered_weight() -> None:
    protocol = load_stage_c_protocol(PROTOCOL)
    kwargs = _train_kwargs(loss="hybrid", residual_weight=1e-2)
    assert_stage_c_train_call(protocol, **kwargs)
    assert_stage_c_train_call(protocol, **_train_kwargs(loss="residual", residual_weight=None))
    with pytest.raises(ValueError, match="does not retrain the data-only"):
        assert_stage_c_train_call(protocol, **_train_kwargs(loss="data", residual_weight=None))
    with pytest.raises(ValueError, match="preregistered hybrid grid"):
        assert_stage_c_train_call(protocol, **_train_kwargs(loss="hybrid", residual_weight=1e-3))
    with pytest.raises(ValueError, match="residual-weight"):
        assert_stage_c_train_call(protocol, **_train_kwargs(loss="residual", residual_weight=1e-2))
    with pytest.raises(ValueError, match="scope"):
        assert_stage_c_train_call(
            protocol,
            **_train_kwargs(loss="hybrid", residual_weight=1e-2, residual_scope="target_interior"),
        )


def test_select_hybrid_weight_uses_validation_mean_and_breaks_ties_toward_the_smaller_weight(
    tmp_path: Path,
) -> None:
    digest = hashlib.sha256(PROTOCOL.read_bytes()).hexdigest()
    paths = []
    for weight in PREREGISTERED_HYBRID_WEIGHTS:
        for seed in (0, 1, 2, 3, 4):
            value = {1e-6: 0.03, 1e-4: 0.01, 1e-2: 0.02}[weight] + seed * 1e-4
            path = tmp_path / f"w{weight}-s{seed}.json"
            path.write_text(json.dumps(_manifest(seed, weight, value, digest)), encoding="utf-8")
            paths.append(path)
    selected = select_hybrid_weight(paths, PROTOCOL)
    assert selected["test_used_for_selection"] is False
    assert selected["test_splits_read"] is False
    assert selected["selected_residual_weight"] == pytest.approx(1e-4)
    selection_path = _write(tmp_path, selected)
    loaded = load_weight_selection(selection_path, load_stage_c_protocol(PROTOCOL), digest)
    assert loaded["selected_residual_weight"] == pytest.approx(1e-4)
    tied = []
    for weight in PREREGISTERED_HYBRID_WEIGHTS:
        for seed in (0, 1, 2, 3, 4):
            value = 0.02 if weight == 1e-2 else 0.01
            path = tmp_path / f"tie-{weight}-{seed}.json"
            path.write_text(json.dumps(_manifest(seed, weight, value, digest)), encoding="utf-8")
            tied.append(path)
    tie = select_hybrid_weight(tied, PROTOCOL)
    assert tie["selected_residual_weight"] == pytest.approx(1e-6)
    tampered = json.loads(json.dumps(tie))
    tampered["selected_residual_weight"] = 1e-2
    bad = tmp_path / "tampered.json"
    bad.write_text(json.dumps(tampered), encoding="utf-8")
    with pytest.raises(ValueError, match="validation table"):
        load_weight_selection(bad, load_stage_c_protocol(PROTOCOL), digest)
    leaked = paths[0]
    leaked.write_text(json.dumps(_manifest(0, 1e-6, 0.03, digest, test_used=True)), encoding="utf-8")
    with pytest.raises(ValueError, match="test split"):
        select_hybrid_weight(paths, PROTOCOL)


def test_slice_residual_means_follow_the_window_mask() -> None:
    dataset = _dataset()
    prediction = dataset.targets.copy()
    residual = np.full((dataset.n_windows(), 2, dataset.n_space()), 2.0, dtype=np.float64)
    target = np.full_like(residual, 0.5)
    scores = score_prediction_slices(
        dataset,
        prediction,
        {1: 0.01, 2: 0.04},
        0.03,
        prediction_residual=residual,
        target_residual=target,
    )
    assert scores["full_test"]["mean_abs_residual"] == pytest.approx(2.0)
    assert scores["full_test"]["residual_mse"] == pytest.approx(4.0)
    assert scores["hard_ood"]["target_mean_abs_residual"] == pytest.approx(0.5)
    assert scores["hard_ood"]["n_instances"] == 1
    with pytest.raises(ValueError, match="together"):
        score_prediction_slices(
            dataset,
            prediction,
            {1: 0.01, 2: 0.04},
            0.03,
            prediction_residual=residual,
        )


def test_aggregate_and_assemble_compare_arms_without_rescoring_data_only(tmp_path: Path) -> None:
    protocol = load_stage_c_protocol(PROTOCOL)
    digest = hashlib.sha256(PROTOCOL.read_bytes()).hexdigest()
    paths = []
    for weight in PREREGISTERED_HYBRID_WEIGHTS:
        for seed in (0, 1, 2, 3, 4):
            value = 0.01 if weight == 1e-2 else 0.02
            path = tmp_path / f"{weight}-{seed}.json"
            path.write_text(json.dumps(_manifest(seed, weight, value, digest)), encoding="utf-8")
            paths.append(path)
    selection = select_hybrid_weight(paths, PROTOCOL)
    selection_path = tmp_path / "selection.json"
    selection_path.write_text(json.dumps(selection), encoding="utf-8")
    loaded = load_weight_selection(selection_path, protocol, digest)
    residual_records = [_slice_record(seed, 0.002 + seed * 1e-4, mode="residual", weight=0.0) for seed in range(5)]
    hybrid_records = [_slice_record(seed, 0.02 + seed * 1e-4, mode="hybrid", weight=1e-2) for seed in range(5)]
    residual = aggregate_stage_c_records(residual_records, protocol, loaded)
    hybrid = aggregate_stage_c_records(hybrid_records, protocol, loaded)
    assert residual["loss_mode"] == "residual"
    assert hybrid["residual_weight"] == pytest.approx(1e-2)
    assert "mean_abs_residual" in hybrid["slices"]["hard_ood"]
    scores = assemble_stage_c_scores(PROTOCOL, selection_path, residual, hybrid)
    hard = scores["versus_stage_b_data_only"]["hard_ood"]
    assert hard["residual"]["arm_mean_is_lower"] is True
    assert hard["hybrid"]["arm_mean_is_lower"] is False
    assert scores["data_only_retrained"] is False
    assert scores["learned_inverse"] is False
    assert scores["selected_residual_weight"] == pytest.approx(1e-2)
    data_mean = hard["hybrid"]["data_only_mean"]
    reference = json.loads((ROOT / "docs" / "v02" / "stage_b_scores.json").read_text(encoding="utf-8"))
    assert data_mean == pytest.approx(reference["slices"]["hard_ood"]["mean_relative_l2"]["mean"])
    with pytest.raises(ValueError, match="validation-selected weight"):
        aggregate_stage_c_records(
            [_slice_record(seed, 0.02, mode="hybrid", weight=1e-6) for seed in range(5)],
            protocol,
            loaded,
        )
    with pytest.raises(ValueError, match="data-only"):
        assert_checkpoint_matches_stage_c(
            {
                "loss_mode": "data",
                "residual_weight": 0.0,
                "residual_scope": "with_input",
                "residual_space": "physical",
                "residual_dt": 0.01,
                "seed": 0,
                "width": 32,
                "modes": 16,
                "layers": 4,
                "input_frames": 8,
                "output_frames": 8,
                "stride": 8,
            },
            protocol,
            loaded,
        )


def test_fno_help_lists_stage_c_commands(capsys: pytest.CaptureFixture[str]) -> None:
    from pinnforge.cli import main

    with pytest.raises(SystemExit) as caught:
        main(["fno", "--help"])
    assert caught.value.code == 0
    text = capsys.readouterr().out
    assert "select-hybrid" in text
    assert "stage-c-scores" in text


def test_train_cli_rejects_data_only_under_the_stage_c_protocol(capsys: pytest.CaptureFixture[str]) -> None:
    from pinnforge.operator.__main__ import main

    with pytest.raises(SystemExit) as caught:
        main(
            [
                "train",
                "--protocol",
                str(PROTOCOL),
                "--pilot",
                "artifacts/burgers_hard_pilot",
                "--manifest",
                "docs/stage_a/pilot_manifest.json",
                "--loss",
                "data",
                "--seed",
                "0",
            ]
        )
    assert caught.value.code == 2
    assert "data-only" in capsys.readouterr().err


def test_aggregate_cli_requires_the_stage_c_selection(capsys: pytest.CaptureFixture[str], tmp_path: Path) -> None:
    from pinnforge.operator.__main__ import main

    dummy = tmp_path / "missing.json"
    with pytest.raises(SystemExit) as caught:
        main(
            [
                "aggregate",
                "--protocol",
                str(PROTOCOL),
                "--inputs",
                str(dummy),
                "--output",
                str(tmp_path / "out.json"),
            ]
        )
    assert caught.value.code == 2
    assert "weight-selection" in capsys.readouterr().err


@pytest.mark.ml
def test_stage_c_protocol_trains_hybrid_and_slices_report_the_residual(tmp_path: Path) -> None:
    pytest.importorskip("torch")
    from pinnforge.cli import main
    from pinnforge.operator.windows import HARD_PILOT_FORMAT
    from pinnforge.reference.numerical.dataset import _field_sha256

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
    protocol = json.loads(PROTOCOL.read_text(encoding="utf-8"))
    protocol["seeds"] = [0, 1, 2]
    protocol["epochs"] = 1
    protocol["width"] = 4
    protocol["modes"] = 2
    protocol["layers"] = 1
    protocol["input_frames"] = 4
    protocol["output_frames"] = 4
    protocol["stride"] = 4
    protocol["batch_size"] = 2
    protocol["pilot"] = str(pilot)
    protocol["manifest"] = str(manifest_path)
    protocol["instance_counts_from_manifest"]["test"] = {"full_test": 2, "hard_ood": 1, "complement": 1}
    protocol_path.write_text(json.dumps(protocol), encoding="utf-8")
    digest = hashlib.sha256(protocol_path.read_bytes()).hexdigest()
    manifests = []
    for weight in PREREGISTERED_HYBRID_WEIGHTS:
        for seed in (0, 1, 2):
            value = 0.01 if weight == 1e-2 else 0.05
            path = tmp_path / f"manifest-{weight}-{seed}.json"
            path.write_text(
                json.dumps(_manifest(seed, weight, value, digest, epochs=1, width=4, modes=2, layers=1, frames=4, batch_size=2)),
                encoding="utf-8",
            )
            manifests.append(path)
    selection_path = tmp_path / "selection.json"
    selection_path.write_text(json.dumps(select_hybrid_weight(manifests, protocol_path)), encoding="utf-8")
    output = tmp_path / "run"
    assert (
        main(
            [
                "fno",
                "train",
                "--protocol",
                str(protocol_path),
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
                "--loss",
                "hybrid",
                "--residual-weight",
                "1e-2",
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
                str(DATA_PROTOCOL),
                "--train-protocol",
                str(protocol_path),
                "--weight-selection",
                str(selection_path),
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
    assert record["loss_mode"] == "hybrid"
    assert record["residual_weight"] == pytest.approx(1e-2)
    assert record["test_used_for_weight_selection"] is False
    assert record["slices"]["hard_ood"]["n_instances"] == 1
    assert record["slices"]["complement"]["n_instances"] == 1
    assert math.isfinite(record["slices"]["hard_ood"]["mean_abs_residual"])
    assert math.isfinite(record["slices"]["hard_ood"]["mean_relative_l2"])


def _train_kwargs(**overrides: object) -> dict[str, object]:
    payload: dict[str, object] = {
        "pilot": Path("artifacts/burgers_hard_pilot"),
        "manifest": Path("docs/stage_a/pilot_manifest.json"),
        "epochs": 30,
        "batch_size": 32,
        "lr": 0.001,
        "width": 32,
        "modes": 16,
        "layers": 4,
        "input_frames": 8,
        "output_frames": 8,
        "stride": 8,
        "seed": 0,
        "loss": "hybrid",
        "residual_weight": 1e-2,
        "residual_scope": "with_input",
        "residual_space": "physical",
        "dt": 0.01,
    }
    payload.update(overrides)
    return payload


def _write(tmp_path: Path, payload: dict[str, object]) -> Path:
    path = tmp_path / "selection.json"
    path.write_text(json.dumps(payload), encoding="utf-8")
    return path


def _manifest(
    seed: int,
    weight: float,
    value: float,
    digest: str,
    *,
    test_used: bool = False,
    epochs: int = 30,
    width: int = 32,
    modes: int = 16,
    layers: int = 4,
    frames: int = 8,
    batch_size: int = 32,
) -> dict[str, object]:
    return {
        "seed": seed,
        "epochs_requested": epochs,
        "selected_epoch": 1,
        "selected_val_relative_l2": value,
        "test_used_for_selection": test_used,
        "test_used_for_training": False,
        "lr": 0.001,
        "batch_size": batch_size,
        "loss_config": {
            "mode": "hybrid",
            "residual_weight": weight,
            "residual_scope": "with_input",
            "residual_space": "physical",
            "dt": 0.01,
        },
        "model": {"width": width, "modes": modes, "n_layers": layers},
        "window": {"input_frames": frames, "output_frames": frames, "stride": frames},
        "training_protocol": {"sha256": digest, "test_metrics_not_read": True},
    }


def _slice_record(seed: int, error: float, *, mode: str, weight: float) -> dict[str, object]:
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
            "mean_abs_residual": value,
            "residual_mse": value / 5,
            "target_mean_abs_residual": 0.0002,
            "worst": [],
        }

    return {
        "format": "pinnforge.fno_slice_eval.v1",
        "seed": seed,
        "selected_epoch": seed + 1,
        "val_relative_l2": error / 5,
        "loss_mode": mode,
        "residual_weight": weight,
        "residual_scope": "with_input",
        "residual_space": "physical",
        "residual_dt": 0.01,
        "threshold_nu": 0.027028120493367818,
        "model": {"width": 32, "modes": 16, "n_layers": 4},
        "window": {"input_frames": 8, "output_frames": 8, "stride": 8},
        "slices": {
            name: section(error if name == "hard_ood" else error / 2, rollout=False)
            for name in ("full_test", "hard_ood", "complement")
        },
        "rollout": {
            name: section(error if name == "hard_ood" else error / 2, rollout=True)
            for name in ("full_test", "hard_ood", "complement")
        },
    }


def _dataset():
    spec = WindowSpec(4, 4, 4)
    trajectories = [
        Trajectory(1, "test", 0.01, _field(1)),
        Trajectory(2, "test", 0.04, _field(2)),
    ]
    return build_window_datasets(trajectories, spec, NORM)["test"]


def _field(instance_id: int, n_times: int = 24, n_space: int = 8) -> np.ndarray:
    x = np.linspace(-1.0, 1.0, n_space, endpoint=False)
    time = np.linspace(0.0, 1.0, n_times)
    return np.sin(math.pi * x)[None, :] * np.cos(time)[:, None] + 0.01 * instance_id
