"""Window cuts, instance splits, and normalization. No torch."""

from __future__ import annotations

import json
import math
import os
import subprocess
import sys
from pathlib import Path

import numpy as np
import pytest

from pinnforge.cli import main
from pinnforge.operator.data import load_split_windows
from pinnforge.operator.metrics import (
    mean_relative_l2,
    persistence_prediction,
    physical_targets,
)
from pinnforge.operator.windows import (
    FieldNorm,
    Trajectory,
    WindowSpec,
    build_window_datasets,
    cut_trajectory,
    field_norm_from_manifest,
    load_pilot_manifest,
    split_sets,
    viscosity_train_stats,
    window_starts,
)
from pinnforge.reference.numerical.dataset import _field_sha256, assign_splits

ROOT = Path(__file__).resolve().parents[1]
MANIFEST = ROOT / "docs" / "stage2" / "pilot_manifest.json"
SPEC = WindowSpec(input_frames=8, output_frames=8, stride=8)
NORM = FieldNorm(u_mean=0.0, u_std=1.0, nu_mean=0.05, nu_std=0.02)


def _field(instance_id: int, n_times: int = 40, n_space: int = 16) -> np.ndarray:
    x = np.linspace(-1.0, 1.0, n_space, endpoint=False)
    time = np.linspace(0.0, 1.0, n_times)
    return np.sin(math.pi * x)[None, :] * np.cos(time)[:, None] + 0.01 * instance_id


def _trajectory(instance_id: int, split: str, nu: float = 0.05) -> Trajectory:
    return Trajectory(instance_id=instance_id, split=split, nu=nu, u=_field(instance_id))


def test_default_window_count_on_the_pilot_grid() -> None:
    starts = window_starts(101, SPEC)
    assert starts == tuple(range(0, 86, 8))
    assert len(starts) == 11
    assert starts[-1] + SPEC.span() == 96


def test_targets_are_the_frames_after_the_input_window() -> None:
    field = _field(3, n_times=40, n_space=8)
    inputs, targets, starts = cut_trajectory(field, WindowSpec(4, 4, 4))
    assert inputs.shape == (9, 4, 8)
    assert targets.shape == (9, 4, 8)
    for index, start in enumerate(starts):
        assert np.allclose(inputs[index], field[start : start + 4])
        assert np.allclose(targets[index], field[start + 4 : start + 8])


def test_short_trajectory_is_rejected() -> None:
    with pytest.raises(ValueError, match="shorter than one window"):
        cut_trajectory(np.zeros((10, 8)), SPEC)


def test_windows_do_not_mix_instances() -> None:
    datasets = build_window_datasets(
        [_trajectory(1, "train", nu=0.04), _trajectory(2, "val", nu=0.08), _trajectory(3, "test", nu=0.06)],
        WindowSpec(4, 4, 4),
        NORM,
    )
    assert datasets["train"].instance_id_set() == {1}
    assert datasets["val"].instance_id_set() == {2}
    assert datasets["test"].instance_id_set() == {3}
    raw = physical_targets(datasets["train"])
    assert np.allclose(raw, datasets["train"].norm.denormalize_u(datasets["train"].targets))
    train_level = float(np.mean(datasets["train"].inputs))
    test_level = float(np.mean(datasets["test"].inputs))
    assert train_level != pytest.approx(test_level)


def test_repeated_instance_id_is_rejected() -> None:
    with pytest.raises(ValueError, match="more than once"):
        build_window_datasets([_trajectory(1, "train"), _trajectory(1, "test")], SPEC, NORM)


def test_normalization_uses_manifest_statistics_not_the_batch() -> None:
    manifest = {
        "format": "pinnforge.burgers_pilot.v1",
        "normalization": {
            "fit_on": "train",
            "applied_to_files": False,
            "u_mean": 0.5,
            "u_std": 2.0,
        },
        "splits": {"train": [1, 2], "val": [3], "test": [4]},
        "instances": [
            {"instance_id": 1, "split": "train", "nu": 0.04},
            {"instance_id": 2, "split": "train", "nu": 0.08},
            {"instance_id": 3, "split": "val", "nu": 0.2},
            {"instance_id": 4, "split": "test", "nu": 10.0},
        ],
    }
    norm = field_norm_from_manifest(manifest)
    assert norm.u_mean == pytest.approx(0.5)
    assert norm.u_std == pytest.approx(2.0)
    nu_mean, nu_std = viscosity_train_stats(manifest)
    assert nu_mean == pytest.approx(0.06)
    assert nu_std == pytest.approx(0.02)
    assert norm.nu_mean == pytest.approx(nu_mean)
    ones = np.ones((4, 8))
    assert np.allclose(norm.normalize_u(ones), 0.25)
    assert float(norm.normalize_nu(0.08)) == pytest.approx(1.0)
    datasets = build_window_datasets(
        [Trajectory(1, "train", 0.04, ones), Trajectory(4, "test", 10.0, ones * 3.0)],
        WindowSpec(2, 2, 2),
        norm,
    )
    assert datasets["test"].instance_id_set() == {4}
    assert float(datasets["test"].nu[0]) == pytest.approx((10.0 - 0.06) / 0.02)
    assert float(np.mean(datasets["train"].inputs)) == pytest.approx(0.25)


def test_overlapping_manifest_splits_are_rejected() -> None:
    manifest = {
        "splits": {"train": [1], "val": [1], "test": [2]},
        "instances": [],
    }
    with pytest.raises(ValueError, match="share instance"):
        split_sets(manifest)


def test_committed_pilot_split_matches_assign_splits() -> None:
    manifest = load_pilot_manifest(MANIFEST)
    sets = split_sets(manifest)
    expected = assign_splits(512, 128, 128, 20260926)
    assert sets["train"] == set(expected["train"])
    assert sets["val"] == set(expected["val"])
    assert sets["test"] == set(expected["test"])
    assert len(sets["train"]) == 512
    assert len(sets["val"]) == 128
    assert len(sets["test"]) == 128
    assert set().union(*sets.values()) == set(range(768))
    norm = field_norm_from_manifest(manifest)
    assert norm.u_mean == pytest.approx(1.2021494137744596e-18)
    assert norm.u_std == pytest.approx(0.3469596293138632)
    assert manifest["normalization"]["n_values"] == 13238272
    assert manifest["normalization"]["fit_on"] == "train"
    assert manifest["pilot"]["windowing"] == "none"
    nu_values = [
        float(item["nu"])
        for item in manifest["instances"]
        if item["split"] == "train"
    ]
    assert norm.nu_mean == pytest.approx(float(np.mean(nu_values)))
    assert norm.nu_std == pytest.approx(float(np.std(np.asarray(nu_values), ddof=0)))


def test_persistence_repeats_the_last_input_frame() -> None:
    datasets = build_window_datasets([_trajectory(7, "test")], WindowSpec(4, 3, 4), NORM)
    dataset = datasets["test"]
    predicted = persistence_prediction(dataset)
    last = dataset.norm.denormalize_u(dataset.inputs[:, -1:, :])
    assert predicted.shape == dataset.targets.shape
    assert np.allclose(predicted, np.repeat(last, 3, axis=1))
    assert mean_relative_l2(physical_targets(dataset), physical_targets(dataset)) == pytest.approx(0.0)


def test_loader_does_not_require_unread_splits(tmp_path: Path) -> None:
    spec = WindowSpec(4, 4, 4)
    _write_pilot(tmp_path, spec, include_test_file=False)
    loaded = load_split_windows(tmp_path, tmp_path / "manifest.json", spec, ("train", "val"))
    assert loaded["train"].instance_id_set() == {1, 4}
    assert loaded["val"].instance_id_set() == {2}
    assert loaded["train"].instance_id_set().isdisjoint(loaded["val"].instance_id_set())
    with pytest.raises(ValueError, match="missing"):
        load_split_windows(tmp_path, tmp_path / "manifest.json", spec, ("test",))


def test_field_hash_mismatch_is_rejected(tmp_path: Path) -> None:
    spec = WindowSpec(4, 4, 4)
    _write_pilot(tmp_path, spec, include_test_file=True)
    manifest_path = tmp_path / "manifest.json"
    payload = json.loads(manifest_path.read_text(encoding="utf-8"))
    payload["instances"][0]["field_sha256"] = "0" * 64
    manifest_path.write_text(json.dumps(payload), encoding="utf-8")
    with pytest.raises(ValueError, match="field_sha256"):
        load_split_windows(tmp_path, manifest_path, spec, ("train",))


def test_operator_help_does_not_import_torch() -> None:
    env = os.environ.copy()
    env["PYTHONPATH"] = str(ROOT / "src")
    code = (
        "import pinnforge.operator, pinnforge.operator.windows, pinnforge.operator.__main__, sys; "
        "assert 'torch' not in sys.modules"
    )
    completed = subprocess.run(
        [sys.executable, "-c", code],
        check=False,
        capture_output=True,
        text=True,
        env=env,
    )
    assert completed.returncode == 0, completed.stderr


@pytest.mark.ml
def test_cli_train_does_not_read_the_test_split(tmp_path: Path) -> None:
    pytest.importorskip("torch")
    from pinnforge.operator.__main__ import main as operator_main

    spec = WindowSpec(4, 4, 8)
    pilot = tmp_path / "pilot"
    pilot.mkdir()
    _write_pilot(pilot, spec, include_test_file=False)
    output = tmp_path / "run"
    code = operator_main(
        [
            "train",
            "--pilot",
            str(pilot),
            "--manifest",
            str(pilot / "manifest.json"),
            "--output",
            str(output),
            "--epochs",
            "1",
            "--width",
            "4",
            "--modes",
            "2",
            "--layers",
            "2",
            "--input-frames",
            "4",
            "--output-frames",
            "4",
            "--stride",
            "8",
            "--batch-size",
            "2",
            "--seed",
            "0",
        ]
    )
    assert code == 0
    manifest = json.loads((output / "manifest.json").read_text(encoding="utf-8"))
    assert manifest["test_used_for_training"] is False
    assert manifest["test_used_for_selection"] is False
    assert manifest["physics_residual"] is False
    assert manifest["loss"] == "mean squared error in normalized u space"
    assert manifest["train_instance_ids"] == [1, 4]
    assert manifest["val_instance_ids"] == [2]
    assert 3 not in manifest["train_instance_ids"]
    assert (output / "checkpoint.pt").is_file()
    _add_test_file(pilot)
    eval_code = operator_main(
        [
            "eval",
            "--pilot",
            str(pilot),
            "--manifest",
            str(pilot / "manifest.json"),
            "--checkpoint",
            str(output / "checkpoint.pt"),
            "--split",
            "test",
            "--batch-size",
            "2",
        ]
    )
    assert eval_code == 0
    record = json.loads((output / "eval_test.json").read_text(encoding="utf-8"))
    assert record["split"] == "test"
    assert record["instance_ids"] == [3]
    assert record["n_instances"] == 1


def test_fno_help_lists_train_and_eval(capsys: pytest.CaptureFixture[str]) -> None:
    with pytest.raises(SystemExit) as caught:
        main(["fno", "--help"])
    assert caught.value.code == 0
    text = capsys.readouterr().out
    assert "train" in text
    assert "eval" in text
    with pytest.raises(SystemExit) as train_help:
        main(["fno", "train", "--help"])
    assert train_help.value.code == 0
    train_text = capsys.readouterr().out
    assert "--manifest" in train_text
    assert "--pilot" in train_text


def _add_test_file(pilot: Path) -> None:
    field = _field(3, n_times=24, n_space=8)
    path = pilot / "test" / "instance_000003.npz"
    path.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(path, u=field, nu=np.float64(0.09), instance_id=np.int64(3))


def _write_pilot(root: Path, spec: WindowSpec, *, include_test_file: bool) -> None:
    rows = [
        (1, "train", 0.04),
        (4, "train", 0.08),
        (2, "val", 0.06),
        (3, "test", 0.09),
    ]
    instances = []
    for instance_id, split, nu in rows:
        relative = Path(split) / f"instance_{instance_id:06d}.npz"
        field = _field(instance_id, n_times=24, n_space=8)
        path = root / relative
        if split != "test" or include_test_file:
            path.parent.mkdir(parents=True, exist_ok=True)
            np.savez_compressed(
                path,
                u=field,
                nu=np.float64(nu),
                instance_id=np.int64(instance_id),
            )
        instances.append(
            {
                "instance_id": instance_id,
                "split": split,
                "nu": nu,
                "path": relative.as_posix(),
                "n": 8,
                "n_times": 24,
                "field_sha256": _field_sha256(field),
            }
        )
    manifest = {
        "format": "pinnforge.burgers_pilot.v1",
        "normalization": {
            "fit_on": "train",
            "applied_to_files": False,
            "u_mean": 0.0,
            "u_std": 1.0,
        },
        "splits": {"train": [1, 4], "val": [2], "test": [3]},
        "instances": instances,
    }
    (root / "manifest.json").write_text(json.dumps(manifest), encoding="utf-8")
    del spec
