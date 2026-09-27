"""Sparse viscosity recovery. No torch."""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pytest

from pinnforge.cli import main
from pinnforge.operator.inverse import (
    ABLATION_OBSERVATION,
    FAILURE_RELATIVE_ERROR,
    PREREGISTERED_BURST_LENGTH,
    PREREGISTERED_BURST_STARTS,
    PREREGISTERED_FRAME_DT,
    PREREGISTERED_N_SENSORS,
    PREREGISTERED_OBSERVATION,
    ObservationSpec,
    dense_reference_observation,
    least_squares_viscosity,
    recover_split,
    score_trajectories,
    training_baseline_nu,
)
from pinnforge.operator.residual import central_burgers_residual
from pinnforge.operator.windows import Trajectory, WindowSpec, window_starts
from pinnforge.reference.numerical.checks import cole_hopf
from pinnforge.reference.numerical.dataset import _field_sha256
from pinnforge.reference.numerical.solver import grid


def test_preregistered_sensors_stay_locked() -> None:
    spec = PREREGISTERED_OBSERVATION
    assert spec.name == "sensors32_bursts"
    assert spec.n_sensors == PREREGISTERED_N_SENSORS == 32
    assert spec.frame_dt == PREREGISTERED_FRAME_DT == 0.01
    assert spec.series == tuple(
        tuple(range(start, start + PREREGISTERED_BURST_LENGTH)) for start in PREREGISTERED_BURST_STARTS
    )
    assert PREREGISTERED_BURST_STARTS == (0, 24, 48, 72)
    assert PREREGISTERED_BURST_LENGTH == 5
    assert FAILURE_RELATIVE_ERROR == 0.5
    assert spec.n_observed_frames() == 20
    assert spec.n_residual_times() == 12
    assert spec.check_grid(101, 256) == 32
    starts = window_starts(101, WindowSpec())
    for burst_start in PREREGISTERED_BURST_STARTS:
        assert burst_start in starts


def test_ablation_uses_the_stage3_stride_as_its_clock() -> None:
    spec = ABLATION_OBSERVATION
    assert spec.name == "sensors32_stride8"
    assert spec.n_sensors == 32
    assert spec.series == (tuple(range(0, 97, 8)),)
    assert spec.series_dt(spec.series[0]) == pytest.approx(0.08)
    assert spec.maximum_index() == 96


def test_observation_spec_rejects_a_bad_series() -> None:
    with pytest.raises(ValueError, match="at least 3"):
        ObservationSpec(name="short", n_sensors=8, series=((0, 1),))
    with pytest.raises(ValueError, match="constant gap"):
        ObservationSpec(name="gap", n_sensors=8, series=((0, 1, 3),))
    with pytest.raises(ValueError, match="share frame"):
        ObservationSpec(name="overlap", n_sensors=8, series=((0, 1, 2), (2, 3, 4)))
    with pytest.raises(ValueError, match="even"):
        ObservationSpec(name="odd", n_sensors=5, series=((0, 1, 2),))
    with pytest.raises(ValueError, match="n_times"):
        PREREGISTERED_OBSERVATION.check_grid(10, 256)


def test_least_squares_matches_the_stage4_residual() -> None:
    field = _cole_field(n_space=32, n_times=5, nu=0.05, dt=0.01)
    spec = ObservationSpec(name="full", n_sensors=None, series=(tuple(range(5)),))
    hat = least_squares_viscosity(field, spec)
    batch = field[None, :, :]
    direct = central_burgers_residual(batch, np.array([hat]), dt=0.01)[0]
    probed = central_burgers_residual(batch, np.array([1.0]), dt=0.01)[0]
    probed_two = central_burgers_residual(batch, np.array([2.0]), dt=0.01)[0]
    diffusion = probed - probed_two
    advection = probed + diffusion
    assert np.allclose(advection - hat * diffusion, direct, rtol=1e-10, atol=1e-12)
    worse = central_burgers_residual(batch, np.array([hat + 0.01]), dt=0.01)
    assert np.mean(direct * direct) < np.mean(worse * worse)


def test_cole_hopf_sparse_recovery_beats_a_constant_baseline() -> None:
    nu = 0.05
    field = _cole_field(n_space=64, n_times=9, nu=nu, dt=0.01)
    spec = ObservationSpec(name="sensors", n_sensors=32, series=((0, 1, 2, 3), (4, 5, 6, 7, 8)))
    hat = least_squares_viscosity(field, spec)
    assert abs(hat - nu) < 1e-3
    baseline = 0.09
    trajectory = Trajectory(instance_id=7, split="test", nu=nu, u=field)
    score = score_trajectories([trajectory], spec, baseline)
    assert score.n_instances == 1
    assert score.correlation is None
    assert score.instances[0].nu_hat == pytest.approx(hat)
    assert score.mean_abs_error < score.baseline_mean_abs_error
    assert score.instances[0].mean_square_residual <= score.instances[0].mean_square_residual_at_truth
    assert score.n_failures == 0
    assert score.n_worse_than_baseline == 0


def test_constant_field_is_not_identifiable() -> None:
    field = np.full((6, 16), 0.2, dtype=np.float64)
    spec = ObservationSpec(name="flat", n_sensors=16, series=(tuple(range(6)),))
    with pytest.raises(ValueError, match="not identifiable"):
        least_squares_viscosity(field, spec)


def test_label_does_not_enter_the_estimate() -> None:
    field = _cole_field(n_space=32, n_times=5, nu=0.04, dt=0.01)
    spec = ObservationSpec(name="full", n_sensors=None, series=(tuple(range(5)),))
    labeled = Trajectory(instance_id=1, split="test", nu=0.04, u=field)
    mislabeled = Trajectory(instance_id=1, split="test", nu=0.09, u=field)
    left = score_trajectories([labeled], spec, 0.06)
    right = score_trajectories([mislabeled], spec, 0.06)
    assert left.instances[0].nu_hat == pytest.approx(right.instances[0].nu_hat)
    assert left.instances[0].abs_error < right.instances[0].abs_error


def test_training_baseline_reads_the_manifest_mean(tmp_path: Path) -> None:
    manifest = tmp_path / "manifest.json"
    manifest.write_text(json.dumps(_manifest_payload()), encoding="utf-8")
    assert training_baseline_nu(manifest) == pytest.approx((0.04 + 0.08) / 2.0)


def test_inverse_cli_scores_test_without_train_files(tmp_path: Path) -> None:
    pilot = tmp_path / "pilot"
    nu = 0.05
    field = _cole_field(n_space=64, n_times=101, nu=nu, dt=0.01)
    relative = Path("test") / "instance_000003.npz"
    path = pilot / relative
    path.parent.mkdir(parents=True)
    np.savez_compressed(path, u=field, nu=np.float64(nu), instance_id=np.int64(3))
    payload = _manifest_payload()
    payload["instances"][3]["field_sha256"] = _field_sha256(field)
    payload["instances"][3]["n"] = 64
    payload["instances"][3]["n_times"] = 101
    manifest = pilot / "manifest.json"
    manifest.write_text(json.dumps(payload), encoding="utf-8")
    output = tmp_path / "eval_test.json"
    code = main(
        [
            "fno",
            "inverse",
            "--pilot",
            str(pilot),
            "--manifest",
            str(manifest),
            "--split",
            "test",
            "--output",
            str(output),
        ]
    )
    assert code == 0
    record = json.loads(output.read_text(encoding="utf-8"))
    assert record["split"] == "test"
    assert record["format"] == "pinnforge.burgers_inverse.v1"
    assert record["primary"]["pattern"] == "sensors32_bursts"
    assert record["baseline_nu"] == pytest.approx(0.06)
    assert record["primary"]["n_instances"] == 1
    assert record["primary"]["mean_abs_error"] < record["primary"]["baseline_mean_abs_error"]
    assert record["primary"]["n_failures"] == 0
    assert abs(record["primary"]["instances"][0]["nu_hat"] - nu) < 1e-3
    assert record["ablation"]["pattern"] == "sensors32_stride8"
    assert record["dense_reference"]["pattern"] == "dense_reference"
    assert not (pilot / "train").exists()
    recovered = recover_split(pilot, manifest, PREREGISTERED_OBSERVATION, split="test")
    assert recovered.mean_abs_error == pytest.approx(record["primary"]["mean_abs_error"])


def test_dense_reference_uses_every_frame() -> None:
    spec = dense_reference_observation(7, frame_dt=0.01)
    assert spec.n_sensors is None
    assert spec.series == (tuple(range(7)),)
    with pytest.raises(ValueError, match="at least 3"):
        dense_reference_observation(2)


def _cole_field(n_space: int, n_times: int, nu: float, dt: float) -> np.ndarray:
    x = grid(n_space)
    times = dt * np.arange(n_times, dtype=np.float64)
    return np.stack([cole_hopf(x, time, nu=nu) for time in times], axis=0)


def _manifest_payload() -> dict[str, object]:
    return {
        "format": "pinnforge.burgers_pilot.v1",
        "normalization": {
            "fit_on": "train",
            "applied_to_files": False,
            "u_mean": 0.0,
            "u_std": 1.0,
        },
        "splits": {"train": [1, 4], "val": [2], "test": [3]},
        "instances": [
            {
                "instance_id": 1,
                "split": "train",
                "nu": 0.04,
                "path": "train/instance_000001.npz",
                "n": 64,
                "n_times": 101,
                "field_sha256": "unused",
            },
            {
                "instance_id": 4,
                "split": "train",
                "nu": 0.08,
                "path": "train/instance_000004.npz",
                "n": 64,
                "n_times": 101,
                "field_sha256": "unused",
            },
            {
                "instance_id": 2,
                "split": "val",
                "nu": 0.07,
                "path": "val/instance_000002.npz",
                "n": 64,
                "n_times": 101,
                "field_sha256": "unused",
            },
            {
                "instance_id": 3,
                "split": "test",
                "nu": 0.05,
                "path": "test/instance_000003.npz",
                "n": 64,
                "n_times": 101,
                "field_sha256": "placeholder",
            },
        ],
    }
