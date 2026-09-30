"""Stage D protocol, sparse window, and validation lambda selection. No torch except where marked."""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pytest

from pinnforge.operator.inverse import (
    PREREGISTERED_OBSERVATION,
    InstanceRecovery,
    recovery_from_rows,
)
from pinnforge.operator.stage_d import (
    NEWTON_AHEAD_OF_FRAME_4,
    RUN_FORMAT,
    extrapolate_initial_window,
    initial_window_from_mask,
    lift_sensors,
    load_objective_selection,
    load_stage_d_protocol,
    mean_std,
    nu_from_objective,
    objective_curve,
    prepare_sparse_batch,
    protocol_sha256,
    read_mask,
    select_objective,
    target_slots,
    viscosity_grid,
    write_stage_d_json,
)
from pinnforge.operator.windows import Trajectory, WindowSpec

ROOT = Path(__file__).resolve().parents[1]
PROTOCOL = ROOT / "docs" / "v02" / "stage_d_inverse_protocol.json"


def test_committed_stage_d_protocol_locks_the_mask_and_the_search() -> None:
    protocol = load_stage_d_protocol(PROTOCOL)
    assert protocol["mask"]["name"] == "sensors32_bursts"
    assert protocol["mask"]["unchanged"] is True
    assert protocol["failure_relative_error"] == 0.5
    assert protocol["hard_ood"]["threshold_nu"] == pytest.approx(0.027028120493367818)
    assert protocol["hard_ood_refit"] is False
    assert protocol["nu_search"]["min"] == pytest.approx(0.005)
    assert protocol["nu_search"]["max"] == pytest.approx(0.1)
    assert protocol["nu_search"]["n_grid"] == 191
    assert protocol["combined_objective"]["lambda_grid"] == [0, 1, 10]
    assert protocol["combined_objective"]["lambda_zero_always_scored"] is True
    assert protocol["combined_objective"]["selection_split"] == "val"
    assert protocol["arms"]["data_only"]["loss"] == "data"
    assert protocol["arms"]["hybrid_1e-2"]["residual_weight"] == pytest.approx(0.01)
    assert protocol["arms"]["data_only"]["seeds"] == [0, 1, 2, 3, 4]
    assert protocol["analysis_ceiling"]["used_for_selection"] is False
    assert protocol["estimator"]["reads_unmasked_field"] is False
    assert protocol["estimator"]["initial_window"]["nu_dependent"] is False
    grid = viscosity_grid(protocol)
    assert grid.size == 191
    assert grid[0] == pytest.approx(0.005)
    assert grid[-1] == pytest.approx(0.1)
    assert "results" not in protocol
    assert "scores" not in protocol


def test_committed_objective_selection_is_validation_only() -> None:
    protocol = load_stage_d_protocol(PROTOCOL)
    selection = load_objective_selection(
        ROOT / "docs" / "v02" / "stage_d_objective_selection.json",
        protocol,
        protocol_sha256(PROTOCOL),
    )
    assert selection["test_used_for_selection"] is False
    assert selection["test_splits_read"] is False
    assert selection["selection_split"] == "val"
    assert selection["selection_slice"] == "hard_ood"
    for arm in ("data_only", "hybrid_1e-2"):
        assert selection["arms"][arm]["selected_lambda"] == pytest.approx(0.0)
        assert selection["arms"][arm]["selected_mean_hard_ood_mean_rel_error"] > 0.0


def test_committed_stage_d_scores_keep_the_stage_a_baselines() -> None:
    scores = json.loads((ROOT / "docs" / "v02" / "stage_d_scores.json").read_text(encoding="utf-8"))
    stress = json.loads((ROOT / "docs" / "v02" / "inverse_stress.json").read_text(encoding="utf-8"))
    assert scores["format"] == "pinnforge.stage_d_scores.v1"
    assert scores["protocol_sha256"] == protocol_sha256(PROTOCOL)
    assert scores["selection_path"] == "docs/v02/stage_d_objective_selection.json"
    assert scores["oracle_used_for_selection"] is False
    assert scores["threshold_nu"] == pytest.approx(0.027028120493367818)
    published = scores["baselines"]["sensors32_bursts"]["hard_ood"]
    reference = stress["hard_ood_sensors32_bursts"]
    assert published["n_instances"] == 30
    assert published["n_failures"] == 9
    assert published["mean_rel_error"] == pytest.approx(reference["mean_rel_error"])
    assert published["mean_abs_error"] == pytest.approx(reference["mean_abs_error"])
    for arm in ("data_only", "hybrid_1e-2"):
        block = scores["arms"][arm]
        assert block["selected_lambda"] == pytest.approx(0.0)
        assert set(block["objectives"]) == {"0.0"}
        hard = block["objectives"]["0.0"]["slices"]["hard_ood"]
        assert hard["n_instances"] == 30
        assert hard["n_failures"]["per_seed"] == [0, 0, 0, 0, 0]
        assert hard["mean_rel_error"]["mean"] < reference["mean_rel_error"]


def test_stage_d_protocol_rejects_a_file_that_already_holds_scores(tmp_path: Path) -> None:
    payload = json.loads(PROTOCOL.read_text(encoding="utf-8"))
    payload["results"] = {"hard_ood": 0.0}
    path = tmp_path / "protocol.json"
    path.write_text(json.dumps(payload), encoding="utf-8")
    with pytest.raises(ValueError, match="test scores"):
        load_stage_d_protocol(path)


def test_lift_roundtrip_keeps_modes_below_the_sensor_nyquist() -> None:
    n_space = 64
    nodes = np.arange(n_space)
    field = np.cos(2.0 * np.pi * 3.0 * nodes / n_space) + 0.25 * np.sin(2.0 * np.pi * 5.0 * nodes / n_space)
    lifted = lift_sensors(field[::4], n_space)
    assert lifted.shape == (n_space,)
    assert np.allclose(lifted, field, atol=1e-10)


def test_quadratic_extrapolation_matches_the_locked_newton_step() -> None:
    frames = (np.arange(5, dtype=np.float64) ** 2)[:, None]
    window = extrapolate_initial_window(frames)
    assert window.shape == (8, 1)
    assert window[5, 0] == pytest.approx(25.0)
    assert window[6, 0] == pytest.approx(36.0)
    assert window[7, 0] == pytest.approx(49.0)
    assert NEWTON_AHEAD_OF_FRAME_4[3] == (1.0, -3.0, 3.0)


def test_sparse_window_ignores_samples_off_the_mask() -> None:
    rng = np.random.default_rng(0)
    field = rng.normal(size=(101, 64))
    poisoned = field.copy()
    poisoned[5, 1] += 10.0
    poisoned[10, 0] += 3.0
    spec = PREREGISTERED_OBSERVATION
    original = initial_window_from_mask(read_mask(field, spec), 64)
    changed = initial_window_from_mask(read_mask(poisoned, spec), 64)
    assert np.allclose(original, changed)
    poisoned[0, 0] += 1.0
    assert not np.allclose(initial_window_from_mask(read_mask(poisoned, spec), 64), original)


def test_target_bursts_land_in_predicted_blocks() -> None:
    slots = target_slots(PREREGISTERED_OBSERVATION, WindowSpec())
    assert slots == ((1, 2, 0, 5), (2, 5, 0, 5), (3, 8, 0, 5))


def test_lambda_zero_ignores_the_residual_and_ties_keep_the_smaller_nu() -> None:
    grid = np.linspace(0.005, 0.1, 191)
    sensor = np.ones(grid.size)
    sensor[10] = 0.2
    sensor[11] = 0.2
    residual = np.linspace(1.0, 0.0, grid.size)
    pure = nu_from_objective(objective_curve(sensor, residual, 1.0, 1.0, 0.0), grid)
    mixed = nu_from_objective(objective_curve(np.ones(grid.size), residual, 1.0, 1.0, 10.0), grid)
    assert pure == pytest.approx(grid[10])
    assert mixed == pytest.approx(grid[-1])


def test_prepare_sparse_batch_uses_only_the_opening_burst_for_the_window() -> None:
    field = np.zeros((101, 64), dtype=np.float64)
    field[0:5, ::2] = 1.0
    trajectory = Trajectory(instance_id=3, split="val", nu=0.04, u=field)
    prepared = prepare_sparse_batch([trajectory], PREREGISTERED_OBSERVATION, WindowSpec())
    assert prepared["windows"].shape == (1, 8, 64)
    assert np.allclose(prepared["windows"][0, 0], 1.0)
    assert prepared["targets"].shape[0] == 1
    assert prepared["targets"].shape[1] == 3


def test_select_objective_uses_validation_hard_ood_and_the_smaller_lambda(tmp_path: Path) -> None:
    digest = protocol_sha256(PROTOCOL)
    paths = []
    for arm in ("data_only", "hybrid_1e-2"):
        for seed in range(5):
            path = tmp_path / f"{arm}_seed_{seed}.json"
            write_stage_d_json(_synthetic_run(arm, seed, digest), path)
            paths.append(path)
    payload = select_objective(paths, PROTOCOL)
    for arm in ("data_only", "hybrid_1e-2"):
        assert payload["arms"][arm]["selected_lambda"] == pytest.approx(1.0)
        assert payload["arms"][arm]["selected_mean_hard_ood_mean_rel_error"] == pytest.approx(0.0)
    assert payload["test_splits_read"] is False
    leaked = tmp_path / "test_run.json"
    write_stage_d_json(_synthetic_run("data_only", 0, digest, split="test"), leaked)
    with pytest.raises(ValueError, match="validation"):
        select_objective([leaked], PROTOCOL)


def test_mean_std_matches_the_sample_definition() -> None:
    center, spread = mean_std([1.0, 2.0, 3.0])
    assert center == pytest.approx(2.0)
    assert spread == pytest.approx(float(np.std([1.0, 2.0, 3.0], ddof=1)))


def test_fno_help_lists_stage_d_commands(capsys: pytest.CaptureFixture[str]) -> None:
    from pinnforge.cli import main

    with pytest.raises(SystemExit) as caught:
        main(["fno", "--help"])
    assert caught.value.code == 0
    text = capsys.readouterr().out
    assert "operator-inverse" in text
    assert "select-objective" in text
    assert "stage-d-scores" in text


def test_recovery_rows_keep_a_supplied_estimate() -> None:
    rows = [
        InstanceRecovery(
            instance_id=2,
            nu=0.02,
            nu_hat=0.03,
            abs_error=0.01,
            rel_error=0.5,
            baseline_abs_error=0.03,
            baseline_rel_error=1.5,
            mean_abs_residual=0.1,
            mean_abs_residual_at_truth=0.2,
            mean_square_residual=0.3,
            mean_square_residual_at_truth=0.4,
            failure=False,
        ),
        InstanceRecovery(
            instance_id=1,
            nu=0.04,
            nu_hat=0.04,
            abs_error=0.0,
            rel_error=0.0,
            baseline_abs_error=0.01,
            baseline_rel_error=0.25,
            mean_abs_residual=0.1,
            mean_abs_residual_at_truth=0.1,
            mean_square_residual=0.2,
            mean_square_residual_at_truth=0.2,
            failure=False,
        ),
    ]
    score = recovery_from_rows("sparse_rollout", 0.05, rows)
    assert [row.instance_id for row in score.instances] == [1, 2]
    assert score.instances[1].nu_hat == pytest.approx(0.03)
    assert score.n_failures == 0
    assert score.mean_rel_error == pytest.approx(0.25)


def _synthetic_run(arm: str, seed: int, digest: str, *, split: str = "val") -> dict[str, object]:
    # lambda 0 minimizes at 0.02 (relative error 1). lambda >= 1 minimizes at 0.01.
    return {
        "format": RUN_FORMAT,
        "split": split,
        "arm": arm,
        "seed": seed,
        "selected_epoch": 1,
        "val_relative_l2": 0.01,
        "loss_mode": "data" if arm == "data_only" else "hybrid",
        "residual_weight": 0.0 if arm == "data_only" else 0.01,
        "checkpoint": f"runs/{arm}/seed_{seed}/checkpoint.pt",
        "baseline_nu": 0.05,
        "grid": [0.01, 0.02, 0.03],
        "protocol_sha256": digest,
        "test_used_for_curves": split == "test",
        "reads_unmasked_field_for_sparse_objective": False,
        "instances": [
            {
                "instance_id": 7,
                "nu": 0.01,
                "alias_rel_l2": 0.4,
                "sensor_mse": [1.0, 0.0, 1.0],
                "sensor_mse_at_baseline": 1.0,
                "residual_mse": [0.0, 1.0, 1.0],
                "residual_mse_at_baseline": 1.0,
                "oracle_sensor_mse": [0.0, 1.0, 1.0],
                "oracle_sensor_mse_at_baseline": 1.0,
                "advection": [0.0, 0.0],
                "diffusion": [1.0, 1.0],
            }
        ],
    }


@pytest.mark.ml
def test_curve_evaluation_changes_with_viscosity() -> None:
    import torch

    from pinnforge.operator.fno import FNO1d
    from pinnforge.operator.stage_d_fit import evaluate_curves
    from pinnforge.operator.windows import FieldNorm

    torch.manual_seed(0)
    model = FNO1d(in_channels=9, out_channels=8, width=4, modes=2, n_layers=1)
    norm = FieldNorm(u_mean=0.0, u_std=1.0, nu_mean=0.05, nu_std=0.02)
    windows = np.zeros((2, 8, 32), dtype=np.float64)
    windows[:, :, :] = np.linspace(-0.2, 0.2, 32)
    targets = np.zeros((2, 1, 5, 32), dtype=np.float64)
    oracle = np.broadcast_to(windows[:, None, :, :], (2, 1, 8, 32)).copy()
    grid = np.asarray([0.01, 0.05, 0.09], dtype=np.float64)
    curves = evaluate_curves(
        model,
        norm,
        windows,
        targets,
        ((0, 0, 0, 5),),
        grid,
        0.05,
        oracle,
        batch_size=2,
        n_sensors=32,
    )
    assert curves["sensor_mse"].shape == (2, 3)
    assert np.isfinite(curves["sensor_mse"]).all()
    assert np.any(np.ptp(curves["sensor_mse"], axis=1) > 0.0)
