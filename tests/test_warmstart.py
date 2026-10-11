"""Warm-started viscosity fit, conformal intervals, and validation-only selection."""

from __future__ import annotations

import json
import math
from pathlib import Path

import numpy as np
import pytest

from pinnforge.operator.warmstart import (
    GROUP_NAMES,
    RefineConfig,
    abstain_threshold,
    calibrate_groups,
    conformal_quantile,
    conformal_score,
    instance_scale,
    interval,
    predicted_group,
    refine_viscosity,
    step_at_cap,
)
from pinnforge.operator.warmstart_run import (
    crossfit_abstain,
    crossfit_interval,
    load_warmstart_protocol,
    select_cap,
    warm_view,
)

ROOT = Path(__file__).resolve().parents[1]


def _linear_residual(nu_star: float, curvature: float = 0.0):
    weights = np.linspace(0.5, 1.5, 40)

    def residual(nu: float) -> np.ndarray:
        shift = math.log(nu / nu_star)
        return weights * shift + curvature * weights[::-1] * shift * shift

    return residual


def test_refine_recovers_linear_minimum_in_three_solves() -> None:
    calls: list[float] = []
    base = _linear_residual(0.02)

    def residual(nu: float) -> np.ndarray:
        calls.append(nu)
        return base(nu)

    result = refine_viscosity(residual, 0.024, RefineConfig(max_evaluations=3))
    assert result.n_evaluations == len(calls) <= 3
    assert abs(result.nu_hat / 0.02 - 1.0) < 1e-9
    assert result.stopped_on_tolerance
    assert not result.at_trust_bound


def test_refine_trace_prefix_matches_smaller_cap() -> None:
    residual = _linear_residual(0.015, curvature=0.8)
    long = refine_viscosity(residual, 0.03, RefineConfig(max_evaluations=6)).to_dict()
    for cap in (2, 3, 4, 5):
        short = refine_viscosity(residual, 0.03, RefineConfig(max_evaluations=cap))
        step = step_at_cap(long, cap)
        assert step["nu_hat"] == short.nu_hat
        assert step["n_evaluations"] == short.n_evaluations
        assert step["log_standard_error"] == short.log_standard_error


def test_refine_stays_in_trust_box() -> None:
    config = RefineConfig(max_evaluations=4, trust_radius=math.log(2.0))
    result = refine_viscosity(_linear_residual(0.006), 0.05, config)
    assert result.at_trust_bound
    assert result.nu_hat == pytest.approx(0.025, rel=1e-12)
    assert all(0.025 * (1 - 1e-12) <= nu <= 0.1 for nu, _ in result.queries)


def test_refine_probe_turns_back_at_upper_bound() -> None:
    result = refine_viscosity(_linear_residual(0.09), 0.1, RefineConfig(max_evaluations=3))
    assert result.queries[1][0] < 0.1
    assert result.nu_hat == pytest.approx(0.09, rel=1e-9)


def test_refine_config_rejects_bad_values() -> None:
    with pytest.raises(ValueError):
        RefineConfig(max_evaluations=1)
    with pytest.raises(ValueError):
        RefineConfig(probe_step=1.0, trust_radius=0.5)
    with pytest.raises(ValueError):
        RefineConfig(return_mode="median")
    config = RefineConfig(max_evaluations=4)
    assert RefineConfig.from_mapping(config.to_dict()) == config


def test_conformal_quantile_rank() -> None:
    scores = [float(value) for value in range(1, 10)]
    assert conformal_quantile(scores, 0.1) == 9.0
    assert conformal_quantile(scores[:5], 0.1) == math.inf
    assert conformal_quantile([float(value) for value in range(1, 20)], 0.1) == 18.0


def test_calibrate_groups_falls_back_to_pooled() -> None:
    rows = [{"score": float(index), "group": GROUP_NAMES[1]} for index in range(30)]
    rows += [{"score": 100.0 + index, "group": GROUP_NAMES[0]} for index in range(3)]
    fitted = calibrate_groups(rows, 0.1, min_group=10)
    assert fitted["groups"][GROUP_NAMES[0]]["pooled_fallback"] is True
    assert fitted["groups"][GROUP_NAMES[0]]["quantile"] == fitted["pooled"]["quantile"]
    assert fitted["groups"][GROUP_NAMES[1]]["pooled_fallback"] is False
    assert fitted["groups"][GROUP_NAMES[1]]["quantile"] == 27.0


def test_interval_score_and_group() -> None:
    low, high = interval(0.02, 0.5, 0.2)
    assert low == pytest.approx(0.02 * math.exp(-0.1))
    assert high == pytest.approx(0.02 * math.exp(0.1))
    assert conformal_score(0.02 * math.exp(0.1), 0.02, 0.5) == pytest.approx(0.2)
    assert predicted_group(0.02, 0.027) == GROUP_NAMES[0]
    assert predicted_group(0.03, 0.027) == GROUP_NAMES[1]
    assert instance_scale("none", log_standard_error=5.0, ensemble_log_std=5.0, floor=1e-3) == 1.0
    assert instance_scale("gauss_newton", log_standard_error=0.0, ensemble_log_std=5.0, floor=1e-3) == pytest.approx(1e-3)


def test_abstain_threshold_rate() -> None:
    values = [float(value) for value in range(100)]
    cut = abstain_threshold(values, 0.1)
    assert sum(value > cut for value in values) == 10
    assert abstain_threshold(values, 0.0) == 99.0


def _fake_row(instance_id: int, nu: float, hats: dict[str, list[float]], cold_seconds: float = 10.0) -> dict:
    warm = {}
    for key, values in hats.items():
        trace = [
            {
                "n_evaluations": index + 2,
                "nu_hat": value,
                "seconds": float(index + 1),
                "loss_hat": 0.0,
                "log_standard_error": 0.01 * (1 + instance_id % 3),
                "relative_residual": 0.01 * (1 + instance_id % 5),
                "at_trust_bound": False,
                "stopped_on_tolerance": False,
            }
            for index, value in enumerate(values)
        ]
        warm[key] = {"nu_guess": nu * 1.1, "first_guess_seconds": 0.1, "seconds": float(len(values)), "trace": trace}
    return {"instance_id": instance_id, "nu": nu, "cold": {"nu_hat": nu, "seconds": cold_seconds, "n_evaluations": 18}, "warm": warm}


def test_select_cap_prefers_smallest_admissible_cap() -> None:
    keys = ["full/hybrid_1e-2/0", "full/hybrid_1e-2/1"]
    rows = []
    for index in range(20):
        nu = 0.01 + 0.0005 * index
        hats = {key: [nu * 1.10, nu * 1.01, nu * 1.0099, nu * 1.0, nu * 1.0] for key in keys}
        rows.append(_fake_row(index, nu, hats))
    chosen = select_cap(rows, keys, 0.027, caps=(2, 3, 4, 5, 6), guard=0.35, slack=1.02)
    assert chosen["selected_cap"] == 3
    times = {item["cap"]: item["time_ratio"] for item in chosen["table"]}
    assert times[2] == pytest.approx(0.11)
    tight = select_cap(rows, keys, 0.027, caps=(2, 3, 4, 5, 6), guard=0.15, slack=1.02)
    assert tight["selected_cap"] == 2


def test_warm_view_uses_full_call_time_at_end_of_trace() -> None:
    row = _fake_row(1, 0.02, {"k": [0.021, 0.0201]})
    row["warm"]["k"]["seconds"] = 2.5
    assert warm_view(row, "k", 3)["seconds"] == pytest.approx(2.6)
    assert warm_view(row, "k", 2)["seconds"] == pytest.approx(1.1)


def test_crossfit_selection_helpers() -> None:
    rng = np.random.default_rng(0)
    views = []
    for index in range(60):
        nu = float(rng.uniform(0.005, 0.1))
        error = float(rng.normal(0.0, 0.02))
        views.append(
            {
                "instance_id": index,
                "nu": nu,
                "nu_hat": nu * math.exp(error),
                "log_standard_error": abs(error) + 0.01,
                "ensemble_log_std": math.nan,
                "relative_residual": abs(error),
                "rel_error": abs(math.exp(error) - 1.0),
            }
        )
    width = crossfit_interval({"a": views}, "none", 0.027)
    assert 0.0 < width["mean_log_half_width"] < 1.0
    assert 0.5 < width["coverage"] <= 1.0
    abstain = crossfit_abstain({"a": views}, "relative_residual")
    overall = float(np.mean([view["rel_error"] for view in views]))
    assert abstain["accepted_mean_rel_error"] < overall


def test_protocol_loader_rejects_scores_and_open_flags(tmp_path: Path) -> None:
    path = tmp_path / "protocol.json"
    path.write_text(json.dumps({"format": "pinnforge.warmstart_protocol.v1", "results": {}}), encoding="utf-8")
    with pytest.raises(ValueError, match="test scores"):
        load_warmstart_protocol(path)
    path.write_text(
        json.dumps(
            {
                "format": "pinnforge.warmstart_protocol.v1",
                "frozen_before_test_evaluation": True,
                "test_used_for_selection": True,
            }
        ),
        encoding="utf-8",
    )
    with pytest.raises(ValueError, match="test_used_for_selection"):
        load_warmstart_protocol(path)


@pytest.mark.ml
def test_batched_sensor_curve_matches_stage_d_objective() -> None:
    import torch

    from pinnforge.operator.fno import FNO1d
    from pinnforge.operator.stage_d_fit import _burst_mse, _rollout_sensors
    from pinnforge.operator.warmstart_fit import first_guess_grid, sensor_curve
    from pinnforge.operator.windows import FieldNorm

    torch.manual_seed(3)
    model = FNO1d(9, 8, width=8, modes=4, n_layers=1)
    norm = FieldNorm(u_mean=0.01, u_std=0.4, nu_mean=0.05, nu_std=0.03)
    rng = np.random.default_rng(1)
    window = rng.normal(size=(8, 64))
    slots = ((0, 2, 0, 5), (1, 5, 3, 5))
    targets = rng.normal(size=(2, 5, 16))
    grid = first_guess_grid("geometric", 7, 0.005, 0.1)
    batched = sensor_curve(model, norm, window, targets, slots, grid, 16)
    reference = []
    for nu in grid:
        blocks = _rollout_sensors(model, norm, window[None], float(nu), 6, 16, 1)
        reference.append(float(_burst_mse(blocks, targets[None], slots, rollout=True)[0]))
    np.testing.assert_allclose(batched, reference, rtol=1e-5)


@pytest.mark.ml
def test_pipeline_smoke(tmp_path: Path) -> None:
    from pinnforge.operator.warmstart_run import cold_view, run_smoke

    row = run_smoke(tmp_path, n_space=64, families=("full",))
    cold = cold_view(row)
    assert cold["n_evaluations"] >= 7
    assert math.isfinite(cold["nu_hat"])
    assert len(row["warm"]) == 12
    view = warm_view(row, "full/hybrid_1e-2/ensemble", 3)
    assert 0.005 <= view["nu_hat"] <= 0.1
    assert view["n_evaluations"] <= 3
    assert view["seconds"] > 0.0
    assert set(row["stage_d_grid"]) == {key for key in row["warm"] if not key.endswith("ensemble")}
