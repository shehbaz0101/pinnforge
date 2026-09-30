"""Stage E protocol, noise model, and breakdown rule. Torch only where marked."""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pytest

from pinnforge.operator.inverse import PREREGISTERED_OBSERVATION
from pinnforge.operator.stage_e import (
    SLICE_NAMES,
    apply_noise,
    axis_conditions,
    breakdown_table,
    coarse_indexes,
    field_std,
    is_failure,
    load_stage_e_protocol,
    local_minimum_count,
    noise_seeds_for,
    observation_spec,
    operator_applicable,
    sensor_columns,
    standard_normal_field,
    unimodal_grid_index,
    window_spec,
    write_stress_chart,
)
from pinnforge.operator.windows import WindowSpec

ROOT = Path(__file__).resolve().parents[1]
PROTOCOL = ROOT / "docs" / "v02" / "stage_e_stress_protocol.json"


def test_committed_stage_e_protocol_locks_the_stress_grid() -> None:
    protocol = load_stage_e_protocol(PROTOCOL)
    assert protocol["frozen_before_test_evaluation"] is True
    assert protocol["test_used_for_selection"] is False
    assert protocol["hard_ood_refit"] is False
    assert protocol["lambda_retuned"] is False
    assert protocol["failure_relative_error"] == 0.5
    assert protocol["hard_ood"]["threshold_nu"] == pytest.approx(0.027028120493367818)
    assert protocol["objective"]["lambda"] == pytest.approx(0.0)
    assert protocol["noise"]["fractions"] == [0.0, 0.001, 0.005, 0.01, 0.02, 0.05]
    assert protocol["noise"]["noise_seeds"] == [0, 1, 2]
    assert protocol["noise"]["primary_noise_seed"] == 0
    assert protocol["noise"]["operator_noise_seeds"] == [0]
    assert protocol["sensors"]["counts"] == [32, 16, 8]
    assert protocol["nu_search"]["n_grid"] == 191
    assert protocol["operator"]["oracle_onestep"] is False
    assert protocol["operator"]["reads_unmasked_field"] is False
    assert "scores" not in protocol
    assert "results" not in protocol
    reference = observation_spec(protocol, 32, 4)
    assert reference == PREREGISTERED_OBSERVATION
    assert [item["id"] for item in axis_conditions(protocol, "noise")][0] == "noise0_sensors32_bursts4"
    assert [int(item["n_sensors"]) for item in axis_conditions(protocol, "sensors")] == [32, 16, 8]
    assert [int(item["n_bursts"]) for item in axis_conditions(protocol, "bursts")] == [4, 2, 1]


def test_protocol_rejects_a_file_that_already_holds_scores(tmp_path: Path) -> None:
    payload = json.loads(PROTOCOL.read_text(encoding="utf-8"))
    payload["scores"] = {"hard_ood": 0.1}
    path = tmp_path / "protocol.json"
    path.write_text(json.dumps(payload), encoding="utf-8")
    with pytest.raises(ValueError, match="test scores"):
        load_stage_e_protocol(path)


def test_fraction_zero_is_the_clean_field_and_positive_fractions_share_epsilon() -> None:
    clean = np.linspace(-1.0, 1.5, 32, dtype=np.float64).reshape(4, 8)
    epsilon = standard_normal_field(20260930, 0, 7, clean.shape)
    again = standard_normal_field(20260930, 0, 7, clean.shape)
    other = standard_normal_field(20260930, 1, 7, clean.shape)
    assert np.array_equal(epsilon, again)
    assert not np.array_equal(epsilon, other)
    untouched = apply_noise(clean, epsilon, 0.0)
    assert np.array_equal(untouched, clean)
    clean[0, 0] = 9.0
    assert untouched[0, 0] != 9.0
    restored = np.linspace(-1.0, 1.5, 32, dtype=np.float64).reshape(4, 8)
    low = apply_noise(restored, epsilon, 0.01)
    high = apply_noise(restored, epsilon, 0.02)
    assert np.allclose(high - restored, 2.0 * (low - restored))
    assert field_std(restored) == pytest.approx(float(np.std(restored, ddof=0)))


def test_coarser_sensors_are_a_subset_of_the_finer_grid() -> None:
    fine = sensor_columns(1024, 32)
    coarse = sensor_columns(1024, 16)
    coarsest = sensor_columns(1024, 8)
    assert fine.size == 32
    assert set(coarse.tolist()).issubset(set(fine.tolist()))
    assert set(coarsest.tolist()).issubset(set(coarse.tolist()))
    assert int(fine[0]) == 0
    assert int(np.diff(fine)[0]) == 1024 // 32


def test_one_burst_has_no_operator_target_and_four_bursts_do() -> None:
    protocol = load_stage_e_protocol(PROTOCOL)
    window = window_spec(protocol)
    assert window == WindowSpec(input_frames=8, output_frames=8, stride=8)
    assert operator_applicable(observation_spec(protocol, 32, 4), window) is True
    assert operator_applicable(observation_spec(protocol, 8, 2), window) is True
    assert operator_applicable(observation_spec(protocol, 32, 1), window) is False
    single = observation_spec(protocol, 32, 1)
    assert single.series == ((0, 1, 2, 3, 4),)


def test_noise_seeds_stay_paired_on_the_primary_seed() -> None:
    protocol = load_stage_e_protocol(PROTOCOL)
    reference = next(item for item in protocol["conditions"] if item["id"] == "noise0_sensors32_bursts4")
    noisy = next(item for item in protocol["conditions"] if item["id"] == "noise0p01_sensors32_bursts4")
    assert noise_seeds_for(protocol, reference, "closed_form_ls") == [0]
    assert noise_seeds_for(protocol, noisy, "closed_form_ls") == [0, 1, 2]
    assert noise_seeds_for(protocol, noisy, "operator_data_only") == [0]


def test_failure_rule_is_unchanged() -> None:
    assert is_failure(0.0, 0.02) is True
    assert is_failure(-0.01, 0.02) is True
    assert is_failure(0.03, 0.02) is False
    assert is_failure(0.031, 0.02) is True


def test_breakdown_thresholds_are_per_slice_and_skip_an_undefined_operator() -> None:
    protocol = load_stage_e_protocol(PROTOCOL)
    conditions = {}
    for condition in protocol["conditions"]:
        rel = 0.04
        failures = 0
        if condition["id"] == "noise0p02_sensors32_bursts4":
            rel = 0.2
            failures = 4
        if condition["id"] == "noise0_sensors8_bursts4":
            rel = 0.11
        block = {
            "applicable": True,
            "slices": {name: {"mean_rel_error": rel, "n_failures": failures} for name in SLICE_NAMES},
        }
        operator = {"applicable": False, "reason": "no_target_burst"} if int(condition["n_bursts"]) == 1 else block
        conditions[condition["id"]] = {
            "closed_form_ls": block,
            "operator_data_only": operator,
            "operator_hybrid_1e-2": operator,
            "training_mean": block,
        }
    table = breakdown_table(protocol, conditions)
    hard = table["noise"]["closed_form_ls"]["hard_ood"]
    assert hard["rel_onset"]["condition_id"] == "noise0p02_sensors32_bursts4"
    assert hard["rel_onset"]["mean_rel_error"] == pytest.approx(0.2)
    assert hard["failure_onset"]["n_failures"] == pytest.approx(4.0)
    assert hard["rel_exceeds_at_reference"] is False
    sensors = table["sensors"]["operator_data_only"]["complement"]
    assert sensors["rel_onset"]["condition_id"] == "noise0_sensors8_bursts4"
    assert sensors["failure_onset"] is None
    bursts = table["bursts"]["operator_data_only"]["full_test"]
    assert bursts["inapplicable_onset"]["condition_id"] == "noise0_sensors32_bursts1"
    assert bursts["rel_onset"] is None


def test_stress_chart_writes_categories_and_the_ten_percent_line(tmp_path: Path) -> None:
    path = tmp_path / "curve.svg"
    write_stress_chart(
        path,
        (
            {"name": "closed-form LS", "x": [0, 1, 2], "y": [0.05, 0.08, 0.2]},
            {"name": "data-only FNO", "x": [0, 1], "y": [0.04, 0.06]},
        ),
        xlabel="noise, fraction of field std",
        ylabel="mean relative error",
        title="hard_ood",
        x_labels=["0", "1%", "5%"],
        hline=0.1,
        ylog=True,
    )
    text = path.read_text(encoding="utf-8")
    assert "<polyline" in text
    assert "stroke-dasharray" in text
    assert "hard_ood" in text
    assert "1%" in text


def test_golden_search_matches_the_leftmost_unimodal_argmin() -> None:
    generator = np.random.default_rng(0)
    for trial in range(40):
        plateau = 1 if trial % 5 else int(generator.integers(1, 6))
        minimum = int(generator.integers(0, 191))
        right_len = 191 - minimum - plateau
        if right_len < 0:
            plateau = 191 - minimum
            right_len = 0
        left = np.cumsum(generator.uniform(0.01, 1.0, minimum))[::-1] if minimum else np.array([])
        right = np.cumsum(generator.uniform(0.01, 1.0, right_len)) if right_len else np.array([])
        curve = np.concatenate([left, np.zeros(plateau), right]) + 1.0
        assert unimodal_grid_index(curve) == int(np.argmin(curve))
    indexes = coarse_indexes(191, 16)
    assert int(indexes[0]) == 0 and int(indexes[-1]) == 190
    valley = np.concatenate([np.arange(8, 0, -1), np.arange(1, 8)])
    assert local_minimum_count(valley.astype(np.float64)) == 1
    two = np.array([3.0, 1.0, 2.0, 0.5, 2.0], dtype=np.float64)
    assert local_minimum_count(two) == 2


@pytest.mark.ml
def test_batched_sensor_curve_matches_the_single_viscosity_reduction() -> None:
    from pinnforge.operator.fno import FNO1d
    from pinnforge.operator.stage_e_fit import sensor_mse_on_grid
    from pinnforge.operator.windows import FieldNorm

    model = FNO1d(in_channels=9, out_channels=8, width=4, modes=2, n_layers=1)
    model.eval()
    norm = FieldNorm(u_mean=0.1, u_std=0.5, nu_mean=0.05, nu_std=0.02)
    generator = np.random.default_rng(0)
    windows = generator.normal(size=(3, 8, 32))
    targets = generator.normal(size=(3, 1, 5, 8))
    slots = ((0, 0, 0, 5),)
    grid = np.linspace(0.005, 0.1, 4)
    single = sensor_mse_on_grid(
        model, norm, windows, targets, slots, grid, n_sensors=8, nu_chunk=1, instance_chunk=1
    )
    batched = sensor_mse_on_grid(
        model, norm, windows, targets, slots, grid, n_sensors=8, nu_chunk=4, instance_chunk=3
    )
    assert single.shape == (3, 4)
    assert np.allclose(single, batched)
    per_row = np.stack([grid[[0, 3]], grid[[1, 2]], grid[[2, 2]]])
    paired = sensor_mse_on_grid(
        model, norm, windows, targets, slots, per_row, n_sensors=8, nu_chunk=2, instance_chunk=3
    )
    assert paired.shape == (3, 2)
    assert np.allclose(paired[0], batched[0, [0, 3]])
    assert np.allclose(paired[1], batched[1, [1, 2]])
    assert np.allclose(paired[2], batched[2, [2, 2]])
    assert np.isfinite(batched).all()
