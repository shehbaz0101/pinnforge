"""Protocol lock and paired comparison. No pilot files and no torch."""

from __future__ import annotations

from pathlib import Path

import numpy as np

from pinnforge.operator.strong_baselines import (
    file_sha256,
    load_strong_protocol,
    paired_against_fno,
)

ROOT = Path(__file__).resolve().parents[1]
PROTOCOL = ROOT / "docs" / "research" / "strong_baselines_protocol.json"


def test_protocol_is_frozen_and_hashed() -> None:
    protocol = load_strong_protocol(PROTOCOL)
    assert protocol["frozen_before_test_evaluation"] is True
    assert protocol["test_used_for_selection"] is False
    assert protocol["hyperparameters_tuned_on_test"] is False
    assert protocol["failure_relative_error"] == 0.5
    assert protocol["mask"]["name"] == "sensors32_bursts"
    assert protocol["mask"]["n_sensors"] == 32
    assert protocol["weak_form"]["interpolation_of_u"] == "none"
    assert protocol["weak_form"]["degree"] == 4
    assert protocol["solver_sensor_ic"]["reads_off_sensor_samples"] is False
    assert protocol["solver_oracle_ic"]["reads_full_initial_condition"] is True
    assert protocol["solver_search"]["n"] == 1024
    assert protocol["solver_search"]["dt"] == 0.00025
    assert protocol["stress"]["solver_sensor_ic"]["condition_ids"] == [
        "noise0p02_sensors32_bursts4",
        "noise0p05_sensors32_bursts4",
        "noise0_sensors16_bursts4",
        "noise0_sensors8_bursts4",
    ]
    assert "results" not in protocol
    assert "nu_hat" not in protocol
    digest = file_sha256(PROTOCOL)
    assert len(digest) == 64


def test_paired_delta_is_fno_minus_baseline() -> None:
    truth = np.array([0.01, 0.02])
    hats = np.array(
        [
            [0.01, 0.03],
            [0.012, 0.02],
        ]
    )
    baseline_rel = np.array([0.5, 0.0])
    baseline_fail = np.array([True, False])
    paired = paired_against_fno(hats, truth, baseline_rel, baseline_fail)
    assert paired["n_instances"] == 2
    assert paired["baseline_n_failures"] == 1
    assert paired["fno_minus_baseline_mean_rel"] < 0.0
    assert paired["per_seed_fno_minus_baseline"][0] < 0.0


def test_solver_stress_calls_the_sensor_lift() -> None:
    source = (ROOT / "src" / "pinnforge" / "operator" / "strong_baselines.py").read_text(encoding="utf-8")
    assert 'initial_condition="solver_sensor_ic"' not in source
    assert 'initial_condition="sensor_lift"' in source
