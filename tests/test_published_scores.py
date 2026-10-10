"""Lock the changelog headlines to the committed score files.

These checks do not retrain. They fail if a published number moves without
the changelog and this test moving together.
"""

from __future__ import annotations

import json
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]


def _load(relative: str) -> dict:
    return json.loads((ROOT / relative).read_text(encoding="utf-8"))


def test_stage_d_hard_ood_headlines_match_the_changelog() -> None:
    scores = _load("docs/v02/stage_d_scores.json")
    data = scores["arms"]["data_only"]
    hybrid = scores["arms"]["hybrid_1e-2"]
    assert data["selected_lambda"] == 0.0
    assert hybrid["selected_lambda"] == 0.0
    data_hard = data["objectives"]["0.0"]["slices"]["hard_ood"]["mean_rel_error"]
    hybrid_hard = hybrid["objectives"]["0.0"]["slices"]["hard_ood"]["mean_rel_error"]
    assert data_hard["mean"] == 0.049487208469027544
    assert data_hard["std"] == 0.0144232271670265
    assert hybrid_hard["mean"] == 0.03368266375533967
    assert hybrid_hard["std"] == 0.0034734247381767768
    assert data["objectives"]["0.0"]["slices"]["hard_ood"]["n_failures"]["per_seed"] == [0, 0, 0, 0, 0]
    assert hybrid["objectives"]["0.0"]["slices"]["hard_ood"]["n_failures"]["per_seed"] == [0, 0, 0, 0, 0]
    closed = scores["baselines"]["sensors32_bursts"]["hard_ood"]
    assert closed["mean_rel_error"] == 0.3494591802034973
    assert closed["n_failures"] == 9
    assert scores["threshold_nu"] == 0.027028120493367818


def test_stage_b_and_c_hard_ood_forward_scores_match_the_changelog() -> None:
    stage_b = _load("docs/v02/stage_b_scores.json")
    stage_c = _load("docs/v02/stage_c_scores.json")
    data = stage_b["slices"]["hard_ood"]["mean_relative_l2"]
    hybrid = stage_c["hybrid"]["slices"]["hard_ood"]["mean_relative_l2"]
    assert data["mean"] == 0.007049252763443842
    assert data["std"] == 0.0003032654479273341
    assert hybrid["mean"] == 0.005215427637735405
    assert hybrid["std"] == 0.0006736875927560861
    assert stage_c["selected_residual_weight"] == 0.01
    assert stage_b["threshold_nu"] == stage_c["threshold_nu"] == 0.027028120493367818


def test_stage_e_noise_and_sparsity_claims_match_the_score_file() -> None:
    scores = _load("docs/v02/stage_e_scores.json")
    assert scores["lambda"] == 0.0
    conditions = scores["conditions"]

    def hard(name: str, arm: str) -> tuple[float, float]:
        row = conditions[name][arm]["slices"]["hard_ood"]
        relative = row["mean_rel_error"]
        failures = row["n_failures"]
        if isinstance(relative, dict):
            relative = relative["mean"]
        if isinstance(failures, dict):
            failures = failures["mean"]
        return float(relative), float(failures)

    closed_rel, closed_fail = hard("noise0_sensors32_bursts4", "closed_form_ls")
    assert closed_rel > 0.10
    assert closed_fail == 9
    for fraction in ("noise0", "noise0p001", "noise0p005", "noise0p01"):
        name = f"{fraction}_sensors32_bursts4"
        for arm in ("operator_data_only", "operator_hybrid_1e-2"):
            relative, failures = hard(name, arm)
            assert relative < 0.10
            assert failures == 0.0
    for arm in ("operator_data_only", "operator_hybrid_1e-2"):
        relative, failures = hard("noise0p02_sensors32_bursts4", arm)
        assert relative > 0.10
        assert failures > 0.0
        sparse_rel, sparse_fail = hard("noise0_sensors16_bursts4", arm)
        assert sparse_rel > 0.10
        assert sparse_fail > 0.0
    closed_noisy, _ = hard("noise0p05_sensors32_bursts4", "closed_form_ls")
    data_noisy, _ = hard("noise0p05_sensors32_bursts4", "operator_data_only")
    hybrid_noisy, _ = hard("noise0p05_sensors32_bursts4", "operator_hybrid_1e-2")
    assert closed_noisy < data_noisy
    assert closed_noisy < hybrid_noisy
