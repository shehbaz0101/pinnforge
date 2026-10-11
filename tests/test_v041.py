"""v0.4.1 protocol freeze, smoke, and CLI wiring."""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from pinnforge.reference.spectral2d.dataset import load_protocol, protocol_sha256
from pinnforge.v041.__main__ import main as v041_main

ROOT = Path(__file__).resolve().parents[1]
PROTOCOL = ROOT / "docs" / "v041" / "protocol.json"
V04_PROTOCOL = ROOT / "docs" / "v04" / "protocol.json"
V04_SCORES = ROOT / "docs" / "v04" / "scores.json"
V04_REPORT = ROOT / "docs" / "v04" / "report.md"


FROZEN_PROTOCOL_SHA256 = "0fd6749a4cfd43154d661d385475f5b019c12e8ff1524c90d2334968360c840a"


def test_protocol_is_frozen_and_has_no_v041_test_scores() -> None:
    protocol = load_protocol(PROTOCOL)
    assert protocol_sha256(PROTOCOL) == FROZEN_PROTOCOL_SHA256
    assert protocol["format"] == "pinnforge.v041_2d_inverse_protocol.v1"
    assert protocol["frozen_before_test_evaluation"] is True
    assert protocol["test_used_for_selection"] is False
    assert protocol["hyperparameters_tuned_on_test"] is False
    assert int(protocol["splits"]["n_cal"]) >= 60
    assert int(protocol["splits"]["n_cal"]) == int(protocol["splits"]["n_val"])
    assert protocol["pass_bars"]["in_distribution_error"]["limit"] == 0.05
    assert protocol["pass_bars"]["hard_ood_gap"]["limit_ratio"] == 3.0
    assert protocol["pass_bars"]["warm_vs_cold"]["error_factor"] == 1.5
    assert protocol["pass_bars"]["warm_vs_cold"]["time_limit_ratio"] == 0.25
    assert protocol["pass_bars"]["coverage"]["range"] == [0.85, 0.95]
    v04 = load_protocol(V04_PROTOCOL)
    assert protocol["pass_bars"]["in_distribution_error"]["limit"] == v04["pass_bars"]["in_distribution_error"]["limit"]
    text = json.dumps(protocol)
    assert "mean_rel_error" not in text
    assert protocol_sha256(PROTOCOL) == protocol_sha256(ROOT / "docs" / "v041" / "protocol.json")


def test_v04_artifacts_were_not_edited() -> None:
    protocol = json.loads(V04_PROTOCOL.read_text(encoding="utf-8"))
    assert protocol["format"] == "pinnforge.v04_2d_inverse_protocol.v1"
    scores = json.loads(V04_SCORES.read_text(encoding="utf-8"))
    assert scores["protocol_sha256"] == protocol_sha256(V04_PROTOCOL)
    assert scores["problems"]["burgers2d"]["bars"]["pass"] is False
    assert "FAIL" in V04_REPORT.read_text(encoding="utf-8") or "fail" in V04_REPORT.read_text(encoding="utf-8")


def test_protocol_hash_command(capsys: pytest.CaptureFixture[str]) -> None:
    assert v041_main(["protocol-hash"]) == 0
    assert capsys.readouterr().out.strip() == protocol_sha256(PROTOCOL)


def test_committed_scores_match_frozen_protocol() -> None:
    scores_path = ROOT / "docs" / "v041" / "scores.json"
    scores = json.loads(scores_path.read_text(encoding="utf-8"))
    assert scores["protocol_sha256"] == protocol_sha256(PROTOCOL)
    assert scores["problems"]["burgers2d"]["bars"]["pass"] is False
    assert scores["problems"]["ns2d"]["bars"]["pass"] is False
    assert scores["problems"]["burgers2d"]["bars"]["checks"]["coverage"]["pass"] is True
    assert scores["problems"]["ns2d"]["bars"]["checks"]["coverage"]["pass"] is True


@pytest.mark.ml
def test_v041_smoke() -> None:
    assert v041_main(["smoke"]) == 0
