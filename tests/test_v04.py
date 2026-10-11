"""v0.4 protocol freeze, smoke, and CLI wiring."""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from pinnforge.reference.spectral2d.dataset import load_protocol, protocol_sha256
from pinnforge.v04.__main__ import main as v04_main

ROOT = Path(__file__).resolve().parents[1]


def test_protocol_is_frozen_and_has_no_test_scores() -> None:
    protocol = load_protocol(ROOT / "docs/v04/protocol.json")
    assert protocol["frozen_before_test_evaluation"] is True
    assert protocol["test_used_for_selection"] is False
    assert protocol["hyperparameters_tuned_on_test"] is False
    assert protocol["pass_bars"]["in_distribution_error"]["limit"] == 0.05
    assert protocol["pass_bars"]["hard_ood_gap"]["limit_ratio"] == 3.0
    assert protocol["pass_bars"]["warm_vs_cold"]["error_factor"] == 1.5
    assert protocol["pass_bars"]["warm_vs_cold"]["time_limit_ratio"] == 0.25
    assert protocol["pass_bars"]["coverage"]["range"] == [0.85, 0.95]
    text = json.dumps(protocol)
    assert "mean_rel_error" not in text
    assert protocol_sha256(ROOT / "docs/v04/protocol.json") == protocol_sha256()


def test_protocol_hash_command(capsys: pytest.CaptureFixture[str]) -> None:
    assert v04_main(["protocol-hash"]) == 0
    assert capsys.readouterr().out.strip() == protocol_sha256()


@pytest.mark.ml
def test_v04_smoke() -> None:
    assert v04_main(["smoke"]) == 0
