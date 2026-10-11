"""v0.6.1 identifiability freeze and parent-artifact lock."""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from pinnforge.reference.spectral2d.dataset import protocol_sha256
from pinnforge.v061.__main__ import main as v061_main

ROOT = Path(__file__).resolve().parents[1]
V06_PROTOCOL = ROOT / "docs" / "v06" / "protocol.json"
V06_SCORES = ROOT / "docs" / "v06" / "scores.json"
V061_PROTOCOL = ROOT / "docs" / "v061" / "protocol.json"
IDENT = ROOT / "docs" / "v061" / "identifiability.json"

V06_PROTOCOL_SHA256 = "07bfbd278b70a53c45d489a8be270aadeca9286be4af4f969e4de9860471f35a"
FROZEN_PROTOCOL_SHA256 = "d3d7af4088e8170ce1555f1ded10b7ade98110446e2a70cebe51cb9ca1a8b052"


def test_parent_v06_was_not_edited() -> None:
    assert protocol_sha256(V06_PROTOCOL) == V06_PROTOCOL_SHA256
    scores = json.loads(V06_SCORES.read_text(encoding="utf-8"))
    assert scores["protocol_sha256"] == V06_PROTOCOL_SHA256
    assert scores["bars"]["overall"] is False


def test_v061_smoke() -> None:
    assert v061_main(["smoke"]) == 0


def test_identifiability_is_not_a_test_score() -> None:
    if not IDENT.is_file():
        pytest.skip("identifiability dump is written on the validation sweep")
    payload = json.loads(IDENT.read_text(encoding="utf-8"))
    assert payload["format"] == "pinnforge.v061_identifiability.v1"
    assert payload["split"] == "valid"
    assert "pinnforge.v061_scores.v1" not in json.dumps(payload)


def test_protocol_is_frozen_and_has_no_v061_test_scores() -> None:
    from pinnforge.reference.spectral2d.dataset import load_protocol

    protocol = load_protocol(V061_PROTOCOL)
    assert protocol_sha256(V061_PROTOCOL) == FROZEN_PROTOCOL_SHA256
    assert protocol["format"] == "pinnforge.v061_shear_flow_protocol.v1"
    assert protocol["frozen_before_test_evaluation"] is True
    assert protocol["test_used_for_selection"] is False
    assert protocol["methods"]["pysr"]["used"] is False
    assert protocol["seeds"] == [0, 1, 2]
    assert protocol["splits"]["from_train_files"]["n_cal"] >= 60
    assert protocol["downsample"]["nx"] == 256
    assert protocol["time_window"]["frame_start"] == 160
    assert "pinnforge.v061_scores.v1" not in json.dumps(protocol)


def test_protocol_hash_after_freeze(capsys: pytest.CaptureFixture[str]) -> None:
    assert v061_main(["protocol-hash"]) == 0
    assert capsys.readouterr().out.strip() == protocol_sha256(V061_PROTOCOL)


def test_committed_scores_match_frozen_protocol() -> None:
    scores_path = ROOT / "docs" / "v061" / "scores.json"
    if not scores_path.is_file():
        pytest.skip("v061 scores are written after the protocol freeze")
    scores = json.loads(scores_path.read_text(encoding="utf-8"))
    assert scores["protocol_sha256"] == protocol_sha256(V061_PROTOCOL)
    report = (ROOT / "docs" / "v061" / "report.md").read_text(encoding="utf-8")
    assert "FAIL" in report or "PASS" in report
