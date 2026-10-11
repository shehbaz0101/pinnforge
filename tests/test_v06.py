"""v0.6 protocol freeze, synthetic smoke, and parent-artifact lock."""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from pinnforge.reference.spectral2d.dataset import load_protocol, protocol_sha256
from pinnforge.v06.__main__ import main as v06_main
from pinnforge.v06.stride_check import listed_stride_is_v052

ROOT = Path(__file__).resolve().parents[1]
PROTOCOL = ROOT / "docs" / "v06" / "protocol.json"
V052_PROTOCOL = ROOT / "docs" / "v052" / "protocol.json"
V052_SCORES = ROOT / "docs" / "v052" / "scores.json"

FROZEN_PROTOCOL_SHA256 = "07bfbd278b70a53c45d489a8be270aadeca9286be4af4f969e4de9860471f35a"
V052_PROTOCOL_SHA256 = "0c2f95cfa15f69fb38bd4b6028b294747b63438d6d752286a4b26ec5088c2cc8"


def test_protocol_is_frozen_and_has_no_v06_test_scores() -> None:
    protocol = load_protocol(PROTOCOL)
    assert protocol_sha256(PROTOCOL) == FROZEN_PROTOCOL_SHA256
    assert protocol["format"] == "pinnforge.v06_shear_flow_protocol.v1"
    assert protocol["frozen_before_test_evaluation"] is True
    assert protocol["test_used_for_selection"] is False
    assert protocol["methods"]["pysr"]["used"] is False
    assert protocol["seeds"] == [0, 1, 2]
    assert protocol["splits"]["from_train_files"]["n_cal"] >= 60
    assert protocol["pass_bars"]["nu_kappa_in_dist"]["max"] == 0.05
    assert protocol["pass_bars"]["nu_kappa_ood"]["max"] == 0.1
    assert protocol["pass_bars"]["coverage"]["min"] == 0.88
    assert protocol["pass_bars"]["coverage"]["max"] == 0.92
    assert protocol["downsample"]["nx"] == 32
    assert protocol["downsample"]["ny"] == 64
    assert "pinnforge.v06_scores.v1" not in json.dumps(protocol)
    assert listed_stride_is_v052(protocol)


def test_parent_v052_was_not_edited() -> None:
    assert protocol_sha256(V052_PROTOCOL) == V052_PROTOCOL_SHA256
    scores = json.loads(V052_SCORES.read_text(encoding="utf-8"))
    assert scores["protocol_sha256"] == V052_PROTOCOL_SHA256
    assert scores["problems"]["ns2d"]["positive_control"]["pass"] is True


def test_protocol_hash_command(capsys: pytest.CaptureFixture[str]) -> None:
    assert v06_main(["protocol-hash"]) == 0
    assert capsys.readouterr().out.strip() == protocol_sha256(PROTOCOL)


def test_v06_smoke() -> None:
    assert v06_main(["smoke"]) == 0


def test_committed_scores_match_frozen_protocol() -> None:
    scores_path = ROOT / "docs" / "v06" / "scores.json"
    if not scores_path.is_file():
        pytest.skip("v06 scores are written after the protocol freeze")
    scores = json.loads(scores_path.read_text(encoding="utf-8"))
    assert scores["protocol_sha256"] == protocol_sha256(PROTOCOL)
    report = (ROOT / "docs" / "v06" / "report.md").read_text(encoding="utf-8")
    assert "FAIL" in report or "PASS" in report
