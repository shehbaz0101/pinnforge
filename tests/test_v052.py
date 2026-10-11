"""v0.5.2 protocol freeze, incommensurate stride, and CLI wiring."""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from pinnforge.reference.spectral2d.dataset import load_protocol, protocol_sha256
from pinnforge.v052.__main__ import main as v052_main
from pinnforge.v052.stride import center_stride, stride_is_incommensurate

ROOT = Path(__file__).resolve().parents[1]
PROTOCOL = ROOT / "docs" / "v052" / "protocol.json"
V051_PROTOCOL = ROOT / "docs" / "v051" / "protocol.json"
V051_SCORES = ROOT / "docs" / "v051" / "scores.json"
V05_PROTOCOL = ROOT / "docs" / "v05" / "protocol.json"
V05_SCORES = ROOT / "docs" / "v05" / "scores.json"

FROZEN_PROTOCOL_SHA256 = "0c2f95cfa15f69fb38bd4b6028b294747b63438d6d752286a4b26ec5088c2cc8"
V051_PROTOCOL_SHA256 = "e1c0d44d1bbd0ac94c448c852f0e069a080a172c3411842abf67b9f43533cf4f"
V05_PROTOCOL_SHA256 = "f4d7f8584cf003bc3b28f5d554e645ff3f5f5160f646538d854cbdae0362e902"


def test_protocol_is_frozen_and_has_no_v052_test_scores() -> None:
    protocol = load_protocol(PROTOCOL)
    assert protocol_sha256(PROTOCOL) == FROZEN_PROTOCOL_SHA256
    assert protocol["format"] == "pinnforge.v052_ns_wsindy_fix_protocol.v1"
    assert protocol["frozen_before_test_evaluation"] is True
    assert protocol["test_used_for_selection"] is False
    assert protocol["methods"]["wsindy_fno"]["used"] is False
    assert protocol["methods"]["pysr"]["used"] is False
    assert protocol["burgers2d"]["scored"] is False
    assert protocol["seeds"] == [0, 1, 2, 3, 4]
    assert [item["n_side"] for item in protocol["lattices"]] == [8, 16, 32, 64]
    assert protocol["reported"]["min_seed_success_rate"] == 0.8
    assert protocol["reported"]["noise"] == 0.01
    assert protocol["positive_control"]["lattice"] == "64x64"
    assert protocol["positive_control"]["noise"] == 0.0
    assert protocol["parent_protocol_sha256"] == V051_PROTOCOL_SHA256
    assert "pinnforge.v052_scores.v1" not in json.dumps(protocol)


def test_registered_stride_is_incommensurate_with_kolmogorov() -> None:
    protocol = load_protocol(PROTOCOL)
    frac = float(protocol["methods"]["wsindy"]["test_function"]["center_stride_fraction"])
    wave = int(protocol["problems"]["ns2d"]["forcing"]["kolmogorov"]["wavenumber"])
    assert frac != 0.25
    for n_side in (8, 16, 32, 64):
        assert stride_is_incommensurate(n_side, frac, wave)
        assert (wave * center_stride(n_side, frac)) % n_side != 0
    assert not stride_is_incommensurate(64, 0.25, 4)


def test_parent_artifacts_were_not_edited() -> None:
    assert protocol_sha256(V051_PROTOCOL) == V051_PROTOCOL_SHA256
    assert protocol_sha256(V05_PROTOCOL) == V05_PROTOCOL_SHA256
    v051 = json.loads(V051_SCORES.read_text(encoding="utf-8"))
    v05 = json.loads(V05_SCORES.read_text(encoding="utf-8"))
    assert v051["protocol_sha256"] == V051_PROTOCOL_SHA256
    assert v051["problems"]["ns2d"]["positive_control"]["pass"] is False
    assert v051["problems"]["burgers2d"]["min_lattice_at_bar"] == "64x64"
    assert v05["protocol_sha256"] == V05_PROTOCOL_SHA256


def test_protocol_hash_command(capsys: pytest.CaptureFixture[str]) -> None:
    assert v052_main(["protocol-hash"]) == 0
    assert capsys.readouterr().out.strip() == protocol_sha256(PROTOCOL)


def test_v052_smoke() -> None:
    assert v052_main(["smoke"]) == 0


def test_committed_scores_match_frozen_protocol() -> None:
    scores_path = ROOT / "docs" / "v052" / "scores.json"
    scores = json.loads(scores_path.read_text(encoding="utf-8"))
    assert scores["protocol_sha256"] == protocol_sha256(PROTOCOL)
    assert scores["protocol_sha256"] == FROZEN_PROTOCOL_SHA256
    assert scores["burgers2d"]["scored"] is False
    assert "burgers2d" not in scores["problems"]
    assert scores["problems"]["ns2d"]["min_lattice_at_bar"] == "32x32"
    assert scores["problems"]["ns2d"]["positive_control"]["pass"] is True
    assert scores["wsindy_fno"]["used"] is False
    report = (ROOT / "docs" / "v052" / "report.md").read_text(encoding="utf-8")
    assert "not rescored" in report
    assert "32x32" in report
