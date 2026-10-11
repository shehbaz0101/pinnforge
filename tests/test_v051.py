"""v0.5.1 protocol freeze, lattice masks, and CLI wiring."""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from pinnforge.reference.spectral2d.dataset import load_protocol, protocol_sha256
from pinnforge.v051.__main__ import main as v051_main
from pinnforge.v051.masks import mask_for_lattice

ROOT = Path(__file__).resolve().parents[1]
PROTOCOL = ROOT / "docs" / "v051" / "protocol.json"
V05_PROTOCOL = ROOT / "docs" / "v05" / "protocol.json"
V05_SCORES = ROOT / "docs" / "v05" / "scores.json"

FROZEN_PROTOCOL_SHA256 = "e1c0d44d1bbd0ac94c448c852f0e069a080a172c3411842abf67b9f43533cf4f"


def test_protocol_is_frozen_and_has_no_v051_test_scores() -> None:
    protocol = load_protocol(PROTOCOL)
    assert protocol_sha256(PROTOCOL) == FROZEN_PROTOCOL_SHA256
    assert protocol["format"] == "pinnforge.v051_sensor_density_protocol.v1"
    assert protocol["frozen_before_test_evaluation"] is True
    assert protocol["test_used_for_selection"] is False
    assert protocol["methods"]["wsindy_fno"]["used"] is False
    assert protocol["methods"]["pysr"]["used"] is False
    assert protocol["seeds"] == [0, 1, 2, 3, 4]
    assert [item["n_side"] for item in protocol["lattices"]] == [8, 16, 32, 64]
    assert protocol["reported"]["min_seed_success_rate"] == 0.8
    assert protocol["reported"]["noise"] == 0.01
    assert protocol["positive_control"]["lattice"] == "64x64"
    assert protocol["positive_control"]["noise"] == 0.0
    assert "pinnforge.v051_scores.v1" not in json.dumps(protocol)


def test_v05_artifacts_were_not_edited() -> None:
    v05 = json.loads(V05_PROTOCOL.read_text(encoding="utf-8"))
    assert v05["format"] == "pinnforge.v05_law_discovery_protocol.v1"
    scores = json.loads(V05_SCORES.read_text(encoding="utf-8"))
    assert scores["protocol_sha256"] == protocol_sha256(V05_PROTOCOL)
    assert scores["overall"]["pass"] is False


def test_protocol_hash_command(capsys: pytest.CaptureFixture[str]) -> None:
    assert v051_main(["protocol-hash"]) == 0
    assert capsys.readouterr().out.strip() == protocol_sha256(PROTOCOL)


def test_lattice_masks_divide_the_64_grid() -> None:
    protocol = load_protocol(PROTOCOL)
    for n_side in (8, 16, 32, 64):
        mask = mask_for_lattice(protocol, n_side)
        assert mask.n == 64
        assert mask.n_side == n_side
        assert mask.stride == 64 // n_side


def test_v051_smoke() -> None:
    assert v051_main(["smoke"]) == 0
