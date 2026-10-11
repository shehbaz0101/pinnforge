"""v0.6.2 solver smoke and parent-artifact lock."""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from pinnforge.reference.spectral2d.dataset import protocol_sha256
from pinnforge.v062.__main__ import main as v062_main
from pinnforge.v062.solver import init_standard_shear, rel_l2

ROOT = Path(__file__).resolve().parents[1]
V061_PROTOCOL = ROOT / "docs" / "v061" / "protocol.json"
V061_PROTOCOL_SHA256 = "d3d7af4088e8170ce1555f1ded10b7ade98110446e2a70cebe51cb9ca1a8b052"
V062_PROTOCOL = ROOT / "docs" / "v062" / "protocol.json"
FROZEN_PROTOCOL_SHA256 = "1a01a419858471174670678cd24ef8b1b0204fda4d1d8e04bb461f38dae87444"


def test_parent_v061_was_not_edited() -> None:
    assert protocol_sha256(V061_PROTOCOL) == V061_PROTOCOL_SHA256
    scores = json.loads((ROOT / "docs" / "v061" / "scores.json").read_text(encoding="utf-8"))
    assert scores["protocol_sha256"] == V061_PROTOCOL_SHA256
    assert scores["bars"]["overall"] is False


def test_v062_smoke() -> None:
    assert v062_main(["smoke"]) == 0


def test_ic_family_is_finite() -> None:
    ux, uy = init_standard_shear(32, 64, n_shear=2, n_blobs=3, width=0.5)
    assert ux.shape == (32, 64)
    assert rel_l2(ux, ux) == 0.0
    assert ux.max() > ux.min()


def test_protocol_is_frozen_and_has_no_v062_test_scores() -> None:
    from pinnforge.reference.spectral2d.dataset import load_protocol

    protocol = load_protocol(V062_PROTOCOL)
    assert protocol_sha256(V062_PROTOCOL) == FROZEN_PROTOCOL_SHA256
    assert protocol["format"] == "pinnforge.v062_sim_train_well_test_protocol.v1"
    assert protocol["frozen_before_test_evaluation"] is True
    assert protocol["test_used_for_selection"] is False
    assert protocol["methods"]["pysr"]["used"] is False
    assert protocol["training"]["n_cal"] >= 60
    assert protocol["compute_reduction"]["sim_nx"] == 64
    assert "pinnforge.v062_scores.v1" not in json.dumps(protocol)


def test_protocol_hash(capsys: pytest.CaptureFixture[str]) -> None:
    assert v062_main(["protocol-hash"]) == 0
    assert capsys.readouterr().out.strip() == protocol_sha256(V062_PROTOCOL)
