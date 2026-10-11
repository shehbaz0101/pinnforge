"""v0.6.2 solver smoke and parent-artifact lock."""

from __future__ import annotations

import json
from pathlib import Path

from pinnforge.reference.spectral2d.dataset import protocol_sha256
from pinnforge.v062.__main__ import main as v062_main
from pinnforge.v062.solver import init_standard_shear, rel_l2

ROOT = Path(__file__).resolve().parents[1]
V061_PROTOCOL = ROOT / "docs" / "v061" / "protocol.json"
V061_PROTOCOL_SHA256 = "d3d7af4088e8170ce1555f1ded10b7ade98110446e2a70cebe51cb9ca1a8b052"


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
