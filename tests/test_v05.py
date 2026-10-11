"""v0.5 protocol freeze, WSINDy recovery, and CLI wiring."""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pytest

from pinnforge.reference.spectral2d.dataset import load_protocol, protocol_sha256
from pinnforge.v05.__main__ import main as v05_main
from pinnforge.v05.wsindy import exact_support, stlsq, support_of

ROOT = Path(__file__).resolve().parents[1]
PROTOCOL = ROOT / "docs" / "v05" / "protocol.json"
V04_PROTOCOL = ROOT / "docs" / "v04" / "protocol.json"
V04_SCORES = ROOT / "docs" / "v04" / "scores.json"
V041_PROTOCOL = ROOT / "docs" / "v041" / "protocol.json"
V041_SCORES = ROOT / "docs" / "v041" / "scores.json"

FROZEN_PROTOCOL_SHA256 = "f4d7f8584cf003bc3b28f5d554e645ff3f5f5160f646538d854cbdae0362e902"


def test_protocol_is_frozen_and_has_no_v05_test_scores() -> None:
    protocol = load_protocol(PROTOCOL)
    assert protocol_sha256(PROTOCOL) == FROZEN_PROTOCOL_SHA256
    assert protocol["format"] == "pinnforge.v05_law_discovery_protocol.v1"
    assert protocol["frozen_before_test_evaluation"] is True
    assert protocol["test_used_for_selection"] is False
    assert protocol["hyperparameters_tuned_on_test"] is False
    assert protocol["hyperparameters_chosen_on_validation"] is True
    assert protocol["methods"]["pysr"]["used"] is False
    assert "Julia" in protocol["methods"]["pysr"]["reason"]
    assert protocol["seeds"] == [0, 1, 2, 3, 4]
    assert protocol["noise"]["fractions"] == [0.0, 0.01, 0.02, 0.05]
    assert protocol["pass_bars"]["min_seed_success_rate"] == 0.8
    assert protocol["pass_bars"]["noise"] == 0.01
    assert "q2" in protocol["library"]["burgers2d"]["distractors"]
    assert "force_y" in protocol["library"]["burgers2d"]["distractors"]
    assert "w_xxx" in protocol["library"]["ns2d"]["distractors"]
    text = json.dumps(protocol)
    assert "pinnforge.v05_scores.v1" not in text
    assert protocol["test_used_for_selection"] is False


def test_v04_and_v041_artifacts_were_not_edited() -> None:
    v04 = json.loads(V04_PROTOCOL.read_text(encoding="utf-8"))
    v041 = json.loads(V041_PROTOCOL.read_text(encoding="utf-8"))
    assert v04["format"] == "pinnforge.v04_2d_inverse_protocol.v1"
    assert v041["format"] == "pinnforge.v041_2d_inverse_protocol.v1"
    assert json.loads(V04_SCORES.read_text(encoding="utf-8"))["protocol_sha256"] == protocol_sha256(V04_PROTOCOL)
    assert json.loads(V041_SCORES.read_text(encoding="utf-8"))["protocol_sha256"] == protocol_sha256(V041_PROTOCOL)


def test_protocol_hash_command(capsys: pytest.CaptureFixture[str]) -> None:
    assert v05_main(["protocol-hash"]) == 0
    assert capsys.readouterr().out.strip() == protocol_sha256(PROTOCOL)


def test_stlsq_recovers_manufactured_support() -> None:
    rng = np.random.default_rng(1)
    truth = np.array([-1.0, -1.0, 0.5, 0.5, 0.0, 0.0, 0.0], dtype=np.float64)
    design = rng.normal(size=(80, truth.size))
    target = design @ truth
    names = ("u_qx", "v_qy", "q_xx", "q_yy", "q2", "q_xxx", "const")
    coef = stlsq(design, target, 0.08, max_iter=10, column_normalize=True)
    assert exact_support(support_of(coef, names), ("u_qx", "v_qy", "q_xx", "q_yy"))


def test_committed_scores_match_frozen_protocol() -> None:
    scores_path = ROOT / "docs" / "v05" / "scores.json"
    scores = json.loads(scores_path.read_text(encoding="utf-8"))
    assert scores["protocol_sha256"] == protocol_sha256(PROTOCOL)
    assert scores["overall"]["pass"] is False
    assert scores["problems"]["burgers2d"]["bars"]["pass"] is False
    assert scores["problems"]["ns2d"]["bars"]["pass"] is False
    assert scores["pysr"]["used"] is False
    report = (ROOT / "docs" / "v05" / "report.md").read_text(encoding="utf-8")
    assert "FAIL" in report


@pytest.mark.ml
def test_v05_smoke() -> None:
    assert v05_main(["smoke"]) == 0
