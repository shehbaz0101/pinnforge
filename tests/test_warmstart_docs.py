"""Committed warm-start protocol, validation record, scores, and note agree."""

from __future__ import annotations

import json
from pathlib import Path

from pinnforge.operator.strong_baselines import file_sha256
from pinnforge.operator.warmstart_report import render_report
from pinnforge.operator.warmstart_run import ERROR_SLACK, TIME_GUARD, load_warmstart_protocol

ROOT = Path(__file__).resolve().parents[1]
PROTOCOL = ROOT / "docs" / "v03" / "warmstart_protocol.json"
RECORD = ROOT / "docs" / "v03" / "warmstart_validation.json"
SCORES = ROOT / "docs" / "v03" / "warmstart_scores.json"
NOTE = ROOT / "docs" / "v03" / "warmstart.md"
STRONG_SCORES = ROOT / "docs" / "research" / "strong_baselines_scores.json"


def _json(path: Path) -> dict:
    return json.loads(path.read_text(encoding="utf-8"))


def test_protocol_is_frozen_and_references_match() -> None:
    protocol = load_warmstart_protocol(PROTOCOL, check_checkpoints=False)
    strong = _json(STRONG_SCORES)
    reference = strong["methods"]["solver_sensor_ic"]["slices"]["hard_ood"]["mean_rel_error"]
    bars = protocol["pass_bars"]
    assert bars["error"]["limit"] == 1.5 * reference
    assert bars["time"]["limit_ratio"] == 0.25
    assert bars["coverage"]["range"] == [0.85, 0.95]
    assert protocol["intervals"]["alpha"] == 0.1
    assert protocol["hard_ood"]["threshold_nu"] == 0.027028120493367818
    assert protocol["retraining"]["retrained"] is True
    assert len(protocol["checkpoints"]) == 20
    assert "results" not in protocol and "scores" not in protocol


def test_validation_record_reads_validation_only_and_cap_follows_rule() -> None:
    record = _json(RECORD)
    assert record["split"] == "val"
    assert record["test_split_read"] is False
    table = record["cap_selection"]["table"]
    admissible = [item for item in table if item["time_ratio"] <= TIME_GUARD]
    assert admissible
    best = min(item["mean_rel_error"] for item in admissible)
    expected = next(item["cap"] for item in admissible if item["mean_rel_error"] <= ERROR_SLACK * best)
    assert record["cap_selection"]["selected_cap"] == expected


def test_scores_match_protocol_and_bars() -> None:
    scores = _json(SCORES)
    protocol = _json(PROTOCOL)
    assert scores["protocol_sha256"] == file_sha256(PROTOCOL)
    assert scores["validation_record_sha256"] == file_sha256(RECORD)
    bars = scores["pass_bars"]
    hard = scores["clean"]["arms"]["hybrid_1e-2"]["slices"]["hard_ood"]
    cold = scores["clean"]["cold"]["hard_ood"]
    assert bars["error"]["value"] == hard["mean_rel_error"]["mean"]
    assert bars["error"]["passed"] == (bars["error"]["value"] <= protocol["pass_bars"]["error"]["limit"])
    assert bars["time"]["value"] == hard["mean_seconds"]["mean"] / cold["mean_seconds"]
    assert bars["time"]["passed"] == (bars["time"]["value"] <= 0.25)
    coverage = scores["clean"]["arms"]["hybrid_1e-2"]["slices"]["full_test"]["coverage"]["mean"]
    assert bars["coverage"]["value"] == coverage
    assert bars["coverage"]["passed"] == (0.85 <= coverage <= 0.95)
    assert bars["all_passed"] == all(bars[name]["passed"] for name in ("error", "time", "coverage"))
    assert len(scores["instance_ids"]) == 128
    assert scores["cold_reproduction"]["n_instances"] == 128


def test_note_is_rendered_from_scores() -> None:
    assert NOTE.read_text(encoding="utf-8") == render_report(_json(SCORES))
