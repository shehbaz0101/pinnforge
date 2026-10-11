"""Resumable v0.5.2 case log keys include the lattice."""

from __future__ import annotations

from pathlib import Path

from pinnforge.v052.checkpoint import CaseLog, feature_key, run_cases


def _double_worker(job: dict) -> dict:
    return {"value": int(job["n"]) * 2}


def test_feature_key_includes_lattice() -> None:
    assert feature_key("ns2d", "16x16", 0.01, 2, 20) == "feat|ns2d|16x16|0.01|2|20"


def test_run_cases_resumes(tmp_path: Path) -> None:
    jobs = [{"key": f"n{i}", "n": i} for i in range(6)]
    serial = run_cases(jobs, log=CaseLog(tmp_path / "serial.jsonl", "tag"), workers=1, worker=_double_worker)
    first = CaseLog(tmp_path / "pool.jsonl", "tag")
    run_cases(jobs[:2], log=first, workers=2, worker=_double_worker)
    pooled = run_cases(jobs, log=CaseLog(tmp_path / "pool.jsonl", "tag"), workers=2, worker=_double_worker)
    assert serial == pooled
