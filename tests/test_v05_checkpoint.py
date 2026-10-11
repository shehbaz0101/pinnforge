"""Resumable v0.5 case log and per-seed noise replay."""

from __future__ import annotations

from pathlib import Path

import numpy as np

from pinnforge.v04.sensors import add_noise
from pinnforge.v05.checkpoint import CaseLog, replay_noise, run_cases


def _double_worker(job: dict) -> dict:
    return {"value": int(job["n"]) * 2, "seed": int(job["seed"])}


def test_case_log_skips_completed_and_ignores_bad_lines(tmp_path: Path) -> None:
    path = tmp_path / "cases.jsonl"
    log = CaseLog(path, "abc")
    log.put("feat|burgers2d|0.01|0|wsindy_raw|16", {"n_rows": 12})
    path.write_text(path.read_text(encoding="utf-8") + "{not json\n", encoding="utf-8")
    with path.open("a", encoding="utf-8") as handle:
        handle.write('{"protocol_sha256": "other", "key": "stale", "payload": {"n_rows": 1}}\n')
    again = CaseLog(path, "abc")
    assert "feat|burgers2d|0.01|0|wsindy_raw|16" in again
    assert "stale" not in again
    assert again.require("feat|burgers2d|0.01|0|wsindy_raw|16")["n_rows"] == 12


def test_run_cases_resumes_and_pool_matches_serial(tmp_path: Path) -> None:
    jobs = [{"key": f"n{i}", "n": i, "seed": i} for i in range(8)]
    serial = run_cases(jobs, log=CaseLog(tmp_path / "serial.jsonl", "tag"), workers=1, worker=_double_worker)
    first = CaseLog(tmp_path / "pool.jsonl", "tag")
    run_cases(jobs[:3], log=first, workers=2, worker=_double_worker)
    pooled = run_cases(jobs, log=CaseLog(tmp_path / "pool.jsonl", "tag"), workers=2, worker=_double_worker)
    assert serial == pooled
    assert all(serial[f"n{i}"]["value"] == 2 * i for i in range(8))
    text = (tmp_path / "pool.jsonl").read_text(encoding="utf-8").strip().splitlines()
    assert len(text) == 8


def test_replay_noise_is_per_seed_and_instance_order() -> None:
    protocol = {"noise": {"noise_seed": 0}}
    fields = np.arange(4 * 2 * 3 * 3, dtype=np.float64).reshape(4, 2, 3, 3)
    noise = 0.02
    seed = 2
    rng = np.random.default_rng(int(protocol["noise"]["noise_seed"]) + 100 * seed)
    serial = {item: add_noise(fields[item], noise, rng) for item in range(fields.shape[0])}
    for item in range(fields.shape[0]):
        replayed = replay_noise(fields, item, noise, seed, protocol)
        np.testing.assert_allclose(replayed, serial[item])
    other = replay_noise(fields, 1, noise, 3, protocol)
    assert not np.allclose(other, serial[1])
    clean = replay_noise(fields, 1, 0.0, seed, protocol)
    np.testing.assert_array_equal(clean, fields[1])
