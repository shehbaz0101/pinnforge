"""HTTP train, eval, and run. These tests import torch."""

from __future__ import annotations

import json
from pathlib import Path

import pytest

pytest.importorskip("fastapi")
pytest.importorskip("httpx2")

pytestmark = pytest.mark.ml

from fastapi.testclient import TestClient  # noqa: E402

from pinnforge.api import create_app  # noqa: E402

ROOT = Path(__file__).resolve().parents[1]


@pytest.fixture(autouse=True)
def _need_torch() -> None:
    pytest.importorskip("torch")


@pytest.fixture
def client(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> TestClient:
    monkeypatch.chdir(tmp_path)
    with TestClient(create_app()) as http:
        yield http


def test_run_config_trains_and_evaluates(client: TestClient, tmp_path: Path) -> None:
    text = (ROOT / "samples/configs/harmonic.yaml").read_text(encoding="utf-8")
    (tmp_path / "harmonic.yaml").write_text(text, encoding="utf-8")
    response = client.post("/run", json={"config": "harmonic.yaml"})
    assert response.status_code == 200
    body = response.json()
    assert body["equation_id"] == "harmonic_oscillator"
    assert body["train"]["epochs"] == 2
    assert body["train"]["seed"] == 0
    assert body["checkpoint"] == "runs/harmonic/checkpoints/checkpoint.pt"
    assert body["log"] == "runs/harmonic/metrics.jsonl"
    assert body["evaluation"]["format"] == "pinnforge.eval.v1"
    assert body["evaluation"]["reference"] == "analytical"
    assert body["evaluation"]["l2"] is not None
    assert body["eval_json"] == "runs/harmonic/eval.json"
    assert (tmp_path / "runs/harmonic/checkpoints/checkpoint.pt").is_file()
    record = json.loads((tmp_path / "runs/harmonic/eval.json").read_text(encoding="utf-8"))
    assert record["l2"] == body["evaluation"]["l2"]
    assert len(record["histogram"]["counts"]) == 4


def test_inline_train_then_eval(client: TestClient, tmp_path: Path) -> None:
    trained = client.post(
        "/train",
        json={
            "equation_id": "harmonic",
            "epochs": 1,
            "n_interior": 4,
            "n_ic": 2,
            "hidden_widths": [4, 4],
            "checkpoint_dir": "checkpoints",
            "log_path": "metrics.jsonl",
        },
    )
    assert trained.status_code == 200
    train_body = trained.json()
    assert train_body["equation_id"] == "harmonic_oscillator"
    assert train_body["checkpoint"] == "checkpoints/checkpoint.pt"
    assert train_body["epochs"] == 1
    assert (tmp_path / "checkpoints/checkpoint.pt").is_file()
    scored = client.post(
        "/eval",
        json={
            "checkpoint": train_body["checkpoint"],
            "equation": "harmonic",
            "n_interior": 4,
            "bins": 4,
            "write_json": "eval.json",
        },
    )
    assert scored.status_code == 200
    evaluation = scored.json()
    assert evaluation["format"] == "pinnforge.eval.v1"
    assert evaluation["reference"] == "analytical"
    assert evaluation["l2"] is not None
    assert evaluation["checkpoint"] == "checkpoints/checkpoint.pt"
    assert evaluation["eval_json"] == "eval.json"
    assert evaluation["bins"] == 4
    written = json.loads((tmp_path / "eval.json").read_text(encoding="utf-8"))
    assert written["format"] == "pinnforge.eval.v1"
    assert written["l2"] == evaluation["l2"]
    mismatch = client.post(
        "/eval",
        json={"checkpoint": "checkpoints/checkpoint.pt", "equation": "burgers"},
    )
    assert mismatch.status_code == 422
    assert "burgers" in mismatch.json()["detail"]
