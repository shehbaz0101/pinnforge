"""HTTP catalog and path sandbox. These tests do not import torch."""

from __future__ import annotations

import sys
from pathlib import Path

import pytest

pytest.importorskip("fastapi")
pytest.importorskip("httpx2")

from fastapi.testclient import TestClient  # noqa: E402

from pinnforge import __version__  # noqa: E402
from pinnforge.api import create_app  # noqa: E402
from pinnforge.api import service as api_service  # noqa: E402

ROOT = Path(__file__).resolve().parents[1]


@pytest.fixture
def client(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> TestClient:
    monkeypatch.chdir(tmp_path)
    with TestClient(create_app()) as http:
        yield http


def test_health(client: TestClient) -> None:
    response = client.get("/health")
    assert response.status_code == 200
    assert response.json() == {"status": "ok", "version": __version__}


def test_equations_catalog(client: TestClient) -> None:
    response = client.get("/equations")
    assert response.status_code == 200
    body = response.json()
    assert [item["equation_id"] for item in body] == [
        "burgers_1d",
        "harmonic_oscillator",
        "poisson_toy",
    ]
    harmonic = body[1]
    assert harmonic["aliases"] == ["harmonic"]
    assert "omega" in harmonic["parameters"]
    assert "u''" in harmonic["summary"]


def test_equation_detail_accepts_alias(client: TestClient) -> None:
    response = client.get("/equations/harmonic")
    assert response.status_code == 200
    body = response.json()
    assert body["equation_id"] == "harmonic_oscillator"
    assert body["aliases"] == ["harmonic"]
    assert body["spec"]["equation_id"] == "harmonic_oscillator"
    assert body["spec"]["omega"] == 1.0
    burgers = client.get("/equations/burgers_1d")
    assert burgers.status_code == 200
    assert burgers.json()["spec"]["equation_id"] == "burgers_1d"


def test_unknown_equation_is_404(client: TestClient) -> None:
    response = client.get("/equations/navier_stokes")
    assert response.status_code == 404
    assert "unknown equation" in response.json()["detail"]


def test_openapi_lists_the_day7_routes(client: TestClient) -> None:
    paths = client.get("/openapi.json").json()["paths"]
    assert "/health" in paths
    assert "/equations" in paths
    assert "/equations/{id_or_alias}" in paths
    assert "/run" in paths
    assert "/train" in paths
    assert "/eval" in paths


def test_run_rejects_a_config_path_outside_the_workdir(client: TestClient, tmp_path: Path) -> None:
    outside = tmp_path.parent / "outside-exp.json"
    outside.write_text('{"equation": "harmonic"}\n', encoding="utf-8")
    escaped = client.post("/run", json={"config": "../outside-exp.json"})
    assert escaped.status_code == 422
    assert "working directory" in escaped.json()["detail"]
    absolute = client.post("/run", json={"config": outside.as_posix()})
    assert absolute.status_code == 422
    assert "relative" in absolute.json()["detail"]
    missing = client.post("/run", json={"config": "missing.yaml"})
    assert missing.status_code == 422
    assert "not found" in missing.json()["detail"]


def test_run_rejects_both_sources_and_unknown_fields(client: TestClient) -> None:
    both = client.post("/run", json={"config": "exp.yaml", "equation": "harmonic"})
    assert both.status_code == 422
    assert "not both" in both.text
    extra = client.post("/run", json={"equation": "harmonic", "steps": 1})
    assert extra.status_code == 422
    empty = client.post("/run", json={})
    assert empty.status_code == 422


def test_run_rejects_an_escaping_log_path_before_torch(client: TestClient) -> None:
    response = client.post(
        "/run",
        json={
            "equation": "harmonic",
            "train": {
                "epochs": 1,
                "n_interior": 4,
                "n_ic": 2,
                "hidden_widths": [4, 4],
                "log_path": "../metrics.jsonl",
            },
        },
    )
    assert response.status_code == 422
    assert "working directory" in response.json()["detail"]


def test_train_and_eval_reject_escaping_paths(client: TestClient) -> None:
    train = client.post(
        "/train",
        json={"equation_id": "harmonic", "epochs": 1, "checkpoint_dir": "../checkpoints"},
    )
    assert train.status_code == 422
    assert "working directory" in train.json()["detail"]
    evaluate = client.post(
        "/eval",
        json={"checkpoint": "../checkpoint.pt", "equation": "harmonic"},
    )
    assert evaluate.status_code == 422
    assert "working directory" in evaluate.json()["detail"]
    suffix = client.post(
        "/eval",
        json={"checkpoint": "checkpoint.json", "equation": "harmonic"},
    )
    assert suffix.status_code == 422
    assert ".pt" in suffix.json()["detail"]
    write_json = client.post(
        "/eval",
        json={
            "checkpoint": "checkpoints/checkpoint.pt",
            "equation": "harmonic",
            "write_json": "../eval.json",
        },
    )
    assert write_json.status_code == 422
    assert "working directory" in write_json.json()["detail"]


def test_valid_run_reports_missing_ml_extra(
    client: TestClient,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    def missing() -> object:
        raise api_service.MlExtraMissing("need the ml extra")

    monkeypatch.setattr(api_service, "_import_run_experiment", missing)
    copied = ROOT / "samples" / "configs" / "harmonic.yaml"
    Path("harmonic.yaml").write_text(copied.read_text(encoding="utf-8"), encoding="utf-8")
    response = client.post("/run", json={"config": "harmonic.yaml"})
    assert response.status_code == 503
    assert "ml extra" in response.json()["detail"]
    assert not Path("runs").exists()


def test_api_import_does_not_load_torch() -> None:
    env_code = (
        "import pinnforge.api.app, sys; "
        "from pinnforge.api.app import create_app; "
        "create_app(); "
        "assert 'torch' not in sys.modules"
    )
    import os
    import subprocess

    env = os.environ.copy()
    env["PYTHONPATH"] = str(ROOT / "src")
    completed = subprocess.run(
        [sys.executable, "-c", env_code],
        check=False,
        capture_output=True,
        text=True,
        env=env,
    )
    assert completed.returncode == 0, completed.stderr
