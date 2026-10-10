"""Burgers inverse demo: path checks without torch, and one CPU run."""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from pinnforge.demo_inverse import (
    DEMO_FORMAT,
    InverseDemoConfig,
    format_result_number,
    results_table_markdown,
    run_burgers_inverse_demo,
)

ROOT = Path(__file__).resolve().parents[1]
_FAST = InverseDemoConfig(
    seed=0,
    n_space=32,
    n_modes=2,
    nu_min=0.03,
    nu_max=0.07,
    n_train=2,
    n_val=1,
    n_test=2,
    dt=0.01,
    t_final=0.14,
    save_dt=0.02,
    n_sensors=16,
    input_frames=4,
    output_frames=4,
    stride=4,
    width=4,
    modes=2,
    layers=1,
    epochs=1,
    batch_size=2,
    lr=1e-3,
    nu_grid=5,
)


def test_demo_help_lists_burgers_inverse(capsys: pytest.CaptureFixture[str]) -> None:
    from pinnforge.cli import main

    with pytest.raises(SystemExit) as caught:
        main(["demo", "--help"])
    assert caught.value.code == 0
    text = capsys.readouterr().out
    assert "burgers-inverse" in text
    assert "--output" in text
    assert "samples/configs/harmonic.yaml" in text


def test_sample_demo_rejects_output(capsys: pytest.CaptureFixture[str]) -> None:
    from pinnforge.cli import main

    with pytest.raises(SystemExit) as caught:
        main(["demo", "--output", "runs/nope"])
    assert caught.value.code == 2
    assert "--output" in capsys.readouterr().err


def test_burgers_inverse_rejects_equation_and_epochs(capsys: pytest.CaptureFixture[str]) -> None:
    from pinnforge.cli import main

    with pytest.raises(SystemExit) as caught:
        main(["demo", "burgers-inverse", "--equation", "poisson"])
    assert caught.value.code == 2
    assert "--equation" in capsys.readouterr().err
    with pytest.raises(SystemExit) as caught:
        main(["demo", "burgers-inverse", "--epochs", "1"])
    assert caught.value.code == 2
    assert "--epochs" in capsys.readouterr().err


def test_burgers_inverse_rejects_an_escaping_output(tmp_path: Path, capsys: pytest.CaptureFixture[str]) -> None:
    from pinnforge.cli import main

    with pytest.raises(SystemExit) as caught:
        main(["demo", "burgers-inverse", "--data-root", str(tmp_path), "--output", "../outside"])
    assert caught.value.code == 2
    assert "working directory" in capsys.readouterr().err
    assert not (tmp_path / "runs").exists()


def test_readme_quotes_the_committed_demo_record() -> None:
    record = json.loads((ROOT / "docs" / "demo" / "result.json").read_text(encoding="utf-8"))
    report = (ROOT / "docs" / "demo" / "result.md").read_text(encoding="utf-8")
    readme = (ROOT / "README.md").read_text(encoding="utf-8")
    table = results_table_markdown(record)
    assert table in report
    assert table in readme
    assert format_result_number(record["val_relative_l2"]) in readme
    assert format_result_number(record["wall_clock_seconds"]) in readme
    assert format_result_number(record["nu_grid_step"]) in readme
    assert record["environment"]["torch"] in readme
    assert record["environment"]["numpy"] in readme
    assert record["command"] in readme
    assert str(record["parameter_count"]) in readme
    for name in ("viscosity_scatter.png", "relative_error.png", "field_sensors.png"):
        blob = (ROOT / "docs" / "demo" / name).read_bytes()
        assert blob.startswith(b"\x89PNG\r\n\x1a\n")
        assert len(blob) < 200_000
        assert f"docs/demo/{name}" in readme


def test_results_table_is_built_from_the_record() -> None:
    record = {
        "config": {"n_sensors": 32},
        "aggregates": {
            "closed_form": {
                "n": 4,
                "mean_abs_error": 0.1,
                "mean_rel_error": 0.2,
                "max_rel_error": 0.3,
                "n_failures": 1,
            },
            "operator": {
                "n": 4,
                "mean_abs_error": 0.4,
                "mean_rel_error": 0.5,
                "max_rel_error": 0.6,
                "n_failures": 0,
            },
            "baseline": {
                "n": 4,
                "mean_abs_error": 0.7,
                "mean_rel_error": 0.8,
                "max_rel_error": 0.9,
                "n_failures": 2,
            },
        },
    }
    table = results_table_markdown(record)
    assert "closed form, 32 sensors" in table
    assert "1.00000000e-01" in table
    assert "| 4 |" in table
    assert table.endswith("\n")


def test_import_of_the_demo_does_not_load_torch() -> None:
    import subprocess
    import sys

    completed = subprocess.run(
        [
            sys.executable,
            "-c",
            "import pinnforge.demo_inverse, sys; assert 'torch' not in sys.modules",
        ],
        check=False,
        capture_output=True,
        text=True,
    )
    assert completed.returncode == 0, completed.stderr


@pytest.mark.ml
def test_fast_demo_writes_finite_scores_and_pngs(tmp_path: Path) -> None:
    result = run_burgers_inverse_demo(tmp_path / "out", output_label="runs/out", config=_FAST)
    record = json.loads((tmp_path / "out" / "result.json").read_text(encoding="utf-8"))
    assert record["format"] == DEMO_FORMAT
    assert record["config"]["n_sensors"] == 16
    assert record["aggregates"]["closed_form"]["n"] == 2
    assert record["scope"]["true_nu_used_in_estimator"] is False
    for name in ("closed_form", "operator", "baseline"):
        assert record["aggregates"][name]["mean_rel_error"] >= 0.0
    table = results_table_markdown(record)
    assert table in (tmp_path / "out" / "result.md").read_text(encoding="utf-8")
    assert table in result.summary
    for name in ("viscosity_scatter.png", "relative_error.png", "field_sensors.png"):
        blob = (tmp_path / "out" / name).read_bytes()
        assert blob.startswith(b"\x89PNG\r\n\x1a\n")
        assert len(blob) < 200_000
    assert (tmp_path / "out" / "checkpoint.pt").is_file()
    assert (tmp_path / "out" / "index.html").is_file()


@pytest.mark.ml
def test_cli_published_demo_exits_zero(tmp_path: Path, capsys: pytest.CaptureFixture[str]) -> None:
    from pinnforge.cli import main

    assert main(["demo", "burgers-inverse", "--data-root", str(tmp_path), "--output", "runs/burgers-inverse"]) == 0
    text = capsys.readouterr().out
    assert text.startswith("pinnforge demo burgers-inverse\n")
    assert "closed form, 32 sensors" in text
    record = json.loads((tmp_path / "runs" / "burgers-inverse" / "result.json").read_text(encoding="utf-8"))
    assert record["config"]["epochs"] == 40
    assert record["config"]["n_sensors"] == 32
    assert results_table_markdown(record) in text


def test_api_rejects_an_escaping_output(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    pytest.importorskip("fastapi")
    pytest.importorskip("httpx2")
    from fastapi.testclient import TestClient

    from pinnforge.api import create_app

    monkeypatch.chdir(tmp_path)
    with TestClient(create_app()) as client:
        response = client.post("/demo/burgers-inverse", json={"output": "../outside"})
    assert response.status_code == 422
    assert "working directory" in response.json()["detail"]


def test_api_reports_missing_ml_extra(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    pytest.importorskip("fastapi")
    pytest.importorskip("httpx2")
    from fastapi.testclient import TestClient

    from pinnforge.api import create_app
    from pinnforge.api import service as api_service
    from pinnforge.demo import DemoDependencyError

    def missing(*_args: object, **_kwargs: object) -> object:
        raise DemoDependencyError("need the ml extra")

    monkeypatch.chdir(tmp_path)
    monkeypatch.setattr(api_service, "_import_burgers_inverse", lambda: missing)
    with TestClient(create_app()) as client:
        response = client.post("/demo/burgers-inverse", json={"output": "runs/burgers-inverse"})
    assert response.status_code == 503
    assert "ml extra" in response.json()["detail"]


@pytest.mark.ml
def test_api_runs_the_fast_config_when_the_runner_is_wrapped(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    pytest.importorskip("fastapi")
    pytest.importorskip("httpx2")
    from fastapi.testclient import TestClient

    from pinnforge.api import create_app
    from pinnforge.api import service as api_service
    from pinnforge.demo_inverse import run_burgers_inverse_demo as real_run

    def wrapped(output_dir: Path, *, output_label: str) -> object:
        return real_run(output_dir, output_label=output_label, config=_FAST)

    monkeypatch.chdir(tmp_path)
    monkeypatch.setattr(api_service, "_import_burgers_inverse", lambda: wrapped)
    with TestClient(create_app()) as client:
        response = client.post("/demo/burgers-inverse", json={"output": "runs/api-demo"})
    assert response.status_code == 200
    body = response.json()
    assert body["output"] == "runs/api-demo"
    assert body["n_test"] == 2
    assert body["closed_form_mean_rel_error"] >= 0.0
    assert Path(body["result_json"]).is_file()
    assert (tmp_path / "runs" / "api-demo" / "index.html").is_file()
