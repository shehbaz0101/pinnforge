"""``pinnforge run --config`` with torch installed."""

from __future__ import annotations

import json
from pathlib import Path

import pytest

pytestmark = pytest.mark.ml

ROOT = Path(__file__).resolve().parents[1]


@pytest.fixture(autouse=True)
def _need_torch() -> None:
    pytest.importorskip("torch")


def test_run_command_trains_and_evaluates(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    capsys: pytest.CaptureFixture[str],
) -> None:
    from pinnforge.cli import main

    monkeypatch.chdir(tmp_path)
    text = (ROOT / "samples/configs/harmonic.yaml").read_text(encoding="utf-8")
    (tmp_path / "harmonic.yaml").write_text(text, encoding="utf-8")
    assert main(["run", "--config", "harmonic.yaml"]) == 0
    lines = capsys.readouterr().out.splitlines()
    assert lines[0] == "equation: harmonic_oscillator"
    assert lines[1] == "seed: 0"
    assert lines[2] == "epochs: 2"
    assert lines[3] == "device: cpu"
    assert lines[4] == "checkpoint: runs/harmonic/checkpoints/checkpoint.pt"
    assert lines[5] == "log: runs/harmonic/metrics.jsonl"
    assert lines[6].startswith("loss: ")
    assert "reference: analytical" in lines
    assert any(line.startswith("l2: ") and "unavailable" not in line for line in lines)
    assert lines[-1] == "json: runs/harmonic/eval.json"
    metrics = (tmp_path / "runs/harmonic/metrics.jsonl").read_text(encoding="utf-8").splitlines()
    assert len(metrics) == 3
    assert (tmp_path / "runs/harmonic/checkpoints/checkpoint.pt").is_file()
    record = json.loads((tmp_path / "runs/harmonic/eval.json").read_text(encoding="utf-8"))
    assert record["format"] == "pinnforge.eval.v1"
    assert record["equation_id"] == "harmonic_oscillator"
    assert record["checkpoint"] == "runs/harmonic/checkpoints/checkpoint.pt"
    assert record["l2"] is not None
    assert len(record["histogram"]["counts"]) == 4


def test_checkpoint_keeps_the_overridden_spec(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    from pinnforge.evaluation import evaluate_checkpoint
    from pinnforge.experiments import load_experiment_config
    from pinnforge.experiments.run import run_experiment
    from pinnforge.training import load_checkpoint

    monkeypatch.chdir(tmp_path)
    (tmp_path / "exp.json").write_text(
        (ROOT / "samples/configs/poisson.json").read_text(encoding="utf-8"),
        encoding="utf-8",
    )
    config = load_experiment_config("exp.json")
    result = run_experiment(config)
    loaded = load_checkpoint("runs/poisson/checkpoints/checkpoint.pt")
    assert loaded.spec == config.equation
    assert loaded.spec.source.value == "sin_pi_x"
    scored = evaluate_checkpoint("runs/poisson/checkpoints/checkpoint.pt", config.eval)
    assert scored.l2 == pytest.approx(result.evaluation.l2)
    assert scored.residual_mean_abs == pytest.approx(result.evaluation.residual_mean_abs)
    assert result.eval_json == (tmp_path / "runs/poisson/eval.json").resolve()


def test_run_rejects_a_log_path_that_escapes(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    from pinnforge.cli import main

    monkeypatch.chdir(tmp_path)
    (tmp_path / "exp.json").write_text(
        json.dumps(
            {
                "equation": "harmonic",
                "train": {
                    "epochs": 1,
                    "n_interior": 4,
                    "n_ic": 2,
                    "hidden_widths": [4, 4],
                    "log_path": "../metrics.jsonl",
                },
            }
        ),
        encoding="utf-8",
    )
    with pytest.raises(SystemExit) as exc:
        main(["run", "--config", "exp.json"])
    assert exc.value.code == 2
