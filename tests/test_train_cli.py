"""``pinnforge train`` with torch installed."""

from __future__ import annotations

import json
from pathlib import Path

import pytest

pytestmark = pytest.mark.ml


@pytest.fixture(autouse=True)
def _need_torch() -> None:
    pytest.importorskip("torch")


def test_train_command_writes_metrics_and_checkpoint(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    capsys: pytest.CaptureFixture[str],
) -> None:
    from pinnforge.cli import main

    monkeypatch.chdir(tmp_path)
    args = [
        "train",
        "--equation",
        "harmonic",
        "--epochs",
        "2",
        "--seed",
        "0",
        "--n-interior",
        "8",
        "--n-ic",
        "4",
        "--hidden-widths",
        "8,8",
        "--checkpoint-dir",
        "ckpts",
        "--log-path",
        "runs/metrics.jsonl",
    ]
    assert main(args) == 0
    captured = capsys.readouterr().out
    lines = captured.splitlines()
    assert lines[0] == "equation: harmonic_oscillator"
    assert lines[1] == "seed: 0"
    assert lines[2] == "epochs: 2"
    assert lines[3] == "device: cpu"
    assert lines[4] == "checkpoint: ckpts/checkpoint.pt"
    assert lines[5] == "log: runs/metrics.jsonl"
    assert lines[6].startswith("loss: ")
    assert lines[7].startswith("loss_pde: ")
    assert lines[8].startswith("loss_ic: ")
    assert lines[9].startswith("loss_bc: ")
    assert lines[10].startswith("lr: ")
    metrics = (tmp_path / "runs" / "metrics.jsonl").read_text(encoding="utf-8").splitlines()
    assert len(metrics) == 3
    first = json.loads(metrics[0])
    assert first["format"] == "pinnforge.metrics.v1"
    assert first["epoch"] == 0
    assert "loss_pde" in first
    assert (tmp_path / "ckpts" / "checkpoint.pt").is_file()
    assert (tmp_path / "ckpts" / "epoch_0002.pt").is_file()


def test_omitted_flags_match_train_config_defaults() -> None:
    from pinnforge.cli import _train_config, build_parser
    from pinnforge.training import TrainConfig

    parser = build_parser()
    harmonic = parser.parse_args(["train", "--equation", "harmonic"])
    assert _train_config(harmonic) == TrainConfig(equation_id="harmonic")
    burgers = parser.parse_args(["train", "--equation", "burgers"])
    assert _train_config(burgers) == TrainConfig(equation_id="burgers")
    poisson = parser.parse_args(["train", "--equation", "poisson"])
    assert _train_config(poisson) == TrainConfig(equation_id="poisson")


def test_train_command_rejects_a_bad_epoch_count() -> None:
    from pinnforge.cli import main

    with pytest.raises(SystemExit) as exc:
        main(["train", "--equation", "harmonic", "--epochs", "0"])
    assert exc.value.code == 2


def test_train_command_rejects_hidden_widths() -> None:
    from pinnforge.cli import main

    with pytest.raises(SystemExit) as exc:
        main(["train", "--equation", "harmonic", "--hidden-widths", "8,nope"])
    assert exc.value.code == 2
