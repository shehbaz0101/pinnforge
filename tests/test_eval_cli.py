"""``pinnforge eval`` with torch installed."""

from __future__ import annotations

import json
from pathlib import Path

import pytest

pytestmark = pytest.mark.ml


@pytest.fixture(autouse=True)
def _need_torch() -> None:
    pytest.importorskip("torch")


def _train_tiny(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    from pinnforge.cli import main

    monkeypatch.chdir(tmp_path)
    assert (
        main(
            [
                "train",
                "--equation",
                "harmonic",
                "--epochs",
                "1",
                "--seed",
                "0",
                "--n-interior",
                "4",
                "--n-ic",
                "2",
                "--hidden-widths",
                "4,4",
                "--checkpoint-dir",
                "ckpts",
                "--log-path",
                "metrics.jsonl",
            ]
        )
        == 0
    )


def test_eval_command_prints_summary_and_writes_json(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    capsys: pytest.CaptureFixture[str],
) -> None:
    from pinnforge.cli import main

    _train_tiny(tmp_path, monkeypatch)
    capsys.readouterr()
    args = [
        "eval",
        "--checkpoint",
        "ckpts/checkpoint.pt",
        "--equation",
        "harmonic",
        "--n-interior",
        "8",
        "--seed",
        "1",
        "--bins",
        "4",
        "--write-json",
        "runs/eval.json",
    ]
    assert main(args) == 0
    lines = capsys.readouterr().out.splitlines()
    assert lines[0] == "equation: harmonic_oscillator"
    assert lines[1] == "checkpoint: ckpts/checkpoint.pt"
    assert lines[2] == "method: uniform"
    assert lines[3] == "seed: 1"
    assert lines[4] == "n_interior: 8"
    assert lines[5] == "reference: analytical"
    assert lines[6].startswith("l2: ")
    assert lines[7].startswith("relative_l2: ")
    assert lines[8].startswith("residual_mean_abs: ")
    assert lines[9].startswith("residual_max_abs: ")
    assert lines[10] == "json: runs/eval.json"
    payload = json.loads((tmp_path / "runs" / "eval.json").read_text(encoding="utf-8"))
    assert payload["format"] == "pinnforge.eval.v1"
    assert payload["equation_id"] == "harmonic_oscillator"
    assert payload["n_interior"] == 8
    assert payload["bins"] == 4
    assert sum(payload["histogram"]["counts"]) == 8
    assert len(payload["histogram"]["edges"]) == 5
    assert payload["l2"] is not None


def test_eval_command_rejects_a_mismatched_equation(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    from pinnforge.cli import main

    _train_tiny(tmp_path, monkeypatch)
    with pytest.raises(SystemExit) as exc:
        main(["eval", "--checkpoint", "ckpts/checkpoint.pt", "--equation", "burgers"])
    assert exc.value.code == 2


def test_eval_command_rejects_paths_outside_the_working_directory(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    from pinnforge.cli import main

    monkeypatch.chdir(tmp_path)
    with pytest.raises(SystemExit) as escaped:
        main(["eval", "--checkpoint", "../checkpoint.pt", "--equation", "harmonic"])
    assert escaped.value.code == 2
    (tmp_path / "checkpoint.pt").write_bytes(b"not a checkpoint")
    with pytest.raises(SystemExit) as outside_json:
        main(
            [
                "eval",
                "--checkpoint",
                "checkpoint.pt",
                "--equation",
                "harmonic",
                "--write-json",
                "../eval.json",
            ]
        )
    assert outside_json.value.code == 2


def test_eval_command_rejects_a_missing_checkpoint(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    from pinnforge.cli import main

    monkeypatch.chdir(tmp_path)
    with pytest.raises(SystemExit) as exc:
        main(["eval", "--checkpoint", "missing.pt", "--equation", "harmonic"])
    assert exc.value.code == 2


def test_burgers_eval_command_marks_the_field_error_unavailable(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    capsys: pytest.CaptureFixture[str],
) -> None:
    from pinnforge.cli import main

    monkeypatch.chdir(tmp_path)
    assert (
        main(
            [
                "train",
                "--equation",
                "burgers",
                "--epochs",
                "1",
                "--n-interior",
                "4",
                "--n-bc",
                "4",
                "--hidden-widths",
                "4,4",
                "--checkpoint-dir",
                "ckpts",
                "--log-path",
                "metrics.jsonl",
            ]
        )
        == 0
    )
    capsys.readouterr()
    assert (
        main(
            [
                "eval",
                "--checkpoint",
                "ckpts/checkpoint.pt",
                "--equation",
                "burgers",
                "--n-interior",
                "4",
            ]
        )
        == 0
    )
    text = capsys.readouterr().out
    assert "reference: unavailable" in text
    assert "l2: unavailable" in text
    assert "relative_l2: unavailable" in text
    assert "residual_mean_abs: " in text
