"""CLI coverage for ``pinnforge sample``."""

from __future__ import annotations

import json
import os
import subprocess
import sys
from pathlib import Path

import numpy as np
import pytest

from pinnforge.cli import main
from pinnforge.sampling import load_sample_record, sample_equation

ROOT = Path(__file__).resolve().parents[1]


def test_sample_help_lists_the_equation_flag(capsys: pytest.CaptureFixture[str]) -> None:
    with pytest.raises(SystemExit) as exc:
        main(["sample", "--help"])
    assert exc.value.code == 0
    help_text = capsys.readouterr().out
    assert "--equation" in help_text
    assert "--n-interior" in help_text
    assert "--seed" in help_text
    assert "--output" in help_text
    assert "harmonic" in help_text


def test_harmonic_sample_summary(capsys: pytest.CaptureFixture[str]) -> None:
    assert main(["sample", "--equation", "harmonic", "--n-interior", "64", "--seed", "0"]) == 0
    text = capsys.readouterr().out
    assert "equation: harmonic_oscillator" in text
    assert "method: uniform" in text
    assert "seed: 0" in text
    assert "counts: interior=64 ic=16 bc=0" in text
    assert "bounds: t=[0.0, 1.0]" in text


def test_burgers_and_poisson_use_equation_defaults(capsys: pytest.CaptureFixture[str]) -> None:
    assert main(["sample", "--equation", "burgers", "--n-interior", "4"]) == 0
    burgers = capsys.readouterr().out
    assert "counts: interior=4 ic=16 bc=16" in burgers
    assert "bounds: x=[-1.0, 1.0]" in burgers
    assert "bounds: t=[0.0, 1.0]" in burgers
    assert main(["sample", "--equation", "poisson", "--n-interior", "4", "--method", "stratified"]) == 0
    poisson = capsys.readouterr().out
    assert "method: stratified" in poisson
    assert "counts: interior=4 ic=0 bc=16" in poisson


def test_explicit_counts_override_defaults(capsys: pytest.CaptureFixture[str]) -> None:
    assert main(["sample", "--equation", "harmonic", "--n-interior", "4", "--n-ic", "0", "--n-bc", "0"]) == 0
    assert "counts: interior=4 ic=0 bc=0" in capsys.readouterr().out


def test_sample_writes_a_relative_json_record(tmp_path: Path, monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]) -> None:
    monkeypatch.chdir(tmp_path)
    code = main(
        [
            "sample",
            "--equation",
            "harmonic",
            "--n-interior",
            "8",
            "--n-ic",
            "2",
            "--n-bc",
            "0",
            "--seed",
            "0",
            "--output",
            "out/batch.json",
        ]
    )
    assert code == 0
    assert "counts: interior=8 ic=2 bc=0" in capsys.readouterr().out
    path = tmp_path / "out" / "batch.json"
    payload = json.loads(path.read_text(encoding="utf-8"))
    assert payload["format"] == "pinnforge.sample.v1"
    assert payload["batch"]["counts"]["interior"] == 8
    spec, config, stored = load_sample_record(path)
    fresh = sample_equation(spec, config)
    np.testing.assert_array_equal(fresh.coordinates(), stored.coordinates())


@pytest.mark.parametrize(
    ("output", "message"),
    [
        ("/tmp/pinnforge-batch.json", "relative"),
        ("../outside.json", "working directory"),
        ("batch.txt", ".json"),
    ],
)
def test_sample_rejects_unsafe_output_paths(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    capsys: pytest.CaptureFixture[str],
    output: str,
    message: str,
) -> None:
    monkeypatch.chdir(tmp_path)
    with pytest.raises(SystemExit) as exc:
        main(["sample", "--equation", "harmonic", "--n-interior", "4", "--output", output])
    assert exc.value.code == 2
    assert message in capsys.readouterr().err
    assert not (tmp_path / "batch.txt").exists()


def test_sample_rejects_bad_counts(capsys: pytest.CaptureFixture[str]) -> None:
    with pytest.raises(SystemExit) as negative:
        main(["sample", "--equation", "harmonic", "--n-interior", "-1"])
    assert negative.value.code == 2
    assert "n_interior" in capsys.readouterr().err
    with pytest.raises(SystemExit) as poisson_ic:
        main(["sample", "--equation", "poisson", "--n-interior", "4", "--n-ic", "2"])
    assert poisson_ic.value.code == 2
    assert "n_ic" in capsys.readouterr().err
    with pytest.raises(SystemExit) as harmonic_bc:
        main(["sample", "--equation", "harmonic_oscillator", "--n-interior", "4", "--n-bc", "2"])
    assert harmonic_bc.value.code == 2
    assert "n_bc" in capsys.readouterr().err
    with pytest.raises(SystemExit) as empty:
        main(["sample", "--equation", "harmonic", "--n-interior", "0", "--n-ic", "0", "--n-bc", "0"])
    assert empty.value.code == 2
    assert "at least one" in capsys.readouterr().err


def test_sample_module_entrypoint(tmp_path: Path) -> None:
    env = os.environ.copy()
    env["PYTHONPATH"] = str(ROOT / "src")
    completed = subprocess.run(
        [
            sys.executable,
            "-m",
            "pinnforge",
            "sample",
            "--equation",
            "harmonic",
            "--n-interior",
            "4",
            "--n-ic",
            "1",
            "--n-bc",
            "0",
            "--seed",
            "0",
        ],
        check=False,
        capture_output=True,
        text=True,
        env=env,
        cwd=tmp_path,
    )
    assert completed.returncode == 0, completed.stderr
    assert "equation: harmonic_oscillator" in completed.stdout
    assert "counts: interior=4 ic=1 bc=0" in completed.stdout
    assert "torch" not in completed.stderr
