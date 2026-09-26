"""Experiment config loading. These tests do not import torch."""

from __future__ import annotations

import os
import subprocess
import sys
from pathlib import Path

import pytest
from pydantic import ValidationError

from pinnforge.equations import Burgers1DSpec, HarmonicOscillatorSpec, PoissonToySpec
from pinnforge.experiments import (
    ExperimentConfig,
    dump_experiment_config,
    load_experiment_config,
)

ROOT = Path(__file__).resolve().parents[1]


@pytest.fixture
def repo_cwd(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.chdir(ROOT)


def test_sample_yaml_and_json_load(repo_cwd: None) -> None:
    harmonic = load_experiment_config("samples/configs/harmonic.yaml")
    assert isinstance(harmonic.equation, HarmonicOscillatorSpec)
    assert harmonic.equation.angular_frequency == 2.0
    assert harmonic.equation.time.upper == 1.0
    assert harmonic.train.equation_id == "harmonic_oscillator"
    assert harmonic.train.epochs == 2
    assert harmonic.train.n_interior == 8
    assert harmonic.train.n_ic == 4
    assert harmonic.train.n_bc == 0
    assert harmonic.train.hidden_widths == (8, 8)
    assert harmonic.train.checkpoint_dir == "runs/harmonic/checkpoints"
    assert harmonic.train.log_path == "runs/harmonic/metrics.jsonl"
    assert harmonic.eval.n_interior == 8
    assert harmonic.eval.seed == 1
    assert harmonic.eval.bins == 4
    assert harmonic.eval_json == "runs/harmonic/eval.json"

    poisson = load_experiment_config("samples/configs/poisson.json")
    assert isinstance(poisson.equation, PoissonToySpec)
    assert poisson.equation.source.value == "sin_pi_x"
    assert poisson.train.equation_id == "poisson_toy"
    assert poisson.train.n_ic == 0
    assert poisson.train.n_bc == 4
    assert poisson.train.epochs == 2
    assert poisson.eval_json == "runs/poisson/eval.json"

    burgers = load_experiment_config("samples/configs/burgers.yaml")
    assert isinstance(burgers.equation, Burgers1DSpec)
    assert burgers.equation.nu == 0.01
    assert burgers.train.equation_id == "burgers_1d"
    assert burgers.train.epochs == 2
    assert burgers.train.n_interior == 8
    assert burgers.train.n_ic == 4
    assert burgers.train.n_bc == 4
    assert burgers.train.hidden_widths == (8, 8)
    assert burgers.train.checkpoint_dir == "runs/burgers/checkpoints"
    assert burgers.eval_json == "runs/burgers/eval.json"


def test_equation_id_alias_and_inline_spec() -> None:
    by_id = ExperimentConfig.model_validate({"equation_id": "burgers", "train": {"epochs": 1}})
    assert by_id.equation.equation_id == "burgers_1d"
    assert by_id.train.n_ic == 16
    assert by_id.train.n_bc == 16
    assert by_id.eval.n_interior == 64
    assert by_id.eval_json is None

    inline = ExperimentConfig.model_validate(
        {
            "equation": {
                "equation_id": "poisson",
                "dimensions": 1,
                "source": "one",
                "x": {"lower": 0.0, "upper": 1.0},
                "boundary_conditions": [
                    {"variable": "x", "kind": "dirichlet", "side": "min", "value": 0.0},
                    {"variable": "x", "kind": "dirichlet", "side": "max", "value": 0.0},
                ],
            },
            "train": {"epochs": 1, "n_bc": 4, "hidden_widths": [4]},
        }
    )
    assert isinstance(inline.equation, PoissonToySpec)
    assert inline.equation.source.value == "one"
    assert inline.train.hidden_widths == (4,)


def test_round_trip_yaml_and_json(tmp_path: Path, monkeypatch: pytest.MonkeyPatch, repo_cwd: None) -> None:
    loaded = load_experiment_config("samples/configs/harmonic.yaml")
    poisson = load_experiment_config("samples/configs/poisson.json")
    monkeypatch.chdir(tmp_path)
    dump_experiment_config(loaded, "harmonic.json")
    dump_experiment_config(loaded, "harmonic.yml")
    dump_experiment_config(poisson, "poisson.yaml")
    assert load_experiment_config("harmonic.json") == loaded
    assert load_experiment_config("harmonic.yml") == loaded
    assert load_experiment_config("poisson.yaml") == poisson
    dump_experiment_config(load_experiment_config("poisson.yaml"), "poisson.json")
    assert load_experiment_config("poisson.json") == poisson


def test_rejects_path_traversal(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.chdir(tmp_path)
    (tmp_path / "exp.json").write_text('{"equation": "harmonic"}\n', encoding="utf-8")
    outside = tmp_path.parent / "outside-exp.json"
    outside.write_text('{"equation": "harmonic"}\n', encoding="utf-8")
    with pytest.raises(ValueError, match="relative"):
        load_experiment_config(outside)
    with pytest.raises(ValueError, match="relative"):
        load_experiment_config("/tmp/exp.json")
    with pytest.raises(ValueError, match="working directory"):
        load_experiment_config("../outside-exp.json")
    with pytest.raises(ValueError, match="working directory"):
        load_experiment_config("nested/../../outside-exp.json")
    link = tmp_path / "link"
    link.symlink_to(tmp_path.parent, target_is_directory=True)
    with pytest.raises(ValueError, match="working directory"):
        load_experiment_config("link/outside-exp.json")
    with pytest.raises(ValueError, match="yaml"):
        load_experiment_config("notes.txt")
    with pytest.raises(ValueError, match="not found"):
        load_experiment_config("missing.yaml")


def test_rejects_invalid_documents(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.chdir(tmp_path)
    (tmp_path / "bad.json").write_text("[1, 2]\n", encoding="utf-8")
    (tmp_path / "bad.yaml").write_text("- just-a-list\n", encoding="utf-8")
    (tmp_path / "empty.yaml").write_text("", encoding="utf-8")
    with pytest.raises(ValueError, match="mapping"):
        load_experiment_config("bad.json")
    with pytest.raises(ValueError, match="mapping"):
        load_experiment_config("bad.yaml")
    with pytest.raises(ValueError, match="mapping"):
        load_experiment_config("empty.yaml")


@pytest.mark.parametrize(
    "payload",
    [
        {},
        {"equation": "heat"},
        {"equation": "harmonic", "equation_params": {"omega": -1}},
        {"equation": "harmonic", "equation_id": "harmonic"},
        {
            "equation": {
                "equation_id": "harmonic_oscillator",
                "omega": 1.0,
                "time": {"lower": 0.0, "upper": 1.0},
                "initial_condition": {"components": {"u": 1.0, "du_dt": 0.0}},
            },
            "equation_params": {"omega": 2.0},
        },
        {"equation": "harmonic", "train": {"equation_id": "poisson"}},
        {"equation": "harmonic", "train": {"epochs": 0}},
        {"equation": "harmonic", "train": {"checkpoint_dir": "/tmp/ckpts"}},
        {"equation": "harmonic", "eval_json": "/tmp/eval.json"},
        {"equation": "harmonic", "eval_json": "eval.txt"},
        {"equation": "poisson", "train": {"n_ic": 4}},
        {"equation": "harmonic", "nope": 1},
    ],
)
def test_schema_rejects_invalid_configs(payload: dict[str, object]) -> None:
    with pytest.raises(ValidationError):
        ExperimentConfig.model_validate(payload)


def test_run_cli_rejects_unsafe_config_paths(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    capsys: pytest.CaptureFixture[str],
) -> None:
    from pinnforge.cli import main

    monkeypatch.chdir(tmp_path)
    with pytest.raises(SystemExit) as absolute:
        main(["run", "--config", "/tmp/exp.yaml"])
    assert absolute.value.code == 2
    with pytest.raises(SystemExit) as escape:
        main(["run", "--config", "../exp.yaml"])
    assert escape.value.code == 2
    err = capsys.readouterr().err
    assert "relative" in err
    assert "working directory" in err


def test_import_does_not_load_torch() -> None:
    env = os.environ.copy()
    env["PYTHONPATH"] = str(ROOT / "src")
    code = (
        "import pinnforge.experiments, pinnforge.specs, sys\n"
        "assert 'torch' not in sys.modules\n"
        "assert 'pinnforge.training' not in sys.modules\n"
        "assert 'pinnforge.evaluation' not in sys.modules\n"
        "assert 'pinnforge.experiments.run' not in sys.modules\n"
    )
    completed = subprocess.run(
        [sys.executable, "-c", code],
        check=False,
        capture_output=True,
        text=True,
        env=env,
    )
    assert completed.returncode == 0, completed.stderr
