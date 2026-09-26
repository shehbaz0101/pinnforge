"""``pinnforge demo``: sample placement without torch, and a short CPU smoke run."""

from __future__ import annotations

import socket
from pathlib import Path

import pytest

from pinnforge.demo import (
    MAX_DEMO_EPOCHS,
    SAMPLE_CONFIGS,
    bundled_samples_dir,
    load_demo_experiment,
    sample_config_path,
)
from pinnforge.offline import OfflineError

ROOT = Path(__file__).resolve().parents[1]


def test_package_data_matches_repo_samples() -> None:
    from importlib.resources import files

    from pinnforge.demo import read_bundled_sample

    packaged = files("pinnforge.data") / "configs"
    for name in ("harmonic.yaml", "burgers.yaml", "poisson.json"):
        text = (packaged / name).read_text(encoding="utf-8")
        assert text == (ROOT / "samples" / "configs" / name).read_text(encoding="utf-8")
        assert read_bundled_sample(f"samples/configs/{name}") == text


def test_sample_paths_match_checked_in_files() -> None:
    samples = bundled_samples_dir()
    assert samples == ROOT / "samples"
    assert sample_config_path("harmonic") == "samples/configs/harmonic.yaml"
    assert sample_config_path("harmonic_oscillator") == sample_config_path("harmonic")
    assert sample_config_path("poisson") == "samples/configs/poisson.json"
    assert sample_config_path("burgers_1d") == "samples/configs/burgers.yaml"
    for relative in SAMPLE_CONFIGS.values():
        assert (ROOT / relative).is_file()


def test_demo_help_lists_equation_data_root_and_epochs(capsys: pytest.CaptureFixture[str]) -> None:
    from pinnforge.cli import main

    with pytest.raises(SystemExit) as caught:
        main(["demo", "--help"])
    assert caught.value.code == 0
    text = capsys.readouterr().out
    assert "--equation" in text
    assert "--data-root" in text
    assert "--epochs" in text
    assert "samples/configs/harmonic.yaml" in text
    assert "samples/configs/poisson.json" in text
    assert "samples/configs/burgers.yaml" in text


def test_demo_rejects_an_unknown_equation() -> None:
    from pinnforge.cli import main

    with pytest.raises(SystemExit) as caught:
        main(["demo", "--equation", "navier-stokes"])
    assert caught.value.code == 2


def test_demo_rejects_too_many_epochs(capsys: pytest.CaptureFixture[str]) -> None:
    from pinnforge.cli import main

    with pytest.raises(SystemExit) as caught:
        main(["demo", "--epochs", str(MAX_DEMO_EPOCHS + 1)])
    assert caught.value.code == 2
    assert "1 to 5" in capsys.readouterr().err


def test_demo_rejects_a_missing_data_root(tmp_path: Path) -> None:
    from pinnforge.cli import main

    with pytest.raises(SystemExit) as caught:
        main(["demo", "--data-root", str(tmp_path / "missing")])
    assert caught.value.code == 2


def test_epoch_override_does_not_rewrite_the_sample(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("PINNFORGE_DATA_ROOT", str(tmp_path))
    original = (ROOT / "samples/configs/harmonic.yaml").read_text(encoding="utf-8")
    path, config = load_demo_experiment("harmonic", epochs=1)
    assert path == "samples/configs/harmonic.yaml"
    assert config.train.epochs == 1
    assert config.equation.equation_id == "harmonic_oscillator"
    copied = tmp_path / path
    assert copied.is_file()
    assert copied.read_text(encoding="utf-8") == original
    assert "epochs: 2" in original


def test_demo_loads_each_sample_inside_an_empty_root(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("PINNFORGE_DATA_ROOT", str(tmp_path))
    for name, equation_id in (
        ("poisson", "poisson_toy"),
        ("burgers", "burgers_1d"),
        ("harmonic_oscillator", "harmonic_oscillator"),
    ):
        path, config = load_demo_experiment(name)
        assert path == SAMPLE_CONFIGS[equation_id]
        assert config.equation.equation_id == equation_id
        assert config.train.epochs == 2
        assert config.train.hidden_widths == (8, 8)
        assert (tmp_path / path).is_file()


def test_demo_rejects_a_sample_that_leaves_the_root(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    configs = tmp_path / "samples" / "configs"
    configs.mkdir(parents=True)
    (configs / "harmonic.yaml").write_text(
        "equation: harmonic\ntrain:\n  epochs: 1\n  checkpoint_dir: ../outside\n",
        encoding="utf-8",
    )
    monkeypatch.setenv("PINNFORGE_DATA_ROOT", str(tmp_path))
    with pytest.raises(ValueError, match="working directory"):
        load_demo_experiment("harmonic")


def test_demo_rejects_a_sample_symlink_that_leaves_the_root(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    outside = tmp_path.parent / "outside-demo.yaml"
    outside.write_text("equation: harmonic\ntrain:\n  epochs: 1\n", encoding="utf-8")
    configs = tmp_path / "samples" / "configs"
    configs.mkdir(parents=True)
    (configs / "poisson.json").symlink_to(outside)
    monkeypatch.setenv("PINNFORGE_DATA_ROOT", str(tmp_path))
    with pytest.raises(ValueError, match="working directory"):
        load_demo_experiment("poisson")


@pytest.mark.ml
def test_demo_trains_harmonic_and_prints_the_reference_error(
    tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    from pinnforge.cli import main

    assert main(["demo", "--data-root", str(tmp_path), "--epochs", "1"]) == 0
    text = capsys.readouterr().out
    assert text.startswith("pinnforge demo\n")
    assert "equation: harmonic_oscillator\n" in text
    assert "config: samples/configs/harmonic.yaml\n" in text
    assert "epochs: 1\n" in text
    assert "device: cpu\n" in text
    assert "reference: analytical\n" in text
    assert "l2: " in text
    assert "unavailable" not in text.split("l2: ", 1)[1].splitlines()[0]
    assert "residual_mean_abs: " in text
    summary = tmp_path / "runs" / "harmonic" / "summary.txt"
    assert summary.read_text(encoding="utf-8") == text
    assert (tmp_path / "runs" / "harmonic" / "checkpoints" / "checkpoint.pt").is_file()
    assert (tmp_path / "runs" / "harmonic" / "eval.json").is_file()
    assert (tmp_path / "runs" / "harmonic" / "metrics.jsonl").is_file()
    with pytest.raises(OfflineError, match="offline-by-design"):
        socket.create_connection(("8.8.8.8", 53), timeout=1)


@pytest.mark.ml
def test_demo_poisson_and_burgers(tmp_path: Path, capsys: pytest.CaptureFixture[str]) -> None:
    from pinnforge.cli import main

    assert main(["demo", "--equation", "poisson", "--data-root", str(tmp_path), "--epochs", "1"]) == 0
    poisson = capsys.readouterr().out
    assert "equation: poisson_toy\n" in poisson
    assert "config: samples/configs/poisson.json\n" in poisson
    assert "reference: analytical\n" in poisson
    assert (tmp_path / "runs" / "poisson" / "summary.txt").read_text(encoding="utf-8") == poisson

    assert main(["demo", "--equation", "burgers", "--data-root", str(tmp_path), "--epochs", "1"]) == 0
    burgers = capsys.readouterr().out
    assert "equation: burgers_1d\n" in burgers
    assert "config: samples/configs/burgers.yaml\n" in burgers
    assert "reference: unavailable\n" in burgers
    assert "l2: unavailable\n" in burgers
    assert "residual_mean_abs: " in burgers
    assert (tmp_path / "runs" / "burgers" / "summary.txt").read_text(encoding="utf-8") == burgers
