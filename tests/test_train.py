"""CPU training loop, metrics, and checkpoints."""

from __future__ import annotations

import math
from pathlib import Path

import pytest

pytestmark = pytest.mark.ml


@pytest.fixture(autouse=True)
def _need_torch() -> None:
    pytest.importorskip("torch")


def _tiny(**overrides: object):
    from pinnforge.training import TrainConfig

    payload: dict[str, object] = {
        "equation_id": "harmonic",
        "epochs": 2,
        "seed": 0,
        "n_interior": 8,
        "n_ic": 4,
        "hidden_widths": (8, 8),
        "checkpoint_dir": "ckpts",
        "log_path": "metrics.jsonl",
    }
    payload.update(overrides)
    present = {key: value for key, value in payload.items() if value is not None}
    return TrainConfig(**present)


def test_harmonic_loss_decreases_from_epoch_zero(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    import torch

    from pinnforge.training import read_metrics, train_loop

    monkeypatch.chdir(tmp_path)
    config = _tiny(epochs=40, n_interior=32, n_ic=16, hidden_widths=(16, 16), lr=1e-2)
    result = train_loop(config)
    start = result.history[0]
    end = result.history[-1]
    assert [row.epoch for row in result.history] == list(range(41))
    assert end.loss < start.loss or end.loss_pde < start.loss_pde
    assert end.lr == pytest.approx(config.lr)
    weighted = config.w_pde * end.loss_pde + config.w_ic * end.loss_ic + config.w_bc * end.loss_bc
    assert end.loss == pytest.approx(weighted)
    assert all(parameter.device.type == "cpu" for parameter in result.model.parameters())
    assert read_metrics(result.log_path) == result.history
    assert not torch.cuda.is_initialized()


def test_checkpoint_round_trip(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    import torch

    from pinnforge.training import load_checkpoint, train_loop

    monkeypatch.chdir(tmp_path)
    config = _tiny(epochs=3, seed=1)
    result = train_loop(config)
    loaded = load_checkpoint(Path("ckpts/checkpoint.pt"))
    tagged = load_checkpoint(Path("ckpts/epoch_0003.pt"))
    initial = load_checkpoint(Path("ckpts/epoch_0000.pt"))
    assert loaded.epoch == tagged.epoch == 3
    assert loaded.config == tagged.config == config
    assert initial.epoch == 0
    assert loaded.path == (tmp_path / "ckpts" / "checkpoint.pt").resolve()
    for key, value in result.model.state_dict().items():
        assert torch.equal(value, loaded.model.state_dict()[key])
        assert torch.equal(value, tagged.model.state_dict()[key])
    changed = [
        key
        for key, value in result.model.state_dict().items()
        if not torch.equal(value, initial.model.state_dict()[key])
    ]
    assert changed
    coords = torch.linspace(0.0, 1.0, 5, dtype=torch.float32).unsqueeze(-1)
    loaded.model.eval()
    result.model.eval()
    with torch.no_grad():
        assert torch.allclose(result.model(coords), loaded.model(coords))
    assert loaded.model.training is False
    assert all(parameter.device.type == "cpu" for parameter in loaded.model.parameters())


def test_same_seed_repeats_the_history(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    from pinnforge.training import train_loop

    monkeypatch.chdir(tmp_path)
    first = train_loop(_tiny(epochs=2, checkpoint_dir="a", log_path="a.jsonl"))
    second = train_loop(_tiny(epochs=2, checkpoint_dir="b", log_path="b.jsonl"))
    assert first.history == second.history


@pytest.mark.parametrize("equation", ["burgers", "poisson"])
def test_other_equations_finish_on_cpu(equation: str, tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    from pinnforge.training import train_loop

    monkeypatch.chdir(tmp_path)
    config = _tiny(
        equation_id=equation,
        epochs=1,
        n_interior=4,
        n_ic=None if equation == "burgers" else 0,
        n_bc=4,
        hidden_widths=(4, 4),
        checkpoint_dir=equation,
        log_path=f"{equation}.jsonl",
    )
    result = train_loop(config)
    assert len(result.history) == 2
    assert all(math.isfinite(row.loss) for row in result.history)
    if equation == "burgers":
        assert result.history[-1].loss_bc > 0.0
    else:
        assert result.config.n_ic == 0


def test_paths_must_stay_inside_the_working_directory(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    from pinnforge.training import load_checkpoint, train_loop

    monkeypatch.chdir(tmp_path)
    with pytest.raises(ValueError, match="working directory"):
        train_loop(_tiny(log_path="../metrics.jsonl"))
    with pytest.raises(ValueError, match="working directory"):
        train_loop(_tiny(checkpoint_dir="../ckpts"))
    with pytest.raises(ValueError, match="relative"):
        load_checkpoint(Path("/tmp/checkpoint.pt"))


def test_checkpoint_interval_skips_intermediate_epochs(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    from pinnforge.training import train_loop

    monkeypatch.chdir(tmp_path)
    train_loop(_tiny(epochs=4, checkpoint_interval=2, checkpoint_dir="ckpts", log_path="metrics.jsonl"))
    assert (tmp_path / "ckpts" / "epoch_0000.pt").is_file()
    assert (tmp_path / "ckpts" / "epoch_0002.pt").is_file()
    assert (tmp_path / "ckpts" / "epoch_0004.pt").is_file()
    assert not (tmp_path / "ckpts" / "epoch_0001.pt").exists()
    assert not (tmp_path / "ckpts" / "epoch_0003.pt").exists()


def test_resume_matches_an_uninterrupted_run(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    import torch

    from pinnforge.training import train_loop

    monkeypatch.chdir(tmp_path)
    full = train_loop(_tiny(epochs=4, seed=1, checkpoint_dir="full", log_path="full.jsonl"))
    train_loop(_tiny(epochs=2, seed=1, checkpoint_dir="mid", log_path="mid.jsonl"))
    resumed = train_loop(
        _tiny(
            epochs=4,
            seed=1,
            checkpoint_dir="resumed",
            log_path="resumed.jsonl",
            resume_from="mid/checkpoint.pt",
        )
    )
    assert [row.epoch for row in resumed.history] == list(range(5))
    assert resumed.history == full.history
    for key, value in full.model.state_dict().items():
        assert torch.equal(value, resumed.model.state_dict()[key])


def test_manifest_logs_distinct_rng_streams(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    import json

    import numpy as np

    from pinnforge.sampling import (
        SampleConfig,
        default_spec,
        points_sha256,
        sample_equation,
        stream_generator,
    )
    from pinnforge.training import train_loop

    monkeypatch.chdir(tmp_path)
    config = _tiny(epochs=1, seed=0, n_interior=8, n_ic=4)
    train_loop(config)
    manifest = json.loads((tmp_path / "ckpts" / "manifest.json").read_text(encoding="utf-8"))
    assert manifest["format"] == "pinnforge.manifest.v1"
    sequences = [manifest["streams"][name]["seed_sequence"] for name in ("train", "validation", "test")]
    assert len({tuple(item) for item in sequences}) == 3
    assert manifest["environment"]["pinnforge"]
    assert "torch" in manifest["environment"]
    spec = default_spec("harmonic")
    sample = SampleConfig(n_interior=8, n_ic=4, n_bc=0, seed=0, method="uniform")
    validation = sample_equation(spec, sample, rng=stream_generator(0, "validation")).interior
    assert points_sha256(validation) == manifest["validation_interior_sha256"]
    train_points = sample_equation(spec, sample, rng=stream_generator(0, "train")).interior
    test_points = sample_equation(spec, sample, rng=stream_generator(0, "test")).interior
    assert not np.array_equal(train_points, test_points)
    assert not np.array_equal(train_points, validation)


def test_load_rejects_a_foreign_payload(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    import torch

    from pinnforge.training import load_checkpoint

    monkeypatch.chdir(tmp_path)
    torch.save({"format": "nope"}, tmp_path / "bad.pt")
    with pytest.raises(ValueError, match="format"):
        load_checkpoint(Path("bad.pt"))

