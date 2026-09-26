"""FNO shapes, a tiny overfit, and the operator CLI. Torch is required."""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest

pytestmark = pytest.mark.ml


@pytest.fixture(autouse=True)
def _need_torch() -> None:
    pytest.importorskip("torch")


def test_fno_output_shape_matches_target_frames() -> None:
    import torch

    from pinnforge.operator.fno import FNO1d, pack_inputs

    model = FNO1d(in_channels=5, out_channels=4, width=4, modes=4, n_layers=2)
    inputs = torch.zeros(3, 4, 16)
    nu = torch.zeros(3)
    packed = pack_inputs(inputs, nu)
    assert packed.shape == (3, 5, 16)
    output = model(packed)
    assert output.shape == (3, 4, 16)
    assert model.parameter_count() > 0


def test_tiny_overfit_drops_the_training_loss() -> None:
    from pinnforge.operator.train import fit_fno
    from pinnforge.operator.windows import FieldNorm, WindowSpec, build_window_datasets

    spec = WindowSpec(input_frames=4, output_frames=4, stride=8)
    norm = FieldNorm(u_mean=0.0, u_std=1.0, nu_mean=0.06, nu_std=0.02)
    datasets = build_window_datasets(
        [
            _smooth_trajectory(1, "train", 0.04),
            _smooth_trajectory(2, "train", 0.08),
            _smooth_trajectory(3, "val", 0.05),
        ],
        spec,
        norm,
    )
    assert datasets["train"].instance_id_set().isdisjoint(datasets["val"].instance_id_set())
    _model, history = fit_fno(
        datasets["train"],
        datasets["val"],
        width=8,
        modes=4,
        n_layers=2,
        epochs=25,
        batch_size=4,
        lr=1e-3,
        seed=0,
    )
    assert history[0].epoch == 0
    assert history[-1].train_mse < history[0].train_mse * 0.5
    assert min(row.val_relative_l2 for row in history) <= history[0].val_relative_l2


def test_checkpoint_roundtrip_and_held_out_ids(tmp_path: Path) -> None:
    from pinnforge.operator.checkpoint import load_fno_checkpoint, save_fno_checkpoint
    from pinnforge.operator.evaluate import score_dataset
    from pinnforge.operator.fno import FNO1d
    from pinnforge.operator.train import fit_fno
    from pinnforge.operator.windows import FieldNorm, WindowSpec, build_window_datasets

    spec = WindowSpec(input_frames=4, output_frames=4, stride=8)
    norm = FieldNorm(u_mean=0.0, u_std=1.0, nu_mean=0.06, nu_std=0.02)
    datasets = build_window_datasets(
        [
            _smooth_trajectory(1, "train", 0.04),
            _smooth_trajectory(2, "val", 0.05),
            _smooth_trajectory(3, "test", 0.07),
        ],
        spec,
        norm,
    )
    model, history = fit_fno(
        datasets["train"],
        datasets["val"],
        width=4,
        modes=2,
        n_layers=2,
        epochs=1,
        batch_size=2,
        lr=1e-3,
        seed=1,
    )
    selected = min(history, key=lambda row: (row.val_relative_l2, row.epoch))
    path = save_fno_checkpoint(
        tmp_path / "checkpoint.pt",
        model,
        epoch=selected.epoch,
        spec=spec,
        norm=norm,
        seed=1,
        val_relative_l2=selected.val_relative_l2,
    )
    loaded = load_fno_checkpoint(path)
    assert loaded.epoch == selected.epoch
    assert loaded.spec == spec
    assert isinstance(loaded.model, FNO1d)
    score = score_dataset(loaded.model, datasets["test"], batch_size=2)
    assert score.instance_ids == (3,)
    assert score.split == "test"
    assert score.n_windows == datasets["test"].n_windows()
    assert math_isfinite(score.mean_relative_l2)


def _smooth_trajectory(instance_id: int, split: str, nu: float):
    from pinnforge.operator.windows import Trajectory

    n_space = 32
    n_times = 24
    x = np.linspace(-1.0, 1.0, n_space, endpoint=False)
    time = np.arange(n_times, dtype=np.float64)
    phase = 0.3 * instance_id
    decay = np.exp(-0.03 * nu * time)
    field = decay[:, None] * np.sin(np.pi * x + phase)[None, :]
    return Trajectory(instance_id=instance_id, split=split, nu=nu, u=field)


def math_isfinite(value: float) -> bool:
    return bool(np.isfinite(value))
