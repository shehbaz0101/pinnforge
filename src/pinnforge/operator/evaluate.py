"""Score an FNO on windows from one instance split.

Predictions are denormalized before relative L2. The training loss is not
recomputed as the selection metric: selection and this report use the mean
per-window relative L2 in physical units. The Burgers residual reported
here uses the same discrete stencil as training.
"""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path

from pinnforge.ml_import import require_torch
from pinnforge.operator.checkpoint import LoadedFNO, load_fno_checkpoint
from pinnforge.operator.data import load_split_windows
from pinnforge.operator.fno import FNO1d, pack_inputs
from pinnforge.operator.metrics import (
    mean_relative_l2,
    median_relative_l2,
    normalized_mse,
    persistence_prediction,
    physical_targets,
    pooled_relative_l2,
)
from pinnforge.operator.residual import LossConfig, prediction_window_residual, residual_stats
from pinnforge.operator.windows import WindowDataset

torch = require_torch()

EVAL_FORMAT = "pinnforge.fno_eval.v1"


@dataclass(frozen=True, slots=True)
class SplitScore:
    """Held-out scores for one split. Arrays are not retained."""

    split: str
    n_windows: int
    n_instances: int
    instance_ids: tuple[int, ...]
    normalized_mse: float
    mean_relative_l2: float
    median_relative_l2: float
    pooled_relative_l2: float
    persistence_mean_relative_l2: float
    mean_abs_residual: float
    residual_mse: float
    target_mean_abs_residual: float
    residual_definition: dict[str, object]

    def as_dict(self) -> dict[str, object]:
        return {
            "format": EVAL_FORMAT,
            "split": self.split,
            "n_windows": self.n_windows,
            "n_instances": self.n_instances,
            "instance_ids": list(self.instance_ids),
            "normalized_mse": self.normalized_mse,
            "mean_relative_l2": self.mean_relative_l2,
            "median_relative_l2": self.median_relative_l2,
            "pooled_relative_l2": self.pooled_relative_l2,
            "persistence_mean_relative_l2": self.persistence_mean_relative_l2,
            "mean_abs_residual": self.mean_abs_residual,
            "residual_mse": self.residual_mse,
            "target_mean_abs_residual": self.target_mean_abs_residual,
            "residual_definition": self.residual_definition,
            "loss_note": (
                "normalized_mse is in manifest-normalized u; relative L2 is on denormalized u; "
                "mean_abs_residual is the mean of |R| under residual_definition"
            ),
        }


def predict_normalized(model: FNO1d, dataset: WindowDataset, *, batch_size: int) -> torch.Tensor:
    """Network output in normalized ``u`` space, shape ``(n_windows, frames, n_space)``."""

    if isinstance(batch_size, bool) or not isinstance(batch_size, int) or batch_size < 1:
        raise ValueError("batch_size must be an integer >= 1")
    model.eval()
    blocks: list[torch.Tensor] = []
    inputs = torch.as_tensor(dataset.inputs, dtype=torch.float32)
    nu = torch.as_tensor(dataset.nu, dtype=torch.float32)
    count = inputs.shape[0]
    with torch.no_grad():
        for start in range(0, count, batch_size):
            stop = min(start + batch_size, count)
            packed = pack_inputs(inputs[start:stop], nu[start:stop])
            blocks.append(model(packed).detach().cpu())
    return torch.cat(blocks, dim=0)


def score_dataset(
    model: FNO1d,
    dataset: WindowDataset,
    *,
    batch_size: int,
    residual: LossConfig | None = None,
) -> SplitScore:
    """Relative L2, normalized MSE, and the Burgers residual on ``dataset``.

    ``residual`` selects the stencil. It defaults to the physical residual
    with the last two input frames included. That default is also what a
    data-only checkpoint records.
    """

    config = residual if residual is not None else LossConfig()
    if not isinstance(config, LossConfig):
        raise TypeError("residual must be a LossConfig")
    prediction = predict_normalized(model, dataset, batch_size=batch_size).numpy()
    reference = physical_targets(dataset)
    predicted = dataset.norm.denormalize_u(prediction)
    persistence = persistence_prediction(dataset)
    ids = tuple(sorted(dataset.instance_id_set()))
    forecast = residual_stats(
        prediction_window_residual(dataset.inputs, prediction, dataset.nu, dataset.norm, config)
    )
    labels = residual_stats(
        prediction_window_residual(dataset.inputs, dataset.targets, dataset.nu, dataset.norm, config)
    )
    return SplitScore(
        split=dataset.split,
        n_windows=dataset.n_windows(),
        n_instances=dataset.n_instances(),
        instance_ids=ids,
        normalized_mse=normalized_mse(prediction, dataset.targets),
        mean_relative_l2=mean_relative_l2(predicted, reference),
        median_relative_l2=median_relative_l2(predicted, reference),
        pooled_relative_l2=pooled_relative_l2(predicted, reference),
        persistence_mean_relative_l2=mean_relative_l2(persistence, reference),
        mean_abs_residual=forecast.mean_abs,
        residual_mse=forecast.mean_square,
        target_mean_abs_residual=labels.mean_abs,
        residual_definition=_residual_definition(config),
    )


def evaluate_checkpoint(
    checkpoint: Path,
    pilot_dir: Path,
    manifest_path: Path,
    *,
    split: str = "test",
    batch_size: int = 32,
    check_field_hash: bool = True,
    residual: LossConfig | None = None,
) -> SplitScore:
    """Load the selected weights and score one split.

    The window spec and normalization stored in the checkpoint must match
    the manifest used to rebuild the windows. Test files are read only
    when ``split`` is ``test``.
    """

    loaded = load_fno_checkpoint(checkpoint)
    datasets = load_split_windows(
        pilot_dir,
        manifest_path,
        loaded.spec,
        (split,),
        check_field_hash=check_field_hash,
    )
    dataset = datasets[split]
    _require_same_norm(loaded, dataset)
    config = residual if residual is not None else loaded.loss
    return score_dataset(loaded.model, dataset, batch_size=batch_size, residual=config)


def write_eval_json(score: SplitScore, path: Path, *, checkpoint: Path, epoch: int) -> Path:
    """Write the eval record. The path's parent directories are created."""

    destination = Path(path)
    destination.parent.mkdir(parents=True, exist_ok=True)
    payload = score.as_dict()
    payload["checkpoint"] = str(checkpoint)
    payload["epoch"] = epoch
    destination.write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")
    return destination


def _residual_definition(config: LossConfig) -> dict[str, object]:
    return {
        "equation": "u_t + u u_x - nu u_xx",
        "time": "second-order central difference on saved frames",
        "space": "spectral_derivative with Nyquist multiplier 0",
        "scope": config.residual_scope,
        "field": config.residual_space,
        "dt": config.dt,
    }


def _require_same_norm(loaded: LoadedFNO, dataset: WindowDataset) -> None:
    pairs = (
        ("u_mean", loaded.norm.u_mean, dataset.norm.u_mean),
        ("u_std", loaded.norm.u_std, dataset.norm.u_std),
        ("nu_mean", loaded.norm.nu_mean, dataset.norm.nu_mean),
        ("nu_std", loaded.norm.nu_std, dataset.norm.nu_std),
    )
    for label, left, right in pairs:
        if abs(left - right) > 1e-12:
            raise ValueError(f"checkpoint {label} does not match the manifest used for evaluation")
