"""Adam training for the Day 1 residuals.

Importing this package requires torch (the optional ``ml`` extra).
The package root does not import it. :func:`train_loop` draws one
seeded batch, minimizes the weighted residual and soft penalties, and
writes ``metrics.jsonl`` plus a checkpoint.
"""

from pinnforge.training.checkpoint import (
    CHECKPOINT_FORMAT,
    LoadedCheckpoint,
    load_checkpoint,
    save_checkpoint,
)
from pinnforge.training.config import TrainConfig
from pinnforge.training.loop import TrainResult, train_loop
from pinnforge.training.metrics import METRICS_FORMAT, EpochMetrics, read_metrics

__all__ = [
    "CHECKPOINT_FORMAT",
    "EpochMetrics",
    "LoadedCheckpoint",
    "METRICS_FORMAT",
    "TrainConfig",
    "TrainResult",
    "load_checkpoint",
    "read_metrics",
    "save_checkpoint",
    "train_loop",
]
