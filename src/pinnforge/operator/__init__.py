"""Data-only Fourier neural operator on Stage 2 Burgers windows.

Window cuts, the instance split, and normalization are numpy-only. The
network and the training loop live in :mod:`pinnforge.operator.fno` and
:mod:`pinnforge.operator.train`. Those modules import torch and need the
optional ``ml`` extra. This package does not add a physics residual and
does not replace :func:`pinnforge.reference.burgers.reference_solution`.
"""

from pinnforge.operator.windows import (
    DEFAULT_INPUT_FRAMES,
    DEFAULT_OUTPUT_FRAMES,
    DEFAULT_STRIDE,
    FieldNorm,
    Trajectory,
    WindowDataset,
    WindowSpec,
    build_window_datasets,
    cut_trajectory,
    field_norm_from_manifest,
    load_pilot_manifest,
    split_sets,
    viscosity_train_stats,
    window_starts,
)

__all__ = [
    "DEFAULT_INPUT_FRAMES",
    "DEFAULT_OUTPUT_FRAMES",
    "DEFAULT_STRIDE",
    "FieldNorm",
    "Trajectory",
    "WindowDataset",
    "WindowSpec",
    "build_window_datasets",
    "cut_trajectory",
    "field_norm_from_manifest",
    "load_pilot_manifest",
    "split_sets",
    "viscosity_train_stats",
    "window_starts",
]
