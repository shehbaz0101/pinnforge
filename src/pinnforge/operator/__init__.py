"""Fourier neural operator on Stage 2 Burgers windows.

Window cuts, the instance split, normalization, and the discrete Burgers
residual live in numpy. The network and the training loop live in
:mod:`pinnforge.operator.fno` and :mod:`pinnforge.operator.train`. Those
modules import torch and need the optional ``ml`` extra. The default loss
is normalized data MSE. A residual or hybrid loss uses the stencil in
:mod:`pinnforge.operator.residual`. Sparse recovery of the scalar
viscosity lives in :mod:`pinnforge.operator.inverse`. It reuses that
stencil. It does not replace
:func:`pinnforge.reference.burgers.reference_solution`.
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
