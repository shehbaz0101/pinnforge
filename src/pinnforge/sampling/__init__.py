"""Seeded collocation, initial-condition, and boundary sampling.

Draws are numpy arrays. Torch is not imported. A :class:`SampleConfig`
names counts, a method, and a seed. :func:`sample_equation` turns a Day 1
spec into a :class:`CollocationBatch` labeled ``interior``, ``ic``, and
``bc``.
"""

from pinnforge.sampling.batch import (
    POINT_LABELS,
    CollocationBatch,
    format_summary,
    sample_burgers,
    sample_equation,
    sample_harmonic,
    sample_poisson,
)
from pinnforge.sampling.config import SAMPLE_METHODS, SampleConfig, SampleMethod
from pinnforge.sampling.defaults import default_spec, resolve_equation_id
from pinnforge.sampling.draw import sample_boundary, sample_collocation
from pinnforge.sampling.io import (
    SAMPLE_RECORD_FORMAT,
    batch_from_dict,
    batch_to_dict,
    load_sample_record,
    resolve_output_path,
    write_sample_record,
)

__all__ = [
    "POINT_LABELS",
    "SAMPLE_METHODS",
    "SAMPLE_RECORD_FORMAT",
    "CollocationBatch",
    "SampleConfig",
    "SampleMethod",
    "batch_from_dict",
    "batch_to_dict",
    "default_spec",
    "format_summary",
    "load_sample_record",
    "resolve_equation_id",
    "resolve_output_path",
    "sample_boundary",
    "sample_burgers",
    "sample_collocation",
    "sample_equation",
    "sample_harmonic",
    "sample_poisson",
    "write_sample_record",
]
