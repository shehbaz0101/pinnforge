"""JSON record for one evaluation.

The format tag is ``pinnforge.eval.v1``. ``l2``, ``relative_l2``, and
``max_abs_error`` are null when the equation has no matching reference
field, and ``relative_l2`` is also null when that field is identically
zero. ``ic_error`` and ``bc_error`` are null when the spec has no such
condition or the eval config asked for no rows. The histogram is
``numpy.histogram`` of the absolute residual on the interior points.
``rng`` identifies the test stream that drew the batch.
"""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path

from pinnforge.sampling import resolve_output_path

EVAL_FORMAT = "pinnforge.eval.v1"


@dataclass(frozen=True, slots=True)
class ResidualHistogram:
    """Bin counts and edges for ``|residual|``.

    ``counts`` has one entry per bin. ``edges`` has one more entry than
    ``counts``, the ``numpy.histogram`` convention.
    """

    counts: tuple[int, ...]
    edges: tuple[float, ...]

    def as_dict(self) -> dict[str, object]:
        """JSON-ready counts and edges."""

        return {"counts": list(self.counts), "edges": list(self.edges)}


@dataclass(frozen=True, slots=True)
class EvalResult:
    """Scalar scores and the residual histogram for one model.

    ``reference`` is ``analytical`` when ``l2`` was compared with a
    closed form or manufactured field, and ``unavailable`` when no
    matched reference exists (Burgers, or a Poisson spec whose boundary
    data does not match a candidate). ``checkpoint`` is the relative
    path passed to :func:`pinnforge.evaluation.evaluate_checkpoint`, or
    ``None`` for an in-memory model. ``l2`` is the root-mean-square
    field error. ``relative_l2`` divides that by the root-mean-square of
    the reference. ``max_abs_error`` is the max absolute field error.
    ``ic_error`` and ``bc_error`` are square roots of the unweighted
    condition penalties on held-out rows. ``bc_errors`` splits that by
    kind. ``rng`` is the test-stream identity.
    """

    equation_id: str
    n_interior: int
    seed: int
    method: str
    bins: int
    reference: str
    l2: float | None
    relative_l2: float | None
    residual_mean_abs: float
    residual_max_abs: float
    histogram: ResidualHistogram
    checkpoint: str | None = None
    max_abs_error: float | None = None
    ic_error: float | None = None
    bc_error: float | None = None
    bc_errors: dict[str, float] | None = None
    rng: dict[str, object] | None = None

    def as_dict(self) -> dict[str, object]:
        """JSON-ready mapping, including the format tag."""

        return {
            "format": EVAL_FORMAT,
            "equation_id": self.equation_id,
            "checkpoint": self.checkpoint,
            "n_interior": self.n_interior,
            "seed": self.seed,
            "method": self.method,
            "bins": self.bins,
            "reference": self.reference,
            "l2": self.l2,
            "relative_l2": self.relative_l2,
            "max_abs_error": self.max_abs_error,
            "ic_error": self.ic_error,
            "bc_error": self.bc_error,
            "bc_errors": None if self.bc_errors is None else dict(self.bc_errors),
            "rng": None if self.rng is None else dict(self.rng),
            "residual_mean_abs": self.residual_mean_abs,
            "residual_max_abs": self.residual_max_abs,
            "histogram": self.histogram.as_dict(),
        }


def write_eval_json(result: EvalResult, path: str | Path) -> Path:
    """Write ``result`` as indented JSON under a relative ``.json`` path.

    The path must stay inside the working directory. Parent directories
    are created.

    Raises:
        TypeError: ``result`` is not an :class:`EvalResult`.
        ValueError: ``path`` is absolute, escapes the working directory,
            or does not end in ``.json``.
    """

    if not isinstance(result, EvalResult):
        raise TypeError("result must be an EvalResult")
    resolved = resolve_output_path(path)
    resolved.parent.mkdir(parents=True, exist_ok=True)
    text = json.dumps(result.as_dict(), indent=2, allow_nan=False) + "\n"
    resolved.write_text(text, encoding="utf-8")
    return resolved
