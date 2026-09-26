"""How many interior points an evaluation scores, and how they are drawn.

:mod:`pinnforge.evaluation` re-exports this model. Importing
:mod:`pinnforge.specs` does not import torch.

Interior points feed the residual histogram and the field error.
Held-out initial-condition and boundary rows are scored separately
when the spec has those conditions. ``n_ic`` and ``n_bc`` default to
``None``, which asks evaluation to choose a count that covers the
spec. Paths are not stored here.
:func:`pinnforge.evaluation.evaluate_checkpoint` checks the checkpoint
path, and :func:`pinnforge.evaluation.write_eval_json` checks the JSON
path.

The integer ``seed`` is the test-stream seed. It is not the training
stream, so the same integer does not redraw the training points.
"""

from __future__ import annotations

from typing import Annotated

from pydantic import BaseModel, BeforeValidator, ConfigDict, Field

from pinnforge.sampling import SampleMethod


def _reject_bool(value: object) -> object:
    if isinstance(value, bool):
        raise ValueError("must be an integer")
    return value


def _optional_count(value: object) -> object:
    if value is None:
        return None
    if isinstance(value, bool) or not isinstance(value, int) or value < 0:
        raise ValueError("must be a non-negative integer")
    return value


Count = Annotated[int, BeforeValidator(_reject_bool)]
OptionalCount = Annotated[int | None, BeforeValidator(_optional_count)]


class EvalConfig(BaseModel):
    """Interior sample used to score one model.

    ``n_interior`` points are drawn on the test RNG stream. ``seed`` is
    that stream's integer, not a shared generator with training.
    ``n_ic`` and ``n_bc`` are held-out condition counts. ``None`` lets
    evaluation cover the spec: initial rows when the equation has an
    initial condition, and one batch row per boundary face when the
    spec prescribes boundaries. ``bins`` is the number of
    ``numpy.histogram`` bins for ``|residual|``.
    """

    model_config = ConfigDict(extra="forbid", frozen=True)

    n_interior: Count = Field(default=64, ge=1)
    n_ic: OptionalCount = None
    n_bc: OptionalCount = None
    seed: Count = Field(default=0, ge=0)
    method: SampleMethod = "uniform"
    bins: Count = Field(default=10, ge=1)
