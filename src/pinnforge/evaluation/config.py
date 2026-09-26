"""How many interior points an evaluation scores, and how they are drawn.

The batch is interior only. Initial and boundary rows are not part of
the residual histogram or the field error. Paths are not stored here;
:func:`pinnforge.evaluation.evaluate_checkpoint` checks the checkpoint
path, and :func:`pinnforge.evaluation.write_eval_json` checks the JSON
path.
"""

from __future__ import annotations

from typing import Annotated

from pydantic import BaseModel, BeforeValidator, ConfigDict, Field

from pinnforge.sampling import SampleMethod


def _reject_bool(value: object) -> object:
    if isinstance(value, bool):
        raise ValueError("must be an integer")
    return value


Count = Annotated[int, BeforeValidator(_reject_bool)]


class EvalConfig(BaseModel):
    """Interior sample used to score one model.

    ``n_interior`` points are drawn with the Day 2 sampler. ``seed`` and
    ``method`` match :class:`~pinnforge.sampling.SampleConfig`. ``bins``
    is the number of ``numpy.histogram`` bins for ``|residual|``.
    """

    model_config = ConfigDict(extra="forbid", frozen=True)

    n_interior: Count = Field(default=64, ge=1)
    seed: Count = Field(default=0, ge=0)
    method: SampleMethod = "uniform"
    bins: Count = Field(default=10, ge=1)
