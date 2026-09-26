"""How many points a sampler draws, and with which reproducible method.

Counts are non-negative integers. A config that asks for no points at all
is rejected. Equation helpers also reject counts the spec cannot host:
Poisson has no initial condition, and a harmonic spec with an empty
boundary list cannot draw boundary points.
"""

from __future__ import annotations

from typing import Annotated, Literal, Self

from pydantic import BaseModel, BeforeValidator, ConfigDict, Field, model_validator

SAMPLE_METHODS: tuple[str, ...] = ("uniform", "latin_hypercube", "stratified")

SampleMethod = Literal["uniform", "latin_hypercube", "stratified"]


def _reject_bool(value: object) -> object:
    if isinstance(value, bool):
        raise ValueError("must be an integer")
    return value


Count = Annotated[int, BeforeValidator(_reject_bool)]


class SampleConfig(BaseModel):
    """Counts, seed, and method for one collocation batch.

    ``n_interior`` points lie in the collocation box. ``n_ic`` points lie
    on the initial-time slice when the equation has one. ``n_bc`` points
    lie on the boundary faces named by the spec. ``seed`` is passed to
    ``numpy.random.default_rng``. ``method`` is ``uniform``,
    ``latin_hypercube``, or ``stratified``.
    """

    model_config = ConfigDict(extra="forbid", frozen=True)

    n_interior: Count = Field(default=64, ge=0)
    n_ic: Count = Field(default=0, ge=0)
    n_bc: Count = Field(default=0, ge=0)
    seed: Count = Field(default=0, ge=0)
    method: SampleMethod = "uniform"

    @model_validator(mode="after")
    def _has_a_point(self) -> Self:
        if self.n_interior + self.n_ic + self.n_bc == 0:
            raise ValueError("at least one of n_interior, n_ic, and n_bc must be positive")
        return self
