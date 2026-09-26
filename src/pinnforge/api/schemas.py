"""JSON bodies for the localhost API.

These models describe the wire format. They do not import FastAPI or
torch. Train and eval fields are checked again by the Day 4 and Day 5
configs before any file is written.
"""

from __future__ import annotations

from typing import Any, Self

from pydantic import BaseModel, ConfigDict, Field, field_validator, model_validator

from pinnforge.sampling import SampleMethod


def _relative_text(value: str, *, label: str) -> str:
    if not isinstance(value, str) or value == "" or value != value.strip() or "\x00" in value:
        raise ValueError(f"{label} must be a non-empty relative path")
    if value.startswith("~"):
        raise ValueError(f"{label} must be a relative path")
    return value


class HealthResponse(BaseModel):
    """Liveness payload. No model and no filesystem access."""

    model_config = ConfigDict(extra="forbid")

    status: str
    version: str


class EquationSummary(BaseModel):
    """One built-in from :func:`pinnforge.equations.list_equations`."""

    model_config = ConfigDict(extra="forbid")

    equation_id: str
    aliases: list[str]
    summary: str
    parameters: list[str]


class EquationDetail(EquationSummary):
    """Catalog entry plus the default spec a config starts from."""

    spec: dict[str, Any]


class TrainSummary(BaseModel):
    """Final Adam step from ``POST /train`` or ``POST /run``."""

    model_config = ConfigDict(extra="forbid")

    equation_id: str
    seed: int
    epochs: int
    device: str
    checkpoint: str
    log: str
    loss: float
    loss_pde: float
    loss_ic: float
    loss_bc: float
    lr: float


class HistogramBody(BaseModel):
    """``numpy.histogram`` counts and edges for ``|residual|``."""

    model_config = ConfigDict(extra="forbid")

    counts: list[int]
    edges: list[float]


class EvalBody(BaseModel):
    """``pinnforge.eval.v1`` record returned by eval and run."""

    model_config = ConfigDict(extra="forbid")

    format: str
    equation_id: str
    checkpoint: str | None
    n_interior: int
    seed: int
    method: str
    bins: int
    reference: str
    l2: float | None
    relative_l2: float | None
    residual_mean_abs: float
    residual_max_abs: float
    histogram: HistogramBody


class EvalResponse(EvalBody):
    """Eval record plus the relative JSON path when one was requested."""

    eval_json: str | None = None


class RunResponse(BaseModel):
    """Train summary and eval record from one experiment."""

    model_config = ConfigDict(extra="forbid")

    equation_id: str
    checkpoint: str
    log: str
    train: TrainSummary
    evaluation: EvalBody
    eval_json: str | None = None


class RunRequest(BaseModel):
    """One experiment, either as a file or as an inline document.

    ``config`` is a relative ``.yaml``, ``.yml``, or ``.json`` path in
    the working directory. The other fields are the experiment document
    ``pinnforge run --config`` reads: ``equation`` or ``equation_id``,
    optional ``equation_params``, and optional ``train``, ``eval``, and
    ``eval_json``. Pass the path or the document, not both.
    """

    model_config = ConfigDict(extra="forbid")

    config: str | None = None
    equation: str | dict[str, Any] | None = None
    equation_id: str | None = None
    equation_params: dict[str, Any] | None = None
    train: dict[str, Any] | None = None
    eval: dict[str, Any] | None = None
    eval_json: str | None = None

    @field_validator("config")
    @classmethod
    def _config_path(cls, value: str | None) -> str | None:
        if value is None:
            return None
        return _relative_text(value, label="config")

    @field_validator("eval_json")
    @classmethod
    def _eval_json_path(cls, value: str | None) -> str | None:
        if value is None:
            return None
        return _relative_text(value, label="eval_json")

    @model_validator(mode="after")
    def _one_source(self) -> Self:
        inline = any(
            value is not None
            for value in (
                self.equation,
                self.equation_id,
                self.equation_params,
                self.train,
                self.eval,
                self.eval_json,
            )
        )
        if self.config is not None and inline:
            raise ValueError("pass a config path or an experiment body, not both")
        if self.config is None and self.equation is None and self.equation_id is None:
            raise ValueError("experiment config requires equation or config")
        return self


class EvalRequest(BaseModel):
    """Score one relative checkpoint. Same inputs as ``pinnforge eval``."""

    model_config = ConfigDict(extra="forbid")

    checkpoint: str
    equation: str
    n_interior: int | None = Field(default=None, ge=1)
    seed: int | None = Field(default=None, ge=0)
    method: SampleMethod | None = None
    bins: int | None = Field(default=None, ge=1)
    write_json: str | None = None

    @field_validator("checkpoint")
    @classmethod
    def _checkpoint_path(cls, value: str) -> str:
        return _relative_text(value, label="checkpoint")

    @field_validator("write_json")
    @classmethod
    def _write_json_path(cls, value: str | None) -> str | None:
        if value is None:
            return None
        return _relative_text(value, label="write_json")

    @field_validator("equation")
    @classmethod
    def _equation_name(cls, value: str) -> str:
        if not isinstance(value, str) or value == "" or value != value.strip():
            raise ValueError("equation must be a non-empty string")
        return value
