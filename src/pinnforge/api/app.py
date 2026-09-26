"""FastAPI application for the localhost research API.

``GET /health`` reports the package version and does not load a model.
``GET /equations`` lists the built-in catalog.
``GET /equations/{id_or_alias}`` returns one entry and its default spec.
``POST /train`` and ``POST /eval`` match the flag-based CLI commands.
``POST /run`` trains and then evaluates an experiment file or an inline
document, the same path as ``pinnforge run --config``.

File paths must stay inside the working directory. A missing ``ml``
extra is HTTP 503. A bad body or a path that escapes is HTTP 422. There
is no authentication and no rate limit.

``pinnforge serve`` binds to ``127.0.0.1:8000`` and refuses ``0.0.0.0``
and ``::`` unless ``--allow-remote`` is set. Starting uvicorn directly
does not apply that check; pass ``--host 127.0.0.1`` in that case.
"""

from __future__ import annotations

from collections.abc import Callable
from typing import TypeVar

from fastapi import FastAPI, HTTPException
from pydantic import ValidationError

from pinnforge import __version__
from pinnforge.api import service
from pinnforge.api.schemas import (
    EquationDetail,
    EquationSummary,
    EvalRequest,
    EvalResponse,
    HealthResponse,
    RunRequest,
    RunResponse,
    TrainSummary,
)
from pinnforge.specs.train import TrainConfig

_T = TypeVar("_T")

_ERRORS: dict[int | str, dict[str, str]] = {
    422: {
        "description": (
            "The body is invalid, a file is missing, or a path leaves the working directory."
        )
    },
    503: {"description": "Torch is not installed. Install the optional ml extra."},
}


def create_app() -> FastAPI:
    """Build the localhost API. Does not import torch."""

    app = FastAPI(
        title="PINNForge",
        version=__version__,
        description=(
            "Local research API for the equation catalog, a CPU train, "
            "an evaluation, and a train-then-eval experiment. "
            "Paths must stay inside the working directory. "
            "pinnforge serve listens on 127.0.0.1 unless --allow-remote is set."
        ),
    )

    @app.get("/health", response_model=HealthResponse)
    def health() -> HealthResponse:
        """Report that the process is up. Does not import torch."""

        return HealthResponse(status="ok", version=__version__)

    @app.get("/equations", response_model=list[EquationSummary])
    def equations() -> list[EquationSummary]:
        """List built-in equations, aliases, and overridable parameters."""

        return service.list_equation_summaries()

    @app.get("/equations/{id_or_alias}", response_model=EquationDetail)
    def equation(id_or_alias: str) -> EquationDetail:
        """Return one built-in and its default spec.

        ``id_or_alias`` is a registry id or a short name such as
        ``harmonic``. Unknown names are HTTP 404.
        """

        try:
            return service.equation_detail(id_or_alias)
        except service.UnknownEquation as exc:
            raise HTTPException(status_code=404, detail=str(exc)) from exc

    @app.post("/run", response_model=RunResponse, responses=_ERRORS)
    def run(body: RunRequest) -> RunResponse:
        """Train then evaluate an experiment file or an inline document.

        ``config`` is a relative YAML or JSON path. The other fields are
        the experiment document. Paths must stay inside the working
        directory. HTTP 503 when torch is not installed.
        """

        return _call(lambda: service.run_request(body))

    @app.post("/train", response_model=TrainSummary, responses=_ERRORS)
    def train(body: TrainConfig) -> TrainSummary:
        """Train one built-in equation and return the final loss.

        The body is a ``TrainConfig``. Checkpoint and log paths must stay
        inside the working directory. HTTP 503 when torch is not installed.
        """

        return _call(lambda: service.train_request(body))

    @app.post("/eval", response_model=EvalResponse, responses=_ERRORS)
    def evaluate(body: EvalRequest) -> EvalResponse:
        """Score a relative checkpoint against its reference and residual.

        ``equation`` must match the checkpoint. ``write_json``, when set,
        is a relative ``.json`` path. HTTP 503 when torch is not installed.
        """

        return _call(lambda: service.eval_request(body))

    return app


def _call(fn: Callable[[], _T]) -> _T:
    try:
        return fn()
    except service.MlExtraMissing as exc:
        raise HTTPException(status_code=503, detail=str(exc)) from exc
    except (TypeError, ValueError, ValidationError, OSError) as exc:
        raise HTTPException(status_code=422, detail=_detail(exc)) from exc


def _detail(exc: Exception) -> str:
    if isinstance(exc, ValidationError):
        return _format_validation(exc)
    return str(exc)


def _format_validation(exc: ValidationError) -> str:
    parts: list[str] = []
    for error in exc.errors():
        location = ".".join(str(item) for item in error["loc"])
        message = str(error["msg"])
        parts.append(f"{location}: {message}" if location else message)
    return "; ".join(parts) if parts else "invalid configuration"


app = create_app()
