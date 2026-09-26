"""FastAPI application for the localhost research API.

``GET /health`` reports the package version and does not load a model.
``GET /equations`` lists the built-in catalog.
``GET /equations/{id_or_alias}`` returns one entry and its default spec.
``POST /train`` and ``POST /eval`` match the flag-based CLI commands.
``POST /run`` trains and then evaluates an experiment file or an inline
document, the same path as ``pinnforge run --config``.

File paths must stay inside the sandbox root (the working directory, or
``PINNFORGE_DATA_ROOT``). A missing ``ml`` extra is HTTP 503. A bad body
or a path that escapes is HTTP 422. ``POST /train``, ``POST /eval``, and
``POST /run`` share a per-client rate limit. Over the limit the response
is HTTP 429 with ``Retry-After``. Catalog reads are not counted. There
is no authentication.

Building the app installs an offline socket guard. Non-loopback TCP
connects raise ``OfflineError``. Loopback stays open.

``pinnforge serve`` binds to ``127.0.0.1:8000`` and refuses ``0.0.0.0``
and ``::`` unless ``--allow-remote`` is set. Starting uvicorn directly
does not apply that check; pass ``--host 127.0.0.1`` in that case. The
offline guard still installs, because importing this module builds
``app``.
"""

from __future__ import annotations

from collections.abc import Awaitable, Callable
from typing import TypeVar

from fastapi import FastAPI, HTTPException
from fastapi.responses import JSONResponse
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
from pinnforge.offline import install_offline_guard
from pinnforge.ratelimit import LIMITED_PATHS, RateLimitConfigError, get_limiter
from pinnforge.specs.train import TrainConfig

_T = TypeVar("_T")

_ERRORS: dict[int | str, dict[str, object]] = {
    422: {
        "description": (
            "The body is invalid, a file is missing, or a path leaves the sandbox root."
        )
    },
    429: {
        "description": (
            "Too many requests on POST /train, POST /eval, and POST /run. "
            "Retry-After is the wait in seconds. GET /health and GET /equations are not limited."
        ),
        "headers": {
            "Retry-After": {
                "description": "Seconds to wait before retrying.",
                "schema": {"type": "integer", "minimum": 1},
            }
        },
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
            "Paths must stay inside the sandbox root. "
            "POST /train, POST /eval, and POST /run share a per-client rate limit. "
            "The process refuses non-loopback TCP connects. "
            "pinnforge serve listens on 127.0.0.1 unless --allow-remote is set."
        ),
    )
    app.add_middleware(_RateLimitMiddleware)

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

    install_offline_guard()
    return app


class _RateLimitMiddleware:
    """Count expensive POSTs. Leave catalog reads, including health, alone."""

    def __init__(self, app: Callable[..., Awaitable[None]]) -> None:
        self._app = app

    async def __call__(self, scope: dict[str, object], receive: object, send: object) -> None:
        if scope.get("type") == "http" and scope.get("method") == "POST":
            path = scope.get("path")
            if isinstance(path, str) and path in LIMITED_PATHS:
                blocked = _limited_response(scope)
                if blocked is not None:
                    await blocked(scope, receive, send)
                    return
        await self._app(scope, receive, send)


def _limited_response(scope: dict[str, object]) -> JSONResponse | None:
    try:
        limiter = get_limiter()
    except RateLimitConfigError as exc:
        return JSONResponse(status_code=500, content={"detail": str(exc)})
    client = scope.get("client")
    if isinstance(client, tuple) and client and isinstance(client[0], str):
        key = client[0]
    else:
        key = "global"
    retry_after = limiter.check(key)
    if retry_after is None:
        return None
    window = limiter.window_seconds
    window_text = str(int(window)) if window == int(window) else str(window)
    return JSONResponse(
        status_code=429,
        content={
            "detail": (
                f"rate limit exceeded: {limiter.limit} requests "
                f"per {window_text} seconds for this client"
            )
        },
        headers={"Retry-After": str(retry_after)},
    )


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
