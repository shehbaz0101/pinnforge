"""Localhost HTTP API.

:func:`create_app` builds the FastAPI application. ``app`` is that
application for ``uvicorn pinnforge.api:app``. Both load FastAPI, so
they need the optional ``api`` extra.

Importing this package does not import FastAPI or torch.
:mod:`pinnforge.api.bind` is the loopback bind check and does not import
them either. ``GET /health`` and ``GET /equations`` do not import torch.
``POST /train``, ``POST /eval``, and ``POST /run`` do.
"""

from __future__ import annotations


def create_app() -> object:
    """Build a new FastAPI app. Requires the optional ``api`` extra."""

    from pinnforge.api.app import create_app as factory

    return factory()


def __getattr__(name: str) -> object:
    if name == "app":
        from pinnforge.api.app import app

        return app
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
