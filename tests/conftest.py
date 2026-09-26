"""Shared fixtures.

The HTTP rate limiter is process-wide. Reset it around every test so one
case cannot spend the budget of the next.
"""

from __future__ import annotations

import pytest


@pytest.fixture(autouse=True)
def _reset_rate_limiter() -> object:
    from pinnforge.ratelimit import reset_rate_limiter

    reset_rate_limiter()
    yield
    reset_rate_limiter()
