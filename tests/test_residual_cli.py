"""``pinnforge residual`` with torch installed."""

from __future__ import annotations

import math

import pytest

pytestmark = pytest.mark.ml


@pytest.fixture(autouse=True)
def _need_torch() -> None:
    pytest.importorskip("torch")


@pytest.mark.parametrize("equation", ["harmonic", "burgers", "poisson"])
def test_residual_command_prints_finite_mse(equation: str, capsys: pytest.CaptureFixture[str]) -> None:
    from pinnforge.cli import main

    args = ["residual", "--equation", equation, "--seed", "0"]
    assert main(args) == 0
    first = capsys.readouterr().out
    assert main(args) == 0
    second = capsys.readouterr().out
    assert first == second
    lines = first.splitlines()
    assert lines[0] == f"equation: {_equation_id(equation)}"
    assert lines[1] == "seed: 0"
    assert lines[2].startswith("residual_mse: ")
    mse = float(lines[2].split(":", 1)[1])
    assert mse >= 0.0
    assert math.isfinite(mse)


def _equation_id(name: str) -> str:
    return {
        "harmonic": "harmonic_oscillator",
        "burgers": "burgers_1d",
        "poisson": "poisson_toy",
    }[name]
