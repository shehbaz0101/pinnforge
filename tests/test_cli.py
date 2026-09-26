"""CLI stub: version, help, and the equation list."""

from __future__ import annotations

import os
import subprocess
import sys
import tomllib
from pathlib import Path

import pytest

from pinnforge import __version__
from pinnforge.cli import main

ROOT = Path(__file__).resolve().parents[1]


def test_version_constant_matches_pyproject() -> None:
    project = tomllib.loads((ROOT / "pyproject.toml").read_text(encoding="utf-8"))
    assert project["project"]["version"] == __version__ == "0.1.0"


def test_version_command(capsys: pytest.CaptureFixture[str]) -> None:
    assert main(["version"]) == 0
    assert capsys.readouterr().out.strip() == f"pinnforge {__version__}"


def test_equations_command(capsys: pytest.CaptureFixture[str]) -> None:
    assert main(["equations"]) == 0
    lines = capsys.readouterr().out.splitlines()
    assert lines == ["burgers_1d", "harmonic_oscillator", "poisson_toy"]


def test_help_and_version_flags(capsys: pytest.CaptureFixture[str]) -> None:
    with pytest.raises(SystemExit) as help_exit:
        main(["--help"])
    assert help_exit.value.code == 0
    help_text = capsys.readouterr().out
    assert "version" in help_text
    assert "equations" in help_text
    with pytest.raises(SystemExit) as version_exit:
        main(["--version"])
    assert version_exit.value.code == 0
    assert capsys.readouterr().out.strip() == f"pinnforge {__version__}"
    with pytest.raises(SystemExit) as missing:
        main([])
    assert missing.value.code == 2


def test_module_entrypoint() -> None:
    env = os.environ.copy()
    env["PYTHONPATH"] = str(ROOT / "src")
    completed = subprocess.run(
        [sys.executable, "-m", "pinnforge", "version"],
        check=False,
        capture_output=True,
        text=True,
        env=env,
    )
    assert completed.returncode == 0
    assert completed.stdout.strip() == f"pinnforge {__version__}"


def test_import_does_not_load_torch() -> None:
    env = os.environ.copy()
    env["PYTHONPATH"] = str(ROOT / "src")
    code = (
        "import pinnforge, pinnforge.reference, sys; "
        "assert 'torch' not in sys.modules; "
        "assert pinnforge.__version__ == '0.1.0'"
    )
    completed = subprocess.run(
        [sys.executable, "-c", code],
        check=False,
        capture_output=True,
        text=True,
        env=env,
    )
    assert completed.returncode == 0, completed.stderr
