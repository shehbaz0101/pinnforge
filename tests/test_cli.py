"""CLI stub: version, help, and the equation list."""

from __future__ import annotations

import argparse
import os
import shutil
import subprocess
import sys
import tomllib
from pathlib import Path

import pytest

from pinnforge import __version__
from pinnforge.cli import build_parser, main

ROOT = Path(__file__).resolve().parents[1]


def test_version_constant_matches_pyproject() -> None:
    project = tomllib.loads((ROOT / "pyproject.toml").read_text(encoding="utf-8"))
    assert project["project"]["version"] == __version__ == "0.3.0"


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
    assert "sample" in help_text
    assert "residual" in help_text
    assert "train" in help_text
    assert "eval" in help_text
    assert "run" in help_text
    assert "demo" in help_text
    assert "serve" in help_text
    assert "fno" in help_text
    with pytest.raises(SystemExit) as version_exit:
        main(["--version"])
    assert version_exit.value.code == 0
    assert capsys.readouterr().out.strip() == f"pinnforge {__version__}"
    with pytest.raises(SystemExit) as missing:
        main([])
    assert missing.value.code == 2


SUBCOMMANDS = (
    "version",
    "equations",
    "sample",
    "residual",
    "train",
    "eval",
    "run",
    "demo",
    "serve",
    "fno",
    "v04",
    "v041",
    "v05",
    "v051",
    "claims",
)


def test_every_subcommand_has_one_line_help() -> None:
    helps = {item.dest: item.help for item in _subcommand_actions()}
    assert tuple(helps) == SUBCOMMANDS
    # argparse starts subcommand help at column 24, so 56 characters fit an 80-column terminal.
    for name, text in helps.items():
        assert text, name
        assert "\n" not in text, name
        assert len(text) <= 56, name
        assert ". " not in text, name


def test_console_script_is_declared() -> None:
    project = tomllib.loads((ROOT / "pyproject.toml").read_text(encoding="utf-8"))
    assert project["project"]["scripts"]["pinnforge"] == "pinnforge.cli:main"


def _help_output(command: list[str]) -> str:
    env = os.environ.copy()
    env["PYTHONPATH"] = str(ROOT / "src")
    env["COLUMNS"] = "80"
    completed = subprocess.run(
        [*command, "--help"],
        check=False,
        capture_output=True,
        text=True,
        env=env,
    )
    assert completed.returncode == 0, completed.stderr
    return completed.stdout


def test_module_help_lists_subcommands_on_one_line_each() -> None:
    text = _help_output([sys.executable, "-m", "pinnforge"])
    helps = {item.dest: item.help for item in _subcommand_actions()}
    for name in SUBCOMMANDS:
        line = next(row for row in text.splitlines() if row.strip().startswith(f"{name} "))
        assert helps[name] in line


def test_console_script_help() -> None:
    script = shutil.which("pinnforge")
    if script is None:
        pytest.skip("the pinnforge console script is not installed in this environment")
    text = _help_output([script])
    for name in SUBCOMMANDS:
        assert name in text


def test_forwarded_subcommand_help() -> None:
    fno = _help_output([sys.executable, "-m", "pinnforge", "fno"])
    assert fno.startswith("usage: pinnforge fno")
    assert "strong-baselines" in fno
    v04 = _help_output([sys.executable, "-m", "pinnforge", "v04"])
    assert v04.startswith("usage: pinnforge v04")
    assert "smoke" in v04
    v041 = _help_output([sys.executable, "-m", "pinnforge", "v041"])
    assert v041.startswith("usage: pinnforge v041")
    assert "smoke" in v041
    v05 = _help_output([sys.executable, "-m", "pinnforge", "v05"])
    assert v05.startswith("usage: pinnforge v05")
    assert "smoke" in v05
    v051 = _help_output([sys.executable, "-m", "pinnforge", "v051"])
    assert v051.startswith("usage: pinnforge v051")
    assert "smoke" in v051
    claims = _help_output([sys.executable, "-m", "pinnforge", "claims"])
    assert claims.startswith("usage: pinnforge claims")
    assert "check" in claims


def test_claims_subcommand_passes_on_the_repository(capsys: pytest.CaptureFixture[str]) -> None:
    assert main(["claims", "check", "--root", str(ROOT)]) == 0
    assert capsys.readouterr().out.strip() == "claim checks passed"


def _subcommand_actions() -> list[argparse.Action]:
    parser = build_parser()
    action = next(item for item in parser._actions if isinstance(item, argparse._SubParsersAction))
    return list(action._choices_actions)


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
        "import pinnforge, pinnforge.reference, pinnforge.sampling, pinnforge.experiments, "
        "pinnforge.specs, pinnforge.ml_import, pinnforge.cli, pinnforge.demo, "
        "pinnforge.demo_inverse, pinnforge.figures, pinnforge.api.bind, sys; "
        "assert 'torch' not in sys.modules; "
        "assert 'fastapi' not in sys.modules; "
        "assert pinnforge.__version__ == '0.3.0'"
    )
    completed = subprocess.run(
        [sys.executable, "-c", code],
        check=False,
        capture_output=True,
        text=True,
        env=env,
    )
    assert completed.returncode == 0, completed.stderr
