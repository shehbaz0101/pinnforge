"""Torch-backed modules stay optional.

These tests run without the ``ml`` extra. When torch is installed they
skip, and ``tests/test_residuals.py`` covers the operators.
"""

from __future__ import annotations

import importlib.util
import os
import subprocess
import sys
from pathlib import Path

import pytest

from pinnforge.ml_import import InstallHint

ROOT = Path(__file__).resolve().parents[1]


def _torch_installed() -> bool:
    return importlib.util.find_spec("torch") is not None


def test_install_hint_is_an_import_error() -> None:
    hint = InstallHint()
    assert isinstance(hint, ImportError)
    text = str(hint)
    assert "ml" in text
    assert 'pip install -e ".[ml]"' in text


def test_torch_modules_raise_install_hint_without_torch() -> None:
    if _torch_installed():
        pytest.skip("torch is installed")
    env = os.environ.copy()
    env["PYTHONPATH"] = str(ROOT / "src")
    code = (
        "import pinnforge\n"
        "from pinnforge.ml_import import InstallHint\n"
        "for name in ('pinnforge.models', 'pinnforge.residuals', 'pinnforge.losses', 'pinnforge.training', 'pinnforge.evaluation'):\n"
        "    try:\n"
        "        __import__(name)\n"
        "    except InstallHint as exc:\n"
        "        text = str(exc)\n"
        "        assert 'ml' in text, text\n"
        "        assert 'pip install' in text, text\n"
        "    else:\n"
        "        raise SystemExit(name + ' imported without torch')\n"
        "from pinnforge.sampling import sample_equation\n"
        "assert pinnforge.__version__ == '0.1.0'\n"
        "assert sample_equation.__name__ == 'sample_equation'\n"
    )
    completed = subprocess.run(
        [sys.executable, "-c", code],
        check=False,
        capture_output=True,
        text=True,
        env=env,
    )
    assert completed.returncode == 0, completed.stderr


def test_train_command_needs_the_ml_extra(capsys: pytest.CaptureFixture[str]) -> None:
    if _torch_installed():
        pytest.skip("torch is installed")
    from pinnforge.cli import main

    assert main(["train", "--equation", "harmonic", "--epochs", "1"]) == 1
    captured = capsys.readouterr()
    assert captured.out == ""
    assert "ml" in captured.err
    assert "pip install" in captured.err


def test_eval_command_needs_the_ml_extra(capsys: pytest.CaptureFixture[str]) -> None:
    if _torch_installed():
        pytest.skip("torch is installed")
    from pinnforge.cli import main

    assert main(["eval", "--checkpoint", "checkpoints/checkpoint.pt", "--equation", "harmonic"]) == 1
    captured = capsys.readouterr()
    assert captured.out == ""
    assert "ml" in captured.err
    assert "pip install" in captured.err


def test_residual_command_needs_the_ml_extra(capsys: pytest.CaptureFixture[str]) -> None:
    if _torch_installed():
        pytest.skip("torch is installed")
    from pinnforge.cli import main

    assert main(["residual", "--equation", "harmonic", "--seed", "0"]) == 1
    captured = capsys.readouterr()
    assert captured.out == ""
    assert "ml" in captured.err
    assert "pip install" in captured.err
