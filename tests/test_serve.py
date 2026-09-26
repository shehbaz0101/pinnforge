"""``pinnforge serve`` bind policy. These tests do not open a port."""

from __future__ import annotations

import sys
from types import ModuleType

import pytest

from pinnforge.api.bind import DEFAULT_HOST, DEFAULT_PORT, resolve_bind_host, resolve_port
from pinnforge.cli import build_parser, main


def test_bind_defaults_to_loopback() -> None:
    assert resolve_bind_host(None) == DEFAULT_HOST == "127.0.0.1"
    assert resolve_port(DEFAULT_PORT) == 8000
    assert resolve_bind_host("localhost") == "localhost"
    assert resolve_bind_host(" 127.0.0.1 ") == "127.0.0.1"


@pytest.mark.parametrize("host", ["0.0.0.0", "::", "[::]", " 0.0.0.0 "])
def test_wildcard_hosts_require_allow_remote(host: str) -> None:
    with pytest.raises(ValueError, match="allow-remote"):
        resolve_bind_host(host)
    assert resolve_bind_host(host, allow_remote=True) == host.strip()


def test_port_must_be_in_range() -> None:
    with pytest.raises(ValueError, match="port"):
        resolve_port(0)
    with pytest.raises(ValueError, match="port"):
        resolve_port(65536)
    with pytest.raises(ValueError, match="port"):
        resolve_port(True)  # type: ignore[arg-type]


def test_serve_parser_defaults() -> None:
    args = build_parser().parse_args(["serve"])
    assert args.host == "127.0.0.1"
    assert args.port == 8000
    assert args.allow_remote is False


@pytest.mark.parametrize("host", ["0.0.0.0", "::"])
def test_serve_refuses_wildcard_without_starting(
    host: str,
    monkeypatch: pytest.MonkeyPatch,
    capsys: pytest.CaptureFixture[str],
) -> None:
    fake = ModuleType("uvicorn")

    def run(*args: object, **kwargs: object) -> None:
        raise AssertionError("server should not start")

    fake.run = run
    monkeypatch.setitem(sys.modules, "uvicorn", fake)
    with pytest.raises(SystemExit) as exc:
        main(["serve", "--host", host])
    assert exc.value.code == 2
    assert "allow-remote" in capsys.readouterr().err


def test_serve_missing_api_extra(monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]) -> None:
    monkeypatch.setitem(sys.modules, "uvicorn", None)
    assert main(["serve"]) == 1
    assert "api extra" in capsys.readouterr().err


def test_serve_passes_loopback_to_uvicorn(monkeypatch: pytest.MonkeyPatch) -> None:
    pytest.importorskip("fastapi")
    captured: dict[str, object] = {}
    fake = ModuleType("uvicorn")

    def run(app: object, **kwargs: object) -> None:
        captured["host"] = kwargs["host"]
        captured["port"] = kwargs["port"]
        captured["title"] = getattr(app, "title", None)

    fake.run = run
    monkeypatch.setitem(sys.modules, "uvicorn", fake)
    assert main(["serve", "--port", "8765"]) == 0
    assert captured == {"host": "127.0.0.1", "port": 8765, "title": "PINNForge"}


def test_serve_allow_remote_wildcard(monkeypatch: pytest.MonkeyPatch) -> None:
    pytest.importorskip("fastapi")
    captured: dict[str, object] = {}
    fake = ModuleType("uvicorn")

    def run(app: object, **kwargs: object) -> None:
        captured["host"] = kwargs["host"]
        captured["port"] = kwargs["port"]

    fake.run = run
    monkeypatch.setitem(sys.modules, "uvicorn", fake)
    assert main(["serve", "--host", "0.0.0.0", "--allow-remote", "--port", "8766"]) == 0
    assert captured == {"host": "0.0.0.0", "port": 8766}
