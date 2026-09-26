"""Day 8 path sandbox, rate limit, and offline guard.

HTTP cases use TestClient and do not open a port. Torch is not required.
"""

from __future__ import annotations

import os
import socket
import subprocess
import sys
from pathlib import Path
from types import ModuleType

import pytest

from pinnforge.offline import OfflineError, install_offline_guard, offline_guard_installed
from pinnforge.ratelimit import RateLimiter, read_rate_settings, reset_rate_limiter
from pinnforge.specs.paths import PathSandboxError, resolve_inside_cwd

ROOT = Path(__file__).resolve().parents[1]


def test_sandbox_allows_a_child_and_dotdot_that_stays_inside(tmp_path: Path) -> None:
    nested = tmp_path / "a" / "b"
    nested.mkdir(parents=True)
    child = resolve_inside_cwd("a/b/metrics.jsonl", root=tmp_path)
    stayed = resolve_inside_cwd("a/../a/b/metrics.jsonl", root=tmp_path)
    assert child == stayed == (nested / "metrics.jsonl").resolve()


def test_sandbox_rejects_parent_traversal_absolute_and_tilde(tmp_path: Path) -> None:
    with pytest.raises(PathSandboxError, match="working directory"):
        resolve_inside_cwd("..", root=tmp_path)
    with pytest.raises(PathSandboxError, match="working directory"):
        resolve_inside_cwd("nested/../../outside", root=tmp_path)
    with pytest.raises(PathSandboxError, match="relative"):
        resolve_inside_cwd("/etc/passwd", root=tmp_path)
    with pytest.raises(PathSandboxError, match="relative"):
        resolve_inside_cwd("~/secret.json", root=tmp_path)
    with pytest.raises(PathSandboxError, match="relative"):
        resolve_inside_cwd("bad\x00name", root=tmp_path)


def test_sandbox_rejects_a_symlink_that_leaves_the_root(tmp_path: Path) -> None:
    outside = tmp_path.parent / "outside-day8.json"
    outside.write_text("{}\n", encoding="utf-8")
    link = tmp_path / "escape.json"
    link.symlink_to(outside)
    with pytest.raises(PathSandboxError, match="working directory"):
        resolve_inside_cwd("escape.json", root=tmp_path)
    directory = tmp_path / "linked-parent"
    directory.symlink_to(tmp_path.parent, target_is_directory=True)
    with pytest.raises(PathSandboxError, match="working directory"):
        resolve_inside_cwd("linked-parent/checkpoint.pt", suffix=".pt", root=tmp_path)


def test_sandbox_allows_a_symlink_that_stays_inside(tmp_path: Path) -> None:
    real = tmp_path / "real"
    real.mkdir()
    link = tmp_path / "alias"
    link.symlink_to(real, target_is_directory=True)
    resolved = resolve_inside_cwd("alias/metrics.jsonl", suffix=".jsonl", root=tmp_path)
    assert resolved == (real / "metrics.jsonl").resolve()


def test_cli_rejects_train_eval_run_and_sample_escapes(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    capsys: pytest.CaptureFixture[str],
) -> None:
    from pinnforge.cli import main

    monkeypatch.chdir(tmp_path)
    with pytest.raises(SystemExit) as train:
        main(["train", "--equation", "harmonic", "--checkpoint-dir", "../ckpts"])
    assert train.value.code == 2
    assert "working directory" in capsys.readouterr().err

    with pytest.raises(SystemExit) as evaluate:
        main(["eval", "--checkpoint", "/etc/passwd", "--equation", "harmonic"])
    assert evaluate.value.code == 2
    assert "relative" in capsys.readouterr().err

    (tmp_path / "exp.json").write_text(
        '{"equation": "harmonic", "train": {"log_path": "../metrics.jsonl"}}\n',
        encoding="utf-8",
    )
    with pytest.raises(SystemExit) as run:
        main(["run", "--config", "exp.json"])
    assert run.value.code == 2
    assert "working directory" in capsys.readouterr().err

    link = tmp_path / "batch.json"
    link.symlink_to(tmp_path.parent / "batch.json")
    with pytest.raises(SystemExit) as sample:
        main(
            [
                "sample",
                "--equation",
                "harmonic",
                "--n-interior",
                "4",
                "--output",
                "batch.json",
            ]
        )
    assert sample.value.code == 2
    assert "working directory" in capsys.readouterr().err


def test_cli_data_root_allows_output_outside_the_cwd(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    capsys: pytest.CaptureFixture[str],
) -> None:
    from pinnforge.cli import main

    work = tmp_path / "work"
    outside = tmp_path / "outside"
    work.mkdir()
    outside.mkdir()
    monkeypatch.chdir(work)
    with pytest.raises(SystemExit) as caught:
        main(
            [
                "sample",
                "--equation",
                "harmonic",
                "--n-interior",
                "4",
                "--output",
                str(outside / "batch.json"),
            ]
        )
    assert caught.value.code == 2
    assert "relative" in capsys.readouterr().err
    assert (
        main(
            [
                "sample",
                "--equation",
                "harmonic",
                "--n-interior",
                "4",
                "--data-root",
                str(outside),
                "--output",
                "batch.json",
            ]
        )
        == 0
    )
    assert (outside / "batch.json").is_file()
    assert not (work / "batch.json").exists()
    assert os.environ.get("PINNFORGE_DATA_ROOT") is None


def test_train_eval_and_run_install_the_offline_guard(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    from pinnforge.cli import main

    monkeypatch.chdir(tmp_path)
    calls: list[str] = []
    monkeypatch.setattr("pinnforge.offline.install_offline_guard", lambda: calls.append("guard"))

    training = ModuleType("pinnforge.training")

    def train_loop(config: object, spec: object = None) -> object:
        raise RuntimeError("stopped-train")

    training.train_loop = train_loop  # type: ignore[attr-defined]
    monkeypatch.setitem(sys.modules, "pinnforge.training", training)
    with pytest.raises(RuntimeError, match="stopped-train"):
        main(["train", "--equation", "harmonic", "--epochs", "1"])

    evaluation = ModuleType("pinnforge.evaluation")

    def evaluate_checkpoint(*args: object, **kwargs: object) -> object:
        raise RuntimeError("stopped-eval")

    evaluation.evaluate_checkpoint = evaluate_checkpoint  # type: ignore[attr-defined]
    evaluation.write_eval_json = lambda *args, **kwargs: None  # type: ignore[attr-defined]
    monkeypatch.setitem(sys.modules, "pinnforge.evaluation", evaluation)
    with pytest.raises(RuntimeError, match="stopped-eval"):
        main(["eval", "--checkpoint", "ckpts/checkpoint.pt", "--equation", "harmonic"])

    (tmp_path / "exp.json").write_text('{"equation": "harmonic"}\n', encoding="utf-8")
    runner = ModuleType("pinnforge.experiments.run")

    def run_experiment(config: object) -> object:
        raise RuntimeError("stopped-run")

    runner.run_experiment = run_experiment  # type: ignore[attr-defined]
    monkeypatch.setitem(sys.modules, "pinnforge.experiments.run", runner)
    with pytest.raises(RuntimeError, match="stopped-run"):
        main(["run", "--config", "exp.json"])

    assert calls == ["guard", "guard", "guard"]


def test_importing_the_package_does_not_install_the_offline_guard() -> None:
    env = os.environ.copy()
    env["PYTHONPATH"] = str(ROOT / "src")
    completed = subprocess.run(
        [
            sys.executable,
            "-c",
            (
                "import pinnforge, pinnforge.cli\n"
                "from pinnforge.offline import offline_guard_installed\n"
                "assert offline_guard_installed() is False\n"
            ),
        ],
        check=False,
        capture_output=True,
        text=True,
        env=env,
    )
    assert completed.returncode == 0, completed.stderr


def test_offline_guard_refuses_remote_connects_and_allows_loopback() -> None:
    install_offline_guard()
    assert offline_guard_installed() is True
    assert os.environ.get("PINNFORGE_OFFLINE") == "1"
    with pytest.raises(OfflineError, match="offline-by-design"):
        socket.create_connection(("example.com", 80), timeout=0.2)
    with pytest.raises(OfflineError, match="8.8.8.8"):
        socket.create_connection(("8.8.8.8", 53), timeout=0.2)
    try:
        connected = socket.create_connection(("127.0.0.1", 9), timeout=0.3)
    except OfflineError:
        raise
    except OSError:
        return
    connected.close()


def test_retry_after_is_the_rest_of_the_window() -> None:
    limiter = RateLimiter(1, 60)
    assert limiter.check("client", now=1000.0) is None
    assert limiter.check("client", now=1000.0) == 60
    assert limiter.check("client", now=1059.2) == 1
    assert limiter.check("client", now=1060.0) is None
    other = RateLimiter(1, 60)
    assert other.check("a", now=1.0) is None
    assert other.check("b", now=1.0) is None
    assert other.check("a", now=1.0) == 60


def test_rate_settings_reject_bad_numbers() -> None:
    with pytest.raises(ValueError, match="PINNFORGE_RATE_LIMIT"):
        read_rate_settings(limit="0")
    with pytest.raises(ValueError, match="PINNFORGE_RATE_WINDOW_SECONDS"):
        read_rate_settings(window="0")
    with pytest.raises(ValueError, match="PINNFORGE_RATE_WINDOW_SECONDS"):
        read_rate_settings(window="inf")


def test_serve_rejects_a_bad_rate_limit_or_data_root(
    capsys: pytest.CaptureFixture[str],
    tmp_path: Path,
) -> None:
    from pinnforge.cli import main

    with pytest.raises(SystemExit) as limit:
        main(["serve", "--rate-limit", "0"])
    assert limit.value.code == 2
    with pytest.raises(SystemExit) as window:
        main(["serve", "--rate-window", "0"])
    assert window.value.code == 2
    with pytest.raises(SystemExit) as root:
        main(["serve", "--data-root", str(tmp_path / "missing")])
    assert root.value.code == 2
    err = capsys.readouterr().err
    assert "PINNFORGE_RATE_LIMIT" in err
    assert "PINNFORGE_RATE_WINDOW_SECONDS" in err
    assert "data root" in err


@pytest.fixture
def client(tmp_path: Path, monkeypatch: pytest.MonkeyPatch):
    pytest.importorskip("fastapi")
    pytest.importorskip("httpx2")
    from fastapi.testclient import TestClient

    from pinnforge.api import create_app

    monkeypatch.chdir(tmp_path)
    with TestClient(create_app()) as http:
        yield http


def test_http_rejects_traversal_absolute_and_symlink(client: object, tmp_path: Path) -> None:
    link = tmp_path / "escape"
    link.symlink_to(tmp_path.parent, target_is_directory=True)
    traversal = client.post(  # type: ignore[attr-defined]
        "/run",
        json={"equation": "harmonic", "train": {"log_path": "../metrics.jsonl"}},
    )
    absolute = client.post(  # type: ignore[attr-defined]
        "/eval",
        json={"checkpoint": "/etc/passwd", "equation": "harmonic"},
    )
    symlink = client.post(  # type: ignore[attr-defined]
        "/train",
        json={"equation_id": "harmonic", "epochs": 1, "checkpoint_dir": "escape"},
    )
    assert traversal.status_code == 422
    assert "working directory" in traversal.json()["detail"]
    assert absolute.status_code == 422
    assert "relative" in absolute.json()["detail"]
    assert symlink.status_code == 422
    assert "working directory" in symlink.json()["detail"]


def test_catalog_requests_do_not_open_sockets(client: object, monkeypatch: pytest.MonkeyPatch) -> None:
    attempts: list[object] = []

    def connect(self: socket.socket, address: object) -> object:
        attempts.append(address)
        raise AssertionError(f"unexpected connect to {address}")

    def create_connection(address: object, *args: object, **kwargs: object) -> object:
        attempts.append(address)
        raise AssertionError(f"unexpected create_connection to {address}")

    monkeypatch.setattr(socket.socket, "connect", connect)
    monkeypatch.setattr(socket, "create_connection", create_connection)
    health = client.get("/health")  # type: ignore[attr-defined]
    equations = client.get("/equations")  # type: ignore[attr-defined]
    assert health.status_code == 200
    assert equations.status_code == 200
    assert attempts == []


def test_rate_limit_returns_429_with_retry_after(
    client: object,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setenv("PINNFORGE_RATE_LIMIT", "2")
    monkeypatch.setenv("PINNFORGE_RATE_WINDOW_SECONDS", "60")
    reset_rate_limiter()
    for _ in range(5):
        assert client.get("/health").status_code == 200  # type: ignore[attr-defined]
        assert client.get("/equations").status_code == 200  # type: ignore[attr-defined]
    first = client.post(  # type: ignore[attr-defined]
        "/train",
        json={"equation_id": "harmonic", "checkpoint_dir": "../ckpts"},
    )
    second = client.post(  # type: ignore[attr-defined]
        "/eval",
        json={"checkpoint": "../checkpoint.pt", "equation": "harmonic"},
    )
    third = client.post("/run", json={"equation": "not-an-equation"})  # type: ignore[attr-defined]
    health = client.get("/health")  # type: ignore[attr-defined]
    assert first.status_code == 422
    assert second.status_code == 422
    assert third.status_code == 429
    assert "rate limit exceeded" in third.json()["detail"]
    assert int(third.headers["retry-after"]) >= 1
    assert health.status_code == 200
    assert health.json()["status"] == "ok"


def test_openapi_documents_429_on_expensive_routes_only(client: object) -> None:
    paths = client.get("/openapi.json").json()["paths"]  # type: ignore[attr-defined]
    for path in ("/train", "/eval", "/run"):
        assert "429" in paths[path]["post"]["responses"]
    assert "429" not in paths["/health"]["get"]["responses"]
    assert "429" not in paths["/equations"]["get"]["responses"]


def test_serve_passes_data_root_and_rate_limit_then_restores(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    pytest.importorskip("fastapi")
    uvicorn = pytest.importorskip("uvicorn")
    from pinnforge.cli import main

    seen: dict[str, object] = {}

    def fake_run(app: object, **kwargs: object) -> None:
        seen["root"] = os.environ.get("PINNFORGE_DATA_ROOT")
        seen["limit"] = os.environ.get("PINNFORGE_RATE_LIMIT")
        seen["window"] = os.environ.get("PINNFORGE_RATE_WINDOW_SECONDS")
        seen["host"] = kwargs.get("host")
        seen["offline"] = offline_guard_installed()

    monkeypatch.setattr(uvicorn, "run", fake_run)
    previous_root = os.environ.get("PINNFORGE_DATA_ROOT")
    previous_limit = os.environ.get("PINNFORGE_RATE_LIMIT")
    previous_window = os.environ.get("PINNFORGE_RATE_WINDOW_SECONDS")
    root = tmp_path / "data-root"
    root.mkdir()
    assert (
        main(
            [
                "serve",
                "--data-root",
                str(root),
                "--rate-limit",
                "3",
                "--rate-window",
                "15",
            ]
        )
        == 0
    )
    assert seen["root"] == str(root.resolve())
    assert seen["limit"] == "3"
    assert seen["window"] == "15.0"
    assert seen["host"] == "127.0.0.1"
    assert seen["offline"] is True
    assert os.environ.get("PINNFORGE_DATA_ROOT") == previous_root
    assert os.environ.get("PINNFORGE_RATE_LIMIT") == previous_limit
    assert os.environ.get("PINNFORGE_RATE_WINDOW_SECONDS") == previous_window
