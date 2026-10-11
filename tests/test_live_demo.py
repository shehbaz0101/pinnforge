"""Public CPU demo: cached Stage D headline, and one live harmonic solve."""

from __future__ import annotations

import json
import subprocess
import sys
from pathlib import Path

import matplotlib.figure
import pytest
import yaml

from pinnforge.live_demo import (
    BUDGETS,
    GITHUB_URL,
    HEADER_MARKDOWN,
    HEALTH_PATH,
    PRECOMPUTED_PHRASE,
    equation_blurb,
    format_error_percent,
    health_body,
    landing_markdown,
    live_button_label,
    precomputed_images,
    precomputed_manifest,
    run_quick_pinn,
    server_bind,
    solve_status,
    stage_d_headline,
    stage_d_markdown,
    viscosity_figure,
    viscosity_image_path,
)

ROOT = Path(__file__).resolve().parents[1]


def test_header_is_one_line_with_the_github_link() -> None:
    body = HEADER_MARKDOWN.split("\n\n", 1)[1]
    assert "\n" not in body
    assert GITHUB_URL in body
    assert "residual" in body
    assert "32 sparse sensors" in body


def test_stage_d_headline_matches_the_published_file() -> None:
    record = json.loads((ROOT / "docs" / "v02" / "stage_d_scores.json").read_text(encoding="utf-8"))
    headline = stage_d_headline()
    closed = record["baselines"]["sensors32_bursts"]["hard_ood"]
    hybrid = record["arms"]["hybrid_1e-2"]["objectives"]["0.0"]["slices"]["hard_ood"]
    data = record["arms"]["data_only"]["objectives"]["0.0"]["slices"]["hard_ood"]
    assert headline.n == 30
    assert headline.closed_form.mean_rel == closed["mean_rel_error"]
    assert headline.hybrid.mean_rel == hybrid["mean_rel_error"]["mean"]
    assert headline.data_only.mean_rel == data["mean_rel_error"]["mean"]
    assert headline.closed_form.failures == 9
    assert headline.hybrid.failures == 0
    assert headline.data_only.failures == 0
    assert format_error_percent(headline.hybrid.mean_rel) == "3.4%"
    assert format_error_percent(headline.closed_form.mean_rel) == "34.9%"
    text = stage_d_markdown(headline)
    assert text.startswith("### Viscosity from 32 sparse sensors\n\n**Precomputed.**")
    assert "3.4%" in text
    assert "34.9%" in text
    assert "about 35%" in text
    figure = viscosity_figure(headline)
    assert isinstance(figure, matplotlib.figure.Figure)


def test_blurbs_name_the_reference_and_burgers_stays_a_short_problem() -> None:
    assert "closed form" in equation_blurb("harmonic")
    assert "sin(πx)" in equation_blurb("poisson")
    burgers = equation_blurb("burgers")
    assert "Fourier" in burgers
    assert "precomputed" in burgers
    assert BUDGETS["burgers_1d"].epochs == 500
    assert BUDGETS["harmonic_oscillator"].epochs == 80
    assert BUDGETS["poisson_toy"].epochs == 250
    assert BUDGETS["poisson_toy"].hidden == (48, 48)
    assert BUDGETS["harmonic_oscillator"].hidden == (32, 32)


def test_gallery_is_a_real_run_and_the_page_says_so() -> None:
    assert PRECOMPUTED_PHRASE == "precomputed from a real run of this code"
    text = landing_markdown()
    assert PRECOMPUTED_PHRASE in text
    assert "3.4%" in text
    assert "34.9%" in text
    assert "does not import PyTorch" in text
    record = precomputed_manifest()
    assert record["phrase"] == PRECOMPUTED_PHRASE
    assert record["generator"] == "pinnforge.live_demo.run_quick_pinn"
    assert record["seed"] == 0
    runs = record["runs"]
    assert runs["harmonic_oscillator"]["epochs"] == 150
    assert runs["poisson_toy"]["epochs"] == 250
    assert runs["burgers_1d"]["epochs"] == 800
    assert runs["harmonic_oscillator"]["relative_l2"] < 0.01
    assert runs["poisson_toy"]["relative_l2"] < 0.01
    assert runs["burgers_1d"]["relative_l2"] < 0.06
    images = precomputed_images()
    assert [path.name for _caption, path in images] == [
        "harmonic.png",
        "burgers.png",
        "poisson.png",
    ]
    for caption, path in images:
        assert PRECOMPUTED_PHRASE in caption
        blob = path.read_bytes()
        assert blob.startswith(b"\x89PNG\r\n\x1a\n")
        assert len(blob) > 10_000
    viscosity = viscosity_image_path().read_bytes()
    assert viscosity.startswith(b"\x89PNG\r\n\x1a\n")
    assert len(viscosity) > 10_000
    assert live_button_label("harmonic") == "Run live on free CPU, about 20 s"
    assert live_button_label("poisson") == "Run live on free CPU, about 25 s"
    assert live_button_label("burgers") == "Run live on free CPU, about 30 s"


def test_readme_points_at_the_render_demo() -> None:
    readme = (ROOT / "README.md").read_text(encoding="utf-8")
    lines = readme.splitlines()
    assert lines[0] == "Live demo: https://pinnforge.onrender.com"
    assert lines[1] == ""
    assert lines[2] == (
        "PINNForge recovers Burgers viscosity from 32 sparse sensors: "
        "3.4% error vs 34.9% for closed form on 30 hard out-of-distribution cases."
    )
    assert "sdk: gradio" in readme
    assert 'sdk_version: "5.50.0"' in readme
    assert 'python_version: "3.12"' in readme
    assert 'suggested_hardware: "cpu-basic"' in readme
    assert "app_file: app.py" in readme
    assert "## Live demo" in readme
    assert "https://pinnforge.onrender.com" in readme
    assert PRECOMPUTED_PHRASE in readme
    assert "about 20 s" in readme
    assert "about 30 s" in readme
    assert (ROOT / "app.py").is_file()
    assert (ROOT / "requirements.txt").is_file()
    assert (ROOT / ".python-version").read_text(encoding="utf-8").strip() == "3.12"
    requirements = (ROOT / "requirements.txt").read_text(encoding="utf-8")
    assert "torch==2.14.1+cpu" in requirements
    assert "gradio==5.50.0" in requirements
    assert "download.pytorch.org/whl/cpu" in requirements


def test_render_blueprint_is_the_free_python_service() -> None:
    document = yaml.safe_load((ROOT / "render.yaml").read_text(encoding="utf-8"))
    services = document["services"]
    assert len(services) == 1
    service = services[0]
    assert service["type"] == "web"
    assert service["name"] == "pinnforge"
    assert service["runtime"] == "python"
    assert service["plan"] == "free"
    assert service["branch"] == "main"
    assert service["healthCheckPath"] == HEALTH_PATH
    assert service["startCommand"] == "python app.py"
    assert "torch==2.14.1+cpu" in service["buildCommand"]
    assert "download.pytorch.org/whl/cpu" in service["buildCommand"]
    assert "pip install -r requirements.txt" in service["buildCommand"]
    env = {item["key"]: item["value"] for item in service["envVars"]}
    assert "PORT" not in env
    assert env["MALLOC_ARENA_MAX"] == "2"
    assert env["OMP_NUM_THREADS"] == "1"
    assert env["OPENBLAS_NUM_THREADS"] == "1"
    assert health_body() == {"status": "ok"}


def test_server_bind_uses_port_and_defaults_locally(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.delenv("PORT", raising=False)
    assert server_bind() == ("0.0.0.0", 7860)
    monkeypatch.setenv("PORT", "10000")
    assert server_bind() == ("0.0.0.0", 10000)
    monkeypatch.setenv("PORT", "  ")
    assert server_bind() == ("0.0.0.0", 7860)
    monkeypatch.setenv("PORT", "0")
    with pytest.raises(ValueError, match="PORT"):
        server_bind()
    monkeypatch.setenv("PORT", "nope")
    with pytest.raises(ValueError, match="PORT"):
        server_bind()


def test_unknown_equation_is_rejected_without_training() -> None:
    with pytest.raises(ValueError, match="live demo equation"):
        run_quick_pinn("navier-stokes")


@pytest.mark.ml
def test_harmonic_live_solve_matches_the_closed_form(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
    pytest.importorskip("torch")
    monkeypatch.chdir(tmp_path)
    solved = run_quick_pinn("harmonic")
    assert solved.equation_id == "harmonic_oscillator"
    assert solved.epochs == 80
    assert solved.relative_l2 < 0.02
    assert solved.seconds < 30.0
    assert solved.x.shape == solved.prediction.shape == solved.reference.shape
    assert not (tmp_path / "runs").exists()
    assert "Live CPU solve" in solve_status(solved)


def test_importing_the_app_does_not_import_torch() -> None:
    pytest.importorskip("gradio")
    code = (
        "import sys\n"
        "from pathlib import Path\n"
        f"root = Path({str(ROOT)!r})\n"
        "sys.path.insert(0, str(root))\n"
        "sys.path.insert(0, str(root / 'src'))\n"
        "import app\n"
        "app.build_demo()\n"
        "assert 'torch' not in sys.modules\n"
        "print('ok')\n"
    )
    completed = subprocess.run(
        [sys.executable, "-c", code],
        check=False,
        capture_output=True,
        text=True,
    )
    assert completed.returncode == 0, completed.stderr
    assert "ok" in completed.stdout


def test_health_route_is_ok_without_training() -> None:
    pytest.importorskip("gradio")
    import importlib
    import sys

    from gradio.routes import App
    from starlette.testclient import TestClient

    if str(ROOT) not in sys.path:
        sys.path.insert(0, str(ROOT))
    app_module = importlib.import_module("app")
    demo = app_module.build_demo()
    demo.queue(max_size=1, default_concurrency_limit=1)
    fastapi_app = App.create_app(demo, app_kwargs=app_module.app_kwargs())
    client = TestClient(fastapi_app)
    response = client.get("/health")
    assert response.status_code == 200
    assert response.json() == {"status": "ok"}
