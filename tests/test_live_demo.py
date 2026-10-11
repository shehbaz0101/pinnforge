"""Public CPU demo: cached Stage D headline, and one live harmonic solve."""

from __future__ import annotations

import json
from pathlib import Path

import matplotlib.figure
import pytest

from pinnforge.live_demo import (
    BUDGETS,
    GITHUB_URL,
    HEADER_MARKDOWN,
    equation_blurb,
    format_error_percent,
    run_quick_pinn,
    solve_status,
    stage_d_headline,
    stage_d_markdown,
    viscosity_figure,
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
    assert BUDGETS["burgers_1d"].epochs == 800
    assert BUDGETS["harmonic_oscillator"].epochs == 150
    assert BUDGETS["poisson_toy"].epochs == 250


def test_readme_points_at_a_placeholder_space() -> None:
    readme = (ROOT / "README.md").read_text(encoding="utf-8")
    assert readme.startswith("---\n")
    assert "sdk: gradio" in readme
    assert 'sdk_version: "5.50.0"' in readme
    assert 'python_version: "3.12"' in readme
    assert 'suggested_hardware: "cpu-basic"' in readme
    assert "app_file: app.py" in readme
    assert "## Live demo" in readme
    assert "https://huggingface.co/spaces/PLACEHOLDER/pinnforge" in readme
    assert (ROOT / "app.py").is_file()
    assert (ROOT / "requirements.txt").is_file()
    requirements = (ROOT / "requirements.txt").read_text(encoding="utf-8")
    assert "torch==2.14.1+cpu" in requirements
    assert "download.pytorch.org/whl/cpu" in requirements


def test_unknown_equation_is_rejected_without_training() -> None:
    with pytest.raises(ValueError, match="live demo equation"):
        run_quick_pinn("navier-stokes")


@pytest.mark.ml
def test_harmonic_live_solve_matches_the_closed_form(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
    pytest.importorskip("torch")
    monkeypatch.chdir(tmp_path)
    solved = run_quick_pinn("harmonic")
    assert solved.equation_id == "harmonic_oscillator"
    assert solved.epochs == 150
    assert solved.relative_l2 < 0.02
    assert solved.seconds < 30.0
    assert solved.x.shape == solved.prediction.shape == solved.reference.shape
    assert not (tmp_path / "runs").exists()
    assert "Live CPU solve" in solve_status(solved)
