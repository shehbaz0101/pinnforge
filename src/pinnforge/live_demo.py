"""CPU page behind the public Gradio demo.

A visitor picks the harmonic oscillator, viscous Burgers, or the Poisson
toy. :func:`run_quick_pinn` trains a small coordinate network for that
problem and returns the prediction next to a reference curve. The
Burgers network has no closed form, so its reference is one Fourier
trajectory of the same initial-value problem. The network, the collocation
budget, and the time window are small enough that the fit finishes in
seconds on a free CPU.

The viscosity chart is not that fit. :func:`stage_d_headline` reads the
published Stage D scores and labels them as precomputed. Importing this
module does not import torch. :func:`run_quick_pinn` and
:func:`prepare_runtime` do.
"""

from __future__ import annotations

import json
import math
import shutil
import threading
import time
import uuid
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np

from pinnforge.equations.base import (
    BoundaryCondition,
    EquationSpec,
    Interval,
    ProfileInitialCondition,
)
from pinnforge.equations.burgers import Burgers1DSpec
from pinnforge.equations.harmonic import HarmonicOscillatorSpec
from pinnforge.equations.poisson import PoissonToySpec
from pinnforge.equations.registry import resolve_equation_id
from pinnforge.sampling.defaults import default_spec

GITHUB_URL = "https://github.com/shehbaz0101/pinnforge"
HEADER_MARKDOWN = (
    "# PINNForge\n\n"
    "A physics-informed network learns a classic equation from its residual on CPU, "
    "and a hybrid operator recovers Burgers viscosity from 32 sparse sensors. "
    f"[GitHub]({GITHUB_URL})"
)
STAGE_D_RELATIVE = Path("docs") / "v02" / "stage_d_scores.json"
_STAGE_D_FORMAT = "pinnforge.stage_d_scores.v1"
_LOCK = threading.Lock()
_REFERENCE = "#1f4e79"
_PINN = "#d06a2b"
_INITIAL = "#8a8f98"
_CLOSED = "#1f4e79"
_DATA = "#5f6b7a"
_HYBRID = "#d06a2b"


@dataclass(frozen=True, slots=True)
class LiveBudget:
    """Adam budget for one button click. The counts stay small on purpose."""

    epochs: int
    n_interior: int
    n_ic: int
    n_bc: int
    hidden: tuple[int, ...]
    lr: float


BUDGETS: dict[str, LiveBudget] = {
    "harmonic_oscillator": LiveBudget(150, 48, 8, 0, (32, 32), 1e-2),
    "poisson_toy": LiveBudget(250, 64, 0, 16, (48, 48), 5e-3),
    "burgers_1d": LiveBudget(800, 128, 32, 16, (32, 32), 1e-2),
}

_BLURBS: dict[str, str] = {
    "harmonic_oscillator": (
        "u'' + ω²u = 0 on t ∈ [0, 1], with ω = 1, u(0) = 1, and u'(0) = 0. "
        "The reference curve is the closed form cos(t). The network is trained when you click Run."
    ),
    "poisson_toy": (
        "−u'' = sin(πx) on [0, 1], with u(0) = u(1) = 0. "
        "The reference curve is sin(πx) / π². The network is trained when you click Run."
    ),
    "burgers_1d": (
        "u_t + u u_x = ν u_xx with ν = 0.1, periodic on x ∈ [−1, 1], t ∈ [0, 0.4], "
        "and initial data −sin(πx). There is no closed form. The reference curve is a "
        "Fourier solution of that same problem. This window is short and smooth so the "
        "fit finishes in seconds. The viscosity chart is a separate precomputed study."
    ),
}

_NOTES: dict[str, str] = {
    "harmonic_oscillator": "Reference: closed form.",
    "poisson_toy": "Reference: manufactured field.",
    "burgers_1d": (
        "Reference: Fourier ETDRK4 on this short problem, not a closed form. "
        "The hard-case viscosity numbers are the precomputed chart, not this curve."
    ),
}

_DISPLAY: dict[str, str] = {
    "harmonic_oscillator": "Harmonic oscillator",
    "poisson_toy": "Poisson",
    "burgers_1d": "Burgers",
}


@dataclass(frozen=True, slots=True)
class LiveSolve:
    """One finished button click."""

    equation_id: str
    display_name: str
    reference_name: str
    note: str
    x_label: str
    y_label: str
    x: np.ndarray
    prediction: np.ndarray
    reference: np.ndarray
    initial: np.ndarray | None
    relative_l2: float
    seconds: float
    epochs: int
    final_loss: float


@dataclass(frozen=True, slots=True)
class HardOodRow:
    """One estimator on the published hard out-of-distribution slice."""

    name: str
    mean_rel: float
    std_rel: float | None
    failures: int
    color: str


@dataclass(frozen=True, slots=True)
class StageDHeadline:
    """Hard-slice means copied out of the published score file."""

    n: int
    threshold_nu: float
    closed_form: HardOodRow
    data_only: HardOodRow
    hybrid: HardOodRow
    source: str

    def rows(self) -> tuple[HardOodRow, HardOodRow, HardOodRow]:
        return (self.closed_form, self.data_only, self.hybrid)


def equation_choices() -> list[tuple[str, str]]:
    """Radio labels and the ids :func:`run_quick_pinn` accepts."""

    return [
        ("Harmonic oscillator", "harmonic"),
        ("Burgers", "burgers"),
        ("Poisson", "poisson"),
    ]


def equation_blurb(equation: str) -> str:
    """One sentence for the selected equation, before any training."""

    return _BLURBS[_canonical(equation)]


def prepare_runtime() -> None:
    """Import torch and the trainer so the first click is only the solve.

    Raises:
        InstallHint: the optional ``ml`` extra is not installed.
    """

    from pinnforge.ml_import import require_torch

    require_torch()
    import pinnforge.training.loop  # noqa: F401


def run_quick_pinn(equation: str) -> LiveSolve:
    """Train one small CPU network and score it on a fixed plot grid.

    The call writes checkpoints under ``runs/live-demo/`` and deletes that
    directory before returning. A lock keeps two clicks from sharing it.

    Raises:
        ValueError: ``equation`` is not one of the three live problems,
            or a plotted value is not finite.
        InstallHint: torch is not installed.
    """

    equation_id = _canonical(equation)
    with _LOCK:
        return _run_locked(equation_id)


def stage_d_scores_path() -> Path:
    """Return the published Stage D score file.

    The search walks parents of this module, then the working directory,
    so a checkout and a Space that copied the repository both resolve.

    Raises:
        FileNotFoundError: the score file is not next to the source tree.
    """

    start = Path(__file__).resolve().parent
    for parent in (start, *start.parents):
        candidate = parent / STAGE_D_RELATIVE
        if candidate.is_file():
            return candidate
    cwd_candidate = Path.cwd() / STAGE_D_RELATIVE
    if cwd_candidate.is_file():
        return cwd_candidate
    raise FileNotFoundError(f"missing published scores at {STAGE_D_RELATIVE.as_posix()}")


def stage_d_headline() -> StageDHeadline:
    """Read the hard-slice means. This does not train or load a checkpoint."""

    path = stage_d_scores_path()
    record = json.loads(path.read_text(encoding="utf-8"))
    if record.get("format") != _STAGE_D_FORMAT:
        raise ValueError(f"unexpected Stage D format in {path}")
    closed_slice = record["baselines"]["sensors32_bursts"]["hard_ood"]
    data_slice = _selected_slice(record, "data_only")
    hybrid_slice = _selected_slice(record, "hybrid_1e-2")
    n = int(closed_slice["n_instances"])
    if int(data_slice["n_instances"]) != n or int(hybrid_slice["n_instances"]) != n:
        raise ValueError("Stage D hard_ood slices do not share n_instances")
    return StageDHeadline(
        n=n,
        threshold_nu=float(record["threshold_nu"]),
        closed_form=HardOodRow(
            name="Closed form",
            mean_rel=_finite(closed_slice["mean_rel_error"], "closed-form mean"),
            std_rel=None,
            failures=_whole_failures(closed_slice["n_failures"], "closed form"),
            color=_CLOSED,
        ),
        data_only=HardOodRow(
            name="Data-only FNO",
            mean_rel=_mean_field(data_slice, "mean_rel_error"),
            std_rel=_std_field(data_slice, "mean_rel_error"),
            failures=_whole_failures(data_slice["n_failures"]["mean"], "data-only"),
            color=_DATA,
        ),
        hybrid=HardOodRow(
            name="Hybrid operator",
            mean_rel=_mean_field(hybrid_slice, "mean_rel_error"),
            std_rel=_std_field(hybrid_slice, "mean_rel_error"),
            failures=_whole_failures(hybrid_slice["n_failures"]["mean"], "hybrid"),
            color=_HYBRID,
        ),
        source=STAGE_D_RELATIVE.as_posix(),
    )


def stage_d_markdown(headline: StageDHeadline | None = None) -> str:
    """Caption for the viscosity chart. The first word says the chart is cached."""

    headline = stage_d_headline() if headline is None else headline
    hybrid = format_error_percent(headline.hybrid.mean_rel)
    closed = format_error_percent(headline.closed_form.mean_rel)
    data = format_error_percent(headline.data_only.mean_rel)
    return (
        "### Viscosity from 32 sparse sensors\n\n"
        f"**Precomputed.** These bars are the published Stage D test scores in "
        f"`{headline.source}`. This page does not retrain the operator.\n\n"
        f"On the hard out-of-distribution slice ({headline.n} cases, viscosity at or below "
        f"{headline.threshold_nu:.3f}), the hybrid operator mean relative error is **{hybrid}**. "
        f"The closed form on the same 32-sensor mask is **{closed}** (about 35%) and fails the "
        f"0.5 rule on {headline.closed_form.failures} cases. The hybrid arm has "
        f"{headline.hybrid.failures} failures. Data-only is {data}."
    )


def format_error_percent(value: float) -> str:
    """One decimal percent for the public chart."""

    number = float(value) * 100.0
    if not math.isfinite(number):
        raise ValueError("percent is not finite")
    return f"{number:.1f}%"


def prediction_figure(solved: LiveSolve) -> Any:
    """Overlay the network on the reference. Burgers also draws the initial data."""

    plt = _pyplot()
    fig, ax = plt.subplots(figsize=(8.0, 3.6), dpi=120)
    if solved.initial is not None:
        ax.plot(solved.x, solved.initial, color=_INITIAL, linewidth=1.5, label="Initial condition")
    ax.plot(solved.x, solved.reference, color=_REFERENCE, linewidth=2.2, label=solved.reference_name)
    ax.plot(
        solved.x,
        solved.prediction,
        color=_PINN,
        linewidth=2.0,
        linestyle="--",
        label="PINN",
    )
    rel = format_error_percent(solved.relative_l2)
    ax.set_title(f"{solved.display_name}: prediction vs reference ({rel} relative L2)")
    ax.set_xlabel(solved.x_label)
    ax.set_ylabel(solved.y_label)
    _polish(ax)
    fig.tight_layout()
    return fig


def viscosity_figure(headline: StageDHeadline | None = None) -> Any:
    """Bar chart of the cached hard-slice errors, in percent."""

    headline = stage_d_headline() if headline is None else headline
    plt = _pyplot()
    fig, ax = plt.subplots(figsize=(5.8, 3.5), dpi=120)
    rows = headline.rows()
    percents = [row.mean_rel * 100.0 for row in rows]
    positions = np.arange(len(rows), dtype=float)
    bars = ax.bar(positions, percents, color=[row.color for row in rows], width=0.62)
    for bar, percent in zip(bars, percents, strict=True):
        ax.text(
            bar.get_x() + bar.get_width() / 2.0,
            bar.get_height(),
            f"{percent:.1f}%",
            ha="center",
            va="bottom",
            fontsize=11,
            color="#1c1c1c",
        )
    ax.set_xticks(positions, [row.name for row in rows])
    ax.set_ylabel("Mean relative error (%)")
    ax.set_title(f"Hard-OOD viscosity error, 32 sensors, n = {headline.n}")
    ax.set_ylim(0.0, max(percents) * 1.28)
    _polish(ax)
    fig.tight_layout()
    return fig


def solve_status(solved: LiveSolve) -> str:
    """One status line under the Run button."""

    rel = f"{solved.relative_l2 * 100.0:.2f}%"
    return (
        f"Live CPU solve, not a cached curve. {solved.display_name}, "
        f"{solved.epochs} Adam steps, {solved.seconds:.2f} s, relative L2 {rel}. {solved.note}"
    )


def _canonical(equation: str) -> str:
    try:
        equation_id = resolve_equation_id(equation)
    except ValueError:
        equation_id = ""
    if equation_id not in BUDGETS:
        known = ", ".join(alias for _label, alias in equation_choices())
        raise ValueError(f"live demo equation must be one of: {known}") from None
    return equation_id


def _run_locked(equation_id: str) -> LiveSolve:
    from pinnforge.ml_import import require_torch
    from pinnforge.specs.train import TrainConfig
    from pinnforge.training.loop import train_loop

    torch = require_torch()
    torch.set_num_threads(1)
    spec = _spec(equation_id)
    budget = BUDGETS[equation_id]
    run_dir = Path("runs") / "live-demo" / uuid.uuid4().hex
    try:
        config = TrainConfig(
            equation_id=spec.equation_id,
            epochs=budget.epochs,
            n_interior=budget.n_interior,
            n_ic=budget.n_ic,
            n_bc=budget.n_bc,
            hidden_widths=budget.hidden,
            lr=budget.lr,
            seed=0,
            checkpoint_dir=f"{run_dir.as_posix()}/checkpoints",
            log_path=f"{run_dir.as_posix()}/metrics.jsonl",
            checkpoint_interval=budget.epochs,
        )
        started = time.perf_counter()
        trained = train_loop(config, spec=spec)
        seconds = time.perf_counter() - started
        curves = _curves(trained.model, spec)
    finally:
        shutil.rmtree(run_dir, ignore_errors=True)
        _remove_if_empty(Path("runs") / "live-demo")
        _remove_if_empty(Path("runs"))
    relative = _relative_l2(curves.prediction, curves.reference)
    final_loss = float(trained.history[-1].loss)
    return LiveSolve(
        equation_id=equation_id,
        display_name=_DISPLAY[equation_id],
        reference_name=curves.reference_name,
        note=_NOTES[equation_id],
        x_label=curves.x_label,
        y_label=curves.y_label,
        x=curves.x,
        prediction=curves.prediction,
        reference=curves.reference,
        initial=curves.initial,
        relative_l2=relative,
        seconds=seconds,
        epochs=budget.epochs,
        final_loss=final_loss,
    )


def _spec(equation_id: str) -> EquationSpec:
    if equation_id == "burgers_1d":
        return Burgers1DSpec(
            nu=0.1,
            x=Interval(lower=-1.0, upper=1.0),
            t=Interval(lower=0.0, upper=0.4),
            initial_condition=ProfileInitialCondition(profile="negative_sin_pi_x"),
            boundary_conditions=[BoundaryCondition(variable="x", kind="periodic")],
        )
    spec = default_spec(equation_id)
    if not isinstance(spec, (HarmonicOscillatorSpec, PoissonToySpec)):
        raise ValueError(f"no live spec for {equation_id}")
    return spec


@dataclass(frozen=True, slots=True)
class _Curves:
    x: np.ndarray
    prediction: np.ndarray
    reference: np.ndarray
    initial: np.ndarray | None
    reference_name: str
    x_label: str
    y_label: str


def _curves(model: object, spec: EquationSpec) -> _Curves:
    if isinstance(spec, HarmonicOscillatorSpec):
        from pinnforge.reference.harmonic import displacement

        grid = np.linspace(float(spec.time.lower), float(spec.time.upper), 200)
        reference = np.asarray(displacement(grid, spec), dtype=np.float64)
        prediction = _predict(model, grid.reshape(-1, 1))
        return _Curves(grid, prediction, reference, None, "Closed form", "t", "u")
    if isinstance(spec, PoissonToySpec):
        from pinnforge.reference.poisson import reference_solution

        grid = np.linspace(float(spec.x.lower), float(spec.x.upper), 200)
        reference = np.asarray(reference_solution(spec, grid), dtype=np.float64)
        prediction = _predict(model, grid.reshape(-1, 1))
        return _Curves(grid, prediction, reference, None, "Manufactured field", "x", "u")
    if isinstance(spec, Burgers1DSpec):
        from pinnforge.reference.numerical.solver import grid as solver_grid
        from pinnforge.reference.numerical.solver import solve

        nodes = np.asarray(solver_grid(128), dtype=np.float64)
        initial = -np.sin(math.pi * nodes)
        final = float(spec.t.upper)
        trajectory = solve(initial, float(spec.nu), dt=1e-3, t_final=final, save_dt=final)
        reference = np.asarray(trajectory.u[-1], dtype=np.float64)
        coords = np.column_stack([nodes, np.full_like(nodes, final)])
        prediction = _predict(model, coords)
        return _Curves(nodes, prediction, reference, initial, "Fourier reference", "x", "u")
    raise ValueError(f"no live curves for {type(spec).__name__}")


def _predict(model: object, coords: np.ndarray) -> np.ndarray:
    from pinnforge.ml_import import require_torch

    torch = require_torch()
    if not hasattr(model, "parameters") or not callable(model):
        raise TypeError("model must be a trained network")
    parameter = next(model.parameters())
    tensor = torch.tensor(np.asarray(coords), dtype=parameter.dtype, device=parameter.device)
    model.eval()
    with torch.no_grad():
        values = model(tensor).detach().cpu().numpy().reshape(-1)
    predicted = np.asarray(values, dtype=np.float64)
    if predicted.shape != (coords.shape[0],) or not np.isfinite(predicted).all():
        raise ValueError("PINN prediction is not finite")
    return predicted


def _relative_l2(prediction: np.ndarray, reference: np.ndarray) -> float:
    if prediction.shape != reference.shape or not np.isfinite(reference).all():
        raise ValueError("reference curve is not finite")
    denom = float(np.linalg.norm(reference))
    if not math.isfinite(denom) or denom == 0.0:
        raise ValueError("reference norm is not positive")
    value = float(np.linalg.norm(prediction - reference) / denom)
    if not math.isfinite(value):
        raise ValueError("relative L2 is not finite")
    return value


def _selected_slice(record: dict[str, Any], arm: str) -> dict[str, Any]:
    arm_record = record["arms"][arm]
    selected = float(arm_record["selected_lambda"])
    objectives = arm_record["objectives"]
    key = next(name for name in objectives if float(name) == selected)
    return objectives[key]["slices"]["hard_ood"]


def _mean_field(slice_record: dict[str, Any], name: str) -> float:
    return _finite(slice_record[name]["mean"], name)


def _std_field(slice_record: dict[str, Any], name: str) -> float:
    return _finite(slice_record[name]["std"], f"{name} std")


def _whole_failures(value: object, label: str) -> int:
    number = _finite(value, f"{label} failures")
    if abs(number - round(number)) > 1e-9:
        raise ValueError(f"{label} failure count is not a whole number")
    return int(round(number))


def _finite(value: object, label: str) -> float:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise ValueError(f"{label} must be a number")
    number = float(value)
    if not math.isfinite(number):
        raise ValueError(f"{label} must be finite")
    return number


def _pyplot() -> Any:
    import matplotlib

    matplotlib.use("Agg")
    from matplotlib import pyplot as plt

    return plt


def _remove_if_empty(path: Path) -> None:
    if path.is_dir() and not any(path.iterdir()):
        path.rmdir()


def _polish(ax: Any) -> None:
    ax.grid(True, color="#e6e8eb", linewidth=0.8)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    ax.tick_params(colors="#1c1c1c")
    handles, _labels = ax.get_legend_handles_labels()
    if handles:
        ax.legend(frameon=False, loc="best")
    ax.title.set_color("#1c1c1c")
    ax.xaxis.label.set_color("#1c1c1c")
    ax.yaxis.label.set_color("#1c1c1c")
