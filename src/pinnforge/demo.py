"""One-command offline demo.

``pinnforge demo`` trains a short CPU run from a checked-in sample under
``samples/configs/`` and scores it with the Day 5 evaluator. The default
is the harmonic oscillator. Poisson uses the manufactured field. Burgers
reports residual metrics only.

Importing this module does not import torch. :func:`execute_demo` does.
The CLI checks the sample path and installs the offline guard before
that import. User paths stay inside the sandbox root
(``PINNFORGE_DATA_ROOT`` or ``--data-root``).
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from pathlib import Path
from typing import TYPE_CHECKING

from pinnforge.equations.registry import resolve_equation_id
from pinnforge.experiments import ExperimentConfig, load_experiment_config
from pinnforge.specs.paths import resolve_inside_cwd

if TYPE_CHECKING:
    from pinnforge.experiments.run import RunResult

# Relative to the sandbox root. ``pinnforge demo`` and the README use these.
SAMPLE_CONFIGS: dict[str, str] = {
    "harmonic_oscillator": "samples/configs/harmonic.yaml",
    "poisson_toy": "samples/configs/poisson.json",
    "burgers_1d": "samples/configs/burgers.yaml",
}
DEFAULT_EQUATION = "harmonic"
MAX_DEMO_EPOCHS = 5
_ANCHOR = "samples/configs/harmonic.yaml"


class DemoDependencyError(RuntimeError):
    """The demo needs the optional ``ml`` extra."""


@dataclass(frozen=True, slots=True)
class DemoResult:
    """Printed summary and the file that stores the same text."""

    config_path: str
    equation_id: str
    epochs: int
    summary_path: Path
    summary: str


def bundled_samples_dir() -> Path | None:
    """Return the checked-in ``samples/`` directory, or ``None`` when it is absent.

    The search walks parents of this file so an editable install finds the
    repository ``samples/`` next to ``src/``.
    """

    start = Path(__file__).resolve().parent
    for parent in (start, *start.parents):
        if (parent / _ANCHOR).is_file():
            return parent / "samples"
    return None


def sample_config_path(equation: str) -> str:
    """Return the sandbox-relative sample path for an id or alias.

    Raises:
        ValueError: ``equation`` is not a built-in.
    """

    equation_id = resolve_equation_id(equation)
    try:
        return SAMPLE_CONFIGS[equation_id]
    except KeyError:
        known = ", ".join(sorted(SAMPLE_CONFIGS))
        raise ValueError(f"no demo sample for {equation_id}; known: {known}") from None


def load_demo_experiment(
    equation: str = DEFAULT_EQUATION,
    *,
    epochs: int | None = None,
) -> tuple[str, ExperimentConfig]:
    """Load the sample experiment for ``equation`` inside the sandbox root.

    When ``samples/configs/...`` is already a file inside the root, that
    file is used. Otherwise the checked-in sample is copied to the same
    relative path under the root. ``epochs``, when set, replaces the
    sample's epoch count. The sample file on disk is not rewritten.
    Either way the count must be from 1 to :data:`MAX_DEMO_EPOCHS`.

    Raises:
        ValueError: the equation is unknown, the epoch count is outside
            the demo cap, the sample is missing, or a path leaves the
            sandbox root.
    """

    if epochs is not None:
        _check_epochs(epochs)
    relative = sample_config_path(equation)
    placed = _place_sample_config(relative)
    config = load_experiment_config(placed)
    if epochs is not None and epochs != config.train.epochs:
        document = config.model_dump(mode="json")
        document["train"]["epochs"] = epochs
        config = ExperimentConfig.model_validate(document)
    _check_epochs(config.train.epochs)
    return placed, config


def execute_demo(config_path: str, config: ExperimentConfig) -> DemoResult:
    """Train ``config`` and write the printed summary next to the eval record.

    Torch is imported here. Callers that have not installed the offline
    guard yet should install it before this function.

    Raises:
        TypeError: ``config`` is not an :class:`ExperimentConfig`.
        DemoDependencyError: the ``ml`` extra is not installed.
        ValueError: a metric is not finite, or a path leaves the sandbox.
    """

    if not isinstance(config, ExperimentConfig):
        raise TypeError("config must be an ExperimentConfig")
    result = _run(config)
    summary_rel = _summary_relative(config)
    text = format_demo_summary(config_path, config, result, summary_rel)
    resolved = resolve_inside_cwd(summary_rel, suffix=".txt", label="summary")
    resolved.parent.mkdir(parents=True, exist_ok=True)
    resolved.write_text(text, encoding="utf-8")
    return DemoResult(
        config_path=config_path,
        equation_id=config.equation.equation_id,
        epochs=config.train.epochs,
        summary_path=resolved,
        summary=text,
    )


def format_demo_summary(
    config_path: str,
    config: ExperimentConfig,
    result: RunResult,
    summary_rel: str,
) -> str:
    """Return the text ``pinnforge demo`` prints, including a trailing newline."""

    final = result.train.history[-1]
    _require_finite("loss", final.loss)
    _require_finite("loss_pde", final.loss_pde)
    _require_finite("loss_ic", final.loss_ic)
    _require_finite("loss_bc", final.loss_bc)
    _require_finite("l2", result.evaluation.l2)
    _require_finite("relative_l2", result.evaluation.relative_l2)
    _require_finite("residual_mean_abs", result.evaluation.residual_mean_abs)
    _require_finite("residual_max_abs", result.evaluation.residual_max_abs)
    lines = [
        "pinnforge demo",
        f"equation: {config.equation.equation_id}",
        f"config: {config_path}",
        f"seed: {config.train.seed}",
        f"epochs: {config.train.epochs}",
        f"device: {config.train.device}",
        f"checkpoint: {result.evaluation.checkpoint}",
        f"log: {Path(config.train.log_path).as_posix()}",
        f"loss: {final.loss:.8e}",
        f"loss_pde: {final.loss_pde:.8e}",
        f"loss_ic: {final.loss_ic:.8e}",
        f"loss_bc: {final.loss_bc:.8e}",
        f"lr: {final.lr:.8e}",
        f"method: {result.evaluation.method}",
        f"eval_seed: {result.evaluation.seed}",
        f"n_interior: {result.evaluation.n_interior}",
        f"reference: {result.evaluation.reference}",
        f"l2: {_format_metric(result.evaluation.l2)}",
        f"relative_l2: {_format_metric(result.evaluation.relative_l2)}",
        f"residual_mean_abs: {result.evaluation.residual_mean_abs:.8e}",
        f"residual_max_abs: {result.evaluation.residual_max_abs:.8e}",
    ]
    if config.eval_json is not None:
        lines.append(f"json: {Path(config.eval_json).as_posix()}")
    lines.append(f"summary: {summary_rel}")
    return "\n".join(lines) + "\n"


def _run(config: ExperimentConfig) -> RunResult:
    from pinnforge.ml_import import InstallHint

    try:
        from pinnforge.experiments.run import run_experiment
    except InstallHint as exc:
        raise DemoDependencyError(str(exc)) from exc
    except ModuleNotFoundError as exc:
        if exc.name != "torch":
            raise
        raise DemoDependencyError(str(InstallHint())) from exc
    return run_experiment(config)


def _place_sample_config(relative: str) -> str:
    """Return ``relative`` once that file exists inside the sandbox root."""

    destination = resolve_inside_cwd(relative, label="config")
    if destination.is_file():
        return relative
    if destination.exists():
        raise ValueError(f"sample config is not a file: {relative}")
    bundled = _bundled_config(relative)
    if bundled is None:
        raise ValueError(f"sample config not found: {relative}")
    try:
        destination.parent.mkdir(parents=True, exist_ok=True)
        destination.write_text(bundled.read_text(encoding="utf-8"), encoding="utf-8")
    except OSError as exc:
        raise ValueError(f"could not place sample config {relative}: {exc}") from exc
    return relative


def _bundled_config(relative: str) -> Path | None:
    samples = bundled_samples_dir()
    if samples is None:
        return None
    # ``relative`` is ``samples/configs/<file>``. The bundle root is ``samples/``.
    suffix = relative.removeprefix("samples/")
    candidate = samples / suffix
    if candidate.is_file():
        return candidate
    return None


def _summary_relative(config: ExperimentConfig) -> str:
    if config.eval_json is not None:
        parent = Path(config.eval_json).parent
    else:
        parent = Path(config.train.log_path).parent
    return (parent / "summary.txt").as_posix()


def _check_epochs(epochs: int) -> None:
    if isinstance(epochs, bool) or not isinstance(epochs, int):
        raise ValueError(f"demo epochs must be an integer from 1 to {MAX_DEMO_EPOCHS}")
    if epochs < 1 or epochs > MAX_DEMO_EPOCHS:
        raise ValueError(f"demo epochs must be an integer from 1 to {MAX_DEMO_EPOCHS}")


def _require_finite(name: str, value: float | None) -> None:
    if value is None:
        return
    if not math.isfinite(value):
        raise ValueError(f"demo {name} is not finite")


def _format_metric(value: float | None) -> str:
    if value is None:
        return "unavailable"
    return f"{value:.8e}"
