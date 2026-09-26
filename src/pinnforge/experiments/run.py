"""Train, then evaluate, one experiment config.

Importing this module requires torch (the optional ``ml`` extra).
:mod:`pinnforge.experiments` does not import it, so loading a config
stays free of torch.
"""

from __future__ import annotations

from dataclasses import dataclass, replace
from pathlib import Path

from pinnforge.evaluation import evaluate_model, write_eval_json
from pinnforge.evaluation.record import EvalResult
from pinnforge.experiments.config import ExperimentConfig
from pinnforge.training import TrainResult, train_loop


@dataclass(frozen=True, slots=True)
class RunResult:
    """Files and scores from :func:`run_experiment`.

    ``evaluation.checkpoint`` is the relative ``checkpoint.pt`` path.
    ``eval_json`` is the resolved record path when the config asked for
    one, and ``None`` when it did not.
    """

    config: ExperimentConfig
    train: TrainResult
    evaluation: EvalResult
    eval_json: Path | None


def run_experiment(config: ExperimentConfig) -> RunResult:
    """Train ``config`` on CPU, then score the same spec.

    The training spec is ``config.equation``, including any parameter
    overrides. Evaluation uses that spec and ``config.eval``, not a
    second built-in lookup. When ``config.eval_json`` is set, the eval
    record is written there.

    Raises:
        TypeError: ``config`` is not an :class:`ExperimentConfig`.
        ValueError: a checkpoint, log, or JSON path escapes the working
            directory, or training or evaluation rejects the config.
    """

    if not isinstance(config, ExperimentConfig):
        raise TypeError("config must be an ExperimentConfig")
    trained = train_loop(config.train, spec=config.equation)
    evaluation = evaluate_model(trained.model, config.equation, config.eval)
    relative = trained.checkpoint_path.resolve().relative_to(Path.cwd().resolve()).as_posix()
    evaluation = replace(evaluation, checkpoint=relative)
    written = None
    if config.eval_json is not None:
        written = write_eval_json(evaluation, config.eval_json)
    return RunResult(
        config=config,
        train=trained,
        evaluation=evaluation,
        eval_json=written,
    )
