"""Equation catalog and train / eval / run calls for the HTTP API.

Path checks use the Day 6 sandbox and run before torch is imported.
Importing this module does not import torch or FastAPI.
"""

from __future__ import annotations

from pathlib import Path

from pinnforge.api.schemas import (
    EquationDetail,
    EquationSummary,
    EvalBody,
    EvalRequest,
    EvalResponse,
    RunRequest,
    RunResponse,
    TrainSummary,
)
from pinnforge.equations import list_equations
from pinnforge.equations.registry import resolve_equation_id
from pinnforge.experiments import ExperimentConfig, load_experiment_config
from pinnforge.ml_import import InstallHint
from pinnforge.sampling import default_spec, resolve_output_path
from pinnforge.specs.eval import EvalConfig
from pinnforge.specs.paths import resolve_inside_cwd
from pinnforge.specs.train import TrainConfig


class UnknownEquation(LookupError):
    """The path id is not a registry id or a short alias."""


class MlExtraMissing(RuntimeError):
    """Train, eval, or run was called without the optional ``ml`` extra."""


def list_equation_summaries() -> list[EquationSummary]:
    """Return the built-in catalog in registry order."""

    return [_summary(info) for info in list_equations()]


def equation_detail(name: str) -> EquationDetail:
    """Return one built-in and the default spec for ``name``.

    ``name`` may be the registry id or a short alias.

    Raises:
        UnknownEquation: ``name`` is not a built-in.
    """

    try:
        equation_id = resolve_equation_id(name)
    except ValueError as exc:
        raise UnknownEquation(str(exc)) from exc
    info = next(item for item in list_equations() if item.equation_id == equation_id)
    spec = default_spec(equation_id)
    summary = _summary(info)
    return EquationDetail(
        equation_id=summary.equation_id,
        aliases=summary.aliases,
        summary=summary.summary,
        parameters=summary.parameters,
        spec=spec.model_dump(mode="json"),
    )


def run_request(body: RunRequest) -> RunResponse:
    """Train then evaluate one experiment file or inline document.

    Config, checkpoint, log, and eval JSON paths are checked before
    torch is imported.

    Raises:
        MlExtraMissing: torch is not installed.
        ValueError: a path escapes the working directory or the file
            is missing.
        ValidationError: the experiment document is rejected.
    """

    config = _experiment(body)
    _check_train_paths(config.train)
    if config.eval_json is not None:
        resolve_output_path(config.eval_json)
    run_experiment = _import_run_experiment()
    result = run_experiment(config)
    final = result.train.history[-1]
    checkpoint = result.evaluation.checkpoint
    if not isinstance(checkpoint, str) or checkpoint == "":
        raise ValueError("run did not record a checkpoint path")
    log = Path(config.train.log_path).as_posix()
    return RunResponse(
        equation_id=config.equation.equation_id,
        checkpoint=checkpoint,
        log=log,
        train=_train_summary(config.train, final, checkpoint, log),
        evaluation=EvalBody.model_validate(result.evaluation.as_dict()),
        eval_json=None if config.eval_json is None else Path(config.eval_json).as_posix(),
    )


def train_request(config: TrainConfig) -> TrainSummary:
    """Train ``config`` and return the final loss and relative paths.

    Raises:
        MlExtraMissing: torch is not installed.
        TypeError: ``config`` is not a :class:`TrainConfig`.
        ValueError: a path escapes the working directory.
    """

    if not isinstance(config, TrainConfig):
        raise TypeError("config must be a TrainConfig")
    _check_train_paths(config)
    train_loop = _import_train_loop()
    result = train_loop(config)
    final = result.history[-1]
    checkpoint = (Path(config.checkpoint_dir) / "checkpoint.pt").as_posix()
    log = Path(config.log_path).as_posix()
    return _train_summary(config, final, checkpoint, log)


def eval_request(body: EvalRequest) -> EvalResponse:
    """Score a checkpoint inside the working directory.

    The checkpoint and ``write_json`` paths are checked before torch is
    imported. ``equation`` must match the checkpoint.

    Raises:
        MlExtraMissing: torch is not installed.
        ValueError: a path escapes the working directory, the file is
            missing, or the equation does not match.
    """

    resolve_inside_cwd(body.checkpoint, suffix=".pt")
    if body.write_json is not None:
        resolve_output_path(body.write_json)
    evaluate_checkpoint, write_eval_json = _import_evaluate()
    result = evaluate_checkpoint(body.checkpoint, _eval_config(body), equation=body.equation)
    written = None
    if body.write_json is not None:
        write_eval_json(result, body.write_json)
        written = Path(body.write_json).as_posix()
    payload = result.as_dict()
    payload["eval_json"] = written
    return EvalResponse.model_validate(payload)


def _experiment(body: RunRequest) -> ExperimentConfig:
    if body.config is not None:
        return load_experiment_config(body.config)
    payload = body.model_dump(exclude_none=True)
    payload.pop("config", None)
    return ExperimentConfig.model_validate(payload)


def _eval_config(body: EvalRequest) -> EvalConfig:
    payload: dict[str, object] = {}
    if body.n_interior is not None:
        payload["n_interior"] = body.n_interior
    if body.seed is not None:
        payload["seed"] = body.seed
    if body.method is not None:
        payload["method"] = body.method
    if body.bins is not None:
        payload["bins"] = body.bins
    return EvalConfig.model_validate(payload)


def _check_train_paths(config: TrainConfig) -> None:
    resolve_inside_cwd(config.checkpoint_dir)
    resolve_inside_cwd(config.log_path, suffix=".jsonl")


def _train_summary(config: TrainConfig, final: object, checkpoint: str, log: str) -> TrainSummary:
    return TrainSummary(
        equation_id=config.equation_id,
        seed=config.seed,
        epochs=config.epochs,
        device=config.device,
        checkpoint=checkpoint,
        log=log,
        loss=float(final.loss),
        loss_pde=float(final.loss_pde),
        loss_ic=float(final.loss_ic),
        loss_bc=float(final.loss_bc),
        lr=float(final.lr),
    )


def _summary(info: object) -> EquationSummary:
    return EquationSummary(
        equation_id=info.equation_id,
        aliases=list(info.aliases),
        summary=info.summary,
        parameters=list(info.parameters),
    )


def _import_run_experiment() -> object:
    try:
        from pinnforge.experiments.run import run_experiment
    except InstallHint as exc:
        raise MlExtraMissing(str(exc)) from exc
    except ModuleNotFoundError as exc:
        if exc.name != "torch":
            raise
        raise MlExtraMissing(str(InstallHint())) from exc
    return run_experiment


def _import_train_loop() -> object:
    try:
        from pinnforge.training import train_loop
    except InstallHint as exc:
        raise MlExtraMissing(str(exc)) from exc
    except ModuleNotFoundError as exc:
        if exc.name != "torch":
            raise
        raise MlExtraMissing(str(InstallHint())) from exc
    return train_loop


def _import_evaluate() -> tuple[object, object]:
    try:
        from pinnforge.evaluation import evaluate_checkpoint, write_eval_json
    except InstallHint as exc:
        raise MlExtraMissing(str(exc)) from exc
    except ModuleNotFoundError as exc:
        if exc.name != "torch":
            raise
        raise MlExtraMissing(str(InstallHint())) from exc
    return evaluate_checkpoint, write_eval_json
