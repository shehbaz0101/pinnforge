"""Experiment files that select an equation and the train and eval settings.

A file is YAML or JSON under the working directory. Absolute paths and
paths that resolve outside that directory are rejected, the same sandbox
as checkpoints and sample records. Loading a file does not import torch.
"""

from __future__ import annotations

import json
from collections.abc import Mapping
from pathlib import Path
from typing import Annotated

import yaml
from pydantic import BaseModel, BeforeValidator, ConfigDict, model_validator

from pinnforge.equations.burgers import Burgers1DSpec
from pinnforge.equations.harmonic import HarmonicOscillatorSpec
from pinnforge.equations.poisson import PoissonToySpec
from pinnforge.equations.registry import build_equation, resolve_equation_id
from pinnforge.specs.eval import EvalConfig
from pinnforge.specs.paths import resolve_inside_cwd
from pinnforge.specs.train import TrainConfig

_CONFIG_SUFFIXES = frozenset({".yaml", ".yml", ".json"})


def _relative_json(value: object) -> str | None:
    if value is None:
        return None
    if not isinstance(value, str):
        raise ValueError("eval_json must be a string")
    if value == "" or value != value.strip() or "\x00" in value:
        raise ValueError("eval_json must be a non-empty relative path")
    raw = Path(value)
    if raw.is_absolute() or value.startswith("~"):
        raise ValueError("eval_json must be a relative path")
    if raw.parts == () or raw == Path("."):
        raise ValueError("eval_json must name a path under the working directory")
    if raw.suffix.lower() != ".json":
        raise ValueError("eval_json must end in .json")
    return value


RelativeJson = Annotated[str | None, BeforeValidator(_relative_json)]


class ExperimentConfig(BaseModel):
    """One train-then-eval run.

    ``equation`` is a built-in spec. The file may name it with an id or
    a short alias (``harmonic``, ``burgers``, ``poisson``) plus optional
    ``equation_params``, or it may give the full Day 1 spec inline.
    ``equation_params`` are merged onto the built-in and validated by
    that schema. They cannot be combined with an inline spec.

    ``train`` holds :class:`~pinnforge.specs.train.TrainConfig` fields
    except that ``equation_id`` comes from ``equation``. Omitted counts
    use the per-equation defaults. ``hidden_widths`` is the MLP width
    list. ``checkpoint_dir`` and ``log_path`` are relative paths.

    ``eval`` holds :class:`~pinnforge.specs.eval.EvalConfig` fields.
    ``eval_json``, when set, is a relative ``.json`` path for the eval
    record. ``pinnforge run --config`` trains, then evaluates, and
    writes that file when it is set.
    """

    model_config = ConfigDict(extra="forbid", frozen=True)

    equation: HarmonicOscillatorSpec | Burgers1DSpec | PoissonToySpec
    train: TrainConfig
    eval: EvalConfig
    eval_json: RelativeJson = None

    @model_validator(mode="before")
    @classmethod
    def _normalize(cls, data: object) -> object:
        if not isinstance(data, dict):
            return data
        raw = dict(data)
        params = raw.pop("equation_params", None)
        has_equation = "equation" in raw
        has_id = "equation_id" in raw
        if has_equation and has_id:
            raise ValueError("pass equation or equation_id, not both")
        if not has_equation and not has_id:
            raise ValueError("experiment config requires equation")
        equation = raw.pop("equation") if has_equation else raw.pop("equation_id")
        if params is not None and (isinstance(params, str) or not isinstance(params, Mapping)):
            raise ValueError("equation_params must be a mapping")
        spec = build_equation(equation, params)
        raw["equation"] = spec
        raw["train"] = _train_payload(raw.get("train"), spec.equation_id)
        raw["eval"] = _eval_payload(raw.get("eval"), "eval" in raw)
        return raw


def _train_payload(train: object, equation_id: str) -> dict[str, object]:
    if train is None:
        payload: dict[str, object] = {}
    elif isinstance(train, TrainConfig):
        payload = train.model_dump(mode="json")
    elif isinstance(train, dict):
        payload = dict(train)
    else:
        raise ValueError("train must be a mapping")
    if "equation_id" in payload:
        raw_id = payload["equation_id"]
        if not isinstance(raw_id, str) or resolve_equation_id(raw_id) != equation_id:
            raise ValueError("train.equation_id does not match equation")
    payload["equation_id"] = equation_id
    return payload


def _eval_payload(eval_config: object, present: bool) -> dict[str, object]:
    if eval_config is None:
        if present:
            raise ValueError("eval must be a mapping")
        return {}
    if isinstance(eval_config, EvalConfig):
        return eval_config.model_dump(mode="json")
    if isinstance(eval_config, dict):
        return dict(eval_config)
    raise ValueError("eval must be a mapping")


def resolve_config_path(path: str | Path) -> Path:
    """Resolve a relative ``.yaml``, ``.yml``, or ``.json`` config path.

    The path must stay inside the working directory after ``..`` and
    symlinks are resolved. Absolute paths and ``~`` are rejected.

    Raises:
        ValueError: the path is absolute, escapes the working directory,
            or has the wrong suffix.
    """

    text = path.as_posix() if isinstance(path, Path) else path
    if not isinstance(text, str):
        raise ValueError("experiment config path must be a string")
    if text.startswith("~") or "\x00" in text:
        raise ValueError("experiment config must be a relative path")
    resolved = resolve_inside_cwd(path)
    if resolved.suffix.lower() not in _CONFIG_SUFFIXES:
        raise ValueError("experiment config must end in .yaml, .yml, or .json")
    return resolved


def load_experiment_config(path: str | Path) -> ExperimentConfig:
    """Read ``path`` and validate it as an :class:`ExperimentConfig`.

    Raises:
        ValueError: the path is outside the working directory, the file
            is missing, or the document is not a mapping.
        ValidationError: the document is a mapping the schema rejects.
    """

    resolved = resolve_config_path(path)
    if not resolved.is_file():
        raise ValueError(f"experiment config not found: {path}")
    text = resolved.read_text(encoding="utf-8")
    data = _parse_document(text, resolved.suffix.lower())
    if not isinstance(data, dict):
        raise ValueError("experiment config must be a mapping")
    config = ExperimentConfig.model_validate(data)
    _sandbox_outputs(config)
    return config


def dump_experiment_config(config: ExperimentConfig, path: str | Path) -> Path:
    """Write the canonical JSON document for ``config``.

    ``path`` follows :func:`resolve_config_path`. YAML output uses the
    expanded spec, not the short ``equation`` alias the file may have
    been loaded from. Parent directories are created.

    Raises:
        TypeError: ``config`` is not an :class:`ExperimentConfig`.
        ValueError: ``path`` escapes the working directory or has the
            wrong suffix.
    """

    if not isinstance(config, ExperimentConfig):
        raise TypeError("config must be an ExperimentConfig")
    resolved = resolve_config_path(path)
    document = config.model_dump(mode="json")
    if resolved.suffix.lower() == ".json":
        text = json.dumps(document, indent=2) + "\n"
    else:
        text = yaml.safe_dump(document, sort_keys=False)
    resolved.parent.mkdir(parents=True, exist_ok=True)
    resolved.write_text(text, encoding="utf-8")
    return resolved


def _sandbox_outputs(config: ExperimentConfig) -> None:
    """Reject checkpoint, metrics, and eval JSON paths that leave the root.

    The schema only checks that those strings look relative. ``..`` and
    symlinks are resolved here, before torch is imported.
    """

    resolve_inside_cwd(config.train.checkpoint_dir, label="checkpoint_dir")
    resolve_inside_cwd(config.train.log_path, suffix=".jsonl", label="log_path")
    if config.eval_json is not None:
        resolve_inside_cwd(config.eval_json, suffix=".json", label="eval_json")


def _parse_document(text: str, suffix: str) -> object:
    if suffix == ".json":
        try:
            return json.loads(text)
        except json.JSONDecodeError as exc:
            raise ValueError(f"experiment config is not valid JSON: {exc}") from exc
    try:
        return yaml.safe_load(text)
    except yaml.YAMLError as exc:
        raise ValueError(f"experiment config is not valid YAML: {exc}") from exc
