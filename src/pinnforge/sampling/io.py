"""JSON records for a sampled batch.

Records are plain JSON so a test fixture or a ``--output`` file needs no
network and no pickle. ``.npy`` files of the stacked coordinates are a
second offline copy of the same points; this module writes the JSON.
"""

from __future__ import annotations

import json
from collections.abc import Mapping
from pathlib import Path

import numpy as np

from pinnforge.equations.base import EquationSpec
from pinnforge.equations.registry import parse_equation
from pinnforge.sampling.batch import CollocationBatch
from pinnforge.sampling.config import SampleConfig

SAMPLE_RECORD_FORMAT = "pinnforge.sample.v1"


def batch_to_dict(batch: CollocationBatch) -> dict[str, object]:
    """JSON-ready mapping of coordinates, labels, and bounds."""

    if not isinstance(batch, CollocationBatch):
        raise TypeError("batch must be a CollocationBatch")
    bounds = [
        {"name": name, "lower": float(lower), "upper": float(upper)}
        for name, lower, upper in zip(batch.axis_names, batch.lower, batch.upper, strict=True)
    ]
    return {
        "equation_id": batch.equation_id,
        "method": batch.method,
        "seed": batch.seed,
        "bounds": bounds,
        "counts": batch.counts,
        "interior": batch.interior.tolist(),
        "ic": batch.ic.tolist(),
        "bc": batch.bc.tolist(),
        "bc_variable": list(batch.bc_variable),
        "bc_side": list(batch.bc_side),
        "labels": batch.labels().tolist(),
    }


def batch_from_dict(data: Mapping[str, object]) -> CollocationBatch:
    """Rebuild a batch from :func:`batch_to_dict`.

    This does not draw new points. Counts and labels, when present, must
    match the coordinate arrays.

    Raises:
        ValueError: the mapping is missing a field or disagrees with itself.
    """

    equation_id = _require(data, "equation_id")
    method = _require(data, "method")
    seed = _require(data, "seed")
    bounds = _require(data, "bounds")
    if not isinstance(equation_id, str) or not isinstance(method, str):
        raise ValueError("equation_id and method must be strings")
    if isinstance(seed, bool) or not isinstance(seed, int):
        raise ValueError("seed must be an integer")
    if not isinstance(bounds, list) or not bounds:
        raise ValueError("bounds must be a non-empty list")
    names: list[str] = []
    lower: list[float] = []
    upper: list[float] = []
    for item in bounds:
        if not isinstance(item, Mapping):
            raise ValueError("each bounds entry must be a mapping")
        name = item.get("name")
        if not isinstance(name, str):
            raise ValueError("bounds name must be a string")
        names.append(name)
        lower.append(_finite_number(item.get("lower"), label=f"{name} lower"))
        upper.append(_finite_number(item.get("upper"), label=f"{name} upper"))
    width = len(names)
    interior = _coords(_require(data, "interior"), width, label="interior")
    ic = _coords(_require(data, "ic"), width, label="ic")
    bc = _coords(_require(data, "bc"), width, label="bc")
    variables = _strings(_require(data, "bc_variable"), label="bc_variable")
    sides = _strings(_require(data, "bc_side"), label="bc_side")
    batch = CollocationBatch(
        equation_id=equation_id,
        axis_names=tuple(names),
        lower=np.array(lower, dtype=np.float64),
        upper=np.array(upper, dtype=np.float64),
        method=method,  # type: ignore[arg-type]
        seed=seed,
        interior=interior,
        ic=ic,
        bc=bc,
        bc_variable=tuple(variables),
        bc_side=tuple(sides),
    )
    counts = data.get("counts")
    if counts is not None:
        if not isinstance(counts, Mapping):
            raise ValueError("counts must be a mapping")
        actual = batch.counts
        for key in ("interior", "ic", "bc"):
            if counts.get(key) != actual[key]:
                raise ValueError("counts do not match coordinate arrays")
    labels = data.get("labels")
    if labels is not None and list(labels) != batch.labels().tolist():
        raise ValueError("labels do not match interior, ic, and bc counts")
    return batch


def sample_record(spec: EquationSpec, config: SampleConfig, batch: CollocationBatch) -> dict[str, object]:
    """JSON-ready record: spec, config, and the batch they produced."""

    if not isinstance(spec, EquationSpec):
        raise TypeError("spec must be an equation spec")
    if not isinstance(config, SampleConfig):
        raise TypeError("config must be a SampleConfig")
    return {
        "format": SAMPLE_RECORD_FORMAT,
        "spec": spec.model_dump(mode="json"),
        "config": config.model_dump(mode="json"),
        "batch": batch_to_dict(batch),
    }


def dumps_sample_record(spec: EquationSpec, config: SampleConfig, batch: CollocationBatch) -> str:
    """Serialize :func:`sample_record` with a trailing newline."""

    return json.dumps(sample_record(spec, config, batch), indent=2) + "\n"


def write_sample_record(
    spec: EquationSpec,
    config: SampleConfig,
    batch: CollocationBatch,
    path: Path,
) -> None:
    """Write a sample record to ``path``, creating parent directories."""

    destination = Path(path)
    destination.parent.mkdir(parents=True, exist_ok=True)
    destination.write_text(dumps_sample_record(spec, config, batch), encoding="utf-8")


def load_sample_record(path: Path) -> tuple[EquationSpec, SampleConfig, CollocationBatch]:
    """Read a record written by :func:`write_sample_record`.

    Raises:
        ValueError: the format tag is missing or the payload does not validate.
    """

    payload = json.loads(Path(path).read_text(encoding="utf-8"))
    if not isinstance(payload, dict):
        raise ValueError("sample record must be a JSON object")
    if payload.get("format") != SAMPLE_RECORD_FORMAT:
        raise ValueError(f"unsupported sample record format {payload.get('format')!r}")
    spec_data = payload.get("spec")
    config_data = payload.get("config")
    batch_data = payload.get("batch")
    if not isinstance(spec_data, dict) or not isinstance(config_data, dict):
        raise ValueError("sample record requires spec and config objects")
    if not isinstance(batch_data, dict):
        raise ValueError("sample record requires a batch object")
    spec = parse_equation(spec_data)
    config = SampleConfig.model_validate(config_data)
    batch = batch_from_dict(batch_data)
    if batch.equation_id != spec.equation_id:
        raise ValueError("batch equation_id does not match the spec")
    if batch.seed != config.seed or batch.method != config.method:
        raise ValueError("batch seed and method must match the config")
    counts = batch.counts
    if (
        counts["interior"] != config.n_interior
        or counts["ic"] != config.n_ic
        or counts["bc"] != config.n_bc
    ):
        raise ValueError("batch counts must match the config")
    return spec, config, batch


def resolve_output_path(path: str | Path) -> Path:
    """Resolve a sandbox-friendly relative ``.json`` output path.

    The path must stay inside the working directory after ``..`` and
    symlinks are resolved. Absolute paths are rejected.

    Raises:
        ValueError: the path is absolute, escapes the working directory,
            or does not end in ``.json``.
    """

    raw = Path(path)
    if raw.is_absolute():
        raise ValueError("output path must be a relative path")
    if raw.parts == () or raw == Path("."):
        raise ValueError("output path must be a file path")
    cwd = Path.cwd().resolve()
    resolved = (cwd / raw).resolve()
    if resolved == cwd or cwd not in resolved.parents:
        raise ValueError("output path must stay inside the working directory")
    if resolved.suffix.lower() != ".json":
        raise ValueError("output path must end in .json")
    return resolved


def _require(data: Mapping[str, object], key: str) -> object:
    if key not in data:
        raise ValueError(f"sample batch is missing {key!r}")
    return data[key]


def _finite_number(value: object, *, label: str) -> float:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise ValueError(f"{label} must be a finite number")
    number = float(value)
    if not np.isfinite(number):
        raise ValueError(f"{label} must be a finite number")
    return number


def _coords(value: object, width: int, *, label: str) -> np.ndarray:
    if not isinstance(value, list):
        raise ValueError(f"{label} coordinates must be a list")
    if len(value) == 0:
        return np.empty((0, width), dtype=np.float64)
    array = np.array(value, dtype=np.float64)
    if array.ndim != 2 or array.shape[1] != width:
        raise ValueError(f"{label} coordinates must have shape (n, {width})")
    return array


def _strings(value: object, *, label: str) -> list[str]:
    if not isinstance(value, list) or not all(isinstance(item, str) for item in value):
        raise ValueError(f"{label} must be a list of strings")
    return list(value)
