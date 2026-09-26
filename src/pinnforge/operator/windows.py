"""Temporal windows from Stage 2 Burgers trajectories.

The split unit is the problem instance. Windows are cut only after an
instance is assigned to train, validation, or test, and every frame in a
window comes from that one trajectory. ``u`` is normalized with the
training mean and population standard deviation recorded in the pilot
manifest. Those two numbers are not refit on the arrays being loaded.

Viscosity is constant on an instance. Its training mean and population
standard deviation (``ddof = 0``) are computed from the training instance
records in the same manifest. Validation and test viscosities use that
affine map and do not enter the statistics.

The default window is 8 input frames and 8 target frames with stride 8.
On the pilot grid that is a lead time of 0.08 (``save_dt = 0.01``). Input
windows on one trajectory do not overlap. The target block of one pair is
the input block of the next pair; that reuse stays inside the instance.
"""

from __future__ import annotations

import json
import math
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from pathlib import Path

import numpy as np

DEFAULT_INPUT_FRAMES = 8
DEFAULT_OUTPUT_FRAMES = 8
DEFAULT_STRIDE = 8

PILOT_FORMAT = "pinnforge.burgers_pilot.v1"
SPLIT_NAMES = ("train", "val", "test")


@dataclass(frozen=True, slots=True)
class WindowSpec:
    """How a trajectory is cut into supervised pairs.

    ``input_frames`` are the history. ``output_frames`` are the following
    frames, which the operator predicts. ``stride`` is the step between
    window starts, in saved frames.
    """

    input_frames: int = DEFAULT_INPUT_FRAMES
    output_frames: int = DEFAULT_OUTPUT_FRAMES
    stride: int = DEFAULT_STRIDE

    def __post_init__(self) -> None:
        _require_positive_int(self.input_frames, "input_frames")
        _require_positive_int(self.output_frames, "output_frames")
        _require_positive_int(self.stride, "stride")

    def span(self) -> int:
        """Input length plus target length, in saved frames."""

        return self.input_frames + self.output_frames


@dataclass(frozen=True, slots=True)
class FieldNorm:
    """Affine maps for ``u`` and for viscosity.

    ``u_mean`` and ``u_std`` come from the pilot manifest. ``nu_mean`` and
    ``nu_std`` are the training-split viscosity statistics. ``u_std`` and
    ``nu_std`` are population standard deviations and must be positive.
    """

    u_mean: float
    u_std: float
    nu_mean: float
    nu_std: float

    def __post_init__(self) -> None:
        for label, value in (
            ("u_mean", self.u_mean),
            ("u_std", self.u_std),
            ("nu_mean", self.nu_mean),
            ("nu_std", self.nu_std),
        ):
            _require_finite(value, label)
        if self.u_std <= 0.0:
            raise ValueError("u_std must be > 0")
        if self.nu_std <= 0.0:
            raise ValueError("nu_std must be > 0")

    def normalize_u(self, values: np.ndarray) -> np.ndarray:
        """Map raw ``u`` to the normalized space the network trains in."""

        array = np.asarray(values, dtype=np.float64)
        return (array - self.u_mean) / self.u_std

    def denormalize_u(self, values: np.ndarray) -> np.ndarray:
        """Map normalized ``u`` back to the stored field units."""

        array = np.asarray(values, dtype=np.float64)
        return array * self.u_std + self.u_mean

    def normalize_nu(self, values: np.ndarray | float) -> np.ndarray:
        """Map viscosity with the training-split mean and standard deviation."""

        array = np.asarray(values, dtype=np.float64)
        return (array - self.nu_mean) / self.nu_std

    def to_dict(self) -> dict[str, float]:
        return {
            "u_mean": self.u_mean,
            "u_std": self.u_std,
            "nu_mean": self.nu_mean,
            "nu_std": self.nu_std,
        }


@dataclass(frozen=True, slots=True)
class Trajectory:
    """One pilot trajectory before windowing."""

    instance_id: int
    split: str
    nu: float
    u: np.ndarray

    def __post_init__(self) -> None:
        _require_instance_id(self.instance_id)
        if self.split not in SPLIT_NAMES:
            raise ValueError("split must be train, val, or test")
        _require_finite(self.nu, "nu")
        if self.nu <= 0.0:
            raise ValueError("nu must be > 0")
        field = np.asarray(self.u, dtype=np.float64)
        if field.ndim != 2 or field.shape[0] < 1 or field.shape[1] < 1:
            raise ValueError("u must have shape (n_times, n_space) with both dimensions >= 1")
        if not np.isfinite(field).all():
            raise ValueError("u must be finite")
        object.__setattr__(self, "u", field)


@dataclass(frozen=True, slots=True)
class WindowDataset:
    """Normalized windows for one split.

    ``inputs`` has shape ``(n_windows, input_frames, n_space)``.
    ``targets`` has shape ``(n_windows, output_frames, n_space)``.
    ``nu``, ``instance_ids``, and ``starts`` have shape ``(n_windows,)``.
    ``starts`` is the first saved-frame index of the input window.
    """

    spec: WindowSpec
    norm: FieldNorm
    split: str
    inputs: np.ndarray
    targets: np.ndarray
    nu: np.ndarray
    instance_ids: np.ndarray
    starts: np.ndarray

    def __post_init__(self) -> None:
        if not isinstance(self.spec, WindowSpec):
            raise TypeError("spec must be a WindowSpec")
        if not isinstance(self.norm, FieldNorm):
            raise TypeError("norm must be a FieldNorm")
        if self.split not in SPLIT_NAMES:
            raise ValueError("split must be train, val, or test")
        inputs = np.asarray(self.inputs, dtype=np.float64)
        targets = np.asarray(self.targets, dtype=np.float64)
        nu = np.asarray(self.nu, dtype=np.float64)
        instance_ids = np.asarray(self.instance_ids, dtype=np.int64)
        starts = np.asarray(self.starts, dtype=np.int64)
        if inputs.ndim != 3 or targets.ndim != 3:
            raise ValueError("inputs and targets must have shape (n_windows, frames, n_space)")
        count = inputs.shape[0]
        if targets.shape[0] != count or nu.shape != (count,) or instance_ids.shape != (count,):
            raise ValueError("window arrays must share n_windows")
        if starts.shape != (count,):
            raise ValueError("window arrays must share n_windows")
        if inputs.shape[1] != self.spec.input_frames:
            raise ValueError("input frame count does not match the window spec")
        if targets.shape[1] != self.spec.output_frames:
            raise ValueError("output frame count does not match the window spec")
        if inputs.shape[2] != targets.shape[2] or inputs.shape[2] < 1:
            raise ValueError("input and target spatial sizes must match and be >= 1")
        if count < 1:
            raise ValueError("a split must contain at least one window")
        if not np.isfinite(inputs).all() or not np.isfinite(targets).all() or not np.isfinite(nu).all():
            raise ValueError("normalized windows must be finite")
        object.__setattr__(self, "inputs", inputs)
        object.__setattr__(self, "targets", targets)
        object.__setattr__(self, "nu", nu)
        object.__setattr__(self, "instance_ids", instance_ids)
        object.__setattr__(self, "starts", starts)

    def n_windows(self) -> int:
        return int(self.inputs.shape[0])

    def n_instances(self) -> int:
        return len(set(int(value) for value in self.instance_ids))

    def n_space(self) -> int:
        return int(self.inputs.shape[2])

    def instance_id_set(self) -> set[int]:
        return {int(value) for value in self.instance_ids}


def window_starts(n_times: int, spec: WindowSpec) -> tuple[int, ...]:
    """Frame indexes where a full input-plus-target window fits.

    The last start satisfies ``start + input_frames + output_frames <= n_times``.
    """

    _require_positive_int(n_times, "n_times")
    if not isinstance(spec, WindowSpec):
        raise TypeError("spec must be a WindowSpec")
    span = spec.span()
    if n_times < span:
        return ()
    last = n_times - span
    return tuple(range(0, last + 1, spec.stride))


def cut_trajectory(
    u: np.ndarray,
    spec: WindowSpec,
) -> tuple[np.ndarray, np.ndarray, tuple[int, ...]]:
    """Cut raw input and target windows from one field.

    ``u`` has shape ``(n_times, n_space)``. The returned arrays keep that
    field's values. They are not normalized. Raises if the trajectory is
    shorter than one window.
    """

    if not isinstance(spec, WindowSpec):
        raise TypeError("spec must be a WindowSpec")
    field = np.asarray(u, dtype=np.float64)
    if field.ndim != 2:
        raise ValueError("u must have shape (n_times, n_space)")
    starts = window_starts(int(field.shape[0]), spec)
    if not starts:
        raise ValueError(
            f"trajectory length {field.shape[0]} is shorter than one window of span {spec.span()}"
        )
    inputs = np.stack([field[start : start + spec.input_frames] for start in starts], axis=0)
    targets = np.stack(
        [field[start + spec.input_frames : start + spec.span()] for start in starts],
        axis=0,
    )
    return inputs, targets, starts


def build_window_datasets(
    trajectories: Sequence[Trajectory],
    spec: WindowSpec,
    norm: FieldNorm,
) -> dict[str, WindowDataset]:
    """Cut and normalize trajectories that are already labeled by split.

    Each instance id may appear once. Two splits that share an id raise,
    and so does a repeated id inside one split. Windows are grouped by the
    trajectory's split. The result has one dataset per split that was
    provided, in ``train``, ``val``, ``test`` order.
    """

    if not isinstance(spec, WindowSpec):
        raise TypeError("spec must be a WindowSpec")
    if not isinstance(norm, FieldNorm):
        raise TypeError("norm must be a FieldNorm")
    if len(trajectories) < 1:
        raise ValueError("at least one trajectory is required")
    seen: dict[int, str] = {}
    grouped: dict[str, list[Trajectory]] = {name: [] for name in SPLIT_NAMES}
    for item in trajectories:
        if not isinstance(item, Trajectory):
            raise TypeError("trajectories must be Trajectory values")
        previous = seen.get(item.instance_id)
        if previous is not None:
            raise ValueError(
                f"instance {item.instance_id} is listed more than once (splits {previous} and {item.split})"
            )
        seen[item.instance_id] = item.split
        grouped[item.split].append(item)
    datasets: dict[str, WindowDataset] = {}
    for name in SPLIT_NAMES:
        items = grouped[name]
        if not items:
            continue
        datasets[name] = _dataset_from_trajectories(items, name, spec, norm)
    _assert_split_disjoint(datasets)
    return datasets


def assert_split_integrity(
    datasets: Mapping[str, WindowDataset],
    splits: Mapping[str, set[int]],
) -> None:
    """Raise if any window's instance is missing from its split or shared.

    ``splits`` is the instance-id assignment, usually from the pilot
    manifest. A dataset may cover a subset of its split. It may not cover
    an id from another split.
    """

    known = {name: set(ids) for name, ids in splits.items()}
    for name in known:
        if name not in SPLIT_NAMES:
            raise ValueError(f"unknown split {name!r}")
    overlap: set[int] = set()
    owned: set[int] = set()
    for name, ids in known.items():
        if owned & ids:
            overlap |= owned & ids
        owned |= ids
    if overlap:
        raise ValueError(f"manifest splits share instance ids {sorted(overlap)[:5]}")
    for name, dataset in datasets.items():
        if name not in known:
            raise ValueError(f"dataset split {name!r} is not in the manifest")
        if dataset.split != name:
            raise ValueError(f"dataset key {name!r} does not match its split label")
        foreign = dataset.instance_id_set() - known[name]
        if foreign:
            sample = sorted(foreign)[:5]
            raise ValueError(f"{name} windows include instances outside that split: {sample}")
        others: set[int] = set()
        for other, ids in known.items():
            if other != name:
                others |= ids
        leaked = dataset.instance_id_set() & others
        if leaked:
            raise ValueError(f"{name} windows leak instances {sorted(leaked)[:5]}")


def load_pilot_manifest(path: Path) -> dict[str, object]:
    """Read a Stage 2 pilot manifest.

    The file is the committed ``docs/stage2/pilot_manifest.json`` or the
    copy written next to a regenerated pilot. The format tag must be
    ``pinnforge.burgers_pilot.v1``.
    """

    source = Path(path)
    payload = json.loads(source.read_text(encoding="utf-8"))
    if not isinstance(payload, dict):
        raise ValueError("pilot manifest must be a JSON object")
    if payload.get("format") != PILOT_FORMAT:
        raise ValueError(f"unsupported pilot manifest format {payload.get('format')!r}")
    return payload


def split_sets(manifest: Mapping[str, object]) -> dict[str, set[int]]:
    """Instance ids for train, validation, and test.

    The three sets are disjoint. Ids inside one list are unique. The
    committed manifest stores each list sorted; this function returns sets,
    so display order does not matter.
    """

    raw = manifest.get("splits")
    if not isinstance(raw, Mapping):
        raise ValueError("pilot manifest is missing splits")
    sets: dict[str, set[int]] = {}
    for name in SPLIT_NAMES:
        if name not in raw:
            raise ValueError(f"pilot manifest splits are missing {name}")
        values = raw[name]
        if not isinstance(values, list) or len(values) < 1:
            raise ValueError(f"{name} split must be a non-empty list")
        parsed: list[int] = []
        for item in values:
            if isinstance(item, bool) or not isinstance(item, int) or item < 0:
                raise ValueError(f"{name} split ids must be non-negative integers")
            parsed.append(item)
        unique = set(parsed)
        if len(unique) != len(parsed):
            raise ValueError(f"{name} split has duplicate instance ids")
        sets[name] = unique
    _assert_sets_disjoint(sets)
    return sets


def field_norm_from_manifest(manifest: Mapping[str, object]) -> FieldNorm:
    """Normalization recorded for ``u``, plus training-split viscosity stats.

    ``u`` statistics are the manifest values. They are not recomputed from
    files. Viscosity statistics are computed here from training records so
    the conditioning channel uses the same split and no validation or test
    viscosity.
    """

    raw = manifest.get("normalization")
    if not isinstance(raw, Mapping):
        raise ValueError("pilot manifest is missing normalization")
    if raw.get("fit_on") != "train":
        raise ValueError("u normalization must be fit on the training split")
    if raw.get("applied_to_files") is not False:
        raise ValueError("pilot files must store raw u; normalization is applied at load")
    u_mean = _require_finite(raw.get("u_mean"), "u_mean")
    u_std = _require_finite(raw.get("u_std"), "u_std")
    nu_mean, nu_std = viscosity_train_stats(manifest)
    return FieldNorm(u_mean=u_mean, u_std=u_std, nu_mean=nu_mean, nu_std=nu_std)


def viscosity_train_stats(manifest: Mapping[str, object]) -> tuple[float, float]:
    """Population mean and standard deviation of training-split viscosity.

    Reads ``instances[].nu`` for records whose split is ``train``. The
    instance ids must match ``splits`` exactly, so a test viscosity cannot
    be included by mis-labeling a record.
    """

    splits = split_sets(manifest)
    records = _instance_records(manifest, splits)
    values = [float(records[instance_id]["nu"]) for instance_id in sorted(splits["train"])]
    array = np.asarray(values, dtype=np.float64)
    if not np.isfinite(array).all():
        raise ValueError("training viscosities must be finite")
    mean = float(np.mean(array))
    std = float(np.std(array, ddof=0))
    if not math.isfinite(std) or std <= 0.0:
        raise ValueError("training viscosity standard deviation must be > 0")
    return mean, std


def _dataset_from_trajectories(
    items: Sequence[Trajectory],
    split: str,
    spec: WindowSpec,
    norm: FieldNorm,
) -> WindowDataset:
    ordered = sorted(items, key=lambda item: item.instance_id)
    input_blocks: list[np.ndarray] = []
    target_blocks: list[np.ndarray] = []
    nu_blocks: list[np.ndarray] = []
    id_blocks: list[np.ndarray] = []
    start_blocks: list[np.ndarray] = []
    n_space: int | None = None
    for item in ordered:
        width = int(item.u.shape[1])
        if n_space is not None and width != n_space:
            raise ValueError("trajectories in one split must share n_space")
        n_space = width
        inputs, targets, starts = cut_trajectory(item.u, spec)
        count = inputs.shape[0]
        input_blocks.append(norm.normalize_u(inputs))
        target_blocks.append(norm.normalize_u(targets))
        nu_blocks.append(np.full(count, float(norm.normalize_nu(item.nu)), dtype=np.float64))
        id_blocks.append(np.full(count, item.instance_id, dtype=np.int64))
        start_blocks.append(np.asarray(starts, dtype=np.int64))
    return WindowDataset(
        spec=spec,
        norm=norm,
        split=split,
        inputs=np.concatenate(input_blocks, axis=0),
        targets=np.concatenate(target_blocks, axis=0),
        nu=np.concatenate(nu_blocks, axis=0),
        instance_ids=np.concatenate(id_blocks, axis=0),
        starts=np.concatenate(start_blocks, axis=0),
    )


def _instance_records(
    manifest: Mapping[str, object],
    splits: Mapping[str, set[int]],
) -> dict[int, Mapping[str, object]]:
    raw = manifest.get("instances")
    if not isinstance(raw, list) or len(raw) < 1:
        raise ValueError("pilot manifest is missing instances")
    records: dict[int, Mapping[str, object]] = {}
    expected = set().union(*(splits[name] for name in SPLIT_NAMES))
    for item in raw:
        if not isinstance(item, Mapping):
            raise ValueError("pilot instance records must be objects")
        instance_id = item.get("instance_id")
        split = item.get("split")
        if isinstance(instance_id, bool) or not isinstance(instance_id, int):
            raise ValueError("instance_id must be an integer")
        if split not in SPLIT_NAMES:
            raise ValueError(f"instance {instance_id} has split {split!r}")
        if instance_id not in splits[split]:
            raise ValueError(f"instance {instance_id} is labeled {split} but is not in that split list")
        if instance_id in records:
            raise ValueError(f"duplicate instance record {instance_id}")
        nu = item.get("nu")
        _require_finite(nu, f"instance {instance_id} nu")
        if float(nu) <= 0.0:
            raise ValueError(f"instance {instance_id} nu must be > 0")
        records[instance_id] = item
    found = set(records)
    if found != expected:
        missing = sorted(expected - found)[:5]
        extra = sorted(found - expected)[:5]
        raise ValueError(f"instance records do not match split lists (missing {missing}, extra {extra})")
    return records


def _assert_split_disjoint(datasets: Mapping[str, WindowDataset]) -> None:
    owned: set[int] = set()
    for name, dataset in datasets.items():
        ids = dataset.instance_id_set()
        shared = owned & ids
        if shared:
            raise ValueError(f"windows leak instance ids {sorted(shared)[:5]} into {name}")
        owned |= ids


def _assert_sets_disjoint(sets: Mapping[str, set[int]]) -> None:
    owned: set[int] = set()
    for name, ids in sets.items():
        shared = owned & ids
        if shared:
            raise ValueError(f"splits share instance ids {sorted(shared)[:5]} at {name}")
        owned |= ids


def _require_positive_int(value: object, label: str) -> int:
    if isinstance(value, bool) or not isinstance(value, (int, np.integer)):
        raise ValueError(f"{label} must be an integer >= 1")
    number = int(value)
    if number < 1:
        raise ValueError(f"{label} must be an integer >= 1")
    return number


def _require_instance_id(value: object) -> int:
    if isinstance(value, bool) or not isinstance(value, (int, np.integer)):
        raise ValueError("instance_id must be a non-negative integer")
    number = int(value)
    if number < 0:
        raise ValueError("instance_id must be a non-negative integer")
    return number


def _require_finite(value: object, label: str) -> float:
    if isinstance(value, bool) or not isinstance(value, (int, float, np.floating)):
        raise ValueError(f"{label} must be a finite number")
    number = float(value)
    if not math.isfinite(number):
        raise ValueError(f"{label} must be a finite number")
    return number
