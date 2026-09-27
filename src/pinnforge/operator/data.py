"""Load a Stage 2 pilot directory into normalized windows.

The split and the ``u`` mean and standard deviation come from the manifest
passed in, which for the reported run is
``docs/stage2/pilot_manifest.json``. A ``manifest.json`` sitting in the
pilot directory is checked against that file when it exists. Field bytes
are checked with the manifest's ``field_sha256``, which does not depend on
zip timestamps.
"""

from __future__ import annotations

import math
from collections.abc import Mapping
from pathlib import Path

import numpy as np

from pinnforge.operator.windows import (
    SPLIT_NAMES,
    FieldNorm,
    Trajectory,
    WindowDataset,
    WindowSpec,
    build_window_datasets,
    field_norm_from_manifest,
    load_pilot_manifest,
    split_sets,
)
from pinnforge.reference.numerical.dataset import _field_sha256

_NORM_ABS = 1e-12
_NU_ABS = 1e-12


def load_split_trajectories(
    pilot_dir: Path,
    manifest_path: Path,
    splits: tuple[str, ...] = SPLIT_NAMES,
    *,
    check_field_hash: bool = True,
) -> list[Trajectory]:
    """Load raw trajectories for the requested splits.

    ``splits`` defaults to train, validation, and test. Pass ``("test",)``
    to leave the train and validation files unread. Every loaded instance
    must belong to the manifest split named in the request. The arrays are
    the stored fields. They are not windowed and not normalized.
    """

    requested = _requested_splits(splits)
    manifest = load_pilot_manifest(manifest_path)
    assignment = split_sets(manifest)
    norm = field_norm_from_manifest(manifest)
    root = Path(pilot_dir)
    if not root.is_dir():
        raise ValueError(f"pilot directory does not exist: {root}")
    _check_local_manifest(root, assignment, norm)
    records = _records_by_id(manifest)
    trajectories: list[Trajectory] = []
    for name in requested:
        for instance_id in sorted(assignment[name]):
            trajectories.append(
                _load_trajectory(
                    root,
                    records[instance_id],
                    expected_split=name,
                    check_field_hash=check_field_hash,
                )
            )
    return trajectories


def load_split_windows(
    pilot_dir: Path,
    manifest_path: Path,
    spec: WindowSpec,
    splits: tuple[str, ...] = SPLIT_NAMES,
    *,
    check_field_hash: bool = True,
) -> dict[str, WindowDataset]:
    """Cut windows for the requested splits.

    ``splits`` defaults to train, validation, and test. Training should
    pass ``("train", "val")`` so test files are not read. Every loaded
    instance must belong to the manifest split, and the windows inherit
    that split.
    """

    if not isinstance(spec, WindowSpec):
        raise TypeError("spec must be a WindowSpec")
    requested = _requested_splits(splits)
    trajectories = load_split_trajectories(
        pilot_dir,
        manifest_path,
        requested,
        check_field_hash=check_field_hash,
    )
    norm = field_norm_from_manifest(load_pilot_manifest(manifest_path))
    datasets = build_window_datasets(trajectories, spec, norm)
    missing = [name for name in requested if name not in datasets]
    if missing:
        raise ValueError(f"no windows were cut for {missing}")
    return {name: datasets[name] for name in requested}


def _requested_splits(splits: tuple[str, ...]) -> tuple[str, ...]:
    if len(splits) < 1:
        raise ValueError("at least one split is required")
    seen: list[str] = []
    for name in splits:
        if name not in SPLIT_NAMES:
            raise ValueError(f"unknown split {name!r}")
        if name in seen:
            raise ValueError(f"split {name!r} was requested twice")
        seen.append(name)
    return tuple(seen)


def _records_by_id(manifest: Mapping[str, object]) -> dict[int, Mapping[str, object]]:
    raw = manifest.get("instances")
    if not isinstance(raw, list):
        raise ValueError("pilot manifest is missing instances")
    records: dict[int, Mapping[str, object]] = {}
    for item in raw:
        if not isinstance(item, Mapping):
            raise ValueError("pilot instance records must be objects")
        instance_id = item.get("instance_id")
        if isinstance(instance_id, bool) or not isinstance(instance_id, int):
            raise ValueError("instance_id must be an integer")
        records[int(instance_id)] = item
    return records


def _load_trajectory(
    root: Path,
    record: Mapping[str, object],
    *,
    expected_split: str,
    check_field_hash: bool,
) -> Trajectory:
    instance_id = int(record["instance_id"])  # type: ignore[arg-type]
    split = record.get("split")
    if split != expected_split:
        raise ValueError(f"instance {instance_id} is {split!r}, not {expected_split!r}")
    relative = record.get("path")
    if not isinstance(relative, str) or relative == "" or Path(relative).is_absolute():
        raise ValueError(f"instance {instance_id} path must be a relative string")
    path = root / relative
    if path.parent.name != expected_split:
        raise ValueError(f"instance {instance_id} file is not under {expected_split}/")
    if not path.is_file():
        raise ValueError(
            f"missing {path}. Regenerate the pilot with "
            "`python -m pinnforge.reference.numerical pilot --output artifacts/burgers_pilot`."
        )
    with np.load(path) as archive:
        field = np.array(archive["u"], dtype=np.float64, copy=True)
        nu = float(archive["nu"])
        stored_id = int(archive["instance_id"])
    if stored_id != instance_id:
        raise ValueError(f"{path} has instance_id {stored_id}, manifest says {instance_id}")
    if not math.isfinite(nu) or abs(nu - float(record["nu"])) > _NU_ABS:  # type: ignore[arg-type]
        raise ValueError(f"instance {instance_id} viscosity does not match the manifest")
    if field.ndim != 2:
        raise ValueError(f"instance {instance_id} field must have shape (n_times, n_space)")
    if int(record["n_times"]) != field.shape[0] or int(record["n"]) != field.shape[1]:  # type: ignore[arg-type]
        raise ValueError(f"instance {instance_id} field shape does not match the manifest")
    if check_field_hash:
        digest = record.get("field_sha256")
        if not isinstance(digest, str) or digest == "":
            raise ValueError(f"instance {instance_id} is missing field_sha256")
        actual = _field_sha256(field)
        if actual != digest:
            raise ValueError(f"instance {instance_id} field_sha256 does not match the manifest")
    return Trajectory(instance_id=instance_id, split=expected_split, nu=nu, u=field)


def _check_local_manifest(root: Path, assignment: Mapping[str, set[int]], norm: FieldNorm) -> None:
    local_path = root / "manifest.json"
    if not local_path.is_file():
        return
    local = load_pilot_manifest(local_path)
    local_splits = split_sets(local)
    for name in SPLIT_NAMES:
        if local_splits[name] != assignment[name]:
            raise ValueError(f"pilot directory {name} split does not match the committed manifest")
    local_norm = local.get("normalization")
    if not isinstance(local_norm, Mapping):
        raise ValueError("pilot directory manifest is missing normalization")
    for label, expected in (("u_mean", norm.u_mean), ("u_std", norm.u_std)):
        got = float(local_norm[label])  # type: ignore[arg-type]
        if abs(got - expected) > _NORM_ABS:
            raise ValueError(
                f"pilot directory {label} ({got}) does not match the committed manifest ({expected})"
            )
