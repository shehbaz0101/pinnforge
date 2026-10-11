"""Validation and test runs for the warm-started viscosity fit.

``run_validation`` reads only the validation split. It times the cold
solver search and every warm-start variant in the same worker, records
each refinement trace up to ``VALIDATION_CAP`` solves, and then applies
the selection rules below. The rules are fixed in code before any number
is read, and they only see validation rows:

* solve cap: among caps whose primary-method ``hard_ood`` time ratio is at
  most ``TIME_GUARD``, the smallest cap whose five-seed mean ``hard_ood``
  relative error is within ``ERROR_SLACK`` of the best admissible cap;
* interval scale: the candidate with the smallest two-fold cross-fitted
  mean log half-width;
* abstain statistic: the candidate with the smallest two-fold cross-fitted
  mean relative error on accepted instances at the fixed abstain rate.

Calibration quantiles and abstain thresholds are then refit on the whole
validation split per method. ``run_test`` refuses to start unless the
frozen protocol, the validation record, and every checkpoint match the
hashes stored in the protocol.
"""

from __future__ import annotations

import json
import math
import os
from collections.abc import Mapping, Sequence
from concurrent.futures import ProcessPoolExecutor, as_completed
from pathlib import Path
from typing import Any

import numpy as np

from pinnforge.operator.data import load_split_trajectories
from pinnforge.operator.inverse import FAILURE_RELATIVE_ERROR
from pinnforge.operator.solver_fit import SolverSearch, build_solver_problem, fit_solver_problem
from pinnforge.operator.stage_e import apply_noise, observation_spec, standard_normal_field
from pinnforge.operator.strong_baselines import (
    file_sha256,
    load_strong_protocol,
    search_from_protocol,
)
from pinnforge.operator.warmstart import (
    RefineConfig,
    abstain_threshold,
    calibrate_groups,
    conformal_score,
    fit_warm_start,
    instance_scale,
    predicted_group,
    step_at_cap,
)
from pinnforge.operator.windows import Trajectory

VALIDATION_FORMAT = "pinnforge.warmstart_validation.v1"
PROTOCOL_FORMAT = "pinnforge.warmstart_protocol.v1"
ARMS = ("data_only", "hybrid_1e-2")
SEEDS = (0, 1, 2, 3, 4)
FAMILIES = ("full", "restricted")
CHECKPOINT_ROOTS = {"full": Path("runs/v03/train"), "restricted": Path("runs/v03/stage_f")}
STRONG_PROTOCOL = Path("docs/research/strong_baselines_protocol.json")
STAGE_D_SELECTION = Path("docs/v02/stage_d_objective_selection.json")
PRIMARY = "full/hybrid_1e-2"
CAPS = (2, 3, 4, 5, 6)
VALIDATION_CAP = 6
ALPHA = 0.1
MIN_GROUP = 10
ABSTAIN_RATE = 0.1
SCALE_FLOOR = 1e-3
TIME_GUARD = 0.20
ERROR_SLACK = 1.02
GRID_KIND = "geometric"
GRID_POINTS = 32
STAGE_D_GRID_POINTS = 191
SCALE_CANDIDATES = {"seed": ("none", "gauss_newton"), "ensemble": ("none", "gauss_newton", "ensemble_spread")}
ABSTAIN_CANDIDATES = {
    "seed": ("relative_residual", "log_standard_error", "guess_shift"),
    "ensemble": ("relative_residual", "log_standard_error", "guess_shift", "ensemble_log_std"),
}
_FORBIDDEN = frozenset({"results", "scores", "metrics", "test_mean_rel_error", "coverage", "mean_rel_error"})
_MODEL_CACHE: dict[str, Any] = {}


def seed_keys(family: str, arm: str) -> tuple[str, ...]:
    return tuple(f"{family}/{arm}/{seed}" for seed in SEEDS)


def ensemble_key(family: str, arm: str) -> str:
    return f"{family}/{arm}/ensemble"


def method_kind(key: str) -> str:
    return "ensemble" if key.endswith("/ensemble") else "seed"


def checkpoint_paths(family: str, roots: Mapping[str, Path] | None = None) -> dict[str, Path]:
    """``family/arm/seed`` to checkpoint path. Every file must exist."""

    base = Path((roots or CHECKPOINT_ROOTS)[family])
    paths = {}
    for arm in ARMS:
        for seed in SEEDS:
            path = base / arm / f"seed_{seed}" / "checkpoint.pt"
            if not path.is_file():
                raise ValueError(f"missing checkpoint {path.as_posix()}")
            paths[f"{family}/{arm}/{seed}"] = path
    return paths


def reference_spec(strong: Mapping[str, Any]) -> Any:
    stage_e = json.loads(Path(strong["stage_e_protocol"]).read_text(encoding="utf-8"))
    spec = observation_spec(stage_e, 32, 4)
    if spec.name != "sensors32_bursts":
        raise ValueError("the reference mask must stay sensors32_bursts")
    return spec


def warm_view(row: Mapping[str, Any], key: str, cap: int) -> dict[str, Any]:
    """One method on one instance as if it had stopped after ``cap`` solves.

    Seconds are the operator first guess plus the refinement. When ``cap``
    reaches the end of the trace the whole timed refinement call is used.
    """

    fit = row["warm"][key]
    step = step_at_cap(fit, cap)
    last = fit["trace"][-1]
    refine_seconds = float(fit["seconds"]) if step == dict(last) else float(step["seconds"])
    nu = float(row["nu"])
    nu_hat = float(step["nu_hat"])
    return {
        "instance_id": int(row["instance_id"]),
        "nu": nu,
        "nu_hat": nu_hat,
        "nu_guess": float(fit["nu_guess"]),
        "seconds": float(fit["first_guess_seconds"]) + refine_seconds,
        "first_guess_seconds": float(fit["first_guess_seconds"]),
        "n_evaluations": int(step["n_evaluations"]),
        "log_standard_error": float(step["log_standard_error"]),
        "relative_residual": float(step["relative_residual"]),
        "guess_shift": abs(math.log(nu_hat / float(fit["nu_guess"]))),
        "ensemble_log_std": float(fit.get("ensemble_log_std", math.nan)),
        "at_trust_bound": bool(step["at_trust_bound"]),
        "rel_error": abs(nu_hat - nu) / nu,
    }


def cold_view(row: Mapping[str, Any]) -> dict[str, Any]:
    cold = row["cold"]
    nu = float(row["nu"])
    nu_hat = float(cold["nu_hat"])
    return {
        "instance_id": int(row["instance_id"]),
        "nu": nu,
        "nu_hat": nu_hat,
        "seconds": float(cold["seconds"]),
        "n_evaluations": int(cold["n_evaluations"]),
        "rel_error": abs(nu_hat - nu) / nu,
    }


def select_cap(
    rows: Sequence[Mapping[str, Any]],
    keys: Sequence[str],
    threshold: float,
    *,
    caps: Sequence[int] = CAPS,
    guard: float = TIME_GUARD,
    slack: float = ERROR_SLACK,
) -> dict[str, Any]:
    """Solve cap from validation ``hard_ood`` rows of the primary seeds."""

    hard = [row for row in rows if float(row["nu"]) <= float(threshold)]
    if not hard:
        raise ValueError("cap selection needs validation hard_ood rows")
    cold_seconds = float(np.mean([float(row["cold"]["seconds"]) for row in hard]))
    cold_error = float(np.mean([cold_view(row)["rel_error"] for row in hard]))
    table = []
    for cap in caps:
        errors = []
        seconds = []
        for key in keys:
            views = [warm_view(row, key, int(cap)) for row in hard]
            errors.append(float(np.mean([item["rel_error"] for item in views])))
            seconds.append(float(np.mean([item["seconds"] for item in views])))
        table.append(
            {
                "cap": int(cap),
                "mean_rel_error": float(np.mean(errors)),
                "per_seed_mean_rel_error": errors,
                "mean_seconds": float(np.mean(seconds)),
                "time_ratio": float(np.mean(seconds)) / cold_seconds,
            }
        )
    admissible = [item for item in table if item["time_ratio"] <= float(guard)]
    if admissible:
        best = min(item["mean_rel_error"] for item in admissible)
        chosen = next(item for item in admissible if item["mean_rel_error"] <= float(slack) * best)
    else:
        chosen = min(table, key=lambda item: (item["time_ratio"], item["cap"]))
    return {
        "rule": (
            f"smallest cap with time_ratio <= {guard!r} whose mean hard_ood relative error is "
            f"within a factor {slack!r} of the best such cap; if none, the fastest cap"
        ),
        "n_hard_ood": len(hard),
        "cold_mean_seconds": cold_seconds,
        "cold_mean_rel_error": cold_error,
        "table": table,
        "selected_cap": int(chosen["cap"]),
    }


def interval_rows(views: Sequence[Mapping[str, Any]], kind: str, threshold: float) -> list[dict[str, Any]]:
    rows = []
    for view in views:
        scale = instance_scale(
            kind,
            log_standard_error=float(view["log_standard_error"]),
            ensemble_log_std=float(view["ensemble_log_std"]),
            floor=SCALE_FLOOR,
        )
        if not math.isfinite(scale):
            raise ValueError("interval scale is not finite")
        rows.append(
            {
                "instance_id": int(view["instance_id"]),
                "scale": scale,
                "score": conformal_score(float(view["nu"]), float(view["nu_hat"]), scale),
                "group": predicted_group(float(view["nu_hat"]), threshold),
            }
        )
    return rows


def crossfit_interval(
    views_by_key: Mapping[str, Sequence[Mapping[str, Any]]],
    kind: str,
    threshold: float,
) -> dict[str, float]:
    """Two folds by instance-id parity. Width is the mean log half-width."""

    widths: list[float] = []
    covered: list[bool] = []
    for views in views_by_key.values():
        rows = interval_rows(views, kind, threshold)
        for fold in (0, 1):
            calibration = [row for row in rows if row["instance_id"] % 2 != fold]
            held = [row for row in rows if row["instance_id"] % 2 == fold]
            fitted = calibrate_groups(calibration, ALPHA, min_group=MIN_GROUP)
            for row in held:
                quantile = float(fitted["groups"][row["group"]]["quantile"])
                widths.append(row["scale"] * quantile)
                covered.append(row["score"] <= quantile)
    return {"mean_log_half_width": float(np.mean(widths)), "coverage": float(np.mean(covered))}


def crossfit_abstain(views_by_key: Mapping[str, Sequence[Mapping[str, Any]]], statistic: str) -> dict[str, float]:
    """Accepted-case error on each fold with the threshold fit on the other."""

    errors: list[float] = []
    rates: list[float] = []
    for views in views_by_key.values():
        for fold in (0, 1):
            calibration = [float(view[statistic]) for view in views if view["instance_id"] % 2 != fold]
            held = [view for view in views if view["instance_id"] % 2 == fold]
            cut = abstain_threshold(calibration, ABSTAIN_RATE)
            accepted = [view for view in held if float(view[statistic]) <= cut]
            rates.append(1.0 - len(accepted) / len(held))
            if accepted:
                errors.append(float(np.mean([view["rel_error"] for view in accepted])))
    return {"accepted_mean_rel_error": float(np.mean(errors)), "abstain_rate": float(np.mean(rates))}


def select_family(
    rows: Sequence[Mapping[str, Any]],
    family: str,
    cap: int,
    threshold: float,
) -> dict[str, Any]:
    """Interval scale and abstain statistic per kind, then per-method calibration."""

    choices: dict[str, Any] = {}
    methods: dict[str, Any] = {}
    for kind in ("seed", "ensemble"):
        keys = [key for arm in ARMS for key in (seed_keys(family, arm) if kind == "seed" else (ensemble_key(family, arm),))]
        views_by_key = {key: [warm_view(row, key, cap) for row in rows] for key in keys}
        scale_table = {name: crossfit_interval(views_by_key, name, threshold) for name in SCALE_CANDIDATES[kind]}
        scale = min(SCALE_CANDIDATES[kind], key=lambda name: (scale_table[name]["mean_log_half_width"], SCALE_CANDIDATES[kind].index(name)))
        abstain_table = {name: crossfit_abstain(views_by_key, name) for name in ABSTAIN_CANDIDATES[kind]}
        statistic = min(
            ABSTAIN_CANDIDATES[kind],
            key=lambda name: (abstain_table[name]["accepted_mean_rel_error"], ABSTAIN_CANDIDATES[kind].index(name)),
        )
        choices[kind] = {
            "scale": scale,
            "scale_crossfit": scale_table,
            "abstain_statistic": statistic,
            "abstain_crossfit": abstain_table,
        }
        for key, views in views_by_key.items():
            calibration = calibrate_groups(interval_rows(views, scale, threshold), ALPHA, min_group=MIN_GROUP)
            methods[key] = {
                "scale": scale,
                "calibration": calibration,
                "abstain_statistic": statistic,
                "abstain_threshold": abstain_threshold([float(view[statistic]) for view in views], ABSTAIN_RATE),
                "n_validation": len(views),
            }
    return {"choices": choices, "methods": methods}


def validation_summary(rows: Sequence[Mapping[str, Any]], keys: Sequence[str], cap: int, threshold: float) -> dict[str, Any]:
    """Validation error and time per method, hard_ood and complement."""

    summary: dict[str, Any] = {}
    for key in keys:
        views = [warm_view(row, key, cap) for row in rows]
        summary[key] = {}
        for name, keep in (("hard_ood", True), ("complement", False)):
            part = [view for view in views if (view["nu"] <= threshold) == keep]
            if not part:
                continue
            summary[key][name] = {
                "n": len(part),
                "mean_rel_error": float(np.mean([view["rel_error"] for view in part])),
                "mean_seconds": float(np.mean([view["seconds"] for view in part])),
            }
    return summary


def stage_d_reproduction(rows: Sequence[Mapping[str, Any]], threshold: float) -> dict[str, Any]:
    """Retrained checkpoints on the Stage D validation selection metric."""

    published = json.loads(STAGE_D_SELECTION.read_text(encoding="utf-8"))
    hard = [row for row in rows if float(row["nu"]) <= threshold]
    report: dict[str, Any] = {"source": STAGE_D_SELECTION.as_posix(), "grid": f"linspace(0.005, 0.1, {STAGE_D_GRID_POINTS})", "arms": {}}
    for arm in ARMS:
        reference = next(item for item in published["arms"][arm]["candidates"] if float(item["lambda"]) == 0.0)
        retrained = []
        for key in seed_keys("full", arm):
            errors = [abs(float(row["stage_d_grid"][key]) - float(row["nu"])) / float(row["nu"]) for row in hard]
            retrained.append(float(np.mean(errors)))
        expected = [float(value) for value in reference["hard_ood_mean_rel_error"]]
        report["arms"][arm] = {
            "published": expected,
            "retrained": retrained,
            "max_abs_difference": float(max(abs(a - b) for a, b in zip(retrained, expected, strict=True))),
        }
    return report


def checkpoint_reproduction(paths: Mapping[str, Path]) -> dict[str, Any]:
    """Selected epoch and validation relative L2 against the published tables."""

    import torch

    sources = {
        ("full", "data_only"): (Path("docs/v02/stage_b_scores.json"), ()),
        ("full", "hybrid_1e-2"): (Path("docs/v02/stage_c_scores.json"), ("hybrid",)),
        ("restricted", "data_only"): (Path("docs/v02/stage_f_scores.json"), ("forward", "data_only")),
        ("restricted", "hybrid_1e-2"): (Path("docs/v02/stage_f_scores.json"), ("forward", "hybrid_1e-2")),
    }
    report: dict[str, Any] = {}
    for key, path in paths.items():
        family, arm, seed = key.split("/")
        source, prefix = sources[(family, arm)]
        block: Any = json.loads(source.read_text(encoding="utf-8"))
        for part in prefix:
            block = block[part]
        payload = torch.load(path, map_location="cpu", weights_only=False)
        epoch = int(payload["epoch"])
        val = float(payload["val_relative_l2"])
        published_epoch = int(block["selected_epoch"]["values"][int(seed)])
        published_val = float(block["val_relative_l2"]["values"][int(seed)])
        report[key] = {
            "sha256": file_sha256(path),
            "epoch": epoch,
            "val_relative_l2": val,
            "published_epoch": published_epoch,
            "published_val_relative_l2": published_val,
            "source": source.as_posix(),
            "matches": bool(epoch == published_epoch and val == published_val),
        }
    return report


def run_validation(
    *,
    pilot_dir: Path,
    manifest_path: Path,
    record_path: Path,
    rows_dir: Path,
    workers: int,
    families: Sequence[str] = FAMILIES,
) -> dict[str, Any]:
    """Validation split only. Rows are cached per family under ``rows_dir``."""

    _single_thread_environment()
    strong = load_strong_protocol(STRONG_PROTOCOL)
    search = search_from_protocol(strong)
    threshold = float(strong["hard_ood"]["threshold_nu"])
    spec = reference_spec(strong)
    trajectories = load_split_trajectories(pilot_dir, manifest_path, ("val",), check_field_hash=True)
    trajectories.sort(key=lambda item: item.instance_id)
    config = RefineConfig(max_evaluations=VALIDATION_CAP)
    rows_by_family: dict[str, list[dict[str, Any]]] = {}
    paths_by_family: dict[str, dict[str, Path]] = {}
    for family in families:
        paths = checkpoint_paths(family)
        paths_by_family[family] = paths
        cache = Path(rows_dir) / f"{family}.json"
        if cache.is_file():
            rows_by_family[family] = json.loads(cache.read_text(encoding="utf-8"))["rows"]
            print(f"reused {cache.as_posix()}", flush=True)
            continue
        items = trajectories if family == "full" else [item for item in trajectories if float(item.nu) > threshold]
        jobs = [
            _job(
                item,
                item.u,
                spec,
                search,
                config,
                paths,
                cold=family == "full",
                stage_d_grid=family == "full" and float(item.nu) <= threshold,
                label=f"val/{family}",
            )
            for item in items
        ]
        tag = f"val/{family}/" + json.dumps(config.to_dict(), sort_keys=True)
        rows = run_jobs(jobs, workers, partial=Path(rows_dir) / f"{family}.partial.jsonl", tag=tag)
        _write_json({"split": "val", "family": family, "rows": rows}, cache)
        rows_by_family[family] = rows
    if set(families) != set(FAMILIES):
        return {"rows": rows_by_family}
    full = rows_by_family["full"]
    cold_by_id = {int(row["instance_id"]): row["cold"] for row in full}
    restricted = [dict(row, cold=cold_by_id[int(row["instance_id"])]) for row in rows_by_family["restricted"]]
    cap = select_cap(full, seed_keys("full", "hybrid_1e-2"), threshold)
    chosen = int(cap["selected_cap"])
    all_keys = {
        family: [key for arm in ARMS for key in (*seed_keys(family, arm), ensemble_key(family, arm))] for family in FAMILIES
    }
    record = {
        "format": VALIDATION_FORMAT,
        "split": "val",
        "test_split_read": False,
        "pilot": Path(pilot_dir).as_posix(),
        "manifest": Path(manifest_path).as_posix(),
        "strong_baselines_protocol": STRONG_PROTOCOL.as_posix(),
        "strong_baselines_protocol_sha256": file_sha256(STRONG_PROTOCOL),
        "threshold_nu": threshold,
        "mask": spec.name,
        "first_guess_grid": {"kind": GRID_KIND, "n_points": GRID_POINTS, "nu_min": search.nu_min, "nu_max": search.nu_max},
        "refine_validation": config.to_dict(),
        "rules": {
            "alpha": ALPHA,
            "min_group": MIN_GROUP,
            "abstain_rate": ABSTAIN_RATE,
            "scale_floor": SCALE_FLOOR,
            "time_guard": TIME_GUARD,
            "error_slack": ERROR_SLACK,
            "caps": list(CAPS),
            "folds": "instance_id parity",
            "scale_candidates": {name: list(value) for name, value in SCALE_CANDIDATES.items()},
            "abstain_candidates": {name: list(value) for name, value in ABSTAIN_CANDIDATES.items()},
            "primary": PRIMARY,
        },
        "n_instances": {"full": len(full), "restricted": len(restricted)},
        "cap_selection": cap,
        "families": {
            "full": select_family(full, "full", chosen, threshold),
            "restricted": select_family(restricted, "restricted", chosen, threshold),
        },
        "summary": {
            "full": validation_summary(full, all_keys["full"], chosen, threshold),
            "restricted": validation_summary(restricted, all_keys["restricted"], chosen, threshold),
            "cold": {
                name: {
                    "n": len(part),
                    "mean_rel_error": float(np.mean([cold_view(row)["rel_error"] for row in part])),
                    "mean_seconds": float(np.mean([float(row["cold"]["seconds"]) for row in part])),
                }
                for name, part in (
                    ("hard_ood", [row for row in full if float(row["nu"]) <= threshold]),
                    ("complement", [row for row in full if float(row["nu"]) > threshold]),
                )
            },
        },
        "stage_d_reproduction": stage_d_reproduction(full, threshold),
        "checkpoints": checkpoint_reproduction({**paths_by_family["full"], **paths_by_family["restricted"]}),
        "instances": {
            family: _compact_instances(rows, all_keys[family], chosen)
            for family, rows in (("full", full), ("restricted", restricted))
        },
    }
    _write_json(record, record_path)
    print(f"validation record: {Path(record_path).as_posix()} cap={chosen}", flush=True)
    return record


def load_warmstart_protocol(path: Path, *, check_checkpoints: bool = True) -> dict[str, Any]:
    """Read the frozen contract. Every referenced file must match its hash.

    Checkpoints are not in git, so ``check_checkpoints=False`` skips only
    their hashes.
    """

    payload = json.loads(Path(path).read_text(encoding="utf-8"))
    if not isinstance(payload, dict) or payload.get("format") != PROTOCOL_FORMAT:
        raise ValueError(f"protocol format must be {PROTOCOL_FORMAT}")
    leaked = _FORBIDDEN & set(payload)
    if leaked:
        raise ValueError(f"protocol must not contain test scores ({', '.join(sorted(leaked))})")
    flags = {
        "frozen_before_test_evaluation": True,
        "test_used_for_selection": False,
        "hyperparameters_tuned_on_test": False,
        "hyperparameters_chosen_on_validation": True,
    }
    for name, expected in flags.items():
        if payload.get(name) is not expected:
            raise ValueError(f"protocol flag {name} must be {expected}")
    if payload.get("failure_relative_error") != FAILURE_RELATIVE_ERROR:
        raise ValueError("failure relative error must stay 0.5")
    if payload.get("device") != "cpu":
        raise ValueError("device must stay cpu")
    RefineConfig.from_mapping(payload["refine"])
    for name in ("strong_baselines_protocol", "strong_baselines_scores", "validation_record", "stage_e_protocol"):
        _require_sha(Path(payload[name]), str(payload[f"{name}_sha256"]))
    record = json.loads(Path(payload["validation_record"]).read_text(encoding="utf-8"))
    if record.get("format") != VALIDATION_FORMAT or record.get("test_split_read") is not False:
        raise ValueError("validation record must come from the validation split")
    if int(record["cap_selection"]["selected_cap"]) != int(payload["refine"]["max_evaluations"]):
        raise ValueError("protocol cap must equal the validation selection")
    if payload["methods"]["primary"] != PRIMARY or payload["pass_bars"]["method"] != PRIMARY:
        raise ValueError(f"the primary method must stay {PRIMARY}")
    if not check_checkpoints:
        return payload
    for key, digest in payload["checkpoints"].items():
        family, arm, seed = key.split("/")
        path = Path(payload["checkpoint_roots"][family]) / arm / f"seed_{seed}" / "checkpoint.pt"
        _require_sha(path, str(digest))
    return payload


def run_test_rows(
    *,
    protocol: Mapping[str, Any],
    protocol_sha256: str,
    pilot_dir: Path,
    manifest_path: Path,
    rows_dir: Path,
    workers: int,
    conditions: Sequence[str],
) -> dict[str, list[dict[str, Any]]]:
    """Per-instance test rows for each condition. Cached by protocol hash."""

    _single_thread_environment()
    strong = json.loads(Path(protocol["strong_baselines_protocol"]).read_text(encoding="utf-8"))
    search = search_from_protocol(strong)
    threshold = float(protocol["hard_ood"]["threshold_nu"])
    spec = reference_spec(strong)
    stage_e = json.loads(Path(protocol["stage_e_protocol"]).read_text(encoding="utf-8"))
    master = int(stage_e["noise"]["master_seed"])
    noise_seed = int(protocol["noise"]["noise_seed"])
    config = RefineConfig.from_mapping(protocol["refine"])
    trajectories = load_split_trajectories(pilot_dir, manifest_path, ("test",), check_field_hash=True)
    trajectories.sort(key=lambda item: item.instance_id)
    roots = {name: Path(value) for name, value in protocol["checkpoint_roots"].items()}
    full = checkpoint_paths("full", roots)
    restricted = checkpoint_paths("restricted", roots)
    output: dict[str, list[dict[str, Any]]] = {}
    for condition in conditions:
        cache = Path(rows_dir) / f"{condition}.json"
        if cache.is_file():
            cached = json.loads(cache.read_text(encoding="utf-8"))
            if cached.get("protocol_sha256") == protocol_sha256:
                output[condition] = cached["rows"]
                print(f"reused {cache.as_posix()}", flush=True)
                continue
        if condition == "clean":
            jobs = [
                _job(item, item.u, spec, search, config, {**full, **restricted}, cold=True, label="test/clean")
                for item in trajectories
            ]
        else:
            fraction = float(protocol["noise"]["fractions"][condition])
            jobs = []
            for item in trajectories:
                if float(item.nu) > threshold:
                    continue
                epsilon = standard_normal_field(master, noise_seed, int(item.instance_id), item.u.shape)
                field = apply_noise(item.u, epsilon, fraction)
                jobs.append(_job(item, field, spec, search, config, full, cold=True, label=f"test/{condition}"))
        rows = run_jobs(jobs, workers, partial=Path(rows_dir) / f"{condition}.partial.jsonl", tag=protocol_sha256)
        for row in rows:
            row["condition"] = condition
        _write_json({"protocol_sha256": protocol_sha256, "condition": condition, "rows": rows}, cache)
        output[condition] = rows
    return output


def run_smoke(
    directory: Path,
    *,
    n_space: int = 128,
    nu: float = 0.02,
    families: Sequence[str] = FAMILIES,
) -> dict[str, Any]:
    """Every worker step on one synthetic field with random-weight operators.

    The field comes from the reference solver on a coarse grid. The numbers
    say nothing about accuracy; this only checks that the pipeline runs.
    """

    import torch

    from pinnforge.operator.checkpoint import save_fno_checkpoint
    from pinnforge.operator.fno import FNO1d
    from pinnforge.operator.warmstart_fit import WINDOW
    from pinnforge.operator.windows import FieldNorm
    from pinnforge.reference.numerical.solver import solve

    _single_thread_environment()
    strong = json.loads(STRONG_PROTOCOL.read_text(encoding="utf-8"))
    spec = reference_spec(strong)
    search = SolverSearch(n=int(n_space))
    grid_x = np.linspace(0.0, 2.0 * np.pi, int(n_space), endpoint=False)
    u0 = np.sin(grid_x) + 0.5 * np.cos(2.0 * grid_x + 0.3)
    field = np.asarray(solve(u0, float(nu), dt=search.dt, t_final=0.8, save_dt=search.save_dt).u, dtype=np.float64)
    norm = FieldNorm(u_mean=0.0, u_std=float(np.std(field)), nu_mean=0.05, nu_std=0.03)
    root = Path(directory)
    roots = {family: root / family for family in FAMILIES}
    for family in families:
        for arm in ARMS:
            for seed in SEEDS:
                torch.manual_seed(1000 * FAMILIES.index(family) + 10 * ARMS.index(arm) + seed)
                model = FNO1d(9, 8, width=8, modes=4, n_layers=1)
                save_fno_checkpoint(
                    roots[family] / arm / f"seed_{seed}" / "checkpoint.pt",
                    model,
                    epoch=0,
                    spec=WINDOW,
                    norm=norm,
                    seed=seed,
                    val_relative_l2=1.0,
                )
    models: dict[str, Path] = {}
    for family in families:
        models.update(checkpoint_paths(family, roots))
    item = Trajectory(instance_id=0, split="val", nu=float(nu), u=field)
    job = _job(item, field, spec, search, RefineConfig(max_evaluations=3), models, cold=True, label="smoke", stage_d_grid=True)
    return instance_worker(job)


def run_jobs(
    jobs: Sequence[dict[str, Any]],
    workers: int,
    partial: Path | None = None,
    tag: str = "",
) -> list[dict[str, Any]]:
    """Run every job. With ``partial``, finished rows are appended there and reused.

    ``tag`` is stored with each partial row so a stale file is ignored.
    """

    done: dict[int, dict[str, Any]] = {}
    if partial is not None and Path(partial).is_file():
        for line in Path(partial).read_text(encoding="utf-8").splitlines():
            if line.strip():
                item = json.loads(line)
                if item.get("tag") == tag:
                    done[int(item["row"]["instance_id"])] = item["row"]
    pending = [job for job in jobs if int(job["instance_id"]) not in done]
    if done:
        print(f"resumed {len(done)} rows, {len(pending)} pending", flush=True)
    if partial is not None:
        Path(partial).parent.mkdir(parents=True, exist_ok=True)

    def keep(row: dict[str, Any]) -> None:
        done[int(row["instance_id"])] = row
        if partial is not None:
            with Path(partial).open("a", encoding="utf-8") as handle:
                handle.write(json.dumps(_json_ready({"tag": tag, "row": row})) + "\n")

    if int(workers) == 1:
        for job in pending:
            keep(instance_worker(job))
    elif pending:
        with ProcessPoolExecutor(max_workers=int(workers)) as pool:
            futures = [pool.submit(instance_worker, job) for job in pending]
            for future in as_completed(futures):
                keep(future.result())
    wanted = {int(job["instance_id"]) for job in jobs}
    return [_restore(done[key]) for key in sorted(wanted)]


def _restore(row: Mapping[str, Any]) -> dict[str, Any]:
    return json.loads(json.dumps(_json_ready(row)))


def instance_worker(job: Mapping[str, Any]) -> dict[str, Any]:
    """Cold search, then every warm start, on one field in one process."""

    from pinnforge.operator.warmstart_fit import (
        first_guess_grid,
        log_std,
        operator_first_guess,
        single_thread,
    )

    single_thread()
    search = SolverSearch(**job["search"])
    config = RefineConfig.from_mapping(job["refine"])
    field = np.asarray(job["field"], dtype=np.float64)
    spec = job["spec"]
    grid = first_guess_grid(GRID_KIND, GRID_POINTS, search.nu_min, search.nu_max)
    models = {key: _cached_model(path) for key, path in job["models"].items()}
    problem = build_solver_problem(field, spec, search, initial_condition="sensor_lift")
    row: dict[str, Any] = {"instance_id": int(job["instance_id"]), "nu": float(job["nu"]), "split": job["split"]}
    if job["cold"]:
        found = fit_solver_problem(problem, search, truth=None)
        row["cold"] = {
            "nu_hat": float(found["nu_hat"]),
            "objective": float(found["objective"]),
            "n_evaluations": int(found["n_evaluations"]),
            "seconds": float(found["seconds"]),
        }
    guesses: dict[str, Any] = {}
    warm: dict[str, Any] = {}
    for key, loaded in models.items():
        guess = operator_first_guess(loaded.model, loaded.norm, field, spec, grid)
        guesses[key] = guess
        fit = fit_warm_start(problem, search, float(guess["nu_hat"]), config)
        fit["first_guess_seconds"] = float(guess["seconds"])
        warm[key] = fit
    families = sorted({key.split("/")[0] for key in models})
    for family in families:
        for arm in ARMS:
            keys = seed_keys(family, arm)
            if not all(key in guesses for key in keys):
                continue
            values = [float(guesses[key]["nu_hat"]) for key in keys]
            center = float(np.exp(np.median(np.log(values))))
            fit = fit_warm_start(problem, search, center, config)
            fit["first_guess_seconds"] = float(sum(float(guesses[key]["seconds"]) for key in keys))
            fit["ensemble_log_std"] = log_std(values)
            fit["ensemble_guesses"] = values
            warm[ensemble_key(family, arm)] = fit
    row["guesses"] = guesses
    row["warm"] = warm
    if job.get("stage_d_grid"):
        linear = first_guess_grid("linear", STAGE_D_GRID_POINTS, search.nu_min, search.nu_max)
        row["stage_d_grid"] = {
            key: float(operator_first_guess(loaded.model, loaded.norm, field, spec, linear)["nu_hat"])
            for key, loaded in models.items()
            if key.startswith("full/")
        }
    cold_text = f" cold={row['cold']['seconds']:.2f}s" if "cold" in row else ""
    primary = warm.get(f"{PRIMARY}/0")
    warm_text = f" warm0={primary['seconds'] + primary['first_guess_seconds']:.2f}s" if primary else ""
    print(f"{job['label']} id={row['instance_id']}{cold_text}{warm_text}", flush=True)
    return row


def _job(
    item: Trajectory,
    field: np.ndarray,
    spec: Any,
    search: SolverSearch,
    config: RefineConfig,
    models: Mapping[str, Path],
    *,
    cold: bool,
    label: str,
    stage_d_grid: bool = False,
) -> dict[str, Any]:
    return {
        "instance_id": int(item.instance_id),
        "nu": float(item.nu),
        "split": str(item.split),
        "field": np.asarray(field, dtype=np.float64),
        "spec": spec,
        "search": _search_dict(search),
        "refine": config.to_dict(),
        "models": {key: str(path) for key, path in models.items()},
        "cold": bool(cold),
        "stage_d_grid": bool(stage_d_grid),
        "label": label,
    }


def _search_dict(search: SolverSearch) -> dict[str, Any]:
    return {
        "nu_min": search.nu_min,
        "nu_max": search.nu_max,
        "n_coarse": search.n_coarse,
        "refine_width": search.refine_width,
        "max_refine": search.max_refine,
        "n": search.n,
        "dt": search.dt,
        "save_dt": search.save_dt,
    }


def _cached_model(path: str) -> Any:
    from pinnforge.operator.checkpoint import load_fno_checkpoint
    from pinnforge.operator.warmstart_fit import WINDOW

    loaded = _MODEL_CACHE.get(path)
    if loaded is None:
        loaded = load_fno_checkpoint(Path(path))
        if loaded.spec != WINDOW:
            raise ValueError(f"{path} window does not match the Stage D window")
        loaded.model.eval()
        _MODEL_CACHE[path] = loaded
    return loaded


def _compact_instances(rows: Sequence[Mapping[str, Any]], keys: Sequence[str], cap: int) -> dict[str, Any]:
    payload: dict[str, Any] = {
        "instance_id": [int(row["instance_id"]) for row in rows],
        "nu": [float(row["nu"]) for row in rows],
        "cold_nu_hat": [float(row["cold"]["nu_hat"]) for row in rows],
        "cold_seconds": [float(row["cold"]["seconds"]) for row in rows],
        "methods": {},
    }
    for key in keys:
        views = [warm_view(row, key, cap) for row in rows]
        payload["methods"][key] = {
            field: [view[field] for view in views]
            for field in ("nu_guess", "nu_hat", "seconds", "log_standard_error", "relative_residual", "ensemble_log_std")
            if not (field == "ensemble_log_std" and method_kind(key) == "seed")
        }
    return payload


def _require_sha(path: Path, expected: str) -> None:
    actual = file_sha256(path)
    if actual != expected:
        raise ValueError(f"{Path(path).as_posix()} sha256 {actual} does not match {expected}")


def _json_ready(value: Any) -> Any:
    if isinstance(value, dict):
        return {str(key): _json_ready(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [_json_ready(item) for item in value]
    if isinstance(value, np.ndarray):
        return _json_ready(value.tolist())
    if isinstance(value, np.floating):
        value = float(value)
    if isinstance(value, np.integer):
        return int(value)
    if isinstance(value, float) and not math.isfinite(value):
        return None if math.isnan(value) else ("inf" if value > 0 else "-inf")
    return value


def _write_json(payload: Mapping[str, Any], path: Path) -> None:
    destination = Path(path)
    destination.parent.mkdir(parents=True, exist_ok=True)
    destination.write_text(json.dumps(_json_ready(payload), indent=1) + "\n", encoding="utf-8")


def _single_thread_environment() -> None:
    for name in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS"):
        os.environ[name] = "1"
