"""Score weak-form and solver-in-the-loop viscosity baselines.

The contract is ``docs/research/strong_baselines_protocol.json``. It is
loaded and hashed before any test curve is reduced. This module does not
train, and it does not edit Stage D or Stage E scores. Fourier neural
operator numbers are aggregated from the committed Stage D per-instance
estimates. A negative comparison for those operators is reported as
measured.
"""

from __future__ import annotations

import hashlib
import json
import math
import os
import time
from collections.abc import Mapping, Sequence
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path
from typing import Any

import numpy as np

from pinnforge.operator.data import load_split_trajectories
from pinnforge.operator.inverse import (
    FAILURE_RELATIVE_ERROR,
    InstanceRecovery,
    RecoveryScore,
    least_squares_viscosity,
    recovery_from_rows,
    training_baseline_nu,
)
from pinnforge.operator.slices import hard_ood_threshold
from pinnforge.operator.solver_fit import (
    SolverSearch,
    build_solver_problem,
    fit_solver_problem,
)
from pinnforge.operator.stage_d import mean_std, slice_masks
from pinnforge.operator.stage_e import (
    apply_noise,
    observation_spec,
    standard_normal_field,
)
from pinnforge.operator.weak_form import WeakFormConfig, weak_form_viscosity
from pinnforge.operator.windows import Trajectory

SCORES_FORMAT = "pinnforge.strong_baselines_scores.v1"
PROTOCOL_FORMAT = "pinnforge.strong_baselines_protocol.v1"
FNO_ARMS = ("data_only", "hybrid_1e-2")
PRIMARY_METHODS = ("closed_form_ls", "weak_form", "solver_sensor_ic", "solver_oracle_ic")
SLICE_NAMES = ("full_test", "hard_ood", "complement", "below_stage2_floor")
_FORBIDDEN = frozenset(
    {
        "results",
        "scores",
        "metrics",
        "nu_hat",
        "mean_rel_error",
        "n_failures",
        "test_mean_rel_error",
    }
)


def file_sha256(path: Path) -> str:
    """SHA-256 of the file bytes."""

    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def load_strong_protocol(path: Path) -> dict[str, Any]:
    """Read the research contract and reject a file that already holds scores."""

    payload = json.loads(Path(path).read_text(encoding="utf-8"))
    if not isinstance(payload, dict):
        raise ValueError("strong-baseline protocol must be a JSON object")
    if payload.get("format") != PROTOCOL_FORMAT:
        raise ValueError(f"protocol format must be {PROTOCOL_FORMAT}")
    leaked = _FORBIDDEN & set(payload)
    if leaked:
        names = ", ".join(sorted(leaked))
        raise ValueError(f"protocol must not contain test scores ({names})")
    if payload.get("frozen_before_test_evaluation") is not True:
        raise ValueError("protocol must be frozen before test evaluation")
    if payload.get("test_used_for_selection") is not False:
        raise ValueError("protocol must not select on the test split")
    if payload.get("hyperparameters_tuned_on_test") is not False:
        raise ValueError("protocol must not tune hyperparameters on test")
    if payload.get("hyperparameters_tuned_on_validation") is not False:
        raise ValueError("protocol must not tune hyperparameters on validation")
    if payload.get("failure_relative_error") != FAILURE_RELATIVE_ERROR:
        raise ValueError("failure relative error must stay 0.5")
    if payload.get("failure_rule") != "nu_hat <= 0 or rel_error > 0.5":
        raise ValueError("failure rule must stay the Stage 5 rule")
    if payload.get("device") != "cpu":
        raise ValueError("device must stay cpu")
    weak = payload.get("weak_form")
    if not isinstance(weak, dict):
        raise ValueError("protocol is missing weak_form")
    WeakFormConfig(
        degree=int(weak["degree"]),
        radius_in_spacings=int(weak["radius_in_spacings"]),
        center_stride=int(weak["center_stride"]),
        projection_floor=float(weak["projection_floor"]),
    )
    if weak.get("interpolation_of_u") != "none":
        raise ValueError("weak form must declare that u is not interpolated")
    search = payload.get("solver_search")
    if not isinstance(search, dict):
        raise ValueError("protocol is missing solver_search")
    SolverSearch(
        nu_min=float(search["nu_min"]),
        nu_max=float(search["nu_max"]),
        n_coarse=int(search["n_coarse"]),
        refine_width=float(search["refine_width"]),
        max_refine=int(search["max_refine"]),
        n=int(search["n"]),
        dt=float(search["dt"]),
        save_dt=float(search["save_dt"]),
    )
    for name in ("solver_sensor_ic", "solver_oracle_ic"):
        block = payload.get(name)
        if not isinstance(block, dict):
            raise ValueError(f"protocol is missing {name}")
    if payload["solver_sensor_ic"].get("reads_off_sensor_samples") is not False:
        raise ValueError("sensor-lift solver must not read off-sensor samples")
    if payload["solver_oracle_ic"].get("reads_full_initial_condition") is not True:
        raise ValueError("oracle solver must be labeled as reading the full initial condition")
    if int(payload.get("workers", 0)) < 1:
        raise ValueError("workers must be an integer >= 1")
    _require_sha(Path(payload["stage_d_scores"]), str(payload["stage_d_scores_sha256"]))
    _require_sha(Path(payload["stage_d_inverse_protocol"]), str(payload["stage_d_inverse_protocol_sha256"]))
    _require_sha(Path(payload["stage_e_protocol"]), str(payload["stage_e_protocol_sha256"]))
    _require_sha(Path(payload["stage_e_scores"]), str(payload["stage_e_scores_sha256"]))
    _require_sha(Path(payload["data_protocol"]), str(payload["data_protocol_sha256"]))
    _require_sha(Path(payload["inverse_stress"]), str(payload["inverse_stress_sha256"]))
    return payload


def weak_config_from_protocol(protocol: Mapping[str, Any]) -> WeakFormConfig:
    block = protocol["weak_form"]
    return WeakFormConfig(
        degree=int(block["degree"]),
        radius_in_spacings=int(block["radius_in_spacings"]),
        center_stride=int(block["center_stride"]),
        projection_floor=float(block["projection_floor"]),
    )


def search_from_protocol(protocol: Mapping[str, Any]) -> SolverSearch:
    block = protocol["solver_search"]
    return SolverSearch(
        nu_min=float(block["nu_min"]),
        nu_max=float(block["nu_max"]),
        n_coarse=int(block["n_coarse"]),
        refine_width=float(block["refine_width"]),
        max_refine=int(block["max_refine"]),
        n=int(block["n"]),
        dt=float(block["dt"]),
        save_dt=float(block["save_dt"]),
    )


def score_method_rows(
    rows: Sequence[Mapping[str, Any]],
    baseline_nu: float,
    pattern: str,
) -> RecoveryScore:
    """Aggregate instance records. Estimates are not refit."""

    built: list[InstanceRecovery] = []
    baseline = float(baseline_nu)
    for row in rows:
        nu = float(row["nu"])
        hat = float(row["nu_hat"])
        absolute = abs(hat - nu)
        relative = absolute / nu
        baseline_absolute = abs(baseline - nu)
        built.append(
            InstanceRecovery(
                instance_id=int(row["instance_id"]),
                nu=nu,
                nu_hat=hat,
                abs_error=absolute,
                rel_error=relative,
                baseline_abs_error=baseline_absolute,
                baseline_rel_error=baseline_absolute / nu,
                mean_abs_residual=0.0,
                mean_abs_residual_at_truth=0.0,
                mean_square_residual=float(row.get("objective", 0.0)),
                mean_square_residual_at_truth=float(row.get("objective_at_truth") or 0.0),
                failure=bool(hat <= 0.0 or relative > FAILURE_RELATIVE_ERROR),
            )
        )
    return recovery_from_rows(pattern, baseline, built)


def slice_public(score: RecoveryScore, protocol: Mapping[str, Any]) -> dict[str, Any]:
    """Per-slice error table. Residual moments are not part of this study."""

    viscosities = np.asarray([row.nu for row in score.instances], dtype=np.float64)
    floor = float(protocol["below_stage2_floor"])
    masks = slice_masks(viscosities, hard_ood_threshold(protocol), floor)
    payload: dict[str, Any] = {}
    for name in SLICE_NAMES:
        part = _subset(score, masks[name], score.pattern)
        payload[name] = _public_slice(part)
    return payload


def paired_against_fno(
    fno_hats: np.ndarray,
    fno_nu: np.ndarray,
    baseline_rel: np.ndarray,
    baseline_fail: np.ndarray,
) -> dict[str, Any]:
    """Per-seed comparison. A negative delta means the FNO has lower error.

    ``fno_hats`` has shape ``(n_seeds, n_instances)``. ``baseline_*`` follow
    the same instance order and do not vary with seed.
    """

    hats = np.asarray(fno_hats, dtype=np.float64)
    truth = np.asarray(fno_nu, dtype=np.float64)
    base = np.asarray(baseline_rel, dtype=np.float64)
    if hats.ndim != 2 or truth.shape != (hats.shape[1],) or base.shape != truth.shape:
        raise ValueError("paired arrays are not aligned")
    if hats.shape[0] < 2:
        raise ValueError("paired comparison needs at least two FNO seeds")
    relative = np.abs(hats - truth[None, :]) / truth[None, :]
    delta = relative - base[None, :]
    failures = (hats <= 0.0) | (relative > FAILURE_RELATIVE_ERROR)
    per_seed_delta = [float(np.mean(delta[index])) for index in range(hats.shape[0])]
    per_seed_lower = [int(np.sum(delta[index] < 0.0)) for index in range(hats.shape[0])]
    per_seed_higher = [int(np.sum(delta[index] > 0.0)) for index in range(hats.shape[0])]
    per_seed_tie = [int(np.sum(delta[index] == 0.0)) for index in range(hats.shape[0])]
    per_seed_fno = [float(np.mean(relative[index])) for index in range(hats.shape[0])]
    per_seed_fail = [int(np.sum(failures[index])) for index in range(hats.shape[0])]
    delta_mean, delta_std = mean_std(per_seed_delta)
    lower_mean, lower_std = mean_std([float(value) for value in per_seed_lower])
    higher_mean, higher_std = mean_std([float(value) for value in per_seed_higher])
    tie_mean, tie_std = mean_std([float(value) for value in per_seed_tie])
    fno_mean, fno_std = mean_std(per_seed_fno)
    fail_mean, fail_std = mean_std([float(value) for value in per_seed_fail])
    return {
        "n_instances": int(truth.size),
        "n_seeds": int(hats.shape[0]),
        "baseline_mean_rel_error": float(np.mean(base)),
        "baseline_n_failures": int(np.sum(baseline_fail)),
        "fno_mean_rel_error": fno_mean,
        "fno_mean_rel_error_std": fno_std,
        "fno_minus_baseline_mean_rel": delta_mean,
        "fno_minus_baseline_mean_rel_std": delta_std,
        "n_fno_lower": lower_mean,
        "n_fno_lower_std": lower_std,
        "n_fno_higher": higher_mean,
        "n_fno_higher_std": higher_std,
        "n_tie": tie_mean,
        "n_tie_std": tie_std,
        "fno_n_failures": fail_mean,
        "fno_n_failures_std": fail_std,
        "per_seed_fno_mean_rel_error": per_seed_fno,
        "per_seed_fno_minus_baseline": per_seed_delta,
    }


def render_report(scores: Mapping[str, Any]) -> str:
    """Markdown whose numeric cells are taken from ``scores``."""

    lines: list[str] = []
    lines.append("# Strong baselines for the viscosity inverse")
    lines.append("")
    lines.append(
        "Research note for a possible arXiv comparison. It does not replace "
        "the Stage D or Stage E reports, and it does not change their numbers. "
        "The question was frozen in "
        f"`{scores['protocol']}` before these test scores were computed. "
        f"Protocol SHA-256: `{scores['protocol_sha256']}`."
    )
    lines.append("")
    lines.append(str(scores["question"]))
    lines.append("")
    lines.append("## Estimators")
    lines.append("")
    lines.append(str(scores["estimator_notes"]))
    lines.append("")
    lines.append("## Primary mask")
    lines.append("")
    lines.append(
        "Mask `sensors32_bursts`: 32 equispaced sensors, four bursts of five "
        "frames at 0, 24, 48, and 72, `Δt = 0.01`. Failure rule: "
        "`ν̂ <= 0` or relative error `> 0.5`. "
        f"`hard_ood` is `ν <= {scores['threshold_nu']}`. "
        "The weak form and both solver fits are deterministic, so they have "
        "no seed standard deviation. Fourier neural operator cells are the "
        "mean ± sample standard deviation (`ddof = 1`) across seeds 0 through 4, "
        "recomputed here from the committed Stage D per-instance `lambda0_nu_hat` "
        "values at `λ = 0`. Those checkpoints were not retrained."
    )
    lines.append("")
    lines.append(_method_table(scores))
    lines.append("")
    lines.append("## Paired comparison")
    lines.append("")
    lines.append(
        "Delta is the FNO relative error minus the baseline relative error "
        "on the same instance. A negative delta is a lower error for the FNO. "
        "Counts are instances, then the mean ± sample standard deviation across seeds."
    )
    lines.append("")
    for slice_name in ("hard_ood", "complement", "below_stage2_floor", "full_test"):
        lines.append(f"### `{slice_name}`")
        lines.append("")
        lines.append(_paired_table(scores, slice_name))
        lines.append("")
    lines.append("## Closed-form failures")
    lines.append("")
    lines.append(
        "Every test instance on which the recomputed `sensors32_bursts` "
        "least squares fails the 0.5 rule. Fourier neural operator cells are "
        "the mean relative error across the five seeds."
    )
    lines.append("")
    lines.append(_failure_table(scores))
    lines.append("")
    lines.append("## Does the Stage D advantage survive?")
    lines.append("")
    lines.append(str(scores["advantage"]["paragraph"]))
    lines.append("")
    lines.append("## Compute")
    lines.append("")
    lines.append(_timing_table(scores))
    lines.append("")
    lines.append(str(scores["timing"]["notes"]))
    lines.append("")
    lines.append("## Stage E noise and sparsity")
    lines.append("")
    lines.append(str(scores["stress"]["notes"]))
    lines.append("")
    lines.append(_stress_table(scores, "weak_form"))
    lines.append("")
    lines.append(_stress_table(scores, "solver_sensor_ic"))
    lines.append("")
    lines.append("## Limitations")
    lines.append("")
    for item in scores["limitations"]:
        lines.append(f"- {item}")
    lines.append("")
    lines.append(
        "Tables in this file were written by `python -m pinnforge.operator strong-baselines` "
        "from `docs/research/strong_baselines_scores.json`. They were not typed by hand."
    )
    lines.append("")
    return "\n".join(lines)


def run_strong_baselines(
    *,
    protocol_path: Path,
    pilot_dir: Path,
    manifest_path: Path,
    output: Path,
    report_path: Path,
) -> dict[str, Any]:
    """Score the frozen protocol on the test split and write the note."""

    os.environ.setdefault("OMP_NUM_THREADS", "1")
    os.environ.setdefault("OPENBLAS_NUM_THREADS", "1")
    os.environ.setdefault("MKL_NUM_THREADS", "1")
    protocol = load_strong_protocol(protocol_path)
    digest = file_sha256(protocol_path)
    trajectories = load_split_trajectories(
        pilot_dir,
        manifest_path,
        ("test",),
        check_field_hash=True,
    )
    trajectories.sort(key=lambda item: item.instance_id)
    baseline_nu = training_baseline_nu(manifest_path)
    stage_d = json.loads(Path(protocol["stage_d_scores"]).read_text(encoding="utf-8"))
    stage_e_protocol = json.loads(Path(protocol["stage_e_protocol"]).read_text(encoding="utf-8"))
    stage_e_scores = json.loads(Path(protocol["stage_e_scores"]).read_text(encoding="utf-8"))
    _require_stage_d_alignment(trajectories, stage_d)
    weak_config = weak_config_from_protocol(protocol)
    search = search_from_protocol(protocol)
    reference_spec = observation_spec(stage_e_protocol, 32, 4)
    if reference_spec.name != "sensors32_bursts":
        raise ValueError("the reference mask must stay sensors32_bursts")

    weak_rows = []
    for item in trajectories:
        started = time.perf_counter()
        hat = weak_form_viscosity(item.u, reference_spec, weak_config)
        weak_rows.append(_row(item, hat, seconds=time.perf_counter() - started))
    print(f"weak form reference done n={len(weak_rows)}", flush=True)

    closed_rows = []
    for item in trajectories:
        hat = least_squares_viscosity(item.u, reference_spec)
        closed_rows.append(_row(item, hat, seconds=0.0))
    _assert_closed_form_matches(closed_rows, stage_d)

    sensor_rows = _fit_solver_rows(
        trajectories,
        reference_spec,
        search,
        initial_condition="sensor_lift",
        workers=int(protocol["workers"]),
        with_truth=True,
        cache_name="reference_sensor_lift",
        protocol_sha256=digest,
    )
    print("solver sensor_lift reference done", flush=True)
    oracle_rows = _fit_solver_rows(
        trajectories,
        reference_spec,
        search,
        initial_condition="oracle_ic",
        workers=int(protocol["workers"]),
        with_truth=True,
        cache_name="reference_oracle_ic",
        protocol_sha256=digest,
    )
    print("solver oracle_ic reference done", flush=True)

    methods = {
        "closed_form_ls": _method_block(closed_rows, baseline_nu, "closed_form_ls", protocol),
        "weak_form": _method_block(weak_rows, baseline_nu, "weak_form", protocol),
        "solver_sensor_ic": _method_block(sensor_rows, baseline_nu, "solver_sensor_ic", protocol),
        "solver_oracle_ic": _method_block(oracle_rows, baseline_nu, "solver_oracle_ic", protocol),
    }
    protocol["_protocol_sha256"] = digest
    fno = _fno_from_stage_d(stage_d, protocol, methods["closed_form_ls"])
    _assert_fno_matches_published(fno, stage_d)
    paired = _all_paired(methods, fno, protocol)
    stress = _run_stress(
        trajectories,
        protocol,
        stage_e_protocol,
        stage_e_scores,
        weak_config,
        search,
        baseline_nu,
    )
    timing = _timing_block(methods)
    timing["fno_stage_d_grid"] = _time_fno_grid(trajectories[0], reference_spec, manifest_path)
    advantage = _advantage_text(methods, fno)
    payload: dict[str, Any] = {
        "format": SCORES_FORMAT,
        "protocol": Path(protocol_path).as_posix(),
        "protocol_sha256": digest,
        "split": "test",
        "device": "cpu",
        "workers": int(protocol["workers"]),
        "failure_rule": "nu_hat <= 0 or rel_error > 0.5",
        "question": protocol["question"],
        "estimator_notes": _estimator_notes(protocol),
        "methods": methods,
        "fno": fno,
        "paired": paired,
        "closed_form_failures": _failure_rows(methods, fno),
        "timing": timing,
        "stress": stress,
        "advantage": advantage,
        "limitations": list(protocol["limitations"]),
        "threshold_nu": float(protocol["hard_ood"]["threshold_nu"]),
        "below_stage2_floor": float(protocol["below_stage2_floor"]),
        "published_stage_d_hard_ood_checked": True,
        "published_closed_form_checked": True,
    }
    _write_json(payload, output)
    report = render_report(payload)
    destination = Path(report_path)
    destination.parent.mkdir(parents=True, exist_ok=True)
    destination.write_text(report, encoding="utf-8")
    print(advantage["paragraph"], flush=True)
    print(f"json: {Path(output).as_posix()}", flush=True)
    print(f"report: {destination.as_posix()}", flush=True)
    return payload


def _solver_worker(payload: dict[str, Any]) -> dict[str, Any]:
    os.environ["OMP_NUM_THREADS"] = "1"
    os.environ["OPENBLAS_NUM_THREADS"] = "1"
    os.environ["MKL_NUM_THREADS"] = "1"
    search = SolverSearch(**payload["search"])
    problem = build_solver_problem(
        payload["field"],
        payload["spec"],
        search,
        initial_condition=str(payload["initial_condition"]),
    )
    found = fit_solver_problem(
        problem,
        search,
        truth=None if payload["truth"] is None else float(payload["truth"]),
    )
    found["instance_id"] = int(payload["instance_id"])
    found["nu"] = float(payload["nu"])
    print(
        f"{payload['initial_condition']} id={payload['instance_id']} "
        f"nu_hat={found['nu_hat']:.16g} evals={found['n_evaluations']} "
        f"seconds={found['seconds']:.2f}",
        flush=True,
    )
    return found


def _fit_solver_rows(
    trajectories: Sequence[Trajectory],
    spec: Any,
    search: SolverSearch,
    *,
    initial_condition: str,
    workers: int,
    with_truth: bool,
    cache_name: str | None = None,
    protocol_sha256: str | None = None,
) -> list[dict[str, Any]]:
    expected = {int(item.instance_id) for item in trajectories}
    if cache_name is not None and protocol_sha256:
        cached = _read_fit_cache(cache_name, protocol_sha256, expected)
        if cached is not None:
            print(f"reused {cache_name} n={len(cached)}", flush=True)
            return cached
    search_payload = {
        "nu_min": search.nu_min,
        "nu_max": search.nu_max,
        "n_coarse": search.n_coarse,
        "refine_width": search.refine_width,
        "max_refine": search.max_refine,
        "n": search.n,
        "dt": search.dt,
        "save_dt": search.save_dt,
    }
    jobs = []
    for item in trajectories:
        jobs.append(
            {
                "field": np.asarray(item.u, dtype=np.float64),
                "spec": spec,
                "search": search_payload,
                "initial_condition": initial_condition,
                "instance_id": int(item.instance_id),
                "nu": float(item.nu),
                "truth": float(item.nu) if with_truth else None,
            }
        )
    if workers == 1:
        found = [_solver_worker(job) for job in jobs]
    else:
        with ProcessPoolExecutor(max_workers=workers) as pool:
            found = list(pool.map(_solver_worker, jobs, chunksize=1))
    by_id = {int(row["instance_id"]): row for row in found}
    rows = []
    for item in trajectories:
        result = by_id[int(item.instance_id)]
        rows.append(
            _row(
                item,
                float(result["nu_hat"]),
                seconds=float(result["seconds"]),
                n_evaluations=int(result["n_evaluations"]),
                objective=float(result["objective"]),
                objective_at_truth=result["objective_at_truth"],
            )
        )
    if cache_name is not None and protocol_sha256:
        _write_fit_cache(cache_name, protocol_sha256, rows)
    return rows


def _fit_cache_path(cache_name: str) -> Path:
    return Path("runs/strong_baselines") / f"{cache_name}.json"


def _read_fit_cache(
    cache_name: str,
    protocol_sha256: str,
    expected_ids: set[int],
) -> list[dict[str, Any]] | None:
    path = _fit_cache_path(cache_name)
    if not path.is_file():
        return None
    payload = json.loads(path.read_text(encoding="utf-8"))
    if payload.get("protocol_sha256") != protocol_sha256:
        return None
    rows = payload.get("rows")
    if not isinstance(rows, list):
        return None
    got = {int(row["instance_id"]) for row in rows}
    if got != expected_ids:
        return None
    return [dict(row) for row in rows]


def _write_fit_cache(cache_name: str, protocol_sha256: str, rows: Sequence[Mapping[str, Any]]) -> None:
    path = _fit_cache_path(cache_name)
    path.parent.mkdir(parents=True, exist_ok=True)
    payload = {"protocol_sha256": protocol_sha256, "rows": _json_ready(list(rows))}
    path.write_text(json.dumps(payload) + "\n", encoding="utf-8")


def _run_stress(
    trajectories: Sequence[Trajectory],
    protocol: Mapping[str, Any],
    stage_e_protocol: Mapping[str, Any],
    stage_e_scores: Mapping[str, Any],
    weak_config: WeakFormConfig,
    search: SolverSearch,
    baseline_nu: float,
) -> dict[str, Any]:
    master = int(stage_e_protocol["noise"]["master_seed"])
    noise_seed = int(protocol["stress"]["noise_seed"])
    epsilons = {
        int(item.instance_id): standard_normal_field(master, noise_seed, int(item.instance_id), item.u.shape)
        for item in trajectories
    }
    weak_blocks: dict[str, Any] = {}
    for condition in stage_e_protocol["conditions"]:
        condition_id = str(condition["id"])
        spec = observation_spec(stage_e_protocol, int(condition["n_sensors"]), int(condition["n_bursts"]))
        fraction = float(condition["noise_fraction"])
        rows = []
        for item in trajectories:
            field = apply_noise(item.u, epsilons[int(item.instance_id)], fraction)
            hat = weak_form_viscosity(field, spec, weak_config)
            rows.append(_row(item, hat))
        block = _method_block(rows, baseline_nu, "weak_form", protocol)
        weak_blocks[condition_id] = {
            "noise_fraction": fraction,
            "n_sensors": int(condition["n_sensors"]),
            "n_bursts": int(condition["n_bursts"]),
            "slices": block["slices"],
        }
        print(
            f"weak stress {condition_id} hard_ood rel "
            f"{block['slices']['hard_ood']['mean_rel_error']}",
            flush=True,
        )

    solver_blocks: dict[str, Any] = {}
    threshold = hard_ood_threshold(protocol)
    hard = [item for item in trajectories if float(item.nu) <= threshold]
    for condition_id in protocol["stress"]["solver_sensor_ic"]["condition_ids"]:
        condition = _condition_by_id(stage_e_protocol, str(condition_id))
        spec = observation_spec(stage_e_protocol, int(condition["n_sensors"]), int(condition["n_bursts"]))
        if len(spec.series) < 2:
            solver_blocks[str(condition_id)] = {"applicable": False, "reason": "no target burst"}
            continue
        fraction = float(condition["noise_fraction"])
        noisy = []
        for item in hard:
            field = apply_noise(item.u, epsilons[int(item.instance_id)], fraction)
            noisy.append(Trajectory(instance_id=item.instance_id, split=item.split, nu=item.nu, u=field))
        rows = _fit_solver_rows(
            noisy,
            spec,
            search,
            initial_condition="sensor_lift",
            workers=int(protocol["workers"]),
            with_truth=False,
            cache_name=f"stress_{condition_id}",
            protocol_sha256=str(protocol.get("_protocol_sha256", "")),
        )
        score = score_method_rows(rows, baseline_nu, "solver_sensor_ic")
        solver_blocks[str(condition_id)] = {
            "applicable": True,
            "noise_fraction": fraction,
            "n_sensors": int(condition["n_sensors"]),
            "n_bursts": int(condition["n_bursts"]),
            "slice": "hard_ood",
            "n_instances": score.n_instances,
            "summary": _public_slice(score),
        }
        print(
            f"solver stress {condition_id} hard_ood rel {score.mean_rel_error}",
            flush=True,
        )
    return {
        "notes": str(protocol["stress"]["notes"]),
        "noise_seed": noise_seed,
        "weak_form": weak_blocks,
        "solver_sensor_ic": solver_blocks,
        "published_stage_e": _published_stage_e(stage_e_scores, stage_e_protocol),
    }


def _time_fno_grid(item: Trajectory, spec: Any, manifest_path: Path) -> dict[str, Any]:
    """Wall clock of one 191-point Stage D rollout. Weights are not trained."""

    from pinnforge.operator.fno import FNO1d
    from pinnforge.operator.stage_d import (
        initial_window_from_mask,
        read_mask,
        target_slots,
        viscosity_grid,
    )
    from pinnforge.operator.stage_d_fit import _rollout_sensors
    from pinnforge.operator.windows import WindowSpec, field_norm_from_manifest, load_pilot_manifest

    window = WindowSpec(input_frames=8, output_frames=8, stride=8)
    mask = read_mask(item.u, spec)
    windows = initial_window_from_mask(mask, int(item.u.shape[1]))[None, :, :]
    slots = target_slots(spec, window)
    n_steps = max(step for _, step, _, _ in slots) + 1
    protocol = {
        "nu_search": {"min": 0.005, "max": 0.1, "n_grid": 191},
    }
    grid = viscosity_grid(protocol)
    norm = field_norm_from_manifest(load_pilot_manifest(manifest_path))
    model = FNO1d(9, 8, width=32, modes=16, n_layers=4)
    model.eval()
    started = time.perf_counter()
    for nu in grid:
        _rollout_sensors(model, norm, windows, float(nu), n_steps, int(spec.n_sensors), 1)
    elapsed = time.perf_counter() - started
    return {
        "seconds_per_instance": float(elapsed),
        "n_instances_timed": 1,
        "instance_id": int(item.instance_id),
        "n_grid": int(grid.size),
        "n_rollout_steps": int(n_steps),
        "architecture": {"width": 32, "modes": 16, "layers": 4, "window": "8/8/8"},
        "weights": "randomly initialized Stage D architecture; not a published checkpoint",
        "device": "cpu",
        "torch": _torch_version(),
    }


def _torch_version() -> str:
    import torch

    return str(torch.__version__)


def _method_block(
    rows: Sequence[Mapping[str, Any]],
    baseline_nu: float,
    pattern: str,
    protocol: Mapping[str, Any],
) -> dict[str, Any]:
    score = score_method_rows(rows, baseline_nu, pattern)
    seconds = [float(row["seconds"]) for row in rows if "seconds" in row]
    evaluations = [int(row["n_evaluations"]) for row in rows if "n_evaluations" in row]
    return {
        "pattern": pattern,
        "deterministic": True,
        "seed_std": None,
        "slices": slice_public(score, protocol),
        "instances": [_json_ready(dict(row)) for row in rows],
        "seconds_per_instance_mean": float(np.mean(seconds)) if seconds else None,
        "seconds_per_instance_max": float(np.max(seconds)) if seconds else None,
        "n_evaluations_mean": float(np.mean(evaluations)) if evaluations else None,
    }


def _fno_from_stage_d(
    stage_d: Mapping[str, Any],
    protocol: Mapping[str, Any],
    closed: Mapping[str, Any],
) -> dict[str, Any]:
    by_id = {int(row["instance_id"]): row for row in stage_d["instances"]}
    closed_by_id = {int(row["instance_id"]): row for row in closed["instances"]}
    if set(by_id) != set(closed_by_id):
        raise ValueError("Stage D instances do not match the recomputed test split")
    arms: dict[str, Any] = {}
    for arm in FNO_ARMS:
        rows_by_seed: list[list[dict[str, Any]]] = []
        n_seeds = len(by_id[next(iter(by_id))][arm]["lambda0_nu_hat"])
        for seed in range(n_seeds):
            seed_rows = []
            for instance_id in sorted(by_id):
                item = by_id[instance_id]
                hat = float(item[arm]["lambda0_nu_hat"][seed])
                seed_rows.append(
                    {
                        "instance_id": instance_id,
                        "nu": float(item["nu"]),
                        "nu_hat": hat,
                        "seconds": 0.0,
                    }
                )
            rows_by_seed.append(seed_rows)
        per_seed_slices = []
        hat_matrix: dict[str, list[float]] = {}
        nu_by_id: dict[str, float] = {}
        for instance_id in sorted(by_id):
            item = by_id[instance_id]
            hat_matrix[str(instance_id)] = [float(value) for value in item[arm]["lambda0_nu_hat"]]
            nu_by_id[str(instance_id)] = float(item["nu"])
        for seed, seed_rows in enumerate(rows_by_seed):
            block = _method_block(seed_rows, 1.0, arm, protocol)
            per_seed_slices.append({"seed": seed, "slices": block["slices"]})
        arms[arm] = {
            "selected_lambda": 0.0,
            "per_seed": per_seed_slices,
            "slices": _average_fno_slices(per_seed_slices),
            "hat_matrix": hat_matrix,
            "nu_by_id": nu_by_id,
        }
    return {"source": "docs/v02/stage_d_scores.json", "arms": arms}


def _average_fno_slices(per_seed: Sequence[Mapping[str, Any]]) -> dict[str, Any]:
    combined: dict[str, Any] = {}
    for name in SLICE_NAMES:
        fields = (
            "mean_abs_error",
            "median_abs_error",
            "max_abs_error",
            "mean_rel_error",
            "median_rel_error",
            "max_rel_error",
            "n_failures",
            "n_nonpositive",
            "n_worse_than_baseline",
        )
        block: dict[str, Any] = {
            "n_instances": int(per_seed[0]["slices"][name]["n_instances"]),
        }
        for field in fields:
            center, spread = mean_std([float(item["slices"][name][field]) for item in per_seed])
            block[field] = center
            block[f"{field}_std"] = spread
        correlations = [item["slices"][name]["correlation"] for item in per_seed]
        if any(value is None for value in correlations):
            block["correlation"] = None
            block["correlation_std"] = None
        else:
            center, spread = mean_std([float(value) for value in correlations])
            block["correlation"] = center
            block["correlation_std"] = spread
        combined[name] = block
    return combined


def _all_paired(
    methods: Mapping[str, Any],
    fno: Mapping[str, Any],
    protocol: Mapping[str, Any],
) -> dict[str, Any]:
    payload: dict[str, Any] = {}
    order = [int(row["instance_id"]) for row in methods["closed_form_ls"]["instances"]]
    for slice_name in SLICE_NAMES:
        payload[slice_name] = {}
        keep = _slice_ids(methods["closed_form_ls"]["instances"], protocol, slice_name)
        for arm in FNO_ARMS:
            payload[slice_name][arm] = {}
            hats, truth = _fno_matrix(fno["arms"][arm], order, keep)
            for method in PRIMARY_METHODS:
                base_rel, base_fail = _baseline_vector(methods[method]["instances"], order, keep)
                payload[slice_name][arm][method] = paired_against_fno(hats, truth, base_rel, base_fail)
    return payload


def _failure_rows(methods: Mapping[str, Any], fno: Mapping[str, Any]) -> list[dict[str, Any]]:
    closed = methods["closed_form_ls"]["instances"]
    weak = {int(row["instance_id"]): row for row in methods["weak_form"]["instances"]}
    sensor = {int(row["instance_id"]): row for row in methods["solver_sensor_ic"]["instances"]}
    oracle = {int(row["instance_id"]): row for row in methods["solver_oracle_ic"]["instances"]}
    data = _fno_hat_lists(fno, "data_only")
    hybrid = _fno_hat_lists(fno, "hybrid_1e-2")
    rows = []
    for row in closed:
        if not row["failure"]:
            continue
        instance_id = int(row["instance_id"])
        nu = float(row["nu"])
        rows.append(
            {
                "instance_id": instance_id,
                "nu": nu,
                "closed_form_rel_error": float(row["rel_error"]),
                "weak_form_rel_error": float(weak[instance_id]["rel_error"]),
                "solver_sensor_ic_rel_error": float(sensor[instance_id]["rel_error"]),
                "solver_oracle_ic_rel_error": float(oracle[instance_id]["rel_error"]),
                "fno_data_only_mean_rel_error": float(np.mean(np.abs(np.asarray(data[instance_id]) - nu) / nu)),
                "fno_hybrid_1e-2_mean_rel_error": float(
                    np.mean(np.abs(np.asarray(hybrid[instance_id]) - nu) / nu)
                ),
                "weak_form_failure": bool(weak[instance_id]["failure"]),
                "solver_sensor_ic_failure": bool(sensor[instance_id]["failure"]),
                "solver_oracle_ic_failure": bool(oracle[instance_id]["failure"]),
            }
        )
    rows.sort(key=lambda item: (-float(item["closed_form_rel_error"]), int(item["instance_id"])))
    return rows


def _advantage_text(methods: Mapping[str, Any], fno: Mapping[str, Any]) -> dict[str, Any]:
    slice_name = "hard_ood"
    closed = float(methods["closed_form_ls"]["slices"][slice_name]["mean_rel_error"])
    numbers = {
        "closed_form_ls": closed,
        "weak_form": float(methods["weak_form"]["slices"][slice_name]["mean_rel_error"]),
        "solver_sensor_ic": float(methods["solver_sensor_ic"]["slices"][slice_name]["mean_rel_error"]),
        "solver_oracle_ic": float(methods["solver_oracle_ic"]["slices"][slice_name]["mean_rel_error"]),
        "fno_data_only": float(fno["arms"]["data_only"]["slices"][slice_name]["mean_rel_error"]),
        "fno_hybrid_1e-2": float(fno["arms"]["hybrid_1e-2"]["slices"][slice_name]["mean_rel_error"]),
    }
    sentences = [
        "On hard_ood, the published comparison is the FNO against closed-form "
        f"residual least squares ({_fmt(numbers['closed_form_ls'])}, "
        f"{int(methods['closed_form_ls']['slices'][slice_name]['n_failures'])} failures)."
    ]
    for arm, label in (("fno_data_only", "data-only"), ("fno_hybrid_1e-2", "hybrid 1e-2")):
        fno_value = numbers[arm]
        for method, method_label in (
            ("weak_form", "weak form"),
            ("solver_sensor_ic", "sensor-lift solver"),
            ("solver_oracle_ic", "oracle-IC solver"),
        ):
            other = numbers[method]
            if fno_value < other:
                relation = "lower error than"
            elif fno_value > other:
                relation = "higher error than"
            else:
                relation = "the same error as"
            sentences.append(
                f"The {label} FNO mean relative error ({_fmt(fno_value)}) is {relation} "
                f"the {method_label} ({_fmt(other)})."
            )
    beats_closed = all(numbers[arm] < closed for arm in ("fno_data_only", "fno_hybrid_1e-2"))
    beats_weak = all(numbers[arm] < numbers["weak_form"] for arm in ("fno_data_only", "fno_hybrid_1e-2"))
    beats_sensor = all(
        numbers[arm] < numbers["solver_sensor_ic"] for arm in ("fno_data_only", "fno_hybrid_1e-2")
    )
    beats_oracle = all(
        numbers[arm] < numbers["solver_oracle_ic"] for arm in ("fno_data_only", "fno_hybrid_1e-2")
    )
    if beats_weak and beats_sensor and beats_oracle:
        sentences.append(
            "Both FNO arms still have lower hard_ood mean relative error than the weak form "
            "and both solver fits. The advantage over the closed form survives these baselines "
            "on this pilot."
        )
    elif beats_closed and not (beats_weak and beats_sensor):
        sentences.append(
            "The advantage over the closed form does not survive as a claim that the operator "
            "beats a weak-form identifier and a solver-in-the-loop fit. "
            "At least one of those baselines has hard_ood mean relative error at or below an FNO arm."
        )
    else:
        sentences.append(
            "The hard_ood ordering is mixed. Read the paired table rather than a single headline."
        )
    if not beats_oracle:
        sentences.append(
            "The oracle initial-condition solver is a ceiling that sees the full field at t = 0. "
            "It is not evidence about a sensor-only method."
        )
    return {"hard_ood": numbers, "paragraph": " ".join(sentences)}


def _timing_block(methods: Mapping[str, Any]) -> dict[str, Any]:
    rows = {}
    for name in ("weak_form", "solver_sensor_ic", "solver_oracle_ic", "closed_form_ls"):
        rows[name] = {
            "seconds_per_instance_mean": methods[name]["seconds_per_instance_mean"],
            "seconds_per_instance_max": methods[name]["seconds_per_instance_max"],
            "n_evaluations_mean": methods[name]["n_evaluations_mean"],
        }
    return {
        "methods": rows,
        "notes": (
            "Solver times are wall-clock seconds per test instance for the full "
            "coarse-grid plus golden-section search, including every forward solve, "
            "on this CPU with the worker count in the protocol. "
            "The FNO time is one instance of the same 191-point rollout the Stage D "
            "inverse evaluates. The network is the Stage D architecture with random "
            "initial weights, because the published checkpoints are not in the "
            "repository. The arithmetic does not depend on those weights. "
            "Closed-form time was not the object of the measurement and is stored as 0."
        ),
    }


def _estimator_notes(protocol: Mapping[str, Any]) -> str:
    weak = protocol["weak_form"]
    sensor = protocol["solver_sensor_ic"]
    oracle = protocol["solver_oracle_ic"]
    return (
        f"Weak form: {weak['description']} "
        f"Solver, sensor lift: {sensor['description']} "
        f"Solver, oracle initial condition: {oracle['description']}"
    )


def _method_table(scores: Mapping[str, Any]) -> str:
    header = (
        "| estimator | slice | n | mean rel | median rel | max rel | failures | correlation |\n"
        "| --- | --- | ---: | ---: | ---: | ---: | ---: | ---: |"
    )
    body = [header]
    for method in (*PRIMARY_METHODS,):
        slices = scores["methods"][method]["slices"]
        for name in SLICE_NAMES:
            row = slices[name]
            body.append(
                "| "
                + " | ".join(
                    [
                        f"`{method}`",
                        f"`{name}`",
                        str(int(row["n_instances"])),
                        _fmt(row["mean_rel_error"]),
                        _fmt(row["median_rel_error"]),
                        _fmt(row["max_rel_error"]),
                        str(int(row["n_failures"])),
                        _fmt_optional(row["correlation"]),
                    ]
                )
                + " |"
            )
    for arm in FNO_ARMS:
        slices = scores["fno"]["arms"][arm]["slices"]
        for name in SLICE_NAMES:
            row = slices[name]
            body.append(
                "| "
                + " | ".join(
                    [
                        f"`fno_{arm}`",
                        f"`{name}`",
                        str(int(row["n_instances"])),
                        _fmt_pm(row["mean_rel_error"], row["mean_rel_error_std"]),
                        _fmt_pm(row["median_rel_error"], row["median_rel_error_std"]),
                        _fmt_pm(row["max_rel_error"], row["max_rel_error_std"]),
                        _fmt_pm(row["n_failures"], row["n_failures_std"]),
                        _fmt_pm(row["correlation"], row["correlation_std"]),
                    ]
                )
                + " |"
            )
    return "\n".join(body)


def _paired_table(scores: Mapping[str, Any], slice_name: str) -> str:
    header = (
        "| FNO arm | baseline | baseline mean rel | FNO mean rel | "
        "FNO − baseline | FNO lower | FNO higher | baseline failures | FNO failures |\n"
        "| --- | --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |"
    )
    body = [header]
    for arm in FNO_ARMS:
        for method in PRIMARY_METHODS:
            row = scores["paired"][slice_name][arm][method]
            body.append(
                "| "
                + " | ".join(
                    [
                        f"`{arm}`",
                        f"`{method}`",
                        _fmt(row["baseline_mean_rel_error"]),
                        _fmt_pm(row["fno_mean_rel_error"], row["fno_mean_rel_error_std"]),
                        _fmt_pm(row["fno_minus_baseline_mean_rel"], row["fno_minus_baseline_mean_rel_std"]),
                        _fmt_pm(row["n_fno_lower"], row["n_fno_lower_std"]),
                        _fmt_pm(row["n_fno_higher"], row["n_fno_higher_std"]),
                        str(int(row["baseline_n_failures"])),
                        _fmt_pm(row["fno_n_failures"], row["fno_n_failures_std"]),
                    ]
                )
                + " |"
            )
    return "\n".join(body)


def _failure_table(scores: Mapping[str, Any]) -> str:
    header = (
        "| instance | ν | closed form | weak form | sensor-lift solver | "
        "oracle-IC solver | data-only mean | hybrid mean |\n"
        "| ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |"
    )
    body = [header]
    rows = scores["closed_form_failures"]
    if not rows:
        body.append("| none |  |  |  |  |  |  |  |")
        return "\n".join(body)
    for row in rows:
        body.append(
            "| "
            + " | ".join(
                [
                    str(int(row["instance_id"])),
                    _fmt(row["nu"]),
                    _fmt(row["closed_form_rel_error"]),
                    _fmt(row["weak_form_rel_error"]),
                    _fmt(row["solver_sensor_ic_rel_error"]),
                    _fmt(row["solver_oracle_ic_rel_error"]),
                    _fmt(row["fno_data_only_mean_rel_error"]),
                    _fmt(row["fno_hybrid_1e-2_mean_rel_error"]),
                ]
            )
            + " |"
        )
    return "\n".join(body)


def _timing_table(scores: Mapping[str, Any]) -> str:
    header = (
        "| estimator | seconds per instance | evaluations per instance |\n"
        "| --- | ---: | ---: |"
    )
    body = [header]
    for name, row in scores["timing"]["methods"].items():
        body.append(
            f"| `{name}` | {_fmt_optional(row['seconds_per_instance_mean'])} | "
            f"{_fmt_optional(row['n_evaluations_mean'])} |"
        )
    fno = scores["timing"]["fno_stage_d_grid"]
    body.append(
        f"| `fno_stage_d_grid` | {_fmt(fno['seconds_per_instance'])} | {int(fno['n_grid'])} |"
    )
    return "\n".join(body)


def _stress_table(scores: Mapping[str, Any], method: str) -> str:
    published = scores["stress"]["published_stage_e"]
    lines = [f"### `{method}`", ""]
    if method == "weak_form":
        lines.append(
            "| condition | slice | weak form mean rel | weak form failures | "
            "closed form | data-only FNO | hybrid FNO |"
        )
        lines.append("| --- | --- | ---: | ---: | ---: | ---: | ---: |")
        for condition_id, block in scores["stress"]["weak_form"].items():
            for slice_name in ("hard_ood", "complement"):
                row = block["slices"][slice_name]
                pub = published[condition_id][slice_name]
                lines.append(
                    "| "
                    + " | ".join(
                        [
                            f"`{condition_id}`",
                            f"`{slice_name}`",
                            _fmt(row["mean_rel_error"]),
                            str(int(row["n_failures"])),
                            _fmt_pub(pub["closed_form_ls"], None),
                            _fmt_pub(pub["operator_data_only"], pub["operator_data_only_std"]),
                            _fmt_pub(pub["operator_hybrid_1e-2"], pub["operator_hybrid_1e-2_std"]),
                        ]
                    )
                    + " |"
                )
        return "\n".join(lines)
    lines.append(
        "| condition | hard_ood n | sensor-lift solver mean rel | failures | "
        "closed form | data-only FNO | hybrid FNO |"
    )
    lines.append("| --- | ---: | ---: | ---: | ---: | ---: | ---: |")
    for condition_id, block in scores["stress"]["solver_sensor_ic"].items():
        pub = published[condition_id]["hard_ood"]
        if not block.get("applicable", False):
            lines.append(f"| `{condition_id}` |  | inapplicable |  |  |  |  |")
            continue
        summary = block["summary"]
        lines.append(
            "| "
            + " | ".join(
                [
                    f"`{condition_id}`",
                    str(int(block["n_instances"])),
                    _fmt(summary["mean_rel_error"]),
                    str(int(summary["n_failures"])),
                    _fmt_pub(pub["closed_form_ls"], None),
                    _fmt_pub(pub["operator_data_only"], pub["operator_data_only_std"]),
                    _fmt_pub(pub["operator_hybrid_1e-2"], pub["operator_hybrid_1e-2_std"]),
                ]
            )
            + " |"
        )
    return "\n".join(lines)


def _published_stage_e(
    stage_e_scores: Mapping[str, Any],
    stage_e_protocol: Mapping[str, Any],
) -> dict[str, Any]:
    payload: dict[str, Any] = {}
    conditions = stage_e_scores["conditions"]
    for condition in stage_e_protocol["conditions"]:
        condition_id = str(condition["id"])
        block = conditions[condition_id]
        payload[condition_id] = {}
        for slice_name in ("hard_ood", "complement"):
            data_mean, data_std = _method_slice_mean(block["operator_data_only"], slice_name)
            hybrid_mean, hybrid_std = _method_slice_mean(block["operator_hybrid_1e-2"], slice_name)
            closed_mean, _closed_std = _method_slice_mean(block["closed_form_ls"], slice_name)
            payload[condition_id][slice_name] = {
                "closed_form_ls": closed_mean,
                "operator_data_only": data_mean,
                "operator_data_only_std": data_std,
                "operator_hybrid_1e-2": hybrid_mean,
                "operator_hybrid_1e-2_std": hybrid_std,
            }
    return payload


def _method_slice_mean(method_block: Mapping[str, Any], slice_name: str) -> tuple[float | None, float | None]:
    if method_block.get("applicable") is False:
        return None, None
    return _stage_e_pair(method_block["slices"][slice_name])


def _stage_e_pair(block: Mapping[str, Any]) -> tuple[float, float | None]:
    if block.get("applicable") is False:
        return (math.nan, None)
    value = block["mean_rel_error"]
    if isinstance(value, dict):
        return float(value["mean"]), float(value["std"])
    std = block.get("mean_rel_error_std")
    return float(value), None if std is None else float(std)


def _row(
    item: Trajectory,
    nu_hat: float,
    *,
    seconds: float = 0.0,
    n_evaluations: int | None = None,
    objective: float | None = None,
    objective_at_truth: float | None = None,
) -> dict[str, Any]:
    nu = float(item.nu)
    absolute = abs(float(nu_hat) - nu)
    relative = absolute / nu
    row: dict[str, Any] = {
        "instance_id": int(item.instance_id),
        "nu": nu,
        "nu_hat": float(nu_hat),
        "abs_error": absolute,
        "rel_error": relative,
        "failure": bool(float(nu_hat) <= 0.0 or relative > FAILURE_RELATIVE_ERROR),
        "seconds": float(seconds),
    }
    if n_evaluations is not None:
        row["n_evaluations"] = int(n_evaluations)
    if objective is not None:
        row["objective"] = float(objective)
    if objective_at_truth is not None:
        row["objective_at_truth"] = float(objective_at_truth)
    return row


def _subset(score: RecoveryScore, mask: np.ndarray, pattern: str) -> RecoveryScore:
    chosen = [row for row, keep in zip(score.instances, mask, strict=True) if bool(keep)]
    if not chosen:
        raise ValueError(f"{pattern} slice is empty")
    return recovery_from_rows(pattern, score.baseline_nu, chosen)


def _public_slice(score: RecoveryScore) -> dict[str, Any]:
    return {
        "n_instances": score.n_instances,
        "mean_abs_error": score.mean_abs_error,
        "median_abs_error": score.median_abs_error,
        "max_abs_error": score.max_abs_error,
        "mean_rel_error": score.mean_rel_error,
        "median_rel_error": score.median_rel_error,
        "max_rel_error": score.max_rel_error,
        "correlation": score.correlation,
        "n_failures": score.n_failures,
        "n_nonpositive": score.n_nonpositive,
        "n_worse_than_baseline": score.n_worse_than_baseline,
    }


def _slice_ids(
    instances: Sequence[Mapping[str, Any]],
    protocol: Mapping[str, Any],
    slice_name: str,
) -> set[int]:
    viscosities = np.asarray([float(row["nu"]) for row in instances], dtype=np.float64)
    masks = slice_masks(viscosities, hard_ood_threshold(protocol), float(protocol["below_stage2_floor"]))
    return {
        int(row["instance_id"])
        for row, keep in zip(instances, masks[slice_name], strict=True)
        if bool(keep)
    }


def _fno_matrix(
    arm: Mapping[str, Any],
    order: Sequence[int],
    keep: set[int],
) -> tuple[np.ndarray, np.ndarray]:
    kept = [instance_id for instance_id in order if instance_id in keep]
    n_seeds = len(arm["per_seed"])
    hats = np.empty((n_seeds, len(kept)), dtype=np.float64)
    truth = np.empty(len(kept), dtype=np.float64)
    # per_seed slices do not store hats. Rebuild from the source lists stored
    # indirectly: the caller must have hats. This function expects arm to
    # carry them. They are recovered from per-seed method blocks only as
    # aggregates. Store hats on the arm in _fno_from_stage_d instead.
    if "hat_matrix" not in arm:
        raise ValueError("FNO arm is missing hat_matrix")
    matrix = arm["hat_matrix"]
    nu_by_id = arm["nu_by_id"]
    for column, instance_id in enumerate(kept):
        hats[:, column] = np.asarray(matrix[str(instance_id)], dtype=np.float64)
        truth[column] = float(nu_by_id[str(instance_id)])
    return hats, truth


def _baseline_vector(
    instances: Sequence[Mapping[str, Any]],
    order: Sequence[int],
    keep: set[int],
) -> tuple[np.ndarray, np.ndarray]:
    by_id = {int(row["instance_id"]): row for row in instances}
    kept = [instance_id for instance_id in order if instance_id in keep]
    rel = np.asarray([float(by_id[instance_id]["rel_error"]) for instance_id in kept], dtype=np.float64)
    fail = np.asarray([bool(by_id[instance_id]["failure"]) for instance_id in kept], dtype=bool)
    return rel, fail


def _fno_hat_lists(fno: Mapping[str, Any], arm: str) -> dict[int, list[float]]:
    matrix = fno["arms"][arm]["hat_matrix"]
    return {int(instance_id): [float(value) for value in hats] for instance_id, hats in matrix.items()}


def _assert_closed_form_matches(rows: Sequence[Mapping[str, Any]], stage_d: Mapping[str, Any]) -> None:
    published = {int(row["instance_id"]): float(row["ls_nu_hat"]) for row in stage_d["instances"]}
    for row in rows:
        gap = abs(float(row["nu_hat"]) - published[int(row["instance_id"])])
        if gap != 0.0:
            raise ValueError(
                f"closed form for instance {row['instance_id']} moved by {gap} "
                "relative to the committed Stage D table"
            )


def _assert_fno_matches_published(fno: Mapping[str, Any], stage_d: Mapping[str, Any]) -> None:
    for arm, key in (("data_only", "data_only"), ("hybrid_1e-2", "hybrid_1e-2")):
        got = float(fno["arms"][arm]["slices"]["hard_ood"]["mean_rel_error"])
        published = float(
            stage_d["arms"][key]["objectives"]["0.0"]["slices"]["hard_ood"]["mean_rel_error"]["mean"]
        )
        if abs(got - published) > 1e-12:
            raise ValueError(f"{arm} hard_ood mean {got} does not match published {published}")


def _require_stage_d_alignment(trajectories: Sequence[Trajectory], stage_d: Mapping[str, Any]) -> None:
    got = {int(item.instance_id) for item in trajectories}
    published = {int(row["instance_id"]) for row in stage_d["instances"]}
    if got != published:
        raise ValueError("test instance ids do not match the Stage D score file")


def _condition_by_id(protocol: Mapping[str, Any], condition_id: str) -> Mapping[str, Any]:
    for condition in protocol["conditions"]:
        if str(condition["id"]) == condition_id:
            return condition
    raise ValueError(f"unknown Stage E condition {condition_id}")


def _require_sha(path: Path, expected: str) -> None:
    actual = file_sha256(path)
    if actual != expected:
        raise ValueError(f"{path.as_posix()} sha256 is {actual}, protocol records {expected}")


def _json_ready(value: Any, path: str = "") -> Any:
    if isinstance(value, dict):
        return {str(key): _json_ready(item, f"{path}.{key}") for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [_json_ready(item, f"{path}[{index}]") for index, item in enumerate(value)]
    if isinstance(value, np.ndarray):
        return _json_ready(value.tolist(), path)
    if isinstance(value, np.floating):
        return _json_ready(float(value), path)
    if isinstance(value, np.integer):
        return int(value)
    if isinstance(value, float):
        if not math.isfinite(value):
            raise ValueError(f"score record has a non-finite number at {path}")
        return value
    return value


def _write_json(payload: Mapping[str, Any], path: Path) -> None:
    destination = Path(path)
    destination.parent.mkdir(parents=True, exist_ok=True)
    destination.write_text(json.dumps(_json_ready(payload), indent=2) + "\n", encoding="utf-8")


def _fmt(value: object) -> str:
    if value is None:
        return "null"
    number = float(value)  # type: ignore[arg-type]
    if math.isnan(number):
        return "inapplicable"
    return format(number, ".16g")


def _fmt_optional(value: object) -> str:
    if value is None:
        return "null"
    return _fmt(value)


def _fmt_pub(center: object, spread: object) -> str:
    if center is None:
        return "inapplicable"
    return _fmt_pm(center, spread)


def _fmt_pm(center: object, spread: object) -> str:
    if center is None:
        return "null"
    if spread is None or (isinstance(spread, float) and math.isnan(spread)):
        return _fmt(center)
    return f"{_fmt(center)} ± {_fmt(spread)}"
