"""Score the frozen warm-start protocol on the test split and render the note.

Calibration quantiles and abstain thresholds come from the validation
record named in the protocol. Nothing is refit on test. Every number in
the markdown is formatted from the scores JSON with ``.16g``.
"""

from __future__ import annotations

import json
import math
import platform
from collections.abc import Mapping, Sequence
from pathlib import Path
from typing import Any

import numpy as np

from pinnforge.operator.inverse import FAILURE_RELATIVE_ERROR
from pinnforge.operator.strong_baselines import file_sha256
from pinnforge.operator.warmstart import instance_scale, interval, mean_std, predicted_group
from pinnforge.operator.warmstart_run import (
    ARMS,
    SCALE_FLOOR,
    cold_view,
    ensemble_key,
    load_warmstart_protocol,
    run_test_rows,
    seed_keys,
    warm_view,
)

SCORES_FORMAT = "pinnforge.warmstart_scores.v1"
SLICES = ("full_test", "hard_ood", "complement")
ARM_LABELS = {"data_only": "FNO data loss", "hybrid_1e-2": "FNO hybrid loss"}


def scored_views(
    rows: Sequence[Mapping[str, Any]],
    key: str,
    cap: int,
    method: Mapping[str, Any],
    threshold: float,
) -> list[dict[str, Any]]:
    """Warm views with the validation interval and abstain flag attached."""

    groups = method["calibration"]["groups"]
    output = []
    for row in rows:
        view = warm_view(row, key, cap)
        scale = instance_scale(
            str(method["scale"]),
            log_standard_error=view["log_standard_error"],
            ensemble_log_std=view["ensemble_log_std"],
            floor=SCALE_FLOOR,
        )
        group = predicted_group(view["nu_hat"], threshold)
        quantile = float(groups[group]["quantile"])
        low, high = interval(view["nu_hat"], scale, quantile)
        view.update(
            {
                "group": group,
                "interval_low": low,
                "interval_high": high,
                "log_half_width": scale * quantile,
                "covered": bool(low <= view["nu"] <= high),
                "abstain": bool(view[str(method["abstain_statistic"])] > float(method["abstain_threshold"])),
                "first_guess_rel_error": abs(view["nu_guess"] - view["nu"]) / view["nu"],
            }
        )
        output.append(view)
    return output


def slice_metrics(views: Sequence[Mapping[str, Any]]) -> dict[str, Any]:
    """Error, time, coverage, and abstention on one slice of one method."""

    errors = np.asarray([view["rel_error"] for view in views], dtype=np.float64)
    accepted = [view for view in views if not view["abstain"]]
    rejected = [view for view in views if view["abstain"]]
    return {
        "n_instances": len(views),
        "mean_rel_error": float(np.mean(errors)),
        "median_rel_error": float(np.median(errors)),
        "max_rel_error": float(np.max(errors)),
        "n_failures": int(np.sum(errors > FAILURE_RELATIVE_ERROR)),
        "mean_seconds": float(np.mean([view["seconds"] for view in views])),
        "mean_first_guess_seconds": float(np.mean([view["first_guess_seconds"] for view in views])),
        "mean_n_evaluations": float(np.mean([view["n_evaluations"] for view in views])),
        "first_guess_mean_rel_error": float(np.mean([view["first_guess_rel_error"] for view in views])),
        "n_at_trust_bound": int(sum(bool(view["at_trust_bound"]) for view in views)),
        "coverage": float(np.mean([view["covered"] for view in views])),
        "mean_log_half_width": float(np.mean([view["log_half_width"] for view in views])),
        "abstain_rate": len(rejected) / len(views),
        "n_abstain": len(rejected),
        "accepted_mean_rel_error": float(np.mean([view["rel_error"] for view in accepted])) if accepted else math.nan,
        "abstained_mean_rel_error": float(np.mean([view["rel_error"] for view in rejected])) if rejected else math.nan,
    }


def cold_metrics(views: Sequence[Mapping[str, Any]]) -> dict[str, Any]:
    errors = np.asarray([view["rel_error"] for view in views], dtype=np.float64)
    return {
        "n_instances": len(views),
        "mean_rel_error": float(np.mean(errors)),
        "median_rel_error": float(np.median(errors)),
        "max_rel_error": float(np.max(errors)),
        "n_failures": int(np.sum(errors > FAILURE_RELATIVE_ERROR)),
        "mean_seconds": float(np.mean([view["seconds"] for view in views])),
        "mean_n_evaluations": float(np.mean([view["n_evaluations"] for view in views])),
    }


def slice_of(views: Sequence[Mapping[str, Any]], name: str, threshold: float) -> list[Mapping[str, Any]]:
    if name == "full_test":
        return list(views)
    hard = name == "hard_ood"
    return [view for view in views if (float(view["nu"]) <= threshold) == hard]


def seed_aggregate(per_seed: Sequence[Mapping[str, Any]]) -> dict[str, Any]:
    """Mean and sample standard deviation across seeds for every metric."""

    payload: dict[str, Any] = {"n_seeds": len(per_seed)}
    for name in per_seed[0]:
        values = [float(item[name]) for item in per_seed]
        if any(math.isnan(value) for value in values):
            payload[name] = {"mean": math.nan, "std": math.nan, "values": values}
            continue
        center, spread = mean_std(values)
        payload[name] = {"mean": center, "std": spread, "values": values}
    return payload


def paired_with_cold(
    warm_by_seed: Sequence[Sequence[Mapping[str, Any]]],
    cold: Sequence[Mapping[str, Any]],
) -> dict[str, Any]:
    """Per-seed paired differences against the cold search on the same instances.

    A negative error delta means the warm start has lower relative error.
    """

    cold_error = np.asarray([view["rel_error"] for view in cold], dtype=np.float64)
    cold_seconds = np.asarray([view["seconds"] for view in cold], dtype=np.float64)
    per_seed = []
    for views in warm_by_seed:
        if [int(view["instance_id"]) for view in views] != [int(view["instance_id"]) for view in cold]:
            raise ValueError("paired views are not aligned")
        error = np.asarray([view["rel_error"] for view in views], dtype=np.float64)
        seconds = np.asarray([view["seconds"] for view in views], dtype=np.float64)
        delta = error - cold_error
        estimate_gap = np.asarray(
            [abs(math.log(float(a["nu_hat"]) / float(b["nu_hat"]))) for a, b in zip(views, cold, strict=True)]
        )
        per_seed.append(
            {
                "mean_rel_error_delta": float(np.mean(delta)),
                "n_warm_lower": float(np.sum(delta < 0.0)),
                "n_warm_higher": float(np.sum(delta > 0.0)),
                "n_tie": float(np.sum(delta == 0.0)),
                "time_ratio_of_means": float(np.mean(seconds) / np.mean(cold_seconds)),
                "mean_instance_time_ratio": float(np.mean(seconds / cold_seconds)),
                "max_instance_time_ratio": float(np.max(seconds / cold_seconds)),
                "mean_abs_log_gap_to_cold": float(np.mean(estimate_gap)),
                "max_abs_log_gap_to_cold": float(np.max(estimate_gap)),
            }
        )
    if len(per_seed) == 1:
        return {"n_seeds": 1, **{name: {"mean": value, "std": math.nan, "values": [value]} for name, value in per_seed[0].items()}}
    return seed_aggregate(per_seed)


def family_block(
    rows: Sequence[Mapping[str, Any]],
    family: str,
    cap: int,
    calibration: Mapping[str, Any],
    threshold: float,
    slices: Sequence[str],
) -> dict[str, Any]:
    """Per-arm seed aggregates and the ensemble, with paired cold comparisons."""

    cold_all = [cold_view(row) for row in rows]
    block: dict[str, Any] = {"arms": {}, "ensembles": {}}
    instances: dict[str, Any] = {}
    for arm in ARMS:
        per_seed_views = []
        for key in seed_keys(family, arm):
            views = scored_views(rows, key, cap, calibration[key], threshold)
            per_seed_views.append(views)
            instances[key] = _instance_arrays(views)
        arm_block: dict[str, Any] = {"slices": {}, "paired_with_cold": {}}
        for name in slices:
            parts = [slice_of(views, name, threshold) for views in per_seed_views]
            arm_block["slices"][name] = seed_aggregate([slice_metrics(part) for part in parts])
            arm_block["paired_with_cold"][name] = paired_with_cold(parts, slice_of(cold_all, name, threshold))
        block["arms"][arm] = arm_block
        key = ensemble_key(family, arm)
        views = scored_views(rows, key, cap, calibration[key], threshold)
        instances[key] = _instance_arrays(views)
        ensemble: dict[str, Any] = {"slices": {}, "paired_with_cold": {}}
        for name in slices:
            part = slice_of(views, name, threshold)
            ensemble["slices"][name] = slice_metrics(part)
            ensemble["paired_with_cold"][name] = paired_with_cold([part], slice_of(cold_all, name, threshold))
        block["ensembles"][arm] = ensemble
    block["cold"] = {name: cold_metrics(slice_of(cold_all, name, threshold)) for name in slices}
    block["instances"] = instances
    return block


def pass_bars(clean: Mapping[str, Any], protocol: Mapping[str, Any]) -> dict[str, Any]:
    bars = protocol["pass_bars"]
    primary = clean["arms"]["hybrid_1e-2"]
    hard = primary["slices"]["hard_ood"]
    cold = clean["cold"]["hard_ood"]
    error_value = float(hard["mean_rel_error"]["mean"])
    error_limit = float(bars["error"]["limit"])
    time_value = float(hard["mean_seconds"]["mean"]) / float(cold["mean_seconds"])
    time_limit = float(bars["time"]["limit_ratio"])
    coverage_value = float(primary["slices"]["full_test"]["coverage"]["mean"])
    low, high = (float(value) for value in bars["coverage"]["range"])
    results = {
        "error": {
            "statement": bars["error"]["statement"],
            "value": error_value,
            "limit": error_limit,
            "per_seed": hard["mean_rel_error"]["values"],
            "passed": bool(error_value <= error_limit),
        },
        "time": {
            "statement": bars["time"]["statement"],
            "warm_mean_seconds": float(hard["mean_seconds"]["mean"]),
            "cold_mean_seconds": float(cold["mean_seconds"]),
            "value": time_value,
            "limit": time_limit,
            "per_seed": [float(value) / float(cold["mean_seconds"]) for value in hard["mean_seconds"]["values"]],
            "passed": bool(time_value <= time_limit),
        },
        "coverage": {
            "statement": bars["coverage"]["statement"],
            "value": coverage_value,
            "low": low,
            "high": high,
            "per_seed": primary["slices"]["full_test"]["coverage"]["values"],
            "passed": bool(low <= coverage_value <= high),
        },
    }
    results["all_passed"] = bool(all(item["passed"] for item in results.values() if isinstance(item, dict)))
    return results


def cold_reproduction(rows: Sequence[Mapping[str, Any]], strong_scores: Mapping[str, Any]) -> dict[str, Any]:
    """Re-timed cold estimates against the committed strong-baseline estimates."""

    published = {int(item["instance_id"]): float(item["nu_hat"]) for item in strong_scores["methods"]["solver_sensor_ic"]["instances"]}
    gaps = [abs(float(row["cold"]["nu_hat"]) - published[int(row["instance_id"])]) for row in rows]
    return {
        "n_instances": len(gaps),
        "n_identical": int(sum(gap == 0.0 for gap in gaps)),
        "max_abs_difference": float(max(gaps)),
        "published_seconds_per_instance_mean": float(strong_scores["timing"]["methods"]["solver_sensor_ic"]["seconds_per_instance_mean"]),
    }


def run_warmstart_test(
    *,
    protocol_path: Path,
    pilot_dir: Path,
    manifest_path: Path,
    rows_dir: Path,
    output: Path,
    report_path: Path,
    workers: int,
) -> dict[str, Any]:
    protocol = load_warmstart_protocol(protocol_path)
    digest = file_sha256(protocol_path)
    if int(workers) != int(protocol["workers"]):
        raise ValueError(f"the protocol fixes workers = {protocol['workers']}")
    conditions = ["clean", *protocol["noise"]["fractions"]]
    rows = run_test_rows(
        protocol=protocol,
        protocol_sha256=digest,
        pilot_dir=pilot_dir,
        manifest_path=manifest_path,
        rows_dir=rows_dir,
        workers=int(workers),
        conditions=conditions,
    )
    payload = build_scores(protocol, digest, Path(protocol_path), rows)
    _write_json(payload, output)
    render_report_file(output, report_path)
    print(json.dumps(_json_ready(payload["pass_bars"]), indent=1), flush=True)
    return payload


def build_scores(
    protocol: Mapping[str, Any],
    digest: str,
    protocol_path: Path,
    rows: Mapping[str, Sequence[Mapping[str, Any]]],
) -> dict[str, Any]:
    record = json.loads(Path(protocol["validation_record"]).read_text(encoding="utf-8"))
    strong_scores = json.loads(Path(protocol["strong_baselines_scores"]).read_text(encoding="utf-8"))
    stage_f = json.loads(Path(protocol["stage_f"]["published_scores"]).read_text(encoding="utf-8"))
    threshold = float(protocol["hard_ood"]["threshold_nu"])
    cap = int(protocol["refine"]["max_evaluations"])
    full_calibration = record["families"]["full"]["methods"]
    restricted_calibration = record["families"]["restricted"]["methods"]
    clean_rows = list(rows["clean"])
    clean = family_block(clean_rows, "full", cap, full_calibration, threshold, SLICES)
    restricted = family_block(clean_rows, "restricted", cap, restricted_calibration, threshold, SLICES)
    noise: dict[str, Any] = {}
    published_noise = strong_scores["stress"]["solver_sensor_ic"]
    for condition, fraction in protocol["noise"]["fractions"].items():
        block = family_block(list(rows[condition]), "full", cap, full_calibration, threshold, ("hard_ood",))
        block.pop("instances")
        block["noise_fraction"] = float(fraction)
        reference = published_noise.get(condition.replace("noise", "noise", 1) + "_sensors32_bursts4")
        if reference and reference.get("applicable"):
            block["published_cold_mean_rel_error"] = float(reference["summary"]["mean_rel_error"])
        noise[condition] = block
    published_stage_f = {
        arm: float(stage_f["inverse"]["arms"][arm]["slices"]["ood"]["mean_rel_error"]["mean"]) for arm in ARMS
    }
    payload = {
        "format": SCORES_FORMAT,
        "protocol": protocol_path.as_posix(),
        "protocol_sha256": digest,
        "validation_record": protocol["validation_record"],
        "validation_record_sha256": protocol["validation_record_sha256"],
        "split": "test",
        "device": "cpu",
        "workers": int(protocol["workers"]),
        "python": platform.python_version(),
        "numpy": np.__version__,
        "torch": _torch_version(),
        "threshold_nu": threshold,
        "cap": cap,
        "nominal_coverage": 1.0 - float(protocol["intervals"]["alpha"]),
        "nominal_coverage_percent": 100.0 * (1.0 - float(protocol["intervals"]["alpha"])),
        "abstain_rate_target": float(protocol["abstain"]["rate"]),
        "n_seeds": len(protocol["seeds"]),
        "first_guess_grid_points": int(protocol["first_guess"]["n_points"]),
        "failure_relative_error": FAILURE_RELATIVE_ERROR,
        "published_cold": {
            "hard_ood_mean_rel_error": float(strong_scores["methods"]["solver_sensor_ic"]["slices"]["hard_ood"]["mean_rel_error"]),
            "seconds_per_instance_mean": float(strong_scores["timing"]["methods"]["solver_sensor_ic"]["seconds_per_instance_mean"]),
            "n_evaluations_mean": float(strong_scores["timing"]["methods"]["solver_sensor_ic"]["n_evaluations_mean"]),
            "oracle_hard_ood_mean_rel_error": float(strong_scores["methods"]["solver_oracle_ic"]["slices"]["hard_ood"]["mean_rel_error"]),
            "fno_hybrid_hard_ood_mean_rel_error": float(strong_scores["fno"]["arms"]["hybrid_1e-2"]["slices"]["hard_ood"]["mean_rel_error"]),
            "fno_data_hard_ood_mean_rel_error": float(strong_scores["fno"]["arms"]["data_only"]["slices"]["hard_ood"]["mean_rel_error"]),
        },
        "cold_reproduction": cold_reproduction(clean_rows, strong_scores),
        "clean": clean,
        "pass_bars": pass_bars(clean, protocol),
        "noise": noise,
        "stage_f": {
            "family": "restricted",
            "calibration_split": "validation instances with nu > threshold_nu",
            "published_restricted_fno_ood_mean_rel_error": published_stage_f,
            **{name: value for name, value in restricted.items() if name != "instances"},
        },
        "restricted_instances": restricted["instances"],
        "validation": {
            "selected_cap": int(record["cap_selection"]["selected_cap"]),
            "cap_table": record["cap_selection"]["table"],
            "cold_mean_seconds": record["cap_selection"]["cold_mean_seconds"],
            "choices": {
                family: {kind: {"scale": item["scale"], "abstain_statistic": item["abstain_statistic"]} for kind, item in record["families"][family]["choices"].items()}
                for family in ("full", "restricted")
            },
        },
        "instance_ids": [int(row["instance_id"]) for row in clean_rows],
        "instance_nu": [float(row["nu"]) for row in clean_rows],
        "cold_nu_hat": [float(row["cold"]["nu_hat"]) for row in clean_rows],
        "cold_seconds": [float(row["cold"]["seconds"]) for row in clean_rows],
        "not_claimed": list(protocol["not_claimed"]),
        "limitations": list(protocol["limitations"]),
    }
    return payload


def render_report_file(scores_path: Path, report_path: Path) -> None:
    scores = json.loads(Path(scores_path).read_text(encoding="utf-8"))
    destination = Path(report_path)
    destination.parent.mkdir(parents=True, exist_ok=True)
    destination.write_text(render_report(scores), encoding="utf-8")


def render_report(scores: Mapping[str, Any]) -> str:
    """Markdown whose numbers are all ``.16g`` leaves of ``scores``."""

    clean = scores["clean"]
    bars = scores["pass_bars"]
    cold = clean["cold"]
    verdict = "passes every preregistered bar" if bars["all_passed"] else "does not pass every preregistered bar"
    lines = [
        "# Warm-started solver fit with calibrated intervals",
        "",
        "Research note. It does not change any Stage D, E, F, or strong-baseline number. "
        "The contract is `docs/v03/warmstart_protocol.json`; it was committed before the test split was scored, "
        "and its SHA-256 is stored as `protocol_sha256` in `docs/v03/warmstart_scores.json`. "
        "Calibration and every hyperparameter come from `docs/v03/warmstart_validation.json`, which reads only the validation split.",
        "",
        "## Question",
        "",
        "Can an FNO first guess followed by a short solver-in-the-loop refinement reach the accuracy of the "
        "cold solver search on the frozen `hard_ood` slice at a fraction of its wall clock, with intervals "
        "whose empirical coverage is near nominal?",
        "",
        "## Answer",
        "",
        f"The primary method (FNO hybrid first guess, then at most {_fmt(scores['cap'])} solver evaluations, "
        f"mean over {_fmt(scores['n_seeds'])} seeds) {verdict}.",
        "",
        "| bar | measured | limit | result |",
        "|---|---|---|---|",
        f"| `hard_ood` mean relative error | {_fmt(bars['error']['value'])} | <= {_fmt(bars['error']['limit'])} | {_pass(bars['error']['passed'])} |",
        f"| `hard_ood` mean wall clock / cold mean wall clock, same run | {_fmt(bars['time']['value'])} | <= {_fmt(bars['time']['limit'])} | {_pass(bars['time']['passed'])} |",
        f"| full-test coverage of the nominal {_fmt(scores['nominal_coverage'])} interval | {_fmt(bars['coverage']['value'])} | [{_fmt(bars['coverage']['low'])}, {_fmt(bars['coverage']['high'])}] | {_pass(bars['coverage']['passed'])} |",
        "",
        f"Cold search re-timed in this run: `hard_ood` mean {_fmt(cold['hard_ood']['mean_seconds'])} s per instance "
        f"and mean relative error {_fmt(cold['hard_ood']['mean_rel_error'])}. Warm start mean {_fmt(bars['time']['warm_mean_seconds'])} s. "
        f"The re-run cold estimates are identical to the committed strong-baseline estimates on "
        f"{_fmt(scores['cold_reproduction']['n_identical'])} of {_fmt(scores['cold_reproduction']['n_instances'])} test instances "
        f"(largest absolute difference {_fmt(scores['cold_reproduction']['max_abs_difference'])}).",
        "",
        "## Headline table on `hard_ood`",
        "",
        "Seed-dependent rows are mean ± sample standard deviation over seeds. The cold search and the ensembles are one run each. "
        "Seconds include the FNO first guess and every refinement solve; model loading is excluded. "
        "Coverage is for the nominal interval. Abstain rate and accepted error use the validation threshold.",
        "",
        "| method | mean rel error | failures | mean seconds | time / cold | coverage | abstain rate | error on accepted |",
        "|---|---|---|---|---|---|---|---|",
        f"| solver, sensor-lift initial condition, cold search | {_fmt(cold['hard_ood']['mean_rel_error'])} | {_fmt(cold['hard_ood']['n_failures'])} | {_fmt(cold['hard_ood']['mean_seconds'])} | 1 | n/a | n/a | n/a |",
    ]
    for arm in ("hybrid_1e-2", "data_only"):
        block = clean["arms"][arm]
        hard = block["slices"]["hard_ood"]
        paired = block["paired_with_cold"]["hard_ood"]
        lines.append(
            f"| warm start, {ARM_LABELS[arm]} first guess | {_pm(hard['mean_rel_error'])} | {_pm(hard['n_failures'])} | "
            f"{_pm(hard['mean_seconds'])} | {_pm(paired['time_ratio_of_means'])} | {_pm(hard['coverage'])} | "
            f"{_pm(hard['abstain_rate'])} | {_pm(hard['accepted_mean_rel_error'])} |"
        )
    for arm in ("hybrid_1e-2", "data_only"):
        block = clean["ensembles"][arm]
        hard = block["slices"]["hard_ood"]
        paired = block["paired_with_cold"]["hard_ood"]
        lines.append(
            f"| warm start, median of the {ARM_LABELS[arm]} seeds | {_fmt(hard['mean_rel_error'])} | {_fmt(hard['n_failures'])} | "
            f"{_fmt(hard['mean_seconds'])} | {_fmt(paired['time_ratio_of_means']['mean'])} | {_fmt(hard['coverage'])} | "
            f"{_fmt(hard['abstain_rate'])} | {_fmt_opt(hard['accepted_mean_rel_error'])} |"
        )
    for arm in ("hybrid_1e-2", "data_only"):
        hard = clean["arms"][arm]["slices"]["hard_ood"]
        lines.append(
            f"| FNO first guess alone, {ARM_LABELS[arm]} | {_pm(hard['first_guess_mean_rel_error'])} | n/a | "
            f"{_pm(hard['mean_first_guess_seconds'])} | n/a | n/a | n/a | n/a |"
        )
    published = scores["published_cold"]
    lines += [
        "",
        f"Published context from `docs/research/strong_baselines_scores.json`: cold sensor-lift solver {_fmt(published['hard_ood_mean_rel_error'])} "
        f"at {_fmt(published['seconds_per_instance_mean'])} s per instance, oracle initial condition {_fmt(published['oracle_hard_ood_mean_rel_error'])}, "
        f"FNO hybrid grid inverse {_fmt(published['fno_hybrid_hard_ood_mean_rel_error'])}, FNO data grid inverse {_fmt(published['fno_data_hard_ood_mean_rel_error'])}. "
        "The first-guess rows above use a coarser grid than the published FNO inverse, so they are not the same estimator.",
        "",
        "## Coverage and abstention by slice",
        "",
        "| method | slice | n | mean rel error | coverage | mean log half-width | abstain rate | error on accepted | error on abstained |",
        "|---|---|---|---|---|---|---|---|---|",
    ]
    for arm in ("hybrid_1e-2", "data_only"):
        for name in SLICES:
            item = clean["arms"][arm]["slices"][name]
            lines.append(
                f"| warm, {ARM_LABELS[arm]} | `{name}` | {_fmt(item['n_instances']['mean'])} | {_pm(item['mean_rel_error'])} | {_pm(item['coverage'])} | "
                f"{_pm(item['mean_log_half_width'])} | {_pm(item['abstain_rate'])} | {_pm(item['accepted_mean_rel_error'])} | {_pm(item['abstained_mean_rel_error'])} |"
            )
    for arm in ("hybrid_1e-2", "data_only"):
        for name in SLICES:
            item = clean["ensembles"][arm]["slices"][name]
            lines.append(
                f"| warm, median of {ARM_LABELS[arm]} seeds | `{name}` | {_fmt(item['n_instances'])} | {_fmt(item['mean_rel_error'])} | {_fmt(item['coverage'])} | "
                f"{_fmt(item['mean_log_half_width'])} | {_fmt(item['abstain_rate'])} | {_fmt_opt(item['accepted_mean_rel_error'])} | {_fmt_opt(item['abstained_mean_rel_error'])} |"
            )
    lines += [
        "",
        "## Paired comparison with the cold search",
        "",
        "Same instances, same worker, same run. Delta is warm minus cold relative error; negative means the warm start has lower error. "
        "Counts are instances per seed, averaged over seeds.",
        "",
        "| method | slice | mean delta | warm lower | warm higher | tie | time ratio of means | max instance time ratio | mean abs log gap to cold estimate |",
        "|---|---|---|---|---|---|---|---|---|",
    ]
    for arm in ("hybrid_1e-2", "data_only"):
        for name in SLICES:
            item = clean["arms"][arm]["paired_with_cold"][name]
            lines.append(
                f"| warm, {ARM_LABELS[arm]} | `{name}` | {_pm(item['mean_rel_error_delta'])} | {_pm(item['n_warm_lower'])} | {_pm(item['n_warm_higher'])} | "
                f"{_pm(item['n_tie'])} | {_pm(item['time_ratio_of_means'])} | {_pm(item['max_instance_time_ratio'])} | {_pm(item['mean_abs_log_gap_to_cold'])} |"
            )
    lines += [
        "",
        "Cold search by slice in this run:",
        "",
        "| slice | n | mean rel error | failures | mean seconds | mean solver evaluations |",
        "|---|---|---|---|---|---|",
    ]
    for name in SLICES:
        item = cold[name]
        lines.append(
            f"| `{name}` | {_fmt(item['n_instances'])} | {_fmt(item['mean_rel_error'])} | {_fmt(item['n_failures'])} | {_fmt(item['mean_seconds'])} | {_fmt(item['mean_n_evaluations'])} |"
        )
    lines += _noise_section(scores)
    lines += _stage_f_section(scores)
    lines += _validation_section(scores)
    lines += [
        "",
        "## What is NOT claimed",
        "",
    ]
    lines += [f"- {item}" for item in scores["not_claimed"]]
    lines += ["", "## Limitations", ""]
    lines += [f"- {item}" for item in scores["limitations"]]
    lines += [
        "",
        "## Reproduce",
        "",
        "Checkpoints are retrained with the frozen Stage B, C, and F commands listed in the protocol; they are not in git. "
        "Their SHA-256 values are in the protocol. Then:",
        "",
        "```bash",
        "python -m pinnforge.operator warmstart validate",
        "python -m pinnforge.operator warmstart test",
        "python -m pinnforge.operator warmstart report",
        "python -m pinnforge.claims check",
        "```",
        "",
        "Python, NumPy, and torch versions are stored as `python`, `numpy`, and `torch` in the scores JSON.",
        "",
    ]
    return "\n".join(lines)


def _noise_section(scores: Mapping[str, Any]) -> list[str]:
    clean = scores["clean"]
    lines = [
        "",
        "## Secondary: Stage E observation noise on `hard_ood`",
        "",
        "Additive Gaussian noise with the Stage E seeds, scaled by the clean-field standard deviation. "
        "Calibration and abstain thresholds are the clean validation ones; nothing is refit for noise. "
        "The cold search is re-run on the same noisy fields.",
        "",
        "| noise fraction | cold mean rel error | published cold | warm, FNO hybrid | warm, FNO data | warm time / cold | coverage, FNO hybrid | abstain rate, FNO hybrid |",
        "|---|---|---|---|---|---|---|---|",
    ]
    hard = clean["arms"]["hybrid_1e-2"]["slices"]["hard_ood"]
    lines.append(
        f"| 0 | {_fmt(clean['cold']['hard_ood']['mean_rel_error'])} | {_fmt(scores['published_cold']['hard_ood_mean_rel_error'])} | {_pm(hard['mean_rel_error'])} | "
        f"{_pm(clean['arms']['data_only']['slices']['hard_ood']['mean_rel_error'])} | {_pm(clean['arms']['hybrid_1e-2']['paired_with_cold']['hard_ood']['time_ratio_of_means'])} | "
        f"{_pm(hard['coverage'])} | {_pm(hard['abstain_rate'])} |"
    )
    for block in scores["noise"].values():
        hybrid = block["arms"]["hybrid_1e-2"]["slices"]["hard_ood"]
        data = block["arms"]["data_only"]["slices"]["hard_ood"]
        published = block.get("published_cold_mean_rel_error")
        lines.append(
            f"| {_fmt(block['noise_fraction'])} | {_fmt(block['cold']['hard_ood']['mean_rel_error'])} | {_fmt_opt(published)} | {_pm(hybrid['mean_rel_error'])} | "
            f"{_pm(data['mean_rel_error'])} | {_pm(block['arms']['hybrid_1e-2']['paired_with_cold']['hard_ood']['time_ratio_of_means'])} | "
            f"{_pm(hybrid['coverage'])} | {_pm(hybrid['abstain_rate'])} |"
        )
    lines.extend(["", _noise_reading(scores)])
    return lines


def _noise_reading(scores: Mapping[str, Any]) -> str:
    low = scores["pass_bars"]["coverage"]["low"]
    worse = []
    undercovered = []
    for block in scores["noise"].values():
        hybrid = block["arms"]["hybrid_1e-2"]["slices"]["hard_ood"]
        if hybrid["mean_rel_error"]["mean"] > block["cold"]["hard_ood"]["mean_rel_error"]:
            worse.append(_fmt(block["noise_fraction"]))
        if hybrid["coverage"]["mean"] < low:
            undercovered.append(f"{_fmt(block['noise_fraction'])} (abstain rate {_fmt(hybrid['abstain_rate']['mean'])})")
    sentences = []
    if worse:
        sentences.append(
            "The FNO hybrid warm start has higher `hard_ood` mean relative error than the cold search on the same noisy fields "
            f"at {_fractions(worse)}; the time ratio stays near its clean value because the solve count is capped."
        )
    else:
        sentences.append("The FNO hybrid warm start does not have higher `hard_ood` mean relative error than the cold search at any noise fraction.")
    if undercovered:
        sentences.append(
            f"Its interval coverage falls below {_fmt(low)} at {_fractions(undercovered)}."
        )
    return " ".join(sentences)


def _fractions(items: Sequence[str]) -> str:
    label = "noise fraction" if len(items) == 1 else "noise fractions"
    if len(items) <= 1:
        return f"{label} {''.join(items)}"
    return f"{label} {', '.join(items[:-1])} and {items[-1]}"


def _stage_f_section(scores: Mapping[str, Any]) -> list[str]:
    block = scores["stage_f"]
    lines = [
        "",
        "## Secondary: Stage F restricted-range operators",
        "",
        "First guesses come from operators trained only on `nu > threshold_nu`, so `hard_ood` is outside their training range. "
        "Intervals and abstain thresholds are calibrated on validation instances above the threshold only. "
        "The refinement and the cold search are unchanged.",
        "",
        "| method | slice | first guess mean rel error | warm mean rel error | failures | time / cold | coverage | abstain rate | error on accepted |",
        "|---|---|---|---|---|---|---|---|---|",
    ]
    for arm in ARMS:
        for name in ("hard_ood", "complement"):
            item = block["arms"][arm]["slices"][name]
            paired = block["arms"][arm]["paired_with_cold"][name]
            lines.append(
                f"| warm, restricted {ARM_LABELS[arm]} | `{name}` | {_pm(item['first_guess_mean_rel_error'])} | {_pm(item['mean_rel_error'])} | {_pm(item['n_failures'])} | "
                f"{_pm(paired['time_ratio_of_means'])} | {_pm(item['coverage'])} | {_pm(item['abstain_rate'])} | {_pm(item['accepted_mean_rel_error'])} |"
            )
    for arm in ARMS:
        for name in ("hard_ood", "complement"):
            item = block["ensembles"][arm]["slices"][name]
            paired = block["ensembles"][arm]["paired_with_cold"][name]
            lines.append(
                f"| warm, median of restricted {ARM_LABELS[arm]} seeds | `{name}` | {_fmt(item['first_guess_mean_rel_error'])} | {_fmt(item['mean_rel_error'])} | {_fmt(item['n_failures'])} | "
                f"{_fmt(paired['time_ratio_of_means']['mean'])} | {_fmt(item['coverage'])} | {_fmt(item['abstain_rate'])} | {_fmt_opt(item['accepted_mean_rel_error'])} |"
            )
    reference = block["published_restricted_fno_ood_mean_rel_error"]
    lines += [
        "",
        f"Published Stage F restricted FNO grid inverse on the out-of-range slice: data loss {_fmt(reference['data_only'])}, hybrid loss {_fmt(reference['hybrid_1e-2'])}. "
        f"Cold search on the same `hard_ood` instances: {_fmt(block['cold']['hard_ood']['mean_rel_error'])}.",
        "",
        _stage_f_reading(scores),
    ]
    return lines


def _stage_f_reading(scores: Mapping[str, Any]) -> str:
    bars = scores["pass_bars"]
    sentences = []
    for arm in ARMS:
        item = scores["stage_f"]["arms"][arm]["slices"]["hard_ood"]
        error = "above" if item["mean_rel_error"]["mean"] > bars["error"]["limit"] else "at or below"
        coverage = "below" if item["coverage"]["mean"] < bars["coverage"]["low"] else "not below"
        sentences.append(
            f"With the restricted {ARM_LABELS[arm]} first guess, the `hard_ood` warm-start error is {error} the error bar limit "
            f"and its coverage is {coverage} {_fmt(bars['coverage']['low'])}."
        )
    sentences.append("These rows are secondary and were not held to the bars.")
    return " ".join(sentences)


def _validation_section(scores: Mapping[str, Any]) -> list[str]:
    validation = scores["validation"]
    lines = [
        "",
        "## Validation-only choices",
        "",
        f"Solver-evaluation cap chosen on validation `hard_ood` for the primary method: {_fmt(validation['selected_cap'])}. "
        f"Validation cold mean wall clock {_fmt(validation['cold_mean_seconds'])} s.",
        "",
        "| cap | validation mean rel error | validation time / cold |",
        "|---|---|---|",
    ]
    for item in validation["cap_table"]:
        lines.append(f"| {_fmt(item['cap'])} | {_fmt(item['mean_rel_error'])} | {_fmt(item['time_ratio'])} |")
    lines += ["", "| family | method kind | interval scale | abstain statistic |", "|---|---|---|---|"]
    for family, kinds in validation["choices"].items():
        for kind, item in kinds.items():
            lines.append(f"| {family} | {kind} | `{item['scale']}` | `{item['abstain_statistic']}` |")
    return lines


def _instance_arrays(views: Sequence[Mapping[str, Any]]) -> dict[str, Any]:
    return {
        field: [view[field] for view in views]
        for field in ("nu_guess", "nu_hat", "seconds", "interval_low", "interval_high", "abstain")
    }


def _pass(flag: object) -> str:
    return "pass" if flag else "fail"


def _fmt(value: object) -> str:
    if value is None:
        return "n/a"
    number = float(value)  # type: ignore[arg-type]
    if math.isnan(number):
        return "n/a"
    return format(number, ".16g")


def _fmt_opt(value: object) -> str:
    return _fmt(value)


def _pm(item: Mapping[str, Any]) -> str:
    center = item.get("mean")
    spread = item.get("std")
    if center is None or (isinstance(center, float) and math.isnan(center)):
        return "n/a"
    if spread is None or (isinstance(spread, float) and math.isnan(spread)):
        return _fmt(center)
    return f"{_fmt(center)} ± {_fmt(spread)}"


def _torch_version() -> str:
    import torch

    return str(torch.__version__)


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
        return None
    return value


def _write_json(payload: Mapping[str, Any], path: Path) -> None:
    destination = Path(path)
    destination.parent.mkdir(parents=True, exist_ok=True)
    destination.write_text(json.dumps(_json_ready(payload), indent=1) + "\n", encoding="utf-8")
