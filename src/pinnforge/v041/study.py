"""Generate, train, calibrate, and score the preregistered v0.4.1 study."""

from __future__ import annotations

import json
import math
import platform
from pathlib import Path
from typing import Any

import numpy as np

from pinnforge.operator.warmstart import (
    abstain_threshold,
    calibrate_groups,
    conformal_quantile,
    conformal_score,
    interval,
    predicted_group,
)
from pinnforge.reference.spectral2d.dataset import generate_problem, load_protocol, protocol_sha256
from pinnforge.v04.baselines import closed_form_ls, relative_error, weak_form
from pinnforge.v04.inverse import fno_inverse, solver_fit
from pinnforge.v04.sensors import SensorMask, add_noise, mask_from_protocol
from pinnforge.v04.train import cut_windows, field_stats, train_fno, viscosity_stats
from pinnforge.v041.inverse import bracket_search

PROTOCOL = Path("docs/v041/protocol.json")
ARTIFACT = Path("artifacts/v041_pilot")
MANIFEST_DIR = Path("docs/v041")
FAILURE_REL = 0.5


def run_generate(output: Path | None = None, *, tiny: bool = False) -> dict[str, Any]:
    protocol = load_protocol(PROTOCOL)
    root = ARTIFACT if output is None else output
    kwargs: dict[str, Any] = {"protocol_path": PROTOCOL}
    if tiny:
        kwargs.update(n=8, n_train=2, n_val=2, n_test=1, t_final=0.05, dt=0.01, save_dt=0.025)
    records = {}
    for problem in ("burgers2d", "ns2d"):
        records[problem] = generate_problem(problem, protocol, root, **kwargs)
    combined = {
        "format": "pinnforge.v041_pilot_index.v1",
        "protocol_sha256": protocol_sha256(PROTOCOL),
        "problems": records,
    }
    MANIFEST_DIR.mkdir(parents=True, exist_ok=True)
    (MANIFEST_DIR / "pilot_manifest.json").write_text(json.dumps(combined, indent=2) + "\n", encoding="utf-8")
    return combined


def run_train(output: Path | None = None, *, seeds: list[int] | None = None) -> dict[str, Any]:
    protocol = load_protocol(PROTOCOL)
    root = ARTIFACT if output is None else output
    use_seeds = protocol["seeds"] if seeds is None else seeds
    summary: dict[str, Any] = {"seeds": use_seeds, "runs": []}
    for problem in ("burgers2d", "ns2d"):
        fields, nu, manifest = _load_problem(root, problem)
        splits = manifest["splits"]
        u_mean, u_std = field_stats(fields[splits["train"]])
        nu_mean, nu_std = viscosity_stats(np.asarray(manifest["nu"])[splits["train"]])
        spec = protocol["fno"]
        train_w = cut_windows(
            fields,
            np.asarray(manifest["nu"]),
            input_frames=spec["input_frames"],
            output_frames=spec["output_frames"],
            stride=spec["stride"],
            instance_ids=splits["train"],
            forcings=manifest["forcing"],
        )
        cal_w = cut_windows(
            fields,
            np.asarray(manifest["nu"]),
            input_frames=spec["input_frames"],
            output_frames=spec["output_frames"],
            stride=spec["stride"],
            instance_ids=splits["val"],
            forcings=manifest["forcing"],
        )
        for loss in ("data", "hybrid"):
            for seed in use_seeds:
                dest = root / "train" / problem / loss / f"seed_{seed}"
                record = train_fno(
                    train_w,
                    cal_w,
                    problem=problem,
                    protocol=protocol,
                    seed=int(seed),
                    loss=loss,
                    output=dest,
                    length=float(manifest["length"]),
                    dt=float(protocol["time"]["save_dt"]),
                    u_mean=u_mean,
                    u_std=u_std,
                    nu_mean=nu_mean,
                    nu_std=nu_std,
                )
                summary["runs"].append(record)
    (MANIFEST_DIR / "train_summary.json").write_text(json.dumps(summary, indent=2) + "\n", encoding="utf-8")
    return summary


def run_calibrate_and_test(output: Path | None = None, *, seeds: list[int] | None = None) -> dict[str, Any]:
    protocol = load_protocol(PROTOCOL)
    root = ARTIFACT if output is None else output
    use_seeds = protocol["seeds"] if seeds is None else seeds
    mask = mask_from_protocol(protocol)
    validation: dict[str, Any] = {"protocol_sha256": protocol_sha256(PROTOCOL), "problems": {}}
    scores: dict[str, Any] = {
        "format": "pinnforge.v041_scores.v1",
        "protocol_sha256": protocol_sha256(PROTOCOL),
        "python": platform.python_version(),
        "numpy": np.__version__,
        "problems": {},
    }
    try:
        import torch

        scores["torch"] = torch.__version__
    except ImportError:
        scores["torch"] = None
    for problem in ("burgers2d", "ns2d"):
        fields, nu, manifest = _load_problem(root, problem)
        threshold = float(manifest["hard_ood"]["threshold_nu"])
        cal_ids = manifest["splits"]["val"]
        test_ids = manifest["splits"]["test"]
        fno_q = _fno_quantiles(protocol, root, problem, fields, nu, manifest, mask, cal_ids, use_seeds)
        n_local = _select_n_local(
            protocol, root, problem, fields, nu, manifest, mask, cal_ids, use_seeds[0], fno_q["hybrid"]
        )
        cal_warm = _score_split(
            protocol,
            root,
            problem,
            fields,
            nu,
            manifest,
            mask,
            cal_ids,
            use_seeds,
            noise=0.0,
            n_local=n_local,
            fno_quantiles=fno_q,
            methods=("fno_data", "fno_hybrid", "solver_sensor_ic", "warm_data", "warm_hybrid"),
        )
        calibration = _calibrate(protocol, cal_warm["warm_hybrid"], threshold)
        validation["problems"][problem] = {
            "threshold_nu": threshold,
            "n_local": n_local,
            "fno_bracket_quantiles": fno_q,
            "calibration": calibration,
            "calibration_preview": {key: cal_warm[key]["summary"] for key in cal_warm},
        }
        problem_scores: dict[str, Any] = {
            "threshold_nu": threshold,
            "n_local": n_local,
            "fno_bracket_quantiles": fno_q,
            "noise": {},
        }
        for fraction in protocol["noise"]["fractions"]:
            scored = _score_split(
                protocol,
                root,
                problem,
                fields,
                nu,
                manifest,
                mask,
                test_ids,
                use_seeds,
                noise=float(fraction),
                n_local=n_local,
                fno_quantiles=fno_q,
                calibration=calibration,
                threshold=threshold,
            )
            problem_scores["noise"][str(fraction)] = scored
        problem_scores["bars"] = _evaluate_bars(protocol, problem_scores["noise"]["0.0"], problem)
        scores["problems"][problem] = problem_scores
    (MANIFEST_DIR / "validation.json").write_text(json.dumps(validation, indent=2) + "\n", encoding="utf-8")
    (MANIFEST_DIR / "scores.json").write_text(json.dumps(scores, indent=2) + "\n", encoding="utf-8")
    return scores


def _load_problem(root: Path, problem: str) -> tuple[np.ndarray, np.ndarray, dict[str, Any]]:
    manifest = json.loads((root / f"{problem}_manifest.json").read_text(encoding="utf-8"))
    fields = np.load(root / manifest["files"]["fields"])
    nu = np.load(root / manifest["files"]["nu"])
    return fields, nu, manifest


def _model_path(root: Path, problem: str, loss: str, seed: int) -> Path:
    return root / "train" / problem / loss / f"seed_{seed}" / "checkpoint.pt"


def _fno_quantiles(
    protocol: dict[str, Any],
    root: Path,
    problem: str,
    fields: np.ndarray,
    nu: np.ndarray,
    manifest: dict[str, Any],
    mask: SensorMask,
    instance_ids: list[int],
    seeds: list[int],
) -> dict[str, float]:
    alpha = float(protocol["methods"]["warm_start"]["bracket"]["alpha"])
    out: dict[str, float] = {}
    for loss in ("data", "hybrid"):
        scores: list[float] = []
        for instance_id in instance_ids:
            hats = []
            for seed in seeds:
                guess = fno_inverse(
                    fields[instance_id],
                    mask,
                    model_path=_model_path(root, problem, loss, int(seed)),
                    protocol=protocol,
                    problem=problem,
                )["nu_hat"]
                hats.append(guess)
            nu_hat = float(np.mean(hats))
            if nu_hat > 0 and float(nu[instance_id]) > 0:
                scores.append(abs(math.log(float(nu[instance_id]) / nu_hat)))
        out[loss] = float(conformal_quantile(scores, alpha))
    return out


def _select_n_local(
    protocol: dict[str, Any],
    root: Path,
    problem: str,
    fields: np.ndarray,
    nu: np.ndarray,
    manifest: dict[str, Any],
    mask: SensorMask,
    instance_ids: list[int],
    seed: int,
    q_log: float,
) -> int:
    candidates = list(protocol["methods"]["warm_start"]["local_search"]["n_local_candidates"])
    threshold = float(manifest["hard_ood"]["threshold_nu"])
    hard_ids = [i for i in instance_ids if float(nu[i]) <= threshold] or list(instance_ids)
    table: list[dict[str, float]] = []
    for n_local in candidates:
        rows = []
        for instance_id in hard_ids:
            field = fields[instance_id]
            guess = fno_inverse(
                field,
                mask,
                model_path=_model_path(root, problem, "hybrid", seed),
                protocol=protocol,
                problem=problem,
            )
            cold = solver_fit(
                field,
                mask,
                problem=problem,
                protocol=protocol,
                forcing=manifest["forcing"][instance_id],
                initial="sensor",
            )
            warm = bracket_search(
                field,
                mask,
                first_guess=guess["nu_hat"],
                q_log=q_log,
                n_local=int(n_local),
                problem=problem,
                protocol=protocol,
                forcing=manifest["forcing"][instance_id],
            )
            rows.append(
                {
                    "rel": relative_error(warm["nu_hat"], float(nu[instance_id])),
                    "time_ratio": (warm["seconds"] + guess["seconds"]) / max(cold["seconds"], 1e-9),
                }
            )
        table.append(
            {
                "n_local": float(n_local),
                "error": float(np.mean([row["rel"] for row in rows])),
                "time_ratio": float(np.mean([row["time_ratio"] for row in rows])),
            }
        )
    feasible = [row for row in table if row["time_ratio"] <= 0.2]
    if not feasible:
        chosen = min(table, key=lambda row: row["time_ratio"])
    else:
        best = min(item["error"] for item in feasible)
        near = [item for item in feasible if item["error"] <= 1.02 * best]
        chosen = min(near, key=lambda row: row["n_local"])
    return int(chosen["n_local"])


def _score_split(
    protocol: dict[str, Any],
    root: Path,
    problem: str,
    fields: np.ndarray,
    nu: np.ndarray,
    manifest: dict[str, Any],
    mask: SensorMask,
    instance_ids: list[int],
    seeds: list[int],
    *,
    noise: float,
    n_local: int,
    fno_quantiles: dict[str, float],
    calibration: dict[str, Any] | None = None,
    threshold: float | None = None,
    methods: tuple[str, ...] | None = None,
) -> dict[str, Any]:
    wanted = set(
        methods
        or (
            "closed_form_ls",
            "weak_form",
            "fno_data",
            "fno_hybrid",
            "solver_sensor_ic",
            "solver_oracle_ic",
            "warm_data",
            "warm_hybrid",
        )
    )
    cut = float(manifest["hard_ood"]["threshold_nu"] if threshold is None else threshold)
    collected: dict[str, list[dict[str, Any]]] = {name: [] for name in wanted}
    rng = np.random.default_rng(int(protocol["noise"]["noise_seed"]) + int(1000 * noise))
    for instance_id in instance_ids:
        field = add_noise(fields[instance_id], noise, rng)
        forcing = manifest["forcing"][instance_id]
        true = float(nu[instance_id])
        if "closed_form_ls" in wanted:
            closed = closed_form_ls(
                field,
                mask,
                problem=problem,
                dt=float(protocol["time"]["save_dt"]),
                forcing=forcing,
                length=float(manifest["length"]),
            )
            collected["closed_form_ls"].append(_row(instance_id, true, closed, 0.0, cut))
        if "weak_form" in wanted:
            weak = weak_form(
                field,
                mask,
                problem=problem,
                dt=float(protocol["time"]["save_dt"]),
                forcing=forcing,
                length=float(manifest["length"]),
            )
            collected["weak_form"].append(_row(instance_id, true, weak, 0.0, cut))
        cold = None
        if "solver_sensor_ic" in wanted or "warm_data" in wanted or "warm_hybrid" in wanted:
            cold = solver_fit(field, mask, problem=problem, protocol=protocol, forcing=forcing, initial="sensor")
        if "solver_sensor_ic" in wanted and cold is not None:
            collected["solver_sensor_ic"].append(_row(instance_id, true, cold["nu_hat"], cold["seconds"], cut))
        if "solver_oracle_ic" in wanted:
            oracle = solver_fit(field, mask, problem=problem, protocol=protocol, forcing=forcing, initial="oracle")
            collected["solver_oracle_ic"].append(_row(instance_id, true, oracle["nu_hat"], oracle["seconds"], cut))
        for loss, key, warm_key in (("data", "fno_data", "warm_data"), ("hybrid", "fno_hybrid", "warm_hybrid")):
            if key not in wanted and warm_key not in wanted:
                continue
            seed_rows = []
            warm_rows = []
            for seed in seeds:
                guess = fno_inverse(
                    field,
                    mask,
                    model_path=_model_path(root, problem, loss, int(seed)),
                    protocol=protocol,
                    problem=problem,
                )
                seed_rows.append(_row(instance_id, true, guess["nu_hat"], guess["seconds"], cut))
                if warm_key in wanted:
                    warm = bracket_search(
                        field,
                        mask,
                        first_guess=guess["nu_hat"],
                        q_log=float(fno_quantiles[loss]),
                        n_local=n_local,
                        problem=problem,
                        protocol=protocol,
                        forcing=forcing,
                    )
                    extra = _row(instance_id, true, warm["nu_hat"], warm["seconds"] + guess["seconds"], cut)
                    extra["bracket_log_half_width"] = warm["bracket_log_half_width"]
                    extra["nu_guess"] = guess["nu_hat"]
                    extra["bracket_low"] = warm["bracket_low"]
                    extra["bracket_high"] = warm["bracket_high"]
                    if cold is not None:
                        extra["cold_seconds"] = cold["seconds"]
                        extra["cold_rel"] = relative_error(cold["nu_hat"], true)
                    if calibration is not None:
                        extra.update(_apply_calibration(extra, calibration, cut))
                    warm_rows.append(extra)
            if key in wanted:
                collected[key].append(_reduce_seeds(seed_rows))
            if warm_key in wanted:
                collected[warm_key].append(_reduce_seeds(warm_rows))
    return {name: {"instances": rows, "summary": _summarize(rows)} for name, rows in collected.items()}


def _row(instance_id: int, true: float, nu_hat: float, seconds: float, threshold: float) -> dict[str, Any]:
    rel = relative_error(nu_hat, true)
    return {
        "instance_id": instance_id,
        "nu": true,
        "nu_hat": nu_hat,
        "rel_error": rel if math.isfinite(rel) else None,
        "failure": (not math.isfinite(nu_hat)) or nu_hat <= 0 or (math.isfinite(rel) and rel > FAILURE_REL),
        "seconds": seconds,
        "hard_ood": true <= threshold,
    }


def _reduce_seeds(rows: list[dict[str, Any]]) -> dict[str, Any]:
    rels = [float(row["rel_error"]) for row in rows if row["rel_error"] is not None]
    payload = dict(rows[0])
    payload["rel_error"] = float(np.mean(rels)) if rels else None
    payload["rel_error_std"] = float(np.std(rels, ddof=1)) if len(rels) > 1 else 0.0
    payload["seconds"] = float(np.mean([row["seconds"] for row in rows]))
    payload["failure"] = float(np.mean([1.0 if row["failure"] else 0.0 for row in rows]))
    if "covered" in rows[0]:
        payload["covered"] = float(np.mean([1.0 if row.get("covered") else 0.0 for row in rows]))
        payload["abstain"] = float(np.mean([1.0 if row.get("abstain") else 0.0 for row in rows]))
    if "bracket_log_half_width" in rows[0]:
        payload["bracket_log_half_width"] = float(np.mean([row["bracket_log_half_width"] for row in rows]))
        payload["cold_seconds"] = float(np.mean([row["cold_seconds"] for row in rows]))
        payload["cold_rel"] = float(np.mean([row["cold_rel"] for row in rows]))
    return payload


def _summarize(rows: list[dict[str, Any]]) -> dict[str, Any]:
    def slice_rows(name: str) -> list[dict[str, Any]]:
        if name == "full_test":
            return rows
        if name == "hard_ood":
            return [row for row in rows if row["hard_ood"]]
        return [row for row in rows if not row["hard_ood"]]

    summary = {}
    for name in ("full_test", "hard_ood", "complement"):
        group = slice_rows(name)
        rels = [float(row["rel_error"]) for row in group if row["rel_error"] is not None]
        summary[name] = {
            "n": len(group),
            "mean_rel_error": float(np.mean(rels)) if rels else None,
            "n_failures": float(sum(row["failure"] or 0 for row in group)),
            "mean_seconds": float(np.mean([row["seconds"] for row in group])) if group else None,
        }
        if group and "covered" in group[0]:
            summary[name]["coverage"] = float(np.mean([row.get("covered") or 0.0 for row in group]))
            summary[name]["abstain_rate"] = float(np.mean([row.get("abstain") or 0.0 for row in group]))
        if group and "cold_seconds" in group[0]:
            summary[name]["time_ratio"] = float(
                np.mean([row["seconds"] / max(row["cold_seconds"], 1e-9) for row in group])
            )
            summary[name]["cold_mean_rel_error"] = float(np.mean([row["cold_rel"] for row in group]))
    return summary


def _calibrate(protocol: dict[str, Any], warm: dict[str, Any], threshold: float) -> dict[str, Any]:
    alpha = float(protocol["methods"]["warm_start"]["intervals"]["alpha"])
    rows = []
    stats = []
    for item in warm["instances"]:
        if item["rel_error"] is None or not math.isfinite(item["nu_hat"]) or item["nu_hat"] <= 0:
            continue
        rows.append(
            {
                "score": conformal_score(item["nu"], item["nu_hat"], 1.0),
                "group": predicted_group(item["nu_hat"], threshold),
            }
        )
        stats.append(float(item.get("bracket_log_half_width") or 0.0))
    groups = calibrate_groups(rows, alpha, min_group=int(protocol["methods"]["warm_start"]["intervals"]["min_group"]))
    return {
        "groups": groups,
        "abstain_threshold": abstain_threshold(stats, float(protocol["methods"]["warm_start"]["abstain"]["rate"]))
        if stats
        else math.inf,
        "scale": 1.0,
    }


def _apply_calibration(row: dict[str, Any], calibration: dict[str, Any], threshold: float) -> dict[str, Any]:
    group = predicted_group(row["nu_hat"], threshold)
    quantile = float(calibration["groups"]["groups"][group]["quantile"])
    low, high = interval(row["nu_hat"], float(calibration["scale"]), quantile)
    return {
        "interval_low": low,
        "interval_high": high,
        "covered": bool(low <= row["nu"] <= high),
        "abstain": bool(float(row.get("bracket_log_half_width") or 0.0) > float(calibration["abstain_threshold"])),
        "scale": float(calibration["scale"]),
        "group": group,
    }


def _evaluate_bars(protocol: dict[str, Any], clean: dict[str, Any], problem: str) -> dict[str, Any]:
    bars = protocol["pass_bars"]
    warm = clean["warm_hybrid"]["summary"]
    id_err = warm["complement"]["mean_rel_error"]
    hard = warm["hard_ood"]["mean_rel_error"]
    comp = warm["complement"]["mean_rel_error"]
    gap = None if not hard or not comp or comp == 0 else hard / comp
    time_ratio = warm["hard_ood"].get("time_ratio")
    cold = warm["hard_ood"].get("cold_mean_rel_error")
    coverage = warm["full_test"].get("coverage")
    checks = {
        "in_distribution_error": {
            "value": id_err,
            "limit": bars["in_distribution_error"]["limit"],
            "pass": id_err is not None and id_err <= bars["in_distribution_error"]["limit"],
        },
        "hard_ood_gap": {
            "value": gap,
            "limit": bars["hard_ood_gap"]["limit_ratio"],
            "pass": gap is not None and gap <= bars["hard_ood_gap"]["limit_ratio"],
        },
        "warm_vs_cold_error": {
            "value": hard,
            "limit": None if cold is None else bars["warm_vs_cold"]["error_factor"] * cold,
            "pass": hard is not None and cold is not None and hard <= bars["warm_vs_cold"]["error_factor"] * cold,
        },
        "warm_vs_cold_time": {
            "value": time_ratio,
            "limit": bars["warm_vs_cold"]["time_limit_ratio"],
            "pass": time_ratio is not None and time_ratio <= bars["warm_vs_cold"]["time_limit_ratio"],
        },
        "coverage": {
            "value": coverage,
            "range": bars["coverage"]["range"],
            "pass": coverage is not None and bars["coverage"]["range"][0] <= coverage <= bars["coverage"]["range"][1],
        },
    }
    return {
        "problem": problem,
        "method": "warm_hybrid",
        "checks": checks,
        "pass": all(item["pass"] for item in checks.values()),
    }
