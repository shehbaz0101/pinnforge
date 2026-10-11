"""Generate, denoise-train, and score the preregistered v0.5 study."""

from __future__ import annotations

import json
import platform
from pathlib import Path
from typing import Any

import numpy as np

from pinnforge.reference.spectral2d.dataset import generate_problem, load_protocol, protocol_sha256
from pinnforge.v04.sensors import mask_from_protocol
from pinnforge.v05.checkpoint import (
    CaseLog,
    case_worker,
    default_workers,
    feature_key,
    replay_noise,
    run_cases,
    single_thread,
)
from pinnforge.v05.library import is_primary_instance, terms_for, true_coefficients, true_support
from pinnforge.v05.wsindy import (
    choose_threshold,
    coeff_rel_error,
    exact_support,
    instance_truth,
    refit_support,
    stlsq,
    support_of,
)

PROTOCOL = Path("docs/v05/protocol.json")
ARTIFACT = Path("artifacts/v05_pilot")
MANIFEST_DIR = Path("docs/v05")


def run_generate(output: Path | None = None, *, tiny: bool = False) -> dict[str, Any]:
    protocol = load_protocol(PROTOCOL)
    root = ARTIFACT if output is None else output
    kwargs: dict[str, Any] = {"protocol_path": PROTOCOL}
    if tiny:
        kwargs.update(n=8, n_train=2, n_val=2, n_test=1, t_final=0.05, dt=0.01, save_dt=0.025)
    records = {}
    for problem in ("burgers2d", "ns2d"):
        manifest_path = root / f"{problem}_manifest.json"
        fields_path = root / f"{problem}_fields.npy"
        if not tiny and manifest_path.is_file() and fields_path.is_file():
            records[problem] = json.loads(manifest_path.read_text(encoding="utf-8"))
            print(f"reused {problem} pilot", flush=True)
            continue
        records[problem] = generate_problem(problem, protocol, root, **kwargs)
    combined = {
        "format": "pinnforge.v05_pilot_index.v1",
        "protocol_sha256": protocol_sha256(PROTOCOL),
        "problems": records,
    }
    MANIFEST_DIR.mkdir(parents=True, exist_ok=True)
    (MANIFEST_DIR / "pilot_manifest.json").write_text(json.dumps(combined, indent=2) + "\n", encoding="utf-8")
    return combined


def run_train(
    output: Path | None = None,
    *,
    seeds: list[int] | None = None,
    workers: int | None = None,
) -> dict[str, Any]:
    protocol = load_protocol(PROTOCOL)
    root = ARTIFACT if output is None else output
    use_seeds = protocol["seeds"] if seeds is None else seeds
    n_workers = default_workers() if workers is None else max(1, int(workers))
    jobs = []
    reused: list[dict[str, Any]] = []
    for problem in ("burgers2d", "ns2d"):
        for seed in use_seeds:
            dest = root / "train" / problem / f"seed_{seed}"
            record_path = dest / "train.json"
            if (dest / "checkpoint.pt").is_file() and record_path.is_file():
                reused.append(json.loads(record_path.read_text(encoding="utf-8")))
                print(f"reused train {problem} seed {seed}", flush=True)
                continue
            jobs.append({"root": str(root), "problem": problem, "seed": int(seed), "dest": str(dest)})
    runs = list(reused)
    if jobs and n_workers <= 1:
        runs.extend(_train_one(job) for job in jobs)
    elif jobs:
        import multiprocessing as mp
        from concurrent.futures import ProcessPoolExecutor, as_completed

        single_thread()
        ctx = mp.get_context("spawn")
        with ProcessPoolExecutor(max_workers=n_workers, mp_context=ctx, initializer=single_thread) as pool:
            futures = [pool.submit(_train_one, job) for job in jobs]
            for future in as_completed(futures):
                runs.append(future.result())
    runs.sort(key=lambda row: (row["problem"], int(row["seed"])))
    summary: dict[str, Any] = {"seeds": use_seeds, "runs": runs}
    MANIFEST_DIR.mkdir(parents=True, exist_ok=True)
    (MANIFEST_DIR / "train_summary.json").write_text(json.dumps(summary, indent=2) + "\n", encoding="utf-8")
    return summary


def _train_one(job: dict[str, Any]) -> dict[str, Any]:
    single_thread()
    from pinnforge.v05.denoise import train_denoiser

    protocol = load_protocol(PROTOCOL)
    root = Path(job["root"])
    problem = str(job["problem"])
    manifest = json.loads((root / f"{problem}_manifest.json").read_text(encoding="utf-8"))
    fields = np.load(root / manifest["files"]["fields"])
    mask = mask_from_protocol(protocol, n=int(manifest["n"]))
    record = train_denoiser(
        fields,
        train_ids=list(manifest["splits"]["train"]),
        val_ids=list(manifest["splits"]["val"]),
        protocol=protocol,
        mask=mask,
        seed=int(job["seed"]),
        output=Path(job["dest"]),
        noisy_field=replay_noise,
    )
    record["problem"] = problem
    return record


def run_score(
    output: Path | None = None,
    *,
    seeds: list[int] | None = None,
    workers: int | None = None,
    checkpoint: Path | None = None,
) -> dict[str, Any]:
    protocol = load_protocol(PROTOCOL)
    root = ARTIFACT if output is None else output
    use_seeds = protocol["seeds"] if seeds is None else seeds
    n_workers = default_workers() if workers is None else max(1, int(workers))
    digest = protocol_sha256(PROTOCOL)
    log = CaseLog(checkpoint or (MANIFEST_DIR / "score_cases.jsonl"), digest)
    jobs = _feature_jobs(root, protocol, use_seeds)
    print(f"extracting {len(jobs)} library cases", flush=True)
    payloads = run_cases(jobs, log=log, workers=n_workers, worker=case_worker)
    validation: dict[str, Any] = {"protocol_sha256": digest, "problems": {}}
    scores: dict[str, Any] = {
        "format": "pinnforge.v05_scores.v1",
        "protocol_sha256": digest,
        "python": platform.python_version(),
        "numpy": np.__version__,
        "pysr": protocol["methods"]["pysr"],
        "problems": {},
        "overall": {},
    }
    try:
        from importlib.metadata import version

        scores["torch"] = version("torch")
    except Exception:
        scores["torch"] = None
    methods = list(protocol["pass_bars"]["methods"])
    noises = [float(item) for item in protocol["noise"]["fractions"]]
    bar_noise = float(protocol["pass_bars"]["noise"])
    min_rate = float(protocol["pass_bars"]["min_seed_success_rate"])
    overall_pass = True
    for problem in ("burgers2d", "ns2d"):
        manifest = json.loads((root / f"{problem}_manifest.json").read_text(encoding="utf-8"))
        names = terms_for(problem)
        val_ids = list(manifest["splits"]["val"])
        test_ids = list(manifest["splits"]["test"])
        forcings = list(manifest["forcing"])
        problem_val: dict[str, Any] = {}
        by_noise: dict[str, Any] = {}
        method_bar: dict[str, Any] = {}
        for method in methods:
            problem_val[method] = {}
            for noise in noises:
                chosen: dict[str, float] = {}
                seed_success: dict[str, bool] = {}
                supports: dict[str, list[str]] = {}
                instance_rates: list[float] = []
                coeff_errors: list[float] = []
                fno_rels: list[float] = []
                decaying_rates: list[float] = []
                for seed in use_seeds:
                    val_systems = []
                    val_truth = true_support(problem, protocol["problems"][problem]["primary_family"])
                    for instance_id in val_ids:
                        if not is_primary_instance(protocol, problem, instance_id, forcings[instance_id]):
                            continue
                        row = payloads[feature_key(problem, noise, int(seed), method, instance_id)]
                        val_systems.append((np.asarray(row["design"]), np.asarray(row["target"])))
                    threshold = choose_threshold(
                        val_systems,
                        names,
                        val_truth,
                        list(protocol["methods"]["wsindy"]["stlsq"]["threshold_candidates"]),
                        max_iter=int(protocol["methods"]["wsindy"]["stlsq"]["max_iter"]),
                        column_normalize=bool(protocol["methods"]["wsindy"]["stlsq"]["column_normalize"]),
                    )
                    chosen[str(seed)] = threshold
                    stacked_x, stacked_y = _stack(
                        payloads,
                        problem,
                        noise,
                        int(seed),
                        method,
                        [item for item in test_ids if is_primary_instance(protocol, problem, item, forcings[item])],
                    )
                    stacked_coef = stlsq(
                        stacked_x,
                        stacked_y,
                        threshold,
                        max_iter=int(protocol["methods"]["wsindy"]["stlsq"]["max_iter"]),
                        column_normalize=bool(protocol["methods"]["wsindy"]["stlsq"]["column_normalize"]),
                    )
                    recovered = support_of(stacked_coef, names)
                    supports[str(seed)] = recovered
                    success = exact_support(recovered, val_truth)
                    seed_success[str(seed)] = success
                    exact_primary = []
                    exact_decaying = []
                    for instance_id in test_ids:
                        row = payloads[feature_key(problem, noise, int(seed), method, instance_id)]
                        coef = stlsq(
                            np.asarray(row["design"]),
                            np.asarray(row["target"]),
                            threshold,
                            max_iter=int(protocol["methods"]["wsindy"]["stlsq"]["max_iter"]),
                            column_normalize=bool(protocol["methods"]["wsindy"]["stlsq"]["column_normalize"]),
                        )
                        recovered_i = support_of(coef, names)
                        truth_i = instance_truth(problem, instance_id, forcings[instance_id])
                        hit = exact_support(recovered_i, truth_i)
                        if is_primary_instance(protocol, problem, instance_id, forcings[instance_id]):
                            exact_primary.append(float(hit))
                            hat = refit_support(np.asarray(row["design"]), np.asarray(row["target"]), names, recovered_i)
                            truth_c = true_coefficients(problem, float(row["nu"]), protocol["problems"][problem]["primary_family"])
                            coeff_errors.append(coeff_rel_error(hat, truth_c, truth_i))
                        elif problem == "ns2d":
                            exact_decaying.append(float(hit))
                        if row.get("fno_rel_l2") is not None:
                            fno_rels.append(float(row["fno_rel_l2"]))
                    instance_rates.append(float(np.mean(exact_primary)) if exact_primary else float("nan"))
                    if exact_decaying:
                        decaying_rates.append(float(np.mean(exact_decaying)))
                rate = float(np.mean([1.0 if seed_success[str(seed)] else 0.0 for seed in use_seeds]))
                block = {
                    "exact_support_rate": rate,
                    "instance_exact_rate": float(np.mean(instance_rates)),
                    "coeff_rel_error": float(np.mean(coeff_errors)) if coeff_errors else float("nan"),
                    "thresholds": chosen,
                    "supports": supports,
                    "seed_success": seed_success,
                    "fno_rel_l2": float(np.mean(fno_rels)) if fno_rels else None,
                }
                if decaying_rates:
                    block["decaying_instance_exact_rate"] = float(np.mean(decaying_rates))
                noise_key = _noise_key(noise)
                by_noise.setdefault(noise_key, {})[method] = block
                problem_val[method][_noise_key(noise)] = {"threshold_by_seed": chosen}
        bar_methods = {}
        problem_pass = False
        for method in methods:
            rate = float(by_noise[_noise_key(bar_noise)][method]["exact_support_rate"])
            method_ok = rate >= min_rate
            bar_methods[method] = {"exact_support_rate": rate, "pass": method_ok}
            problem_pass = problem_pass or method_ok
        method_bar = bar_methods
        overall_pass = overall_pass and problem_pass
        scores["problems"][problem] = {
            "bars": {
                "pass": problem_pass,
                "noise": bar_noise,
                "min_seed_success_rate": min_rate,
                "methods": method_bar,
            },
            "by_noise": by_noise,
        }
        validation["problems"][problem] = problem_val
    scores["overall"] = {"pass": overall_pass}
    MANIFEST_DIR.mkdir(parents=True, exist_ok=True)
    (MANIFEST_DIR / "validation.json").write_text(json.dumps(validation, indent=2) + "\n", encoding="utf-8")
    (MANIFEST_DIR / "scores.json").write_text(json.dumps(scores, indent=2) + "\n", encoding="utf-8")
    return scores


def _feature_jobs(root: Path, protocol: dict[str, Any], seeds: list[int]) -> list[dict[str, Any]]:
    jobs: list[dict[str, Any]] = []
    methods = list(protocol["pass_bars"]["methods"])
    noises = [float(item) for item in protocol["noise"]["fractions"]]
    for problem in ("burgers2d", "ns2d"):
        manifest = json.loads((root / f"{problem}_manifest.json").read_text(encoding="utf-8"))
        instance_ids = list(manifest["splits"]["val"]) + list(manifest["splits"]["test"])
        for seed in seeds:
            for noise in noises:
                for method in methods:
                    for instance_id in instance_ids:
                        jobs.append(
                            {
                                "key": feature_key(problem, noise, int(seed), method, int(instance_id)),
                                "kind": "feature",
                                "protocol": str(PROTOCOL),
                                "root": str(root),
                                "problem": problem,
                                "noise": float(noise),
                                "seed": int(seed),
                                "method": method,
                                "instance_id": int(instance_id),
                            }
                        )
    return jobs


def _stack(
    payloads: dict[str, dict[str, Any]],
    problem: str,
    noise: float,
    seed: int,
    method: str,
    instance_ids: list[int],
) -> tuple[np.ndarray, np.ndarray]:
    blocks_x = []
    blocks_y = []
    for instance_id in instance_ids:
        row = payloads[feature_key(problem, noise, seed, method, instance_id)]
        blocks_x.append(np.asarray(row["design"], dtype=np.float64))
        blocks_y.append(np.asarray(row["target"], dtype=np.float64))
    return np.concatenate(blocks_x, axis=0), np.concatenate(blocks_y, axis=0)


def _noise_key(noise: float) -> str:
    return f"{float(noise):.2f}"
