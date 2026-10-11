"""Generate and score the preregistered v0.5.1 sensor-density study."""

from __future__ import annotations

import json
import platform
from pathlib import Path
from typing import Any

import numpy as np

from pinnforge.reference.spectral2d.dataset import generate_problem, load_protocol, protocol_sha256
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
from pinnforge.v051.checkpoint import CaseLog, case_worker, default_workers, feature_key, run_cases
from pinnforge.v051.masks import lattices

PROTOCOL = Path("docs/v051/protocol.json")
ARTIFACT = Path("artifacts/v051_pilot")
MANIFEST_DIR = Path("docs/v051")


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
        "format": "pinnforge.v051_pilot_index.v1",
        "protocol_sha256": protocol_sha256(PROTOCOL),
        "problems": records,
    }
    MANIFEST_DIR.mkdir(parents=True, exist_ok=True)
    (MANIFEST_DIR / "pilot_manifest.json").write_text(json.dumps(combined, indent=2) + "\n", encoding="utf-8")
    return combined


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
    log = CaseLog(checkpoint or (ARTIFACT / "score_cases.jsonl"), digest)
    jobs = _feature_jobs(root, protocol, use_seeds)
    print(f"extracting {len(jobs)} library cases", flush=True)
    payloads = run_cases(jobs, log=log, workers=n_workers, worker=case_worker)
    validation: dict[str, Any] = {"protocol_sha256": digest, "problems": {}}
    scores: dict[str, Any] = {
        "format": "pinnforge.v051_scores.v1",
        "protocol_sha256": digest,
        "python": platform.python_version(),
        "numpy": np.__version__,
        "pysr": protocol["methods"]["pysr"],
        "wsindy_fno": protocol["methods"]["wsindy_fno"],
        "problems": {},
    }
    try:
        from importlib.metadata import version

        scores["torch"] = version("torch")
    except Exception:
        scores["torch"] = None
    noises = [float(item) for item in protocol["noise"]["fractions"]]
    bar_noise = float(protocol["reported"]["noise"])
    min_rate = float(protocol["reported"]["min_seed_success_rate"])
    control_noise = float(protocol["positive_control"]["noise"])
    control_lattice = str(protocol["positive_control"]["lattice"])
    control_rate = float(protocol["positive_control"]["min_seed_success_rate"])
    for problem in ("burgers2d", "ns2d"):
        manifest = json.loads((root / f"{problem}_manifest.json").read_text(encoding="utf-8"))
        names = terms_for(problem)
        val_ids = list(manifest["splits"]["val"])
        test_ids = list(manifest["splits"]["test"])
        forcings = list(manifest["forcing"])
        by_lattice: dict[str, Any] = {}
        problem_val: dict[str, Any] = {}
        for lattice in lattices(protocol):
            name = str(lattice["name"])
            by_lattice[name] = {}
            problem_val[name] = {}
            for noise in noises:
                chosen: dict[str, float] = {}
                seed_success: dict[str, bool] = {}
                supports: dict[str, list[str]] = {}
                instance_rates: list[float] = []
                coeff_errors: list[float] = []
                decaying_rates: list[float] = []
                for seed in use_seeds:
                    val_truth = true_support(problem, protocol["problems"][problem]["primary_family"])
                    val_systems = []
                    for instance_id in val_ids:
                        if not is_primary_instance(protocol, problem, instance_id, forcings[instance_id]):
                            continue
                        row = payloads[feature_key(problem, name, noise, int(seed), instance_id)]
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
                        name,
                        noise,
                        int(seed),
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
                    seed_success[str(seed)] = exact_support(recovered, val_truth)
                    exact_primary = []
                    exact_decaying = []
                    for instance_id in test_ids:
                        row = payloads[feature_key(problem, name, noise, int(seed), instance_id)]
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
                            truth_c = true_coefficients(
                                problem,
                                float(row["nu"]),
                                protocol["problems"][problem]["primary_family"],
                            )
                            coeff_errors.append(coeff_rel_error(hat, truth_c, truth_i))
                        elif problem == "ns2d":
                            exact_decaying.append(float(hit))
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
                }
                if decaying_rates:
                    block["decaying_instance_exact_rate"] = float(np.mean(decaying_rates))
                by_lattice[name][_noise_key(noise)] = block
                problem_val[name][_noise_key(noise)] = {"threshold_by_seed": chosen}
        min_lattice = "none"
        for lattice in lattices(protocol):
            name = str(lattice["name"])
            rate = float(by_lattice[name][_noise_key(bar_noise)]["exact_support_rate"])
            if rate >= min_rate:
                min_lattice = name
                break
        control = by_lattice[control_lattice][_noise_key(control_noise)]
        control_ok = float(control["exact_support_rate"]) >= control_rate
        scores["problems"][problem] = {
            "min_lattice_at_bar": min_lattice,
            "positive_control": {
                "lattice": control_lattice,
                "noise": control_noise,
                "exact_support_rate": float(control["exact_support_rate"]),
                "pass": control_ok,
                "reading": "control passed" if control_ok else "method/implementation failure",
            },
            "by_lattice": by_lattice,
        }
        validation["problems"][problem] = problem_val
    MANIFEST_DIR.mkdir(parents=True, exist_ok=True)
    (MANIFEST_DIR / "validation.json").write_text(json.dumps(validation, indent=2) + "\n", encoding="utf-8")
    (MANIFEST_DIR / "scores.json").write_text(json.dumps(scores, indent=2) + "\n", encoding="utf-8")
    return scores


def _feature_jobs(root: Path, protocol: dict[str, Any], seeds: list[int]) -> list[dict[str, Any]]:
    jobs: list[dict[str, Any]] = []
    noises = [float(item) for item in protocol["noise"]["fractions"]]
    for problem in ("burgers2d", "ns2d"):
        manifest = json.loads((root / f"{problem}_manifest.json").read_text(encoding="utf-8"))
        instance_ids = list(manifest["splits"]["val"]) + list(manifest["splits"]["test"])
        for lattice in lattices(protocol):
            for seed in seeds:
                for noise in noises:
                    for instance_id in instance_ids:
                        jobs.append(
                            {
                                "key": feature_key(problem, str(lattice["name"]), float(noise), int(seed), int(instance_id)),
                                "kind": "feature",
                                "protocol": str(PROTOCOL),
                                "root": str(root),
                                "problem": problem,
                                "lattice": str(lattice["name"]),
                                "n_side": int(lattice["n_side"]),
                                "noise": float(noise),
                                "seed": int(seed),
                                "instance_id": int(instance_id),
                            }
                        )
    return jobs


def _stack(
    payloads: dict[str, dict[str, Any]],
    problem: str,
    lattice: str,
    noise: float,
    seed: int,
    instance_ids: list[int],
) -> tuple[np.ndarray, np.ndarray]:
    blocks_x = []
    blocks_y = []
    for instance_id in instance_ids:
        row = payloads[feature_key(problem, lattice, noise, seed, instance_id)]
        blocks_x.append(np.asarray(row["design"], dtype=np.float64))
        blocks_y.append(np.asarray(row["target"], dtype=np.float64))
    return np.concatenate(blocks_x, axis=0), np.concatenate(blocks_y, axis=0)


def _noise_key(noise: float) -> str:
    return f"{float(noise):.2f}"
