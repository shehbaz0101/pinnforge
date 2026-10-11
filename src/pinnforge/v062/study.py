"""Score frozen v0.6.2 methods: sims train, The Well test."""

from __future__ import annotations

import json
import math
import platform
from pathlib import Path
from typing import Any

import numpy as np

from pinnforge.reference.spectral2d.dataset import load_protocol, protocol_sha256
from pinnforge.v06.hybrid import clip_to_bracket, conformal_quantile, covered, interval
from pinnforge.v06.masks import ShearMask, replay_noise
from pinnforge.v06.wsindy import fit_coefficients, relative_error
from pinnforge.v061.checkpoint import CaseLog, default_workers, feature_key, run_cases
from pinnforge.v061.io import load_frames, parse_reynolds_schmidt, resolve_raw
from pinnforge.v062.sitl import search_kappa

PROTOCOL = Path("docs/v062/protocol.json")
ARTIFACT = Path("artifacts/v062_pilot")
DOCS = Path("docs/v062")
RAW_ROOTS = [Path("artifacts/v061_raw"), Path("artifacts/v06_raw"), Path("artifacts/v062_raw")]


def run_pack_well() -> dict[str, Any]:
    protocol = load_protocol(PROTOCOL)
    window = protocol["time_window"]
    frame_ids = [int(window["frame_start"] + k) for k in range(int(window["n_frames"]))]
    combo = {(item["reynolds"], item["schmidt"]): (item["nu"], item["kappa"]) for item in protocol["subset"]["in_distribution"]}
    combo.update({(item["reynolds"], item["schmidt"]): (item["nu"], item["kappa"]) for item in protocol["subset"]["ood"]})
    instances = []
    blocks = []
    files = protocol["subset"]["files"]
    for relative, split in [(name, "test") for name in files["test"]] + [(name, "ood") for name in files["ood_test"]]:
        path = resolve_raw(relative, RAW_ROOTS)
        packed, meta = load_frames(
            path,
            frame_ids=frame_ids,
            nx=int(protocol["source_grid"]["nx"]),
            ny=int(protocol["source_grid"]["ny"]),
            trajectory_ids=None,
        )
        re_val, sc_val = parse_reynolds_schmidt(relative)
        nu, kappa = combo[(re_val, sc_val)]
        for local, traj in enumerate(meta["trajectory_ids"]):
            instances.append(
                {
                    "instance_id": len(instances),
                    "file": relative,
                    "trajectory": int(traj),
                    "split": split,
                    "reynolds": re_val,
                    "schmidt": sc_val,
                    "nu": nu,
                    "kappa": kappa,
                }
            )
            blocks.append(packed[local])
    fields = np.stack(blocks, axis=0)
    np.save(ARTIFACT / "well_fields.npy", fields)
    manifest = {
        "format": "pinnforge.v062_well_pack.v1",
        "protocol_sha256": protocol_sha256(PROTOCOL),
        "n_instances": int(fields.shape[0]),
        "splits": {
            "test": [row["instance_id"] for row in instances if row["split"] == "test"],
            "ood": [row["instance_id"] for row in instances if row["split"] == "ood"],
        },
        "instances": instances,
        "files": {"well_fields": "well_fields.npy"},
    }
    ARTIFACT.mkdir(parents=True, exist_ok=True)
    (ARTIFACT / "well_manifest.json").write_text(json.dumps(manifest, indent=2) + "\n", encoding="utf-8")
    return manifest


def run_score(*, workers: int | None = None, checkpoint: Path | None = None) -> dict[str, Any]:
    protocol = load_protocol(PROTOCOL)
    protocol.setdefault("downsample", {"length_x": 1.0, "length_y": 2.0})
    digest = protocol_sha256(PROTOCOL)
    sim = json.loads((ARTIFACT / "sim_manifest.json").read_text(encoding="utf-8"))
    well = json.loads((ARTIFACT / "well_manifest.json").read_text(encoding="utf-8"))
    seeds = [int(s) for s in protocol["seeds"]]
    n_workers = default_workers() if workers is None else max(1, int(workers))
    log = CaseLog(checkpoint or (ARTIFACT / "score_cases.jsonl"), digest)
    jobs = _jobs(sim, well, protocol, seeds)
    train_jobs = [job for job in jobs if job["kind"] == "train"]
    eval_jobs = [job for job in jobs if job["kind"] == "eval"]
    print(f"training {len(train_jobs)} FNO checkpoints", flush=True)
    run_cases(train_jobs, log=log, workers=1, worker=case_worker)
    print(f"scoring {len(eval_jobs)} inverse cases", flush=True)
    payloads = run_cases(eval_jobs, log=log, workers=n_workers, worker=case_worker)
    scores = _aggregate(payloads, protocol, sim, well, seeds, digest)
    (DOCS / "validation.json").write_text(
        json.dumps({"protocol_sha256": digest, "n_cal": len(sim["splits"]["cal"])}, indent=2) + "\n",
        encoding="utf-8",
    )
    (DOCS / "scores.json").write_text(json.dumps(scores, indent=2) + "\n", encoding="utf-8")
    return scores


def _jobs(sim: dict[str, Any], well: dict[str, Any], protocol: dict[str, Any], seeds: list[int]) -> list[dict[str, Any]]:
    jobs = []
    densities = [item["name"] for item in protocol["sensors"]["densities"]]
    noises = [float(item) for item in protocol["noise"]["fractions"]]
    for seed in seeds:
        for density in densities:
            jobs.append({"key": f"train|{seed}|{density}", "kind": "train", "seed": seed, "density": density})
    needed = [("sim", i) for i in sim["splits"]["cal"]] + [("well", i) for i in well["splits"]["test"] + well["splits"]["ood"]]
    for seed in seeds:
        for density in densities:
            for noise in noises:
                for source, instance_id in needed:
                    jobs.append(
                        {
                            "key": feature_key("eval", density, noise, seed, instance_id if source == "sim" else instance_id + 1000),
                            "kind": "eval",
                            "seed": seed,
                            "density": density,
                            "noise": noise,
                            "instance_id": instance_id,
                            "source": source,
                        }
                    )
    return jobs


_WORKER: dict[str, Any] = {}


def _worker_state() -> dict[str, Any]:
    if _WORKER:
        return _WORKER
    from pinnforge.v041.checkpoint import single_thread

    single_thread()
    protocol = load_protocol(PROTOCOL)
    protocol.setdefault("downsample", {"length_x": 1.0, "length_y": 2.0})
    _WORKER.update(
        {
            "protocol": protocol,
            "sim": json.loads((ARTIFACT / "sim_manifest.json").read_text(encoding="utf-8")),
            "well": json.loads((ARTIFACT / "well_manifest.json").read_text(encoding="utf-8")),
            "sim_fields": np.load(ARTIFACT / "sim_fields.npy", mmap_mode="r"),
            "well_fields": np.load(ARTIFACT / "well_fields.npy", mmap_mode="r"),
        }
    )
    return _WORKER


def _mask(protocol: dict[str, Any], density_name: str, nx: int, ny: int) -> ShearMask:
    density = next(item for item in protocol["sensors"]["densities"] if item["name"] == density_name)
    window = protocol["time_window"]
    return ShearMask(
        nx=nx,
        ny=ny,
        n_side_x=int(density["n_side_x"]),
        n_side_y=int(density["n_side_y"]),
        burst_starts=tuple(int(item) for item in window["burst_starts"]),
        burst_length=int(window["burst_length"]),
    )


def case_worker(job: dict[str, Any]) -> dict[str, Any]:
    state = _worker_state()
    protocol = state["protocol"]
    density = str(job["density"])
    if job["kind"] == "train":
        from pinnforge.v061.fno_inverse import pack_lattice, train_fno

        seed = int(job["seed"])
        sim = state["sim"]
        fields = state["sim_fields"]
        mask = _mask(protocol, density, fields.shape[-2], fields.shape[-1])
        xs, nu, kappa = [], [], []
        for noise in protocol["noise"]["fractions"]:
            for instance_id in sim["splits"]["train"]:
                xs.append(pack_lattice(replay_noise(fields, instance_id, float(noise), seed, protocol), mask))
                row = sim["instances"][instance_id]
                nu.append(row["nu"])
                kappa.append(row["kappa"])
        val_x, val_nu, val_ka = [], [], []
        for instance_id in sim["splits"]["val"]:
            val_x.append(pack_lattice(replay_noise(fields, instance_id, 0.01, seed, protocol), mask))
            row = sim["instances"][instance_id]
            val_nu.append(row["nu"])
            val_ka.append(row["kappa"])
        train_x = np.stack(xs, axis=0)
        val_x = np.stack(val_x, axis=0)
        dummy = np.zeros((1, 1))
        return train_fno(
            dummy,
            np.asarray(nu + val_nu),
            np.asarray(kappa + val_ka),
            list(range(train_x.shape[0])),
            list(range(train_x.shape[0], train_x.shape[0] + val_x.shape[0])),
            protocol=protocol,
            mask=mask,
            seed=seed,
            checkpoint=ARTIFACT / f"fno_seed{seed}_{density}.pt",
            train_x=train_x,
            val_x=val_x,
        )

    source = str(job["source"])
    instance_id = int(job["instance_id"])
    seed = int(job["seed"])
    noise = float(job["noise"])
    if source == "sim":
        fields = state["sim_fields"]
        row = state["sim"]["instances"][instance_id]
    else:
        fields = state["well_fields"]
        row = state["well"]["instances"][instance_id]
    mask = _mask(protocol, density, fields.shape[-2], fields.shape[-1])
    observed = replay_noise(fields, instance_id, noise, seed, protocol)
    ws = fit_coefficients(observed, protocol=protocol, mask=mask, full_field=False)
    oracle = fit_coefficients(np.asarray(fields[instance_id]), protocol=protocol, mask=mask, full_field=True)
    from pinnforge.v061.fno_inverse import predict_logs

    log_nu, log_ka = predict_logs(observed, mask=mask, checkpoint=ARTIFACT / f"fno_seed{seed}_{density}.pt")
    fno = {"nu": float(math.exp(log_nu)), "kappa": float(math.exp(log_ka))}
    sitl_kappa = float("nan")
    if source == "well":
        train_rows = state["sim"]["instances"]
        train_ids = state["sim"]["splits"]["train"]
        med_k = float(np.median([train_rows[i]["kappa"] for i in train_ids]))
        lo = min(float(protocol["training"]["schmidt_max"]) ** -1 * 1e-4, med_k / 4.0)
        hi = max(float(protocol["training"]["schmidt_min"]) ** -1 * 1e-4, med_k * 4.0)
        sitl_kappa = search_kappa(observed, mask=mask, nu=float(row["nu"]), lo=lo, hi=hi, protocol=protocol)
    return {
        "wsindy": ws,
        "oracle": oracle,
        "fno_data": fno,
        "sitl_cold": {"nu": float(row["nu"]), "kappa": sitl_kappa},
        "nu": row["nu"],
        "kappa": row["kappa"],
        "split": row["split"],
        "log_nu_fno": log_nu,
        "log_kappa_fno": log_ka,
        "density": density,
        "source": source,
    }


def _aggregate(
    payloads: dict[str, dict[str, Any]],
    protocol: dict[str, Any],
    sim: dict[str, Any],
    well: dict[str, Any],
    seeds: list[int],
    digest: str,
) -> dict[str, Any]:
    alpha = float(protocol["methods"]["hybrid"]["alpha"])
    bars = protocol["pass_bars"]
    noises = [float(item) for item in protocol["noise"]["fractions"]]
    densities = [item["name"] for item in protocol["sensors"]["densities"]]
    train_nu = float(np.median([sim["instances"][i]["nu"] for i in sim["splits"]["train"]]))
    train_ka = float(np.median([sim["instances"][i]["kappa"] for i in sim["splits"]["train"]]))
    by_noise: dict[str, Any] = {}
    for noise in noises:
        density_rows = []
        for density in densities:
            seed_rows = []
            for seed in seeds:
                cal_nu, cal_ka = [], []
                for instance_id in sim["splits"]["cal"]:
                    row = payloads[feature_key("eval", density, noise, seed, instance_id)]
                    cal_nu.append(abs(float(row["log_nu_fno"]) - math.log(float(row["nu"]))))
                    cal_ka.append(abs(float(row["log_kappa_fno"]) - math.log(float(row["kappa"]))))
                q_nu = conformal_quantile(np.asarray(cal_nu), alpha)
                q_ka = conformal_quantile(np.asarray(cal_ka), alpha)
                packed: dict[str, Any] = {"q_log_nu": q_nu, "q_log_kappa": q_ka}
                for slice_name, ids in (("in_dist", well["splits"]["test"]), ("ood", well["splits"]["ood"])):
                    methods = {"wsindy": [], "fno_data": [], "hybrid": [], "sitl_cold": [], "oracle": [], "constant": []}
                    cover = []
                    for instance_id in ids:
                        row = payloads[feature_key("eval", density, noise, seed, instance_id + 1000)]
                        fno_nu, fno_ka = float(row["fno_data"]["nu"]), float(row["fno_data"]["kappa"])
                        low_nu, high_nu = interval(fno_nu, q_nu)
                        low_ka, high_ka = interval(fno_ka, q_ka)
                        hy_nu = clip_to_bracket(float(row["wsindy"]["nu"]), low_nu, high_nu)
                        hy_ka = clip_to_bracket(float(row["sitl_cold"]["kappa"]), low_ka, high_ka)
                        truth_nu, truth_ka = float(row["nu"]), float(row["kappa"])
                        methods["wsindy"].append(
                            (relative_error(row["wsindy"]["nu"], truth_nu), relative_error(row["wsindy"]["kappa"], truth_ka))
                        )
                        methods["fno_data"].append((relative_error(fno_nu, truth_nu), relative_error(fno_ka, truth_ka)))
                        methods["hybrid"].append((relative_error(hy_nu, truth_nu), relative_error(hy_ka, truth_ka)))
                        methods["sitl_cold"].append(
                            (relative_error(row["sitl_cold"]["nu"], truth_nu), relative_error(row["sitl_cold"]["kappa"], truth_ka))
                        )
                        methods["oracle"].append(
                            (relative_error(row["oracle"]["nu"], truth_nu), relative_error(row["oracle"]["kappa"], truth_ka))
                        )
                        methods["constant"].append((relative_error(train_nu, truth_nu), relative_error(train_ka, truth_ka)))
                        cover.append(covered(truth_nu, low_nu, high_nu) and covered(truth_ka, low_ka, high_ka))
                    packed[slice_name] = {
                        name: {
                            "median_rel_nu": float(np.median([a for a, _b in vals])),
                            "median_rel_kappa": float(np.median([b for _a, b in vals])),
                            "median_rel": float(0.5 * np.median([a for a, _b in vals]) + 0.5 * np.median([b for _a, b in vals])),
                        }
                        for name, vals in methods.items()
                    }
                    packed[slice_name]["hybrid"]["coverage"] = float(np.mean(cover)) if cover else float("nan")
                seed_rows.append(packed)
            density_rows.append(_mean_seeds(seed_rows))
        by_noise[f"{float(noise):.2f}"] = _mean_seeds(density_rows)
    primary = by_noise["0.01"]
    in_dist = primary["in_dist"]["hybrid"]["median_rel"]
    ood = primary["ood"]["hybrid"]["median_rel"]
    gap = ood / max(in_dist, 1e-12)
    coverage = primary["in_dist"]["hybrid"]["coverage"]
    beats = ood < primary["ood"]["wsindy"]["median_rel"] and ood < primary["ood"]["fno_data"]["median_rel"]
    robust = by_noise["0.05"]["in_dist"]["hybrid"]["median_rel"] <= by_noise["0.05"]["in_dist"]["wsindy"]["median_rel"]
    vs_const = in_dist < primary["in_dist"]["constant"]["median_rel"]
    bar_hits = {
        "in_dist_median_rel": in_dist <= float(bars["nu_kappa_in_dist"]["max"]),
        "ood_median_rel": ood <= float(bars["nu_kappa_ood"]["max"]),
        "ood_over_in_dist": gap <= float(bars["gap"]["max"]),
        "interval_coverage": float(bars["coverage"]["min"]) <= coverage <= float(bars["coverage"]["max"]),
        "hybrid_beats_baselines_ood": beats,
        "noise_5pct_not_worse_than_wsindy": robust,
        "beats_constant_predictor": vs_const,
    }
    scores = {
        "format": "pinnforge.v062_scores.v1",
        "protocol_sha256": digest,
        "python": platform.python_version(),
        "numpy": np.__version__,
        "pysr": protocol["methods"]["pysr"],
        "constant_train_median": {"nu": train_nu, "kappa": train_ka},
        "by_noise": by_noise,
        "bars": {
            "in_dist_median_rel": in_dist,
            "ood_median_rel": ood,
            "ood_over_in_dist": gap,
            "interval_coverage": coverage,
            "hybrid_beats_baselines_ood": beats,
            "noise_5pct_not_worse_than_wsindy": robust,
            "beats_constant_predictor": vs_const,
            "pass": bar_hits,
            "overall": all(bar_hits.values()),
        },
    }
    try:
        from importlib.metadata import version

        scores["torch"] = version("torch")
    except Exception:
        scores["torch"] = None
    return scores


def _mean_seeds(rows: list[dict[str, Any]]) -> dict[str, Any]:
    out: dict[str, Any] = {}
    for slice_name in ("in_dist", "ood"):
        out[slice_name] = {}
        for method in ("wsindy", "fno_data", "hybrid", "sitl_cold", "oracle", "constant"):
            keys = ["median_rel_nu", "median_rel_kappa", "median_rel"]
            cell = {key: float(np.mean([row[slice_name][method][key] for row in rows])) for key in keys}
            if method == "hybrid":
                cell["coverage"] = float(np.mean([row[slice_name][method]["coverage"] for row in rows]))
            out[slice_name][method] = cell
    out["q_log_nu"] = float(np.mean([row["q_log_nu"] for row in rows]))
    out["q_log_kappa"] = float(np.mean([row["q_log_kappa"] for row in rows]))
    return out
