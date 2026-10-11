"""Pack The Well subset and score the frozen v0.6 inverse."""

from __future__ import annotations

import json
import math
import platform
from pathlib import Path
from typing import Any

import numpy as np

from pinnforge.reference.spectral2d.dataset import load_protocol, protocol_sha256
from pinnforge.v06.checkpoint import CaseLog, default_workers, feature_key, run_cases
from pinnforge.v06.hybrid import clip_to_bracket, conformal_quantile, covered, interval
from pinnforge.v06.masks import mask_from_protocol, replay_noise
from pinnforge.v06.wsindy import fit_coefficients, relative_error

PROTOCOL = Path("docs/v06/protocol.json")
RAW = Path("artifacts/v06_raw")
ARTIFACT = Path("artifacts/v06_pilot")
DOCS = Path("docs/v06")


def run_pack(output: Path | None = None) -> dict[str, Any]:
    from pinnforge.v06.io import load_window

    protocol = load_protocol(PROTOCOL)
    root = ARTIFACT if output is None else output
    root.mkdir(parents=True, exist_ok=True)
    instances: list[dict[str, Any]] = []
    blocks: list[np.ndarray] = []
    files = protocol["subset"]["files"]
    combo_nu = {(item["reynolds"], item["schmidt"]): (item["nu"], item["kappa"]) for item in protocol["subset"]["in_distribution"]}
    combo_nu.update({(item["reynolds"], item["schmidt"]): (item["nu"], item["kappa"]) for item in protocol["subset"]["ood"]})

    def _add(relative: str, split: str, ood: bool, ids: list[int] | None = None) -> None:
        path = RAW / relative
        if not path.is_file():
            # huggingface_hub local_dir keeps the repo-relative path
            path = RAW / Path(relative).name if (RAW / Path(relative).name).is_file() else path
        packed, meta = load_window(path, protocol=protocol, trajectory_ids=ids)
        re_val, sc_val = _parse_name(relative)
        nu, kappa = combo_nu[(re_val, sc_val)]
        for local, traj in enumerate(meta["trajectory_ids"]):
            instances.append(
                {
                    "instance_id": len(instances),
                    "file": relative,
                    "trajectory": int(traj),
                    "split_source": split,
                    "ood": ood,
                    "reynolds": re_val,
                    "schmidt": sc_val,
                    "nu": nu,
                    "kappa": kappa,
                }
            )
            blocks.append(packed[local])

    train_per = int(protocol["subset"]["trajectories_per_train_file"])
    for relative in files["train"]:
        _add(relative, "train_file", False, list(range(train_per)))
    for relative in files["valid"]:
        _add(relative, "valid", False)
    for relative in files["test"]:
        _add(relative, "test", False)
    for relative in files["ood_valid"]:
        _add(relative, "ood_valid", True)
    for relative in files["ood_test"]:
        _add(relative, "ood_test", True)

    fields = np.stack(blocks, axis=0)
    np.save(root / "fields.npy", fields)
    train_ids, cal_ids = _assign_train_cal(instances, protocol)
    val_ids = [row["instance_id"] for row in instances if row["split_source"] == "valid"]
    test_ids = [row["instance_id"] for row in instances if row["split_source"] == "test"]
    ood_ids = [row["instance_id"] for row in instances if row["ood"]]
    for row in instances:
        iid = row["instance_id"]
        if iid in train_ids:
            row["split"] = "train"
        elif iid in cal_ids:
            row["split"] = "cal"
        elif iid in val_ids:
            row["split"] = "val"
        elif iid in test_ids:
            row["split"] = "test"
        elif iid in ood_ids:
            row["split"] = "ood"
        else:
            row["split"] = "unused"
    manifest = {
        "format": "pinnforge.v06_pilot.v1",
        "protocol_sha256": protocol_sha256(PROTOCOL),
        "n_instances": int(fields.shape[0]),
        "field_shape": list(fields.shape),
        "splits": {"train": train_ids, "cal": cal_ids, "val": val_ids, "test": test_ids, "ood": ood_ids},
        "instances": instances,
        "files": {"fields": "fields.npy"},
    }
    (root / "manifest.json").write_text(json.dumps(manifest, indent=2) + "\n", encoding="utf-8")
    DOCS.mkdir(parents=True, exist_ok=True)
    (DOCS / "pilot_manifest.json").write_text(json.dumps(manifest, indent=2) + "\n", encoding="utf-8")
    return manifest


def run_score(
    output: Path | None = None,
    *,
    seeds: list[int] | None = None,
    workers: int | None = None,
    checkpoint: Path | None = None,
) -> dict[str, Any]:
    protocol = load_protocol(PROTOCOL)
    root = ARTIFACT if output is None else output
    manifest = json.loads((root / "manifest.json").read_text(encoding="utf-8"))
    use_seeds = protocol["seeds"] if seeds is None else seeds
    n_workers = default_workers() if workers is None else max(1, int(workers))
    digest = protocol_sha256(PROTOCOL)
    log = CaseLog(checkpoint or (root / "score_cases.jsonl"), digest)
    jobs = _jobs(manifest, protocol, use_seeds)
    train_jobs = [job for job in jobs if job["kind"] == "train"]
    eval_jobs = [job for job in jobs if job["kind"] == "eval"]
    print(f"training {len(train_jobs)} FNO seeds", flush=True)
    run_cases(train_jobs, log=log, workers=1, worker=case_worker)
    print(f"scoring {len(eval_jobs)} inverse cases", flush=True)
    payloads = run_cases(eval_jobs, log=log, workers=n_workers, worker=case_worker)
    scores = _aggregate(payloads, protocol, manifest, use_seeds, digest)
    validation = {"protocol_sha256": digest, "n_cal": len(manifest["splits"]["cal"])}
    (DOCS / "validation.json").write_text(json.dumps(validation, indent=2) + "\n", encoding="utf-8")
    (DOCS / "scores.json").write_text(json.dumps(scores, indent=2) + "\n", encoding="utf-8")
    return scores


def _assign_train_cal(instances: list[dict[str, Any]], protocol: dict[str, Any]) -> tuple[list[int], list[int]]:
    train_ids: list[int] = []
    cal_ids: list[int] = []
    n_train = int(protocol["splits"]["from_train_files"]["n_train"])
    n_cal = int(protocol["splits"]["from_train_files"]["n_cal"])
    per_file = n_train // 3
    by_file: dict[str, list[int]] = {}
    for row in instances:
        if row["split_source"] != "train_file":
            continue
        by_file.setdefault(row["file"], []).append(row["instance_id"])
    leftovers: list[int] = []
    for _file, ids in by_file.items():
        train_ids.extend(ids[:per_file])
        leftovers.extend(ids[per_file:])
    cal_ids.extend(leftovers[:n_cal])
    if len(cal_ids) < int(protocol["splits"]["from_train_files"]["n_cal_min"]):
        raise ValueError("calibration split is smaller than the frozen n_cal_min")
    return train_ids, cal_ids


def _parse_name(relative: str) -> tuple[float, float]:
    name = Path(relative).name
    # shear_flow_Reynolds_1e4_Schmidt_5e-1.hdf5
    parts = name.replace(".hdf5", "").split("_")
    re_tok = parts[parts.index("Reynolds") + 1]
    sc_tok = parts[parts.index("Schmidt") + 1]
    return float(re_tok), float(sc_tok)


def _jobs(manifest: dict[str, Any], protocol: dict[str, Any], seeds: list[int]) -> list[dict[str, Any]]:
    jobs = []
    noises = [float(item) for item in protocol["noise"]["fractions"]]
    needed = manifest["splits"]["cal"] + manifest["splits"]["val"] + manifest["splits"]["test"] + manifest["splits"]["ood"]
    # FNO train jobs first (one per seed)
    for seed in seeds:
        jobs.append({"key": f"train|{seed}", "kind": "train", "seed": int(seed)})
    for seed in seeds:
        for noise in noises:
            for instance_id in needed:
                jobs.append(
                    {
                        "key": feature_key("eval", noise, int(seed), int(instance_id)),
                        "kind": "eval",
                        "seed": int(seed),
                        "noise": float(noise),
                        "instance_id": int(instance_id),
                    }
                )
    return jobs


_WORKER: dict[str, Any] = {}


def _worker_state() -> dict[str, Any]:
    """Load the packed fields once per process. The 75 MB array is not re-read per case."""

    if _WORKER:
        return _WORKER
    from pinnforge.v041.checkpoint import single_thread

    single_thread()
    protocol = load_protocol(PROTOCOL)
    root = ARTIFACT
    manifest = json.loads((root / "manifest.json").read_text(encoding="utf-8"))
    _WORKER.update(
        {
            "protocol": protocol,
            "root": root,
            "manifest": manifest,
            "fields": np.load(root / manifest["files"]["fields"]),
            "mask": mask_from_protocol(protocol),
        }
    )
    return _WORKER


def case_worker(job: dict[str, Any]) -> dict[str, Any]:
    state = _worker_state()
    protocol = state["protocol"]
    root = state["root"]
    manifest = state["manifest"]
    fields = state["fields"]
    mask = state["mask"]
    if job["kind"] == "train":
        from pinnforge.v06.fno_inverse import train_fno

        seed = int(job["seed"])
        # train on every frozen noise fraction
        xs = []
        nu = []
        kappa = []
        for noise in protocol["noise"]["fractions"]:
            for instance_id in manifest["splits"]["train"]:
                xs.append(replay_noise(fields, instance_id, float(noise), seed, protocol))
                row = manifest["instances"][instance_id]
                nu.append(row["nu"])
                kappa.append(row["kappa"])
        packed = np.stack(xs, axis=0)
        val_fields = []
        val_nu = []
        val_ka = []
        for instance_id in manifest["splits"]["val"]:
            val_fields.append(replay_noise(fields, instance_id, 0.01, seed, protocol))
            row = manifest["instances"][instance_id]
            val_nu.append(row["nu"])
            val_ka.append(row["kappa"])
        train_ids = list(range(packed.shape[0]))
        val_block = np.stack(val_fields, axis=0)
        combined = np.concatenate([packed, val_block], axis=0)
        val_ids = list(range(packed.shape[0], combined.shape[0]))
        ckpt = root / f"fno_seed{seed}.pt"
        summary = train_fno(
            combined,
            np.asarray(nu + val_nu),
            np.asarray(kappa + val_ka),
            train_ids,
            val_ids,
            protocol=protocol,
            mask=mask,
            seed=seed,
            checkpoint=ckpt,
        )
        return summary

    seed = int(job["seed"])
    noise = float(job["noise"])
    instance_id = int(job["instance_id"])
    row = manifest["instances"][instance_id]
    observed = replay_noise(fields, instance_id, noise, seed, protocol)
    ws = fit_coefficients(observed, protocol=protocol, mask=mask, full_field=False)
    oracle = fit_coefficients(fields[instance_id], protocol=protocol, mask=mask, full_field=True)
    from pinnforge.v06.fno_inverse import predict_logs

    ckpt = root / f"fno_seed{seed}.pt"
    log_nu, log_ka = predict_logs(observed, mask=mask, checkpoint=ckpt)
    fno = {"nu": float(math.exp(log_nu)), "kappa": float(math.exp(log_ka))}
    return {
        "wsindy": ws,
        "oracle": oracle,
        "fno_data": fno,
        "nu": row["nu"],
        "kappa": row["kappa"],
        "ood": row["ood"],
        "split": row["split"],
        "log_nu_fno": log_nu,
        "log_kappa_fno": log_ka,
    }


def _aggregate(
    payloads: dict[str, dict[str, Any]],
    protocol: dict[str, Any],
    manifest: dict[str, Any],
    seeds: list[int],
    digest: str,
) -> dict[str, Any]:
    alpha = float(protocol["methods"]["hybrid"]["alpha"])
    bars = protocol["pass_bars"]
    noises = [float(item) for item in protocol["noise"]["fractions"]]
    by_noise: dict[str, Any] = {}
    for noise in noises:
        seed_rows = []
        for seed in seeds:
            cal_nu = []
            cal_ka = []
            for instance_id in manifest["splits"]["cal"]:
                row = payloads[feature_key("eval", noise, int(seed), int(instance_id))]
                # conformal on the hybrid-to-be FNO logs vs truth
                cal_nu.append(abs(float(row["log_nu_fno"]) - math.log(float(row["nu"]))))
                cal_ka.append(abs(float(row["log_kappa_fno"]) - math.log(float(row["kappa"]))))
            q_nu = conformal_quantile(np.asarray(cal_nu), alpha)
            q_ka = conformal_quantile(np.asarray(cal_ka), alpha)
            slices = {"in_dist": manifest["splits"]["test"], "ood": manifest["splits"]["ood"]}
            packed: dict[str, Any] = {"q_log_nu": q_nu, "q_log_kappa": q_ka}
            for slice_name, ids in slices.items():
                methods = {"wsindy": [], "fno_data": [], "hybrid": [], "oracle": []}
                cover = []
                for instance_id in ids:
                    row = payloads[feature_key("eval", noise, int(seed), int(instance_id))]
                    fno_nu, fno_ka = float(row["fno_data"]["nu"]), float(row["fno_data"]["kappa"])
                    low_nu, high_nu = interval(fno_nu, q_nu)
                    low_ka, high_ka = interval(fno_ka, q_ka)
                    hy_nu = clip_to_bracket(float(row["wsindy"]["nu"]), low_nu, high_nu)
                    hy_ka = clip_to_bracket(float(row["wsindy"]["kappa"]), low_ka, high_ka)
                    truth_nu, truth_ka = float(row["nu"]), float(row["kappa"])
                    methods["wsindy"].append(
                        (relative_error(row["wsindy"]["nu"], truth_nu), relative_error(row["wsindy"]["kappa"], truth_ka))
                    )
                    methods["fno_data"].append((relative_error(fno_nu, truth_nu), relative_error(fno_ka, truth_ka)))
                    methods["hybrid"].append((relative_error(hy_nu, truth_nu), relative_error(hy_ka, truth_ka)))
                    methods["oracle"].append(
                        (relative_error(row["oracle"]["nu"], truth_nu), relative_error(row["oracle"]["kappa"], truth_ka))
                    )
                    cover.append(covered(truth_nu, low_nu, high_nu) and covered(truth_ka, low_ka, high_ka))
                packed[slice_name] = {
                    name: {
                        "median_rel_nu": float(np.median([a for a, _b in vals])),
                        "median_rel_kappa": float(np.median([b for _a, b in vals])),
                        "median_rel": float(
                            0.5 * np.median([a for a, _b in vals]) + 0.5 * np.median([b for _a, b in vals])
                        ),
                    }
                    for name, vals in methods.items()
                }
                packed[slice_name]["hybrid"]["coverage"] = float(np.mean(cover)) if cover else float("nan")
            seed_rows.append(packed)
        by_noise[_noise_key(noise)] = _mean_seeds(seed_rows)
    primary = by_noise["0.01"]
    in_dist = primary["in_dist"]["hybrid"]["median_rel"]
    ood = primary["ood"]["hybrid"]["median_rel"]
    gap = ood / max(in_dist, 1e-12)
    coverage = primary["in_dist"]["hybrid"]["coverage"]
    beats = ood < primary["ood"]["wsindy"]["median_rel"] and ood < primary["ood"]["fno_data"]["median_rel"]
    robust = by_noise["0.05"]["in_dist"]["hybrid"]["median_rel"] <= by_noise["0.05"]["in_dist"]["wsindy"]["median_rel"]
    bar_hits = {
        "in_dist_median_rel": in_dist <= float(bars["nu_kappa_in_dist"]["max"]),
        "ood_median_rel": ood <= float(bars["nu_kappa_ood"]["max"]),
        "ood_over_in_dist": gap <= float(bars["gap"]["max"]),
        "interval_coverage": float(bars["coverage"]["min"]) <= coverage <= float(bars["coverage"]["max"]),
        "hybrid_beats_baselines_ood": beats,
        "noise_5pct_not_worse_than_wsindy": robust,
    }
    scores = {
        "format": "pinnforge.v06_scores.v1",
        "protocol_sha256": digest,
        "python": platform.python_version(),
        "numpy": np.__version__,
        "pysr": protocol["methods"]["pysr"],
        "by_noise": by_noise,
        "bars": {
            "in_dist_median_rel": in_dist,
            "ood_median_rel": ood,
            "ood_over_in_dist": gap,
            "interval_coverage": coverage,
            "hybrid_beats_baselines_ood": beats,
            "noise_5pct_not_worse_than_wsindy": robust,
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
        for method in ("wsindy", "fno_data", "hybrid", "oracle"):
            keys = ["median_rel_nu", "median_rel_kappa", "median_rel"]
            cell = {key: float(np.mean([row[slice_name][method][key] for row in rows])) for key in keys}
            if method == "hybrid":
                cell["coverage"] = float(np.mean([row[slice_name][method]["coverage"] for row in rows]))
            out[slice_name][method] = cell
    out["q_log_nu"] = float(np.mean([row["q_log_nu"] for row in rows]))
    out["q_log_kappa"] = float(np.mean([row["q_log_kappa"] for row in rows]))
    return out


def _noise_key(noise: float) -> str:
    return f"{float(noise):.2f}"
