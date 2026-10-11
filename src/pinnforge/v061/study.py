"""Pack the identifiable subset and score the frozen v0.6.1 inverse."""

from __future__ import annotations

import json
import math
import os
import platform
from pathlib import Path
from typing import Any

import numpy as np

from pinnforge.reference.spectral2d.dataset import load_protocol, protocol_sha256
from pinnforge.v06.download import download_file, file_sha256
from pinnforge.v06.hybrid import clip_to_bracket, conformal_quantile, covered, interval
from pinnforge.v06.masks import ShearMask, replay_noise
from pinnforge.v06.wsindy import fit_coefficients, relative_error
from pinnforge.v061.checkpoint import CaseLog, default_workers, feature_key, run_cases
from pinnforge.v061.io import load_frames, parse_reynolds_schmidt, resolve_raw

PROTOCOL = Path("docs/v061/protocol.json")
RAW_ROOTS = [Path("artifacts/v061_raw"), Path("artifacts/v06_raw")]
RAW = Path("artifacts/v061_raw")
ARTIFACT = Path("artifacts/v061_pilot")
DOCS = Path("docs/v061")
EXISTING = Path("artifacts/v06_raw")


def run_download() -> dict[str, Any]:
    protocol = load_protocol(PROTOCOL)
    RAW.mkdir(parents=True, exist_ok=True)
    records = []
    for relative in _all_files(protocol):
        dest = RAW / relative
        dest.parent.mkdir(parents=True, exist_ok=True)
        prior = EXISTING / relative
        if dest.is_file():
            path = dest
        elif prior.is_file():
            os.link(prior, dest)
            path = dest
        else:
            path = download_file(relative, RAW)
        records.append(
            {
                "path": relative,
                "local": str(path),
                "bytes": int(path.stat().st_size),
                "sha256": file_sha256(path),
            }
        )
        print(f"have {relative} {records[-1]['bytes']} {records[-1]['sha256'][:12]}", flush=True)
    payload = {
        "format": "pinnforge.v061_download_manifest.v1",
        "protocol_sha256": protocol_sha256(PROTOCOL),
        "repo": "polymathic-ai/shear_flow",
        "anonymous": True,
        "files": records,
    }
    (DOCS / "download_manifest.json").write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")
    return payload


def run_pack(output: Path | None = None) -> dict[str, Any]:
    protocol = load_protocol(PROTOCOL)
    root = ARTIFACT if output is None else output
    root.mkdir(parents=True, exist_ok=True)
    window = protocol["time_window"]
    frame_ids = [
        int(window["frame_start"] + k * int(window["frame_stride"])) for k in range(int(window["n_frames"]))
    ]
    combo = {(item["reynolds"], item["schmidt"]): (item["nu"], item["kappa"]) for item in protocol["subset"]["in_distribution"]}
    combo.update({(item["reynolds"], item["schmidt"]): (item["nu"], item["kappa"]) for item in protocol["subset"]["ood"]})
    combo.update(
        {(item["reynolds"], item["schmidt"]): (item["nu"], item["kappa"]) for item in protocol["subset"]["ood_re_reported_not_bar"]}
    )
    instances: list[dict[str, Any]] = []
    blocks: list[np.ndarray] = []

    def _add(relative: str, split: str, ood: bool, ids: list[int] | None = None) -> None:
        path = resolve_raw(relative, RAW_ROOTS)
        packed, meta = load_frames(
            path,
            frame_ids=frame_ids,
            nx=int(protocol["downsample"]["nx"]),
            ny=int(protocol["downsample"]["ny"]),
            trajectory_ids=ids,
        )
        re_val, sc_val = parse_reynolds_schmidt(relative)
        nu, kappa = combo[(re_val, sc_val)]
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
    files = protocol["subset"]["files"]
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
    for relative in files["ood_re_valid"]:
        _add(relative, "ood_re_valid", True)
    for relative in files["ood_re_test"]:
        _add(relative, "ood_re_test", True)

    fields = np.stack(blocks, axis=0)
    np.save(root / "fields.npy", fields)
    train_ids, cal_ids = _assign_train_cal(instances, protocol)
    val_ids = [row["instance_id"] for row in instances if row["split_source"] == "valid"]
    test_ids = [row["instance_id"] for row in instances if row["split_source"] == "test"]
    ood_ids = [row["instance_id"] for row in instances if row["split_source"] in {"ood_valid", "ood_test"}]
    ood_re_ids = [row["instance_id"] for row in instances if row["split_source"] in {"ood_re_valid", "ood_re_test"}]
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
        elif iid in ood_re_ids:
            row["split"] = "ood_re"
        else:
            row["split"] = "unused"
    manifest = {
        "format": "pinnforge.v061_pilot.v1",
        "protocol_sha256": protocol_sha256(PROTOCOL),
        "n_instances": int(fields.shape[0]),
        "field_shape": list(fields.shape),
        "splits": {
            "train": train_ids,
            "cal": cal_ids,
            "val": val_ids,
            "test": test_ids,
            "ood": ood_ids,
            "ood_re": ood_re_ids,
        },
        "instances": instances,
        "files": {"fields": "fields.npy"},
    }
    (root / "manifest.json").write_text(json.dumps(manifest, indent=2) + "\n", encoding="utf-8")
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
    print(f"training {len(train_jobs)} FNO checkpoints", flush=True)
    run_cases(train_jobs, log=log, workers=1, worker=case_worker)
    print(f"scoring {len(eval_jobs)} inverse cases", flush=True)
    payloads = run_cases(eval_jobs, log=log, workers=n_workers, worker=case_worker)
    scores = _aggregate(payloads, protocol, manifest, use_seeds, digest)
    validation = {
        "protocol_sha256": digest,
        "n_cal": len(manifest["splits"]["cal"]),
        "identifiability": "docs/v061/identifiability.json",
        "chosen_window": protocol["time_window"]["name"],
        "chosen_grid": [protocol["downsample"]["nx"], protocol["downsample"]["ny"]],
    }
    (DOCS / "validation.json").write_text(json.dumps(validation, indent=2) + "\n", encoding="utf-8")
    (DOCS / "scores.json").write_text(json.dumps(scores, indent=2) + "\n", encoding="utf-8")
    return scores


def _all_files(protocol: dict[str, Any]) -> list[str]:
    files = protocol["subset"]["files"]
    names: list[str] = []
    for key in ("train", "valid", "test", "ood_valid", "ood_test", "ood_re_valid", "ood_re_test"):
        names.extend(str(item) for item in files[key])
    return names


def _assign_train_cal(instances: list[dict[str, Any]], protocol: dict[str, Any]) -> tuple[list[int], list[int]]:
    train_ids: list[int] = []
    cal_ids: list[int] = []
    n_train = int(protocol["splits"]["from_train_files"]["n_train"])
    n_cal = int(protocol["splits"]["from_train_files"]["n_cal"])
    per_file_train = n_train // 4
    per_file_cal = n_cal // 4
    by_file: dict[str, list[int]] = {}
    for row in instances:
        if row["split_source"] != "train_file":
            continue
        by_file.setdefault(row["file"], []).append(row["instance_id"])
    for _file, ids in by_file.items():
        train_ids.extend(ids[:per_file_train])
        cal_ids.extend(ids[per_file_train : per_file_train + per_file_cal])
    if len(cal_ids) < int(protocol["splits"]["from_train_files"]["n_cal_min"]):
        raise ValueError("calibration split is smaller than the frozen n_cal_min")
    return train_ids, cal_ids


def _jobs(manifest: dict[str, Any], protocol: dict[str, Any], seeds: list[int]) -> list[dict[str, Any]]:
    jobs = []
    noises = [float(item) for item in protocol["noise"]["fractions"]]
    densities = [item["name"] for item in protocol["sensors"]["densities"]]
    needed = manifest["splits"]["cal"] + manifest["splits"]["val"] + manifest["splits"]["test"]
    needed += manifest["splits"]["ood"] + manifest["splits"]["ood_re"]
    for seed in seeds:
        for density in densities:
            jobs.append({"key": f"train|{seed}|{density}", "kind": "train", "seed": int(seed), "density": density})
    for seed in seeds:
        for density in densities:
            for noise in noises:
                for instance_id in needed:
                    jobs.append(
                        {
                            "key": feature_key("eval", density, noise, int(seed), int(instance_id)),
                            "kind": "eval",
                            "seed": int(seed),
                            "density": density,
                            "noise": float(noise),
                            "instance_id": int(instance_id),
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
    root = ARTIFACT
    manifest = json.loads((root / "manifest.json").read_text(encoding="utf-8"))
    _WORKER.update(
        {
            "protocol": protocol,
            "root": root,
            "manifest": manifest,
            "fields": np.load(root / manifest["files"]["fields"], mmap_mode="r"),
            "oracles": {},
        }
    )
    return _WORKER


def _mask(protocol: dict[str, Any], density_name: str) -> ShearMask:
    density = next(item for item in protocol["sensors"]["densities"] if item["name"] == density_name)
    window = protocol["time_window"]
    return ShearMask(
        nx=int(protocol["downsample"]["nx"]),
        ny=int(protocol["downsample"]["ny"]),
        n_side_x=int(density["n_side_x"]),
        n_side_y=int(density["n_side_y"]),
        burst_starts=tuple(int(item) for item in window["burst_starts"]),
        burst_length=int(window["burst_length"]),
    )


def case_worker(job: dict[str, Any]) -> dict[str, Any]:
    state = _worker_state()
    protocol = state["protocol"]
    root = state["root"]
    manifest = state["manifest"]
    fields = state["fields"]
    density = str(job["density"])
    mask = _mask(protocol, density)
    if job["kind"] == "train":
        from pinnforge.v061.fno_inverse import pack_lattice, train_fno

        seed = int(job["seed"])
        xs = []
        nu = []
        kappa = []
        for noise in protocol["noise"]["fractions"]:
            for instance_id in manifest["splits"]["train"]:
                xs.append(pack_lattice(replay_noise(fields, instance_id, float(noise), seed, protocol), mask))
                row = manifest["instances"][instance_id]
                nu.append(row["nu"])
                kappa.append(row["kappa"])
        val_x = []
        val_nu = []
        val_ka = []
        for instance_id in manifest["splits"]["val"]:
            val_x.append(pack_lattice(replay_noise(fields, instance_id, 0.01, seed, protocol), mask))
            row = manifest["instances"][instance_id]
            val_nu.append(row["nu"])
            val_ka.append(row["kappa"])
        train_x = np.stack(xs, axis=0)
        val_x = np.stack(val_x, axis=0)
        train_ids = list(range(train_x.shape[0]))
        val_ids = list(range(train_x.shape[0], train_x.shape[0] + val_x.shape[0]))
        ckpt = root / f"fno_seed{seed}_{density}.pt"
        dummy = np.zeros((1, 1))
        return train_fno(
            dummy,
            np.asarray(nu + val_nu),
            np.asarray(kappa + val_ka),
            train_ids,
            val_ids,
            protocol=protocol,
            mask=mask,
            seed=seed,
            checkpoint=ckpt,
            train_x=train_x,
            val_x=val_x,
        )

    seed = int(job["seed"])
    noise = float(job["noise"])
    instance_id = int(job["instance_id"])
    row = manifest["instances"][instance_id]
    observed = replay_noise(fields, instance_id, noise, seed, protocol)
    ws = fit_coefficients(observed, protocol=protocol, mask=mask, full_field=False)
    if instance_id not in state["oracles"]:
        state["oracles"][instance_id] = fit_coefficients(fields[instance_id], protocol=protocol, mask=mask, full_field=True)
    oracle = state["oracles"][instance_id]
    from pinnforge.v061.fno_inverse import predict_logs

    ckpt = root / f"fno_seed{seed}_{density}.pt"
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
        "density": density,
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
    densities = [item["name"] for item in protocol["sensors"]["densities"]]
    train_nu = float(np.median([manifest["instances"][i]["nu"] for i in manifest["splits"]["train"]]))
    train_ka = float(np.median([manifest["instances"][i]["kappa"] for i in manifest["splits"]["train"]]))
    by_noise: dict[str, Any] = {}
    for noise in noises:
        density_rows = []
        for density in densities:
            seed_rows = []
            for seed in seeds:
                cal_nu = []
                cal_ka = []
                for instance_id in manifest["splits"]["cal"]:
                    row = payloads[feature_key("eval", density, noise, int(seed), int(instance_id))]
                    cal_nu.append(abs(float(row["log_nu_fno"]) - math.log(float(row["nu"]))))
                    cal_ka.append(abs(float(row["log_kappa_fno"]) - math.log(float(row["kappa"]))))
                q_nu = conformal_quantile(np.asarray(cal_nu), alpha)
                q_ka = conformal_quantile(np.asarray(cal_ka), alpha)
                slices = {
                    "in_dist": manifest["splits"]["test"],
                    "ood": manifest["splits"]["ood"],
                    "ood_re": manifest["splits"]["ood_re"],
                }
                packed: dict[str, Any] = {"q_log_nu": q_nu, "q_log_kappa": q_ka, "density": density}
                for slice_name, ids in slices.items():
                    methods = {"wsindy": [], "fno_data": [], "hybrid": [], "oracle": [], "constant": []}
                    cover = []
                    for instance_id in ids:
                        row = payloads[feature_key("eval", density, noise, int(seed), int(instance_id))]
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
                        methods["constant"].append((relative_error(train_nu, truth_nu), relative_error(train_ka, truth_ka)))
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
            density_rows.append(_mean_seeds(seed_rows))
        by_noise[_noise_key(noise)] = _mean_densities(density_rows)
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
        "format": "pinnforge.v061_scores.v1",
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
    for slice_name in ("in_dist", "ood", "ood_re"):
        out[slice_name] = {}
        for method in ("wsindy", "fno_data", "hybrid", "oracle", "constant"):
            keys = ["median_rel_nu", "median_rel_kappa", "median_rel"]
            cell = {key: float(np.mean([row[slice_name][method][key] for row in rows])) for key in keys}
            if method == "hybrid":
                cell["coverage"] = float(np.mean([row[slice_name][method]["coverage"] for row in rows]))
            out[slice_name][method] = cell
    out["q_log_nu"] = float(np.mean([row["q_log_nu"] for row in rows]))
    out["q_log_kappa"] = float(np.mean([row["q_log_kappa"] for row in rows]))
    return out


def _mean_densities(rows: list[dict[str, Any]]) -> dict[str, Any]:
    return _mean_seeds(rows)


def _noise_key(noise: float) -> str:
    return f"{float(noise):.2f}"
