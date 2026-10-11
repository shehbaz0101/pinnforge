"""Generate continuously sampled Sc trajectories at 64 by 128."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import numpy as np

from pinnforge.reference.spectral2d.dataset import load_protocol, protocol_sha256
from pinnforge.v062.solver import SF_IC_FAMILY, init_standard_shear, solve_shear

PROTOCOL = Path("docs/v062/protocol.json")
ARTIFACT = Path("artifacts/v062_pilot")
DOCS = Path("docs/v062")


def run_generate(*, n_workers: int = 4) -> dict[str, Any]:
    from concurrent.futures import ProcessPoolExecutor

    protocol = load_protocol(PROTOCOL)
    train = protocol["training"]
    grid = protocol["sim_grid"]
    n_sim = int(protocol["compute_reduction"]["n_sim"])
    rng = np.random.default_rng(int(train["split_seed"]))
    jobs = []
    for i in range(n_sim):
        log_a = np.log(float(train["schmidt_min"]))
        log_b = np.log(float(train["schmidt_max"]))
        sc = float(np.exp(rng.uniform(log_a, log_b)))
        spec = SF_IC_FAMILY[int(rng.integers(0, len(SF_IC_FAMILY)))]
        jobs.append(
            {
                "index": i,
                "schmidt": sc,
                "n_shear": spec.n_shear,
                "n_blobs": spec.n_blobs,
                "width": spec.width,
                "nx": int(grid["nx"]),
                "ny": int(grid["ny"]),
                "dt": float(grid["dt"]),
                "t_final": float(grid["t_final"]),
                "save_dt": float(grid["save_dt"]),
                "reynolds": float(train["reynolds"]),
            }
        )
    ARTIFACT.mkdir(parents=True, exist_ok=True)
    fields = np.zeros((n_sim, 9, 4, int(grid["nx"]), int(grid["ny"])), dtype=np.float64)
    records = [None] * n_sim
    with ProcessPoolExecutor(max_workers=max(1, int(n_workers))) as pool:
        for record, late in pool.map(_one_traj, jobs):
            idx = int(record["index"])
            fields[idx] = late
            records[idx] = record
            print(f"sim {idx} Sc={record['schmidt']:.4g}", flush=True)
    n_train = int(train["n_train"])
    n_cal = int(train["n_cal"])
    n_val = int(train["n_val"])
    splits = {
        "train": list(range(0, n_train)),
        "cal": list(range(n_train, n_train + n_cal)),
        "val": list(range(n_train + n_cal, n_train + n_cal + n_val)),
    }
    for row in records:
        iid = int(row["index"])
        if iid in splits["train"]:
            row["split"] = "train"
        elif iid in splits["cal"]:
            row["split"] = "cal"
        elif iid in splits["val"]:
            row["split"] = "val"
    np.save(ARTIFACT / "sim_fields.npy", fields)
    manifest = {
        "format": "pinnforge.v062_sim_manifest.v1",
        "protocol_sha256": protocol_sha256(PROTOCOL),
        "n_sim": n_sim,
        "field_shape": list(fields.shape),
        "splits": splits,
        "instances": records,
        "files": {"sim_fields": "sim_fields.npy"},
    }
    (ARTIFACT / "sim_manifest.json").write_text(json.dumps(manifest, indent=2) + "\n", encoding="utf-8")
    (DOCS / "sim_manifest.json").write_text(json.dumps(manifest, indent=2) + "\n", encoding="utf-8")
    return manifest


def _one_traj(job: dict[str, Any]) -> tuple[dict[str, Any], np.ndarray]:
    nu = 1.0 / float(job["reynolds"])
    kappa = nu / float(job["schmidt"])
    ux, uy = init_standard_shear(
        int(job["nx"]),
        int(job["ny"]),
        n_shear=int(job["n_shear"]),
        n_blobs=int(job["n_blobs"]),
        width=float(job["width"]),
    )
    rolled = solve_shear(
        ux,
        uy,
        ux,
        nu=nu,
        kappa=kappa,
        dt=float(job["dt"]),
        t_final=float(job["t_final"]),
        save_dt=float(job["save_dt"]),
    )
    late = rolled["fields"][160:169]
    record = {
        "index": int(job["index"]),
        "reynolds": float(job["reynolds"]),
        "schmidt": float(job["schmidt"]),
        "nu": nu,
        "kappa": kappa,
        "n_shear": int(job["n_shear"]),
        "n_blobs": int(job["n_blobs"]),
        "width": float(job["width"]),
        "split": "pending",
    }
    return record, late
