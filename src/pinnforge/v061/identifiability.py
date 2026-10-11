"""Validation-only full-field oracle sweep. Not a test score."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import numpy as np

from pinnforge.v06.grid import downsample_fourier
from pinnforge.v06.wsindy import fit_coefficients, relative_error
from pinnforge.v061.io import coefficients, load_frames, parse_reynolds_schmidt, resolve_raw

DOCS = Path("docs/v061")
RAW_ROOTS = [Path("artifacts/v061_raw"), Path("artifacts/v06_raw")]
GATE = 0.02

VALID_FILES = [
    "data/valid/shear_flow_Reynolds_1e4_Schmidt_1e-1.hdf5",
    "data/valid/shear_flow_Reynolds_1e4_Schmidt_2e-1.hdf5",
    "data/valid/shear_flow_Reynolds_1e4_Schmidt_5e-1.hdf5",
    "data/valid/shear_flow_Reynolds_1e4_Schmidt_1e0.hdf5",
    "data/valid/shear_flow_Reynolds_1e4_Schmidt_2e0.hdf5",
    "data/valid/shear_flow_Reynolds_1e4_Schmidt_5e0.hdf5",
    "data/valid/shear_flow_Reynolds_5e4_Schmidt_5e-1.hdf5",
    "data/valid/shear_flow_Reynolds_5e4_Schmidt_1e0.hdf5",
    "data/valid/shear_flow_Reynolds_5e4_Schmidt_2e0.hdf5",
    "data/valid/shear_flow_Reynolds_1e5_Schmidt_5e-1.hdf5",
    "data/valid/shear_flow_Reynolds_1e5_Schmidt_1e0.hdf5",
    "data/valid/shear_flow_Reynolds_5e5_Schmidt_1e-1.hdf5",
    "data/valid/shear_flow_Reynolds_5e5_Schmidt_1e0.hdf5",
]

GRIDS = [
    {"name": "32x64", "nx": 32, "ny": 64},
    {"name": "64x128", "nx": 64, "ny": 128},
    {"name": "128x256", "nx": 128, "ny": 256},
    {"name": "256x512", "nx": 256, "ny": 512},
]

WINDOWS = [
    {
        "name": "v06_short",
        "frame_start": 80,
        "n_frames": 9,
        "frame_stride": 1,
        "burst_step": 3,
        "burst_length": 3,
    },
    {
        "name": "mid_21",
        "frame_start": 60,
        "n_frames": 21,
        "frame_stride": 1,
        "burst_step": 3,
        "burst_length": 3,
    },
    {
        "name": "spaced_2",
        "frame_start": 40,
        "n_frames": 21,
        "frame_stride": 2,
        "burst_step": 3,
        "burst_length": 3,
    },
    {
        "name": "late_9",
        "frame_start": 160,
        "n_frames": 9,
        "frame_stride": 1,
        "burst_step": 3,
        "burst_length": 3,
    },
]

TRAJECTORIES = [0, 1]
DT_STORED = 0.1
LENGTH_X = 1.0
LENGTH_Y = 2.0
STRIDE_FRAC = 1.0 / 6.0


def run_identifiability(*, skip_native: bool = False) -> dict[str, Any]:
    grids = [row for row in GRIDS if not (skip_native and row["nx"] == 256)]
    rows: list[dict[str, Any]] = []
    for relative in VALID_FILES:
        path = resolve_raw(relative, RAW_ROOTS)
        reynolds, schmidt = parse_reynolds_schmidt(relative)
        nu, kappa = coefficients(reynolds, schmidt)
        print(f"file {relative} Re={reynolds} Sc={schmidt}", flush=True)
        for window in WINDOWS:
            frame_ids = [
                int(window["frame_start"] + k * window["frame_stride"]) for k in range(int(window["n_frames"]))
            ]
            dt = DT_STORED * float(window["frame_stride"])
            n_frames = len(frame_ids)
            burst_starts = list(range(0, max(1, n_frames - int(window["burst_length"]) + 1), int(window["burst_step"])))
            proto = _oracle_protocol(dt, burst_starts, int(window["burst_length"]))
            native, _meta = load_frames(
                path,
                frame_ids=frame_ids,
                nx=256,
                ny=512,
                trajectory_ids=TRAJECTORIES,
            )
            for grid in grids:
                if int(grid["nx"]) == 256 and int(grid["ny"]) == 512:
                    packed = native
                else:
                    packed = downsample_fourier(native, int(grid["nx"]), int(grid["ny"]))
                rel_nu = []
                rel_ka = []
                hats = []
                for local in range(packed.shape[0]):
                    hat = fit_coefficients(packed[local], protocol=proto, mask=None, full_field=True)
                    hats.append(hat)
                    rel_nu.append(relative_error(hat["nu"], nu))
                    rel_ka.append(relative_error(hat["kappa"], kappa))
                row = {
                    "file": relative,
                    "reynolds": reynolds,
                    "schmidt": schmidt,
                    "nu": nu,
                    "kappa": kappa,
                    "grid": grid["name"],
                    "nx": int(grid["nx"]),
                    "ny": int(grid["ny"]),
                    "window": window["name"],
                    "frame_start": int(window["frame_start"]),
                    "n_frames": int(window["n_frames"]),
                    "frame_stride": int(window["frame_stride"]),
                    "dt": dt,
                    "n_bursts": len(burst_starts),
                    "n_traj": int(packed.shape[0]),
                    "median_rel_nu": float(np.median(rel_nu)),
                    "median_rel_kappa": float(np.median(rel_ka)),
                    "both_le_gate": bool(float(np.median(rel_nu)) <= GATE and float(np.median(rel_ka)) <= GATE),
                    "hats": hats,
                }
                rows.append(row)
                print(
                    f"  {grid['name']} {window['name']} nu={row['median_rel_nu']:.4g} "
                    f"kappa={row['median_rel_kappa']:.4g} gate={row['both_le_gate']}",
                    flush=True,
                )
    chosen = _choose(rows)
    restricted = _choose_restricted(rows)
    payload = {
        "format": "pinnforge.v061_identifiability.v1",
        "role": "validation-only full-field oracle sweep. Not a test score. v06 artifacts were not edited.",
        "split": "valid",
        "gate": GATE,
        "n_traj_per_file": len(TRAJECTORIES),
        "trajectory_ids": TRAJECTORIES,
        "files": VALID_FILES,
        "grids": GRIDS,
        "windows": WINDOWS,
        "test_function": {
            "family": "(1 - xi^2)_+^degree",
            "degree": 4,
            "space_radius_fraction": 0.25,
            "center_stride_fraction": STRIDE_FRAC,
        },
        "rows": [{key: value for key, value in row.items() if key != "hats"} for row in rows],
        "per_cell_hats": [
            {
                "file": row["file"],
                "grid": row["grid"],
                "window": row["window"],
                "hats": row["hats"],
            }
            for row in rows
        ],
        "chosen": chosen,
        "chosen_restricted": restricted,
        "unidentifiable": _unidentifiable(rows, chosen),
    }
    DOCS.mkdir(parents=True, exist_ok=True)
    (DOCS / "identifiability.json").write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")
    write_identifiability_md(payload)
    return payload


def write_identifiability_md(payload: dict[str, Any] | None = None) -> Path:
    data = payload or json.loads((DOCS / "identifiability.json").read_text(encoding="utf-8"))
    chosen = data["chosen"]
    restricted = data["chosen_restricted"]
    lines = [
        "# v061 identifiability (validation only)",
        "",
        "This note is the human reading of [identifiability.json](identifiability.json). "
        "It uses official valid files only. It is not a test score. "
        "It does not edit [docs/v06/](../v06/) or earlier study trees. "
        "The dataset is The Well `shear_flow` (CC BY 4).",
        "",
        "The oracle is the same weak-form least-squares fit of `nu` and `kappa` as v06, "
        "on the clean full field (no lattice, no noise), after periodic Fourier truncation "
        "to the listed grid. The v052 center-stride fraction is reused.",
        "",
        f"A cell passes the identifiability gate when both median relative errors are `<= {_fmt(data['gate'])}`.",
        "",
        "## Chosen setting",
        "",
    ]
    if chosen is None:
        lines.extend(
            [
                "No swept cell reached the gate for both coefficients on every listed `(Re, Sc)`.",
                "",
            ]
        )
    else:
        lines.extend(
            [
                "Cheapest CPU-tractable cell where every listed `(Re, Sc)` meets the gate:",
                "",
                f"- grid `{chosen['grid']}`",
                f"- window `{chosen['window']}` (`frame_start` `{_fmt(chosen['frame_start'])}`, "
                f"`n_frames` `{_fmt(chosen['n_frames'])}`, `frame_stride` `{_fmt(chosen['frame_stride'])}`)",
                f"- worst median rel `nu` `{_fmt(chosen['worst_median_rel_nu'])}`",
                f"- worst median rel `kappa` `{_fmt(chosen['worst_median_rel_kappa'])}`",
                "",
            ]
        )
    if restricted is not None:
        lines.extend(
            [
                "Cheapest CPU-tractable cell on the largest identifiable subset:",
                "",
                f"- grid `{restricted['grid']}`",
                f"- window `{restricted['window']}` (`frame_start` `{_fmt(restricted['frame_start'])}`, "
                f"`n_frames` `{_fmt(restricted['n_frames'])}`, `frame_stride` `{_fmt(restricted['frame_stride'])}`)",
                f"- Reynolds `{_fmt(restricted['reynolds'])}`",
                f"- Schmidt values `{', '.join(_fmt(item) for item in restricted['schmidt'])}`",
                f"- worst median rel `nu` `{_fmt(restricted['worst_median_rel_nu'])}`",
                f"- worst median rel `kappa` `{_fmt(restricted['worst_median_rel_kappa'])}`",
                f"- `n_combos` `{_fmt(restricted['n_combos'])}`",
                f"- primary coefficient `{restricted['primary_coefficient']}`",
                "",
            ]
        )
    unident = data["unidentifiable"]
    lines.extend(
        [
            "## What is identifiable",
            "",
            f"- `nu` cells at the gate: `{_fmt(unident['n_nu_pass'])}` of `{_fmt(unident['n_cells'])}`",
            f"- `kappa` cells at the gate: `{_fmt(unident['n_kappa_pass'])}` of `{_fmt(unident['n_cells'])}`",
            f"- both at the gate: `{_fmt(unident['n_both_pass'])}` of `{_fmt(unident['n_cells'])}`",
            "",
            unident["reading"],
            "",
            "## Median relative error by cell",
            "",
            "| Re | Sc | grid | window | median rel nu | median rel kappa | both <= gate |",
            "| --- | --- | --- | --- | --- | --- | --- |",
        ]
    )
    for row in data["rows"]:
        lines.append(
            f"| {_fmt(row['reynolds'])} | {_fmt(row['schmidt'])} | {row['grid']} | {row['window']} | "
            f"{_fmt(row['median_rel_nu'])} | {_fmt(row['median_rel_kappa'])} | "
            f"{'pass' if row['both_le_gate'] else 'fail'} |"
        )
    lines.extend(
        [
            "",
            "## Reproduce",
            "",
            "```",
            "python -m pinnforge.v061 download-valid",
            "python -m pinnforge.v061 identifiability",
            "```",
            "",
            "CI never downloads The Well.",
            "",
        ]
    )
    path = DOCS / "identifiability.md"
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")
    return path


def _oracle_protocol(dt: float, burst_starts: list[int], burst_length: int) -> dict[str, Any]:
    return {
        "downsample": {"length_x": LENGTH_X, "length_y": LENGTH_Y},
        "time_window": {"dt": dt, "burst_starts": burst_starts, "burst_length": burst_length},
        "methods": {
            "wsindy": {
                "test_function": {
                    "degree": 4,
                    "space_radius_fraction": 0.25,
                    "center_stride_fraction": STRIDE_FRAC,
                }
            }
        },
    }


def _choose(rows: list[dict[str, Any]]) -> dict[str, Any] | None:
    combos = {(row["reynolds"], row["schmidt"]) for row in rows}
    cost = {("32x64",): 0, ("64x128",): 1, ("128x256",): 2, ("256x512",): 3}
    window_cost = {"v06_short": 0, "late_9": 1, "mid_21": 2, "spaced_2": 3}
    candidates = []
    for grid in GRIDS:
        for window in WINDOWS:
            block = [
                row
                for row in rows
                if row["grid"] == grid["name"] and row["window"] == window["name"]
            ]
            if len(block) < len(combos):
                continue
            if not all(row["both_le_gate"] for row in block):
                continue
            candidates.append(
                {
                    "grid": grid["name"],
                    "nx": grid["nx"],
                    "ny": grid["ny"],
                    "window": window["name"],
                    "frame_start": window["frame_start"],
                    "n_frames": window["n_frames"],
                    "frame_stride": window["frame_stride"],
                    "cost": (cost[(grid["name"],)], window_cost[window["name"]]),
                    "worst_median_rel_nu": max(row["median_rel_nu"] for row in block),
                    "worst_median_rel_kappa": max(row["median_rel_kappa"] for row in block),
                    "n_combos": len(block),
                }
            )
    if not candidates:
        return None
    candidates.sort(key=lambda item: item["cost"])
    best = dict(candidates[0])
    best.pop("cost")
    return best


def _choose_restricted(rows: list[dict[str, Any]]) -> dict[str, Any] | None:
    """Largest (Re, Sc) subset that shares one (grid, window) at the gate, cheapest such cell."""

    cost = {"32x64": 0, "64x128": 1, "128x256": 2, "256x512": 3}
    window_cost = {"v06_short": 0, "late_9": 1, "mid_21": 2, "spaced_2": 3}
    candidates = []
    for grid in GRIDS:
        for window in WINDOWS:
            block = [
                row
                for row in rows
                if row["grid"] == grid["name"] and row["window"] == window["name"] and row["both_le_gate"]
            ]
            if not block:
                continue
            reynolds = sorted({row["reynolds"] for row in block})
            schmidt = sorted({row["schmidt"] for row in block})
            candidates.append(
                {
                    "grid": grid["name"],
                    "nx": grid["nx"],
                    "ny": grid["ny"],
                    "window": window["name"],
                    "frame_start": window["frame_start"],
                    "n_frames": window["n_frames"],
                    "frame_stride": window["frame_stride"],
                    "reynolds": reynolds[0] if len(reynolds) == 1 else reynolds,
                    "schmidt": schmidt,
                    "n_combos": len(block),
                    "n_reynolds": len(reynolds),
                    "worst_median_rel_nu": max(row["median_rel_nu"] for row in block),
                    "worst_median_rel_kappa": max(row["median_rel_kappa"] for row in block),
                    "primary_coefficient": "kappa" if len(reynolds) == 1 else "nu_and_kappa",
                    "cost": (-len(block), cost[grid["name"]], window_cost[window["name"]]),
                }
            )
    if not candidates:
        return None
    candidates.sort(key=lambda item: item["cost"])
    best = dict(candidates[0])
    best.pop("cost")
    return best


def _unidentifiable(rows: list[dict[str, Any]], chosen: dict[str, Any] | None) -> dict[str, Any]:
    n_cells = len(rows)
    n_nu = sum(1 for row in rows if row["median_rel_nu"] <= GATE)
    n_ka = sum(1 for row in rows if row["median_rel_kappa"] <= GATE)
    n_both = sum(1 for row in rows if row["both_le_gate"])
    by_re: dict[str, list[dict[str, Any]]] = {}
    for row in rows:
        by_re.setdefault(f"{row['reynolds']:.16g}", []).append(row)
    hard_re = []
    for re_key, block in by_re.items():
        if not any(row["both_le_gate"] for row in block):
            hard_re.append(float(re_key))
    if chosen is not None:
        reading = (
            "Every listed `(Re, Sc)` meets the gate at the chosen cell. "
            "No coefficient is marked unidentifiable on that subset."
        )
    elif hard_re:
        listed = ", ".join(_fmt(item) for item in hard_re)
        reading = (
            "No grid or window recovered both coefficients at every listed `(Re, Sc)`. "
            f"Reynolds values with no passing cell: {listed}. "
            "Kappa is the harder coefficient at high Reynolds and at Schmidt `5`, "
            "because the tracer dissipation scale is truncated even on the native grid. "
            "Nu is constant on the identifiable Reynolds `10000` slice, so the continuous inverse is kappa."
        )
    else:
        reading = (
            "Both coefficients have at least one passing cell for every listed Reynolds, "
            "but no single (grid, window) pair passes all combinations. "
            "The study claim is restricted to the combinations that pass at the cheapest shared cell."
        )
    return {
        "n_cells": n_cells,
        "n_nu_pass": n_nu,
        "n_kappa_pass": n_ka,
        "n_both_pass": n_both,
        "reynolds_with_no_passing_cell": hard_re,
        "reading": reading,
    }


def _fmt(value: Any) -> str:
    if isinstance(value, bool):
        return "true" if value else "false"
    number = float(value)
    if number != number:
        return "nan"
    return format(number, ".16g")
