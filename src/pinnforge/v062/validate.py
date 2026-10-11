"""Compare the numpy solver to The Well at the v0.6.1 identifiable setting."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import numpy as np

from pinnforge.v061.io import load_frames, resolve_raw
from pinnforge.v062.solver import (
    SF_IC_FAMILY,
    init_standard_shear,
    rel_l2,
    solve_shear,
)

DOCS = Path("docs/v062")
RAW_ROOTS = [Path("artifacts/v061_raw"), Path("artifacts/v06_raw"), Path("artifacts/v062_raw")]
WELL_FILE = "data/valid/shear_flow_Reynolds_1e4_Schmidt_1e0.hdf5"
REYNOLDS = 10000.0
SCHMIDT = 1.0
NU = 1.0 / REYNOLDS
KAPPA = NU / SCHMIDT
NX = 256
NY = 512
IDENT_START = 160
IDENT_FRAMES = 9
# Pre-stated gates, committed before the comparison numbers were written.
IC_REL_L2_MAX = 0.02
WINDOW_REL_L2_MAX = 0.15
FIELDS = ("ux", "uy", "pressure", "tracer")


def match_ic(ux: np.ndarray, uy: np.ndarray) -> dict[str, Any]:
    best: dict[str, Any] | None = None
    for spec in SF_IC_FAMILY:
        su, sv = init_standard_shear(ux.shape[0], ux.shape[1], n_shear=spec.n_shear, n_blobs=spec.n_blobs, width=spec.width)
        score = rel_l2(su, ux) + rel_l2(sv, uy)
        row = {
            "n_shear": spec.n_shear,
            "n_blobs": spec.n_blobs,
            "width": spec.width,
            "rel_l2_ux": rel_l2(su, ux),
            "rel_l2_uy": rel_l2(sv, uy),
            "score": score,
        }
        if best is None or score < best["score"]:
            best = row
    assert best is not None
    return best


def run_validation(*, dt: float = 0.002, traj: int = 3) -> dict[str, Any]:
    path = resolve_raw(WELL_FILE, RAW_ROOTS)
    t0, _meta = load_frames(path, frame_ids=[0], nx=NX, ny=NY, trajectory_ids=[traj])
    window, _wmeta = load_frames(
        path,
        frame_ids=list(range(IDENT_START, IDENT_START + IDENT_FRAMES)),
        nx=NX,
        ny=NY,
        trajectory_ids=[traj],
    )
    well0 = t0[0, 0]
    well_w = window[0]
    ic = match_ic(well0[0], well0[1])
    su, sv = init_standard_shear(NX, NY, n_shear=int(ic["n_shear"]), n_blobs=int(ic["n_blobs"]), width=float(ic["width"]))
    ic_l2 = {
        "ux": rel_l2(su, well0[0]),
        "uy": rel_l2(sv, well0[1]),
        "tracer": rel_l2(su, well0[3]),
        "pressure": rel_l2(np.zeros_like(well0[2]), well0[2]) if float(np.linalg.norm(well0[2])) > 1e-12 else 0.0,
    }
    # Short-horizon: start from The Well frame 160, integrate the late_9 window.
    rolled = solve_shear(
        well_w[0, 0],
        well_w[0, 1],
        well_w[0, 3],
        nu=NU,
        kappa=KAPPA,
        dt=dt,
        t_final=0.8,
        save_dt=0.1,
    )
    window_l2 = {}
    per_frame = []
    for k, name in enumerate(FIELDS):
        # Compare the last saved frame (t=0.8) to The Well frame 168.
        window_l2[name] = rel_l2(rolled["fields"][-1, k], well_w[-1, k])
    for frame in range(rolled["fields"].shape[0]):
        per_frame.append({name: rel_l2(rolled["fields"][frame, k], well_w[frame, k]) for k, name in enumerate(FIELDS)})
    ic_pass = ic_l2["ux"] <= IC_REL_L2_MAX and ic_l2["uy"] <= IC_REL_L2_MAX and ic_l2["tracer"] <= IC_REL_L2_MAX
    window_pass = all(window_l2[name] <= WINDOW_REL_L2_MAX for name in ("ux", "uy", "tracer"))
    payload = {
        "format": "pinnforge.v062_solver_validation.v1",
        "role": "solver vs The Well at the v0.6.1 identifiable setting. Not a test score.",
        "source": {
            "dataset": "The Well shear_flow",
            "file": WELL_FILE,
            "trajectory": traj,
            "generator": "RudyMorel/the-well-rbc-sf src/generate_sf.py",
            "dedalus_paper": "Burns et al., Phys. Rev. Research 2, 023068, 2020",
            "dataset_url": "https://polymathic-ai.org/the_well/datasets/shear_flow/",
        },
        "setting": {
            "reynolds": REYNOLDS,
            "schmidt": SCHMIDT,
            "nu": NU,
            "kappa": KAPPA,
            "nx": NX,
            "ny": NY,
            "dt": dt,
            "ident_frame_start": IDENT_START,
            "ident_n_frames": IDENT_FRAMES,
            "window_t": 0.8,
        },
        "gates": {
            "ic_rel_l2_max": IC_REL_L2_MAX,
            "window_rel_l2_max": WINDOW_REL_L2_MAX,
            "window_note": "Start from The Well frame 160 and integrate 0.8. Pressure is recovered, not evolved.",
        },
        "matched_ic": ic,
        "ic_rel_l2": ic_l2,
        "window_rel_l2": window_l2,
        "window_per_frame": per_frame,
        "ic_pass": ic_pass,
        "window_pass": window_pass,
        "overall_pass": bool(ic_pass and window_pass),
    }
    DOCS.mkdir(parents=True, exist_ok=True)
    (DOCS / "solver_validation.json").write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")
    write_validation_md(payload)
    return payload


def write_validation_md(payload: dict[str, Any] | None = None) -> Path:
    data = payload or json.loads((DOCS / "solver_validation.json").read_text(encoding="utf-8"))
    lines = [
        "# v062 solver validation",
        "",
        "This note is the human reading of [solver_validation.json](solver_validation.json). "
        "It is not a test score. It does not edit [docs/v061/](../v061/) or earlier study trees. "
        "The dataset is The Well `shear_flow` (CC BY 4).",
        "",
        "The numpy solver uses the Dedalus IC family in `RudyMorel/the-well-rbc-sf` "
        "(`init_standard_shear`) and the periodic incompressible NS + tracer equations "
        "on the The Well shear_flow domain.",
        "",
        f"Pre-stated IC gate: each of `ux`, `uy`, tracer rel L2 `<= {_fmt(data['gates']['ic_rel_l2_max'])}`.",
        f"Pre-stated window gate: start from The Well frame `{_fmt(data['setting']['ident_frame_start'])}`, "
        f"integrate `{_fmt(data['setting']['window_t'])}`, require `ux`, `uy`, tracer rel L2 `<= {_fmt(data['gates']['window_rel_l2_max'])}`.",
        "",
        f"Overall: `{'pass' if data['overall_pass'] else 'fail'}`.",
        "",
        "## Matched IC",
        "",
        f"- `n_shear` `{_fmt(data['matched_ic']['n_shear'])}`",
        f"- `n_blobs` `{_fmt(data['matched_ic']['n_blobs'])}`",
        f"- `width` `{_fmt(data['matched_ic']['width'])}`",
        f"- IC rel L2 ux `{_fmt(data['ic_rel_l2']['ux'])}`",
        f"- IC rel L2 uy `{_fmt(data['ic_rel_l2']['uy'])}`",
        f"- IC rel L2 tracer `{_fmt(data['ic_rel_l2']['tracer'])}`",
        f"- IC rel L2 pressure `{_fmt(data['ic_rel_l2']['pressure'])}`",
        "",
        "## Identifiable window",
        "",
        f"- ux `{_fmt(data['window_rel_l2']['ux'])}`",
        f"- uy `{_fmt(data['window_rel_l2']['uy'])}`",
        f"- pressure `{_fmt(data['window_rel_l2']['pressure'])}`",
        f"- tracer `{_fmt(data['window_rel_l2']['tracer'])}`",
        "",
    ]
    path = DOCS / "solver_validation.md"
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")
    return path


def write_stop_report(payload: dict[str, Any]) -> Path:
    """Human stop note when the solver does not meet the pre-stated gates."""

    lines = [
        "# v062 stop: solver does not match The Well",
        "",
        "The pre-stated validation gates failed. Scoring was not run. "
        "No v061 or earlier artifact was edited.",
        "",
        f"IC pass: `{'true' if payload['ic_pass'] else 'false'}`.",
        f"Window pass: `{'true' if payload['window_pass'] else 'false'}`.",
        "",
        "Numbers are in [solver_validation.json](solver_validation.json).",
        "",
    ]
    path = DOCS / "STOP.md"
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")
    return path


def _fmt(value: Any) -> str:
    if isinstance(value, bool):
        return "true" if value else "false"
    number = float(value)
    if number != number:
        return "nan"
    return format(number, ".16g")
