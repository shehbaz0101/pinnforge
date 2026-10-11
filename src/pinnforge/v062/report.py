"""Write the v0.6.2 report from committed scores."""

from __future__ import annotations

import hashlib
import json
from pathlib import Path
from typing import Any

DOCS = Path("docs/v062")


def write_report(scores_path: Path | None = None) -> Path:
    scores = json.loads((scores_path or DOCS / "scores.json").read_text(encoding="utf-8"))
    protocol = json.loads((DOCS / "protocol.json").read_text(encoding="utf-8"))
    bars = scores["bars"]
    overall = "PASS" if bars["overall"] else "FAIL"
    lines = [
        "# v062 sim-train / The Well-test inverse",
        "",
        "Research note. The contract is [protocol.json](protocol.json). "
        "Its SHA-256 is stored as `protocol_sha256` in [scores.json](scores.json). "
        "Solver validation is [solver_validation.md](solver_validation.md). "
        "No v061 or earlier artifact was edited. "
        "The dataset is The Well `shear_flow` (CC BY 4).",
        "",
        "## Question",
        "",
        protocol["question"],
        "",
        f"## Result: {overall}",
        "",
        "| bar | value | pass |",
        "| --- | --- | --- |",
        f"| in-dist median rel | {_fmt(bars['in_dist_median_rel'])} | {_yn(bars['pass']['in_dist_median_rel'])} |",
        f"| OOD median rel | {_fmt(bars['ood_median_rel'])} | {_yn(bars['pass']['ood_median_rel'])} |",
        f"| OOD / in-dist | {_fmt(bars['ood_over_in_dist'])} | {_yn(bars['pass']['ood_over_in_dist'])} |",
        f"| interval coverage | {_fmt(bars['interval_coverage'])} | {_yn(bars['pass']['interval_coverage'])} |",
        f"| hybrid OOD below both baselines | {_yn(bars['hybrid_beats_baselines_ood'])} | {_yn(bars['pass']['hybrid_beats_baselines_ood'])} |",
        f"| noise 0.05 not worse than WSINDy | {_yn(bars['noise_5pct_not_worse_than_wsindy'])} | {_yn(bars['pass']['noise_5pct_not_worse_than_wsindy'])} |",
        f"| below constant predictor | {_yn(bars['beats_constant_predictor'])} | {_yn(bars['pass']['beats_constant_predictor'])} |",
        "",
        "## Methods by noise",
        "",
        "| noise | slice | method | median rel nu | median rel kappa | combined | coverage |",
        "| --- | --- | --- | --- | --- | --- | --- |",
    ]
    for noise in ("0.00", "0.01", "0.05"):
        block = scores["by_noise"][noise]
        for slice_name in ("in_dist", "ood"):
            for method in ("wsindy", "fno_data", "hybrid", "sitl_cold", "oracle", "constant"):
                cell = block[slice_name][method]
                cov = cell.get("coverage")
                lines.append(
                    f"| {noise} | {slice_name} | {method} | {_fmt(cell['median_rel_nu'])} | "
                    f"{_fmt(cell['median_rel_kappa'])} | {_fmt(cell['median_rel'])} | {_fmt(cov)} |"
                )
    lines.extend(
        [
            "",
            "## What is not claimed",
            "",
            "- a result on Petrobras 3W",
            "- 3D",
            "- the full ensemble",
            "- that Dedalus was run in this study",
            "- that PySR was scored",
            "- identifiability at Reynolds 50000 or higher",
            "- a retuned bar after seeing test scores",
            "- a change to any v061 or earlier number",
            "",
            "## Attribution",
            "",
            "The Well `shear_flow` is CC BY 4 (Ohana et al.). "
            "The generator family is Dedalus (Burns et al.). "
            "[dataset page](https://polymathic-ai.org/the_well/datasets/shear_flow/).",
            "",
            "## Reproduce",
            "",
            "```",
            "python -m pinnforge.v062 generate",
            "python -m pinnforge.v062 pack-well",
            "python -m pinnforge.v062 score --checkpoint artifacts/v062_pilot/score_cases.jsonl",
            "python -m pinnforge.v062 report",
            "```",
            "",
            "CI runs `python -m pinnforge.v062 smoke` only.",
            "",
        ]
    )
    path = DOCS / "report.md"
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")
    files = [
        ("protocol.json", DOCS / "protocol.json"),
        ("scores.json", DOCS / "scores.json"),
        ("solver_validation.json", DOCS / "solver_validation.json"),
    ]
    lines = [
        "# v062 reproducibility",
        "",
        f"- python: `{scores.get('python')}`",
        f"- numpy: `{scores.get('numpy')}`",
        f"- torch: `{scores.get('torch')}`",
        f"- protocol_sha256: `{scores['protocol_sha256']}`",
        "",
        "| file | SHA-256 |",
        "| --- | --- |",
    ]
    for name, path_i in files:
        if path_i.is_file():
            lines.append(f"| [{name}]({name}) | `{hashlib.sha256(path_i.read_bytes()).hexdigest()}` |")
    (DOCS / "reproducibility.md").write_text("\n".join(lines) + "\n", encoding="utf-8")
    return path


def _fmt(value: Any) -> str:
    if value is None:
        return "—"
    if isinstance(value, bool):
        return "true" if value else "false"
    number = float(value)
    if number != number:
        return "nan"
    return format(number, ".16g")


def _yn(flag: bool) -> str:
    return "pass" if flag else "fail"
