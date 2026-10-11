"""Write the v0.6 report from committed scores."""

from __future__ import annotations

import hashlib
import json
from pathlib import Path
from typing import Any

DOCS = Path("docs/v06")


def write_report(scores_path: Path | None = None) -> Path:
    scores = json.loads((scores_path or DOCS / "scores.json").read_text(encoding="utf-8"))
    protocol = json.loads((DOCS / "protocol.json").read_text(encoding="utf-8"))
    bars = scores["bars"]
    overall = "PASS" if bars["overall"] else "FAIL"
    lines = [
        "# v06 The Well shear_flow inverse",
        "",
        "Research note. The contract is [protocol.json](protocol.json). "
        "Its SHA-256 is stored as `protocol_sha256` in [scores.json](scores.json). "
        "Hashes of the committed JSON files are in [reproducibility.md](reproducibility.md). "
        "No v052 or earlier artifact was edited. "
        "Raw HDF5 is not committed. "
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
        f"| hybrid beats OOD baselines | {_yn(bars['hybrid_beats_baselines_ood'])} | {_yn(bars['pass']['hybrid_beats_baselines_ood'])} |",
        f"| noise 0.05 not worse than WSINDy | {_yn(bars['noise_5pct_not_worse_than_wsindy'])} | {_yn(bars['pass']['noise_5pct_not_worse_than_wsindy'])} |",
        "",
        "## Methods by noise",
        "",
        "| noise | slice | method | median rel nu | median rel kappa | combined | coverage |",
        "| --- | --- | --- | --- | --- | --- | --- |",
    ]
    for noise in ("0.00", "0.01", "0.05"):
        block = scores["by_noise"][noise]
        for slice_name in ("in_dist", "ood"):
            for method in ("wsindy", "fno_data", "hybrid", "oracle"):
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
            "- that a Dedalus solver was run",
            "- that PySR was scored",
            "- a retuned bar after seeing test scores",
            "- a change to any v052, v051, v05, v041, or v04 number",
            "",
            "## Attribution",
            "",
            "The Well `shear_flow` is CC BY 4. Ohana et al., arXiv:2412.00568. "
            "The generator is Dedalus (Burns et al., Phys. Rev. Research 2, 023068, 2020). "
            "[dataset page](https://polymathic-ai.org/the_well/datasets/shear_flow/).",
            "",
            "## Reproduce",
            "",
            "```",
            "python -m pinnforge.v06 download",
            "python -m pinnforge.v06 pack",
            "python -m pinnforge.v06 score --checkpoint artifacts/v06_pilot/score_cases.jsonl",
            "python -m pinnforge.v06 report",
            "```",
            "",
            "CI runs `python -m pinnforge.v06 smoke` only. It never downloads The Well.",
            "",
        ]
    )
    path = DOCS / "report.md"
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")
    _write_reproducibility(scores)
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


def _write_reproducibility(scores: dict[str, Any]) -> None:
    files = [
        ("protocol.json", DOCS / "protocol.json"),
        ("scores.json", DOCS / "scores.json"),
        ("validation.json", DOCS / "validation.json"),
        ("pilot_manifest.json", DOCS / "pilot_manifest.json"),
        ("download_manifest.json", DOCS / "download_manifest.json"),
    ]
    lines = [
        "# v06 reproducibility",
        "",
        "CPU-only run. No paid GPU. Raw HDF5 stays under `artifacts/v06_raw/` and is not committed.",
        "",
        "The protocol [protocol.json](protocol.json) was committed before any v06 test score.",
        "",
        f"- python: `{scores.get('python')}`",
        f"- numpy: `{scores.get('numpy')}`",
        f"- torch: `{scores.get('torch')}`",
        f"- protocol_sha256: `{scores['protocol_sha256']}`",
        "",
        "| file | SHA-256 |",
        "| --- | --- |",
    ]
    for name, path in files:
        if path.is_file():
            digest = hashlib.sha256(path.read_bytes()).hexdigest()
            lines.append(f"| [{name}]({name}) | `{digest}` |")
    (DOCS / "reproducibility.md").write_text("\n".join(lines) + "\n", encoding="utf-8")
