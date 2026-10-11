"""Write the v0.5.1 report from committed scores."""

from __future__ import annotations

import hashlib
import json
from pathlib import Path
from typing import Any

DOCS = Path("docs/v051")


def write_report(scores_path: Path | None = None) -> Path:
    scores = json.loads((scores_path or DOCS / "scores.json").read_text(encoding="utf-8"))
    protocol = json.loads((DOCS / "protocol.json").read_text(encoding="utf-8"))
    validation = json.loads((DOCS / "validation.json").read_text(encoding="utf-8"))
    bar_noise = _fmt(protocol["reported"]["noise"])
    bar_rate = _fmt(protocol["reported"]["min_seed_success_rate"])
    lines = [
        "# v051 sensor-density study",
        "",
        "Research note. The contract is [protocol.json](protocol.json). "
        "Its SHA-256 is stored as `protocol_sha256` in [scores.json](scores.json). "
        "Hashes of the committed JSON files are in [reproducibility.md](reproducibility.md). "
        "Thresholds chosen on validation are [validation.json](validation.json). "
        "No v05, v041, or v04 artifact was edited. "
        "No hyperparameter was chosen on the test split. "
        "The FNO-denoise arm and PySR were not run; those skips are in the protocol.",
        "",
        "## Question",
        "",
        protocol["question"],
        "",
        "## Reported quantity",
        "",
        f"Smallest lattice whose `wsindy_raw` seed exact-support rate at noise `{bar_noise}` "
        f"is `>= {bar_rate}`, or `none`. The bar was not retuned.",
        "",
        "| problem | min lattice @ 0.01 | 64x64 noise 0 control | control reading |",
        "| --- | --- | --- | --- |",
    ]
    for problem in ("burgers2d", "ns2d"):
        block = scores["problems"][problem]
        control = block["positive_control"]
        lines.append(
            f"| {problem} | {block['min_lattice_at_bar']} | "
            f"{_fmt(control['exact_support_rate'])} | {control['reading']} |"
        )
    lines.extend(
        [
            "",
            "## Density by noise",
            "",
            "| problem | lattice | noise | seed exact | instance exact | coeff rel err |",
            "| --- | --- | --- | --- | --- | --- |",
        ]
    )
    for problem in ("burgers2d", "ns2d"):
        by_lattice = scores["problems"][problem]["by_lattice"]
        for lattice in protocol["lattices"]:
            name = str(lattice["name"])
            for noise in ("0.00", "0.01", "0.02", "0.05"):
                cell = by_lattice[name][noise]
                lines.append(
                    f"| {problem} | {name} | {noise} | {_fmt(cell['exact_support_rate'])} | "
                    f"{_fmt(cell['instance_exact_rate'])} | {_fmt(cell['coeff_rel_error'])} |"
                )
    lines.extend(
        [
            "",
            "## What is not claimed",
            "",
            "- a result on The Well shear_flow or Petrobras 3W",
            "- 3D",
            "- that the FNO-denoise arm was scored",
            "- that PySR was scored",
            "- a retuned bar after seeing test scores",
            "- a change to any v05, v041, or v04 number",
            "",
            "## Reproduce",
            "",
            "```",
            "python -m pinnforge.v051 generate",
            "python -m pinnforge.v051 score --checkpoint artifacts/v051_pilot/score_cases.jsonl",
            "python -m pinnforge.v051 report",
            "```",
            "",
            "CI runs `python -m pinnforge.v051 smoke` only. Pilot arrays and the library-matrix "
            "JSONL stay under `artifacts/v051_pilot/` and are not committed. "
            "Scoring is resumable: finished JSONL rows are skipped.",
            "",
        ]
    )
    path = DOCS / "report.md"
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")
    _write_reproducibility(scores, validation)
    return path


def _fmt(value: Any) -> str:
    if value is None:
        return "—"
    number = float(value)
    if number != number:
        return "nan"
    return format(number, ".16g")


def _write_reproducibility(scores: dict[str, Any], validation: dict[str, Any]) -> None:
    files = [
        ("protocol.json", DOCS / "protocol.json"),
        ("scores.json", DOCS / "scores.json"),
        ("validation.json", DOCS / "validation.json"),
        ("pilot_manifest.json", DOCS / "pilot_manifest.json"),
    ]
    lines = [
        "# v051 reproducibility",
        "",
        "CPU-only run in this repository. No paid GPU. No secrets. "
        "The detached runner is [scripts/v051_run.sh](../../scripts/v051_run.sh).",
        "",
        "The protocol [protocol.json](protocol.json) was committed before any v051 test score. "
        "It was not edited after that commit. No v0.5 artifact was edited.",
        "",
        f"- python: `{scores.get('python')}`",
        f"- numpy: `{scores.get('numpy')}`",
        f"- torch: `{scores.get('torch')}`",
        f"- protocol_sha256: `{scores['protocol_sha256']}`",
        f"- validation protocol_sha256: `{validation['protocol_sha256']}`",
        f"- PySR used: `{scores['pysr']['used']}`",
        f"- FNO-denoise used: `{scores['wsindy_fno']['used']}`",
        "",
        "SHA-256 of committed files:",
        "",
        "| file | SHA-256 |",
        "| --- | --- |",
    ]
    for name, path in files:
        if path.is_file():
            digest = hashlib.sha256(path.read_bytes()).hexdigest()
            lines.append(f"| [{name}]({name}) | `{digest}` |")
    lines.extend(
        [
            "",
            "Pilot arrays and the library-matrix JSONL stay under `artifacts/v051_pilot/` "
            "and are not committed.",
            "",
            "```bash",
            "python -m pinnforge.v051 generate",
            "python -m pinnforge.v051 score --checkpoint artifacts/v051_pilot/score_cases.jsonl",
            "python -m pinnforge.v051 report",
            "```",
            "",
        ]
    )
    (DOCS / "reproducibility.md").write_text("\n".join(lines) + "\n", encoding="utf-8")
