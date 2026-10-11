"""Write the v0.5.2 report from committed scores."""

from __future__ import annotations

import hashlib
import json
from pathlib import Path
from typing import Any

DOCS = Path("docs/v052")
V051 = Path("docs/v051")


def write_report(scores_path: Path | None = None) -> Path:
    scores = json.loads((scores_path or DOCS / "scores.json").read_text(encoding="utf-8"))
    protocol = json.loads((DOCS / "protocol.json").read_text(encoding="utf-8"))
    validation = json.loads((DOCS / "validation.json").read_text(encoding="utf-8"))
    parent = json.loads((V051 / "scores.json").read_text(encoding="utf-8"))
    bar_noise = _fmt(protocol["reported"]["noise"])
    bar_rate = _fmt(protocol["reported"]["min_seed_success_rate"])
    ns = scores["problems"]["ns2d"]
    burgers = parent["problems"]["burgers2d"]
    lines = [
        "# v052 Navier-Stokes WSINDy fix",
        "",
        "Research note. The contract is [protocol.json](protocol.json). "
        "Its SHA-256 is stored as `protocol_sha256` in [scores.json](scores.json). "
        "Hashes of the committed JSON files are in [reproducibility.md](reproducibility.md). "
        "Thresholds chosen on validation are [validation.json](validation.json). "
        "The diagnosis is [diagnosis.md](diagnosis.md). "
        "No v051, v05, v041, or v04 artifact was edited. "
        "Burgers was not rescored. "
        "No hyperparameter was chosen on the test split. "
        "The FNO-denoise arm and PySR were not run; those skips are in the protocol.",
        "",
        "## Question",
        "",
        protocol["question"],
        "",
        "## Root cause",
        "",
        "The v051 `64x64` noise-0 control failed because `center_stride_fraction` "
        "`0.25` placed every bump center one Kolmogorov period apart, so `force_y`, "
        "`const`, and `force_x` were rank one. The generator PDE and the library "
        "true support already matched. The registered fix is the incommensurate "
        "stride in protocol.json. Kolmogorov forcing stays in the library.",
        "",
        "## Reported quantity",
        "",
        f"Smallest lattice whose `wsindy_raw` seed exact-support rate at noise `{bar_noise}` "
        f"is `>= {bar_rate}`, or `none`. The bar was not retuned. Burgers is the "
        "frozen v051 number.",
        "",
        "| problem | min lattice @ 0.01 | 64x64 noise 0 control | control reading | source |",
        "| --- | --- | --- | --- | --- |",
        f"| burgers2d | {burgers['min_lattice_at_bar']} | "
        f"{_fmt(burgers['positive_control']['exact_support_rate'])} | "
        f"{burgers['positive_control']['reading']} | v051, not rescored |",
        f"| ns2d | {ns['min_lattice_at_bar']} | "
        f"{_fmt(ns['positive_control']['exact_support_rate'])} | "
        f"{ns['positive_control']['reading']} | this study |",
        "",
        "## Density by noise (ns2d only)",
        "",
        "| problem | lattice | noise | seed exact | instance exact | coeff rel err |",
        "| --- | --- | --- | --- | --- | --- |",
    ]
    by_lattice = ns["by_lattice"]
    for lattice in protocol["lattices"]:
        name = str(lattice["name"])
        for noise in ("0.00", "0.01", "0.02", "0.05"):
            cell = by_lattice[name][noise]
            lines.append(
                f"| ns2d | {name} | {noise} | {_fmt(cell['exact_support_rate'])} | "
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
            "- that Kolmogorov forcing was treated as a known right-hand-side term",
            "- that Burgers was rescored",
            "- a retuned bar after seeing test scores",
            "- a change to any v051, v05, v041, or v04 number",
            "",
            "## Reproduce",
            "",
            "```",
            "python -m pinnforge.v052 generate",
            "python -m pinnforge.v052 score --checkpoint artifacts/v052_pilot/score_cases.jsonl",
            "python -m pinnforge.v052 report",
            "```",
            "",
            "CI runs `python -m pinnforge.v052 smoke` only. Pilot arrays and the library-matrix "
            "JSONL stay under `artifacts/v052_pilot/` and are not committed. "
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
        ("diagnosis_evidence.json", DOCS / "diagnosis_evidence.json"),
    ]
    lines = [
        "# v052 reproducibility",
        "",
        "CPU-only run in this repository. No paid GPU. No secrets. "
        "The detached runner is [scripts/v052_run.sh](../../scripts/v052_run.sh).",
        "",
        "The protocol [protocol.json](protocol.json) was committed before any v052 test score. "
        "It was not edited after that commit. No v0.5.1 or v0.5 artifact was edited. "
        "Burgers was not rescored.",
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
            "Pilot arrays and the library-matrix JSONL stay under `artifacts/v052_pilot/` "
            "and are not committed.",
            "",
            "```bash",
            "python -m pinnforge.v052 generate",
            "python -m pinnforge.v052 score --checkpoint artifacts/v052_pilot/score_cases.jsonl",
            "python -m pinnforge.v052 report",
            "```",
            "",
        ]
    )
    (DOCS / "reproducibility.md").write_text("\n".join(lines) + "\n", encoding="utf-8")
