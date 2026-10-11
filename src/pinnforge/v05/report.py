"""Write the v0.5 report from committed scores."""

from __future__ import annotations

import hashlib
import json
from pathlib import Path
from typing import Any

DOCS = Path("docs/v05")


def write_report(scores_path: Path | None = None) -> Path:
    scores = json.loads((scores_path or DOCS / "scores.json").read_text(encoding="utf-8"))
    protocol = json.loads((DOCS / "protocol.json").read_text(encoding="utf-8"))
    validation = json.loads((DOCS / "validation.json").read_text(encoding="utf-8"))
    burgers = scores["problems"]["burgers2d"]["bars"]
    ns = scores["problems"]["ns2d"]["bars"]
    overall = "PASS" if scores["overall"]["pass"] else "FAIL"
    lines = [
        "# v0.5 2D PDE term discovery",
        "",
        "Research note. The contract is [protocol.json](protocol.json). "
        "Its SHA-256 is stored as `protocol_sha256` in [scores.json](scores.json). "
        "Hashes of the committed JSON files are in [reproducibility.md](reproducibility.md). "
        "Thresholds chosen on validation are [validation.json](validation.json). "
        "No v0.4 or v0.4.1 artifact was edited. "
        "No hyperparameter was chosen on the test split. "
        "PySR was not run; that skip is in the protocol.",
        "",
        "## Question",
        "",
        protocol["question"],
        "",
        "## Pass / fail",
        "",
        f"**{overall}.** The frozen bar is exact support in `>= 80%` of seeds at `1%` noise "
        "for at least one of `wsindy_raw` and `wsindy_fno` on each problem. "
        "Navier-Stokes uses the Kolmogorov family. The bar was not retuned.",
        "",
        "| problem | method | exact support @ 1% | bar |",
        "| --- | --- | --- | --- |",
    ]
    for problem in ("burgers2d", "ns2d"):
        bars = scores["problems"][problem]["bars"]
        for method, block in bars["methods"].items():
            lines.append(
                f"| {problem} | `{method}` | {_fmt(block['exact_support_rate'])} | "
                f"{'pass' if block['pass'] else 'fail'} |"
            )
        lines.append(
            f"| {problem} | overall |  | {'pass' if bars['pass'] else 'fail'} |"
        )
    lines.extend(
        [
            "",
            _overall_sentence(burgers, ns, scores["overall"]["pass"]),
            "",
            "## Every noise fraction",
            "",
            "| problem | noise | method | seed exact | instance exact | coeff rel err | FNO rel L2 |",
            "| --- | --- | --- | --- | --- | --- | --- |",
        ]
    )
    for problem in ("burgers2d", "ns2d"):
        by_noise = scores["problems"][problem]["by_noise"]
        for noise in ("0.00", "0.01", "0.02", "0.05"):
            for method, block in by_noise[noise].items():
                lines.append(
                    "| {problem} | {noise} | `{method}` | {rate} | {inst} | {coef} | {fno} |".format(
                        problem=problem,
                        noise=noise,
                        method=method,
                        rate=_fmt(block["exact_support_rate"]),
                        inst=_fmt(block["instance_exact_rate"]),
                        coef=_fmt(block["coeff_rel_error"]),
                        fno=_fmt(block["fno_rel_l2"]) if block["fno_rel_l2"] is not None else "—",
                    )
                )
    lines.extend(
        [
            "",
            "## What is not claimed",
            "",
        ]
    )
    lines.extend(f"- {item}" for item in protocol["not_claimed"])
    lines.extend(
        [
            "",
            "## Reproduce",
            "",
            "```",
            "python -m pinnforge.v05 generate",
            "python -m pinnforge.v05 train",
            "python -m pinnforge.v05 score --checkpoint docs/v05/score_cases.jsonl",
            "python -m pinnforge.v05 report",
            "```",
            "",
            "CI runs `python -m pinnforge.v05 smoke` only. Pilot arrays are regenerated, not committed. "
            "Scoring is resumable: finished JSONL rows are skipped.",
            "",
        ]
    )
    path = DOCS / "report.md"
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")
    _write_reproducibility(scores, validation)
    return path


def _overall_sentence(burgers: dict[str, Any], ns: dict[str, Any], overall: bool) -> str:
    if overall:
        return "Both problems pass the preregistered bar."
    parts = []
    for name, block in (("burgers2d", burgers), ("ns2d", ns)):
        parts.append(f"{name} {'passes' if block['pass'] else 'fails'}")
    prefix = "Both problems fail overall. " if (not burgers["pass"] and not ns["pass"]) else "The preregistered bar fails overall. "
    return prefix + "; ".join(parts) + "."


def _fmt(value: Any) -> str:
    if value is None:
        return "—"
    number = float(value)
    if number != number:
        return "nan"
    return f"{number:.4f}"


def _write_reproducibility(scores: dict[str, Any], validation: dict[str, Any]) -> None:
    files = {
        "protocol.json": DOCS / "protocol.json",
        "scores.json": DOCS / "scores.json",
        "validation.json": DOCS / "validation.json",
        "report.md": DOCS / "report.md",
    }
    lines = [
        "# v0.5 reproducibility",
        "",
        f"- python: `{scores.get('python')}`",
        f"- numpy: `{scores.get('numpy')}`",
        f"- torch: `{scores.get('torch')}`",
        f"- protocol_sha256: `{scores['protocol_sha256']}`",
        f"- validation protocol_sha256: `{validation['protocol_sha256']}`",
        f"- PySR used: `{scores['pysr']['used']}`",
        "",
        "SHA-256 of committed files:",
        "",
    ]
    for name, path in files.items():
        if path.is_file():
            digest = hashlib.sha256(path.read_bytes()).hexdigest()
            lines.append(f"- `{name}`: `{digest}`")
    (DOCS / "reproducibility.md").write_text("\n".join(lines) + "\n", encoding="utf-8")
