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
    bar_noise = _fmt(protocol["pass_bars"]["noise"])
    bar_rate = _fmt(protocol["pass_bars"]["min_seed_success_rate"])
    lines = [
        "# v0.5 2D PDE term discovery",
        "",
        "Research note. The contract is [protocol.json](protocol.json). "
        "Its SHA-256 is stored as `protocol_sha256` in [scores.json](scores.json). "
        "Hashes of the committed JSON files are in [reproducibility.md](reproducibility.md). "
        "Thresholds chosen on validation are [validation.json](validation.json). "
        "No v04 or v041 artifact was edited. "
        "No hyperparameter was chosen on the test split. "
        "PySR was not run; that skip is in the protocol.",
        "",
        "## Question",
        "",
        "With sparse noisy sensors of the existing 2D Burgers and vorticity-form "
        "Navier-Stokes generators, can weak-form sparse regression (WSINDy) recover "
        "the exact PDE support — the true terms and no extras — on raw probes and on "
        f"fields denoised by a reused 2D FNO, against a frozen `{bar_rate}` "
        f"seed-success bar at noise `{bar_noise}`?",
        "",
        "## Pass / fail",
        "",
        f"**{overall}.** The frozen bar is exact support in `>= {bar_rate}` of seeds "
        f"at noise `{bar_noise}` for at least one of `wsindy_raw` and `wsindy_fno` "
        "on each problem. Navier-Stokes uses the Kolmogorov family. The bar was not retuned.",
        "",
        f"| problem | method | exact support @ {bar_noise} | bar |",
        "| --- | --- | --- | --- |",
    ]
    for problem in ("burgers2d", "ns2d"):
        bars = scores["problems"][problem]["bars"]
        for method, block in bars["methods"].items():
            lines.append(
                f"| {problem} | `{method}` | {_fmt(block['exact_support_rate'])} | "
                f"{'pass' if block['pass'] else 'fail'} |"
            )
        lines.append(f"| {problem} | overall |  | {'pass' if bars['pass'] else 'fail'} |")
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
            "- a result on The Well shear_flow or Petrobras 3W",
            "- 3D",
            "- that PySR was scored",
            "- a retuned bar after seeing test scores",
            "- a change to any v04 or v041 number",
            "",
            "## Reproduce",
            "",
            "```",
            "python -m pinnforge.v05 generate",
            "python -m pinnforge.v05 train",
            "python -m pinnforge.v05 score --checkpoint artifacts/v05_pilot/score_cases.jsonl",
            "python -m pinnforge.v05 report",
            "```",
            "",
            "CI runs `python -m pinnforge.v05 smoke` only. Pilot arrays and the library-matrix "
            "JSONL stay under `artifacts/v05_pilot/` and are not committed. "
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
    prefix = (
        "Both problems fail overall. "
        if (not burgers["pass"] and not ns["pass"])
        else "The preregistered bar fails overall. "
    )
    return prefix + "; ".join(parts) + "."


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
        ("train_summary.json", DOCS / "train_summary.json"),
    ]
    lines = [
        "# v0.5 reproducibility",
        "",
        "CPU-only run in this repository. No paid GPU. No secrets. "
        "The detached runner is [scripts/v05_run.sh](../../scripts/v05_run.sh).",
        "",
        "The protocol [protocol.json](protocol.json) was committed in `4aed3558` "
        "before any v0.5 test score. It was not edited after that commit.",
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
            "Checkpoints, pilot arrays, and the library-matrix JSONL stay under "
            "`artifacts/v05_pilot/` and are not committed. The JSONL holds full design "
            "matrices and is too large to keep next to the scores.",
            "",
            "```bash",
            "python -m pinnforge.v05 generate",
            "python -m pinnforge.v05 train",
            "python -m pinnforge.v05 score --checkpoint artifacts/v05_pilot/score_cases.jsonl",
            "python -m pinnforge.v05 report",
            "```",
            "",
        ]
    )
    (DOCS / "reproducibility.md").write_text("\n".join(lines) + "\n", encoding="utf-8")
