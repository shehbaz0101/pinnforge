"""Write the v0.4.1 report and figures from committed scores."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

DOCS = Path("docs/v041")


def write_report(scores_path: Path | None = None) -> Path:
    scores = json.loads((scores_path or DOCS / "scores.json").read_text(encoding="utf-8"))
    protocol = json.loads((DOCS / "protocol.json").read_text(encoding="utf-8"))
    figures = _figures(scores)
    burgers = scores["problems"]["burgers2d"]["bars"]
    ns = scores["problems"]["ns2d"]["bars"]
    lines = [
        "# v0.4.1 2D viscosity inverse",
        "",
        "Research note. The contract is [protocol.json](protocol.json). "
        "Its SHA-256 is stored as `protocol_sha256` in [scores.json](scores.json). "
        "Hashes of the committed JSON files are in [reproducibility.md](reproducibility.md). "
        "Calibration is [validation.json](validation.json). No v0.4 artifact was edited. "
        "No hyperparameter was chosen on the test split.",
        "",
        "## Question",
        "",
        protocol["question"],
        "",
        "## Pass / fail",
        "",
        "Bars are the frozen v0.4 bars. They are not retuned.",
        "",
        "| problem | in-dist `ν` error `<= 0.05` | hard_ood / complement `<= 3` | warm error `<= 1.5×` cold | warm time `<= 0.25×` cold | coverage in `[0.85, 0.95]` | overall |",
        "| --- | --- | --- | --- | --- | --- | --- |",
    ]
    for problem in ("burgers2d", "ns2d"):
        bars = scores["problems"][problem]["bars"]
        cells = [
            problem,
            _cell(bars["checks"]["in_distribution_error"]),
            _cell(bars["checks"]["hard_ood_gap"]),
            _cell(bars["checks"]["warm_vs_cold_error"]),
            _cell(bars["checks"]["warm_vs_cold_time"]),
            _cell(bars["checks"]["coverage"]),
            "pass" if bars["pass"] else "fail",
        ]
        lines.append("| " + " | ".join(cells) + " |")
    lines.extend(
        [
            "",
            "Primary method: hybrid-FNO bracket plus local solver search, five-seed mean. "
            "Primary problem: `burgers2d`. "
            + _overall_sentence(burgers, ns),
            "",
            "## Headline tables (clean sensors)",
            "",
        ]
    )
    for problem in ("burgers2d", "ns2d"):
        lines.append(f"### {problem}")
        lines.append("")
        lines.append("| method | complement rel err | hard_ood rel err | failures | mean seconds |")
        lines.append("| --- | ---: | ---: | ---: | ---: |")
        clean = scores["problems"][problem]["noise"]["0.0"]
        for name, label in (
            ("closed_form_ls", "closed-form residual LS"),
            ("weak_form", "weak form"),
            ("fno_data", "FNO inverse, data loss"),
            ("fno_hybrid", "FNO inverse, hybrid loss"),
            ("solver_sensor_ic", "cold solver, sensor-lift IC"),
            ("solver_oracle_ic", "oracle-IC solver (ceiling)"),
            ("warm_data", "warm start, data-loss FNO bracket"),
            ("warm_hybrid", "warm start, hybrid FNO bracket"),
        ):
            summary = clean[name]["summary"]
            lines.append(
                "| {label} | {comp} | {hard} | {fail} | {sec} |".format(
                    label=label,
                    comp=_fmt(summary["complement"]["mean_rel_error"]),
                    hard=_fmt(summary["hard_ood"]["mean_rel_error"]),
                    fail=_fmt(summary["hard_ood"]["n_failures"]),
                    sec=_fmt(summary["full_test"]["mean_seconds"]),
                )
            )
        lines.append("")
        warm = clean["warm_hybrid"]["summary"]
        lines.append(
            f"Warm-start coverage on full test: {_fmt(warm['full_test'].get('coverage'))}. "
            f"Abstain rate: {_fmt(warm['full_test'].get('abstain_rate'))}. "
            f"Time / cold on hard_ood: {_fmt(warm['hard_ood'].get('time_ratio'))}. "
            f"n_local: {_fmt(scores['problems'][problem].get('n_local'))}."
        )
        lines.append("")
    lines.extend(
        [
            "## Noise",
            "",
            "Calibration is the clean calibration split. Nothing is refit.",
            "",
            "| problem | noise | cold hard_ood | warm hybrid hard_ood | warm time / cold | coverage |",
            "| --- | ---: | ---: | ---: | ---: | ---: |",
        ]
    )
    for problem in ("burgers2d", "ns2d"):
        for fraction, block in scores["problems"][problem]["noise"].items():
            cold = block["solver_sensor_ic"]["summary"]["hard_ood"]
            warm = block["warm_hybrid"]["summary"]["hard_ood"]
            lines.append(
                f"| {problem} | {fraction} | {_fmt(cold['mean_rel_error'])} | {_fmt(warm['mean_rel_error'])} | {_fmt(warm.get('time_ratio'))} | {_fmt(block['warm_hybrid']['summary']['full_test'].get('coverage'))} |"
            )
    lines.extend(
        [
            "",
            "## Figures",
            "",
            f"![Prediction versus reference]({figures['field']})",
            "",
            f"![Relative error against viscosity]({figures['error']})",
            "",
            f"![Interval coverage]({figures['coverage']})",
            "",
            "## What is NOT claimed",
            "",
        ]
    )
    lines.extend(f"- {item}" for item in protocol["not_claimed"])
    lines.extend(
        [
            "",
            "## Reproduce",
            "",
            "```bash",
            "python -m pinnforge.v041 generate",
            "python -m pinnforge.v041 train",
            "python -m pinnforge.v041 score",
            "python -m pinnforge.v041 report",
            "python -m pinnforge.claims check",
            "```",
            "",
            "CI runs `python -m pinnforge.v041 smoke` only. Pilot arrays are regenerated, not committed.",
            "",
        ]
    )
    path = DOCS / "report.md"
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")
    return path


def _overall_sentence(burgers: dict[str, Any], ns: dict[str, Any]) -> str:
    parts = []
    if burgers["pass"] and ns["pass"]:
        parts.append("Both problems pass overall.")
    elif not burgers["pass"] and not ns["pass"]:
        parts.append("Both problems fail overall.")
    else:
        parts.append(
            f"burgers2d {'passes' if burgers['pass'] else 'fails'}; ns2d {'passes' if ns['pass'] else 'fails'}."
        )
    for name, bars in (("burgers2d", burgers), ("ns2d", ns)):
        failed = [key for key, check in bars["checks"].items() if not check["pass"]]
        passed = [key for key, check in bars["checks"].items() if check["pass"]]
        if failed:
            parts.append(f"{name} missed {', '.join(failed)}.")
        if passed:
            parts.append(f"{name} passed {', '.join(passed)}.")
    return " ".join(parts)


def _cell(check: dict[str, Any]) -> str:
    mark = "pass" if check["pass"] else "fail"
    return f"{mark} ({_fmt(check.get('value'))})"


def _fmt(value: Any) -> str:
    if value is None:
        return "n/a"
    if isinstance(value, bool):
        return "true" if value else "false"
    number = float(value)
    if not np.isfinite(number):
        return "inf"
    return format(number, ".16g")


def _figures(scores: dict[str, Any]) -> dict[str, str]:
    dest = DOCS / "figures"
    dest.mkdir(parents=True, exist_ok=True)
    warm = scores["problems"]["burgers2d"]["noise"]["0.0"]["warm_hybrid"]["instances"]
    nus = [row["nu"] for row in warm]
    errs = [row["rel_error"] if row["rel_error"] is not None else np.nan for row in warm]
    fig, ax = plt.subplots(figsize=(5, 3.4))
    ax.scatter(nus, errs, c=["C1" if row["hard_ood"] else "C0" for row in warm], s=28)
    ax.set_xlabel("true ν")
    ax.set_ylabel("relative error")
    ax.set_title("Burgers warm start, clean sensors")
    fig.tight_layout()
    error_path = dest / "error_vs_nu.png"
    fig.savefig(error_path, dpi=120)
    plt.close(fig)

    fig, ax = plt.subplots(figsize=(5, 3.4))
    covered = [row.get("covered") for row in warm]
    lows = [row.get("interval_low") for row in warm]
    highs = [row.get("interval_high") for row in warm]
    order = np.argsort(nus)
    ax.vlines(
        [nus[i] for i in order],
        [lows[i] if lows[i] is not None else np.nan for i in order],
        [highs[i] if highs[i] is not None else np.nan for i in order],
        color="0.7",
    )
    ax.scatter([nus[i] for i in order], [nus[i] for i in order], s=16, c="k")
    ax.scatter(
        [row["nu"] for row in warm],
        [row["nu_hat"] for row in warm],
        s=20,
        c=["C2" if item else "C3" for item in covered],
    )
    ax.set_xlabel("true ν")
    ax.set_ylabel("ν")
    ax.set_title("nominal 0.9 intervals, Burgers warm start")
    fig.tight_layout()
    coverage_path = dest / "coverage.png"
    fig.savefig(coverage_path, dpi=120)
    plt.close(fig)

    fig, axes = plt.subplots(1, 2, figsize=(7.2, 3.2))
    field_path = dest / "prediction_vs_reference.png"
    artifact = Path("artifacts/v041_pilot/burgers2d_fields.npy")
    if artifact.is_file():
        from pinnforge.v04.sensors import lift_sensors, mask_from_protocol, read_sensors

        protocol = json.loads((DOCS / "protocol.json").read_text(encoding="utf-8"))
        fields = np.load(artifact)
        idx = int(warm[0]["instance_id"]) if warm else 0
        truth = fields[idx, -1, 0]
        mask = mask_from_protocol(protocol, n=truth.shape[0])
        lifted = lift_sensors(read_sensors(fields[idx], mask), mask)[-1, 0]
        axes[0].imshow(truth, cmap="RdBu_r", origin="lower")
        axes[0].set_title("reference u, last frame")
        axes[1].imshow(lifted, cmap="RdBu_r", origin="lower")
        axes[1].set_title("sensor-lifted u")
        for ax in axes:
            ax.set_xticks([])
            ax.set_yticks([])
    else:
        axes[0].text(0.5, 0.5, "pilot arrays not in git", ha="center")
        axes[1].axis("off")
    fig.tight_layout()
    fig.savefig(field_path, dpi=120)
    plt.close(fig)
    return {
        "error": "figures/error_vs_nu.png",
        "coverage": "figures/coverage.png",
        "field": "figures/prediction_vs_reference.png",
    }
