"""CLI for the v0.5 2D PDE term-discovery study."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from pinnforge.reference.spectral2d.dataset import load_protocol, protocol_sha256

PROTOCOL = Path("docs/v05/protocol.json")


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="pinnforge v05",
        description="v0.5 2D PDE term discovery with WSINDy on raw and FNO-denoised fields.",
    )
    sub = parser.add_subparsers(dest="command", required=True)
    sub.add_parser("protocol-hash", help="Print the SHA-256 of the frozen protocol")
    sub.add_parser("generate", help="Generate the 64x64 pilots")
    train = sub.add_parser("train", help="Train the 2D FNO denoisers")
    train.add_argument("--seeds", default=None, help="Comma-separated seeds (default: protocol seeds)")
    train.add_argument(
        "--workers",
        type=int,
        default=None,
        help="Process-pool size across problem/seed runs (default: CPU count)",
    )
    score = sub.add_parser("score", help="Choose thresholds on validation, then score the test split")
    score.add_argument(
        "--workers",
        type=int,
        default=None,
        help="Process-pool size across cases (default: PINNFORGE_V05_WORKERS or CPU count)",
    )
    score.add_argument(
        "--checkpoint",
        default=None,
        help="JSONL case log used to skip completed cases (default: artifacts/v05_pilot/score_cases.jsonl)",
    )
    sub.add_parser("report", help="Write docs/v05/report.md")
    sub.add_parser("smoke", help="Tiny CPU smoke that does not score the study")
    return parser


def main(argv: list[str] | None = None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)
    if args.command == "protocol-hash":
        print(protocol_sha256(PROTOCOL))
        return 0
    if args.command == "generate":
        from pinnforge.v05.study import run_generate

        record = run_generate()
        print(json.dumps({"problems": list(record["problems"]), "protocol_sha256": record["protocol_sha256"]}))
        return 0
    if args.command == "train":
        from pinnforge.v05.study import run_train

        seeds = None if args.seeds is None else [int(item) for item in args.seeds.split(",")]
        run_train(seeds=seeds, workers=args.workers)
        return 0
    if args.command == "score":
        from pinnforge.v05.study import run_score

        scores = run_score(
            workers=args.workers,
            checkpoint=None if args.checkpoint is None else Path(args.checkpoint),
        )
        print("overall", "pass" if scores["overall"]["pass"] else "fail")
        for problem, block in scores["problems"].items():
            print(problem, "pass" if block["bars"]["pass"] else "fail")
        return 0
    if args.command == "report":
        from pinnforge.v05.report import write_report

        path = write_report()
        print(path)
        return 0
    if args.command == "smoke":
        return _smoke()
    raise ValueError(args.command)


def _smoke() -> int:
    import numpy as np

    from pinnforge.v05.wsindy import exact_support, stlsq, support_of

    protocol = load_protocol(PROTOCOL)
    assert protocol["frozen_before_test_evaluation"]
    assert protocol["test_used_for_selection"] is False
    assert protocol["methods"]["pysr"]["used"] is False
    assert protocol["pass_bars"]["min_seed_success_rate"] == 0.8
    assert protocol["pass_bars"]["noise"] == 0.01
    rng = np.random.default_rng(0)
    truth = np.array([-1.0, -1.0, 0.5, 0.5, 0.0, 0.0, 0.0], dtype=np.float64)
    design = rng.normal(size=(80, truth.size))
    target = design @ truth
    names = ("u_qx", "v_qy", "q_xx", "q_yy", "q2", "q_xxx", "const")
    coef = stlsq(design, target, 0.08, max_iter=10, column_normalize=True)
    recovered = support_of(coef, names)
    assert exact_support(recovered, ("u_qx", "v_qy", "q_xx", "q_yy"))
    from pinnforge.ml_import import require_torch
    from pinnforge.v04.fno2d import FNO2d

    torch = require_torch()
    model = FNO2d(6, 6, width=4, modes=2, n_layers=1)
    pred = model(torch.zeros(1, 6, 8, 8))
    assert pred.shape == (1, 6, 8, 8)
    print("v05 smoke ok", protocol_sha256(PROTOCOL)[:12])
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
