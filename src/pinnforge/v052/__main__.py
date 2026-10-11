"""CLI for the v0.5.2 Navier-Stokes WSINDy fix."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from pinnforge.reference.spectral2d.dataset import load_protocol, protocol_sha256

PROTOCOL = Path("docs/v052/protocol.json")


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="pinnforge v052",
        description="v0.5.2 Navier-Stokes WSINDy center-stride fix.",
    )
    sub = parser.add_subparsers(dest="command", required=True)
    sub.add_parser("protocol-hash", help="Print the SHA-256 of the frozen protocol")
    sub.add_parser("generate", help="Generate the 64x64 Navier-Stokes pilots")
    score = sub.add_parser("score", help="Choose thresholds on validation, then score the test split")
    score.add_argument(
        "--workers",
        type=int,
        default=None,
        help="Process-pool size across cases (default: PINNFORGE_V052_WORKERS or CPU count)",
    )
    score.add_argument(
        "--checkpoint",
        default=None,
        help="JSONL case log used to skip completed cases (default: artifacts/v052_pilot/score_cases.jsonl)",
    )
    sub.add_parser("report", help="Write docs/v052/report.md")
    sub.add_parser("smoke", help="Tiny CPU smoke that does not score the study")
    return parser


def main(argv: list[str] | None = None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)
    if args.command == "protocol-hash":
        print(protocol_sha256(PROTOCOL))
        return 0
    if args.command == "generate":
        from pinnforge.v052.study import run_generate

        record = run_generate()
        print(json.dumps({"problems": list(record["problems"]), "protocol_sha256": record["protocol_sha256"]}))
        return 0
    if args.command == "score":
        from pinnforge.v052.study import run_score

        scores = run_score(
            workers=args.workers,
            checkpoint=None if args.checkpoint is None else Path(args.checkpoint),
        )
        block = scores["problems"]["ns2d"]
        print("ns2d", "min_lattice", block["min_lattice_at_bar"], "control", block["positive_control"]["reading"])
        return 0
    if args.command == "report":
        from pinnforge.v052.report import write_report

        path = write_report()
        print(path)
        return 0
    if args.command == "smoke":
        return _smoke()
    raise ValueError(args.command)


def _smoke() -> int:
    import numpy as np

    from pinnforge.v05.wsindy import exact_support, stlsq, support_of
    from pinnforge.v051.masks import mask_for_lattice
    from pinnforge.v052.stride import stride_is_incommensurate

    protocol = load_protocol(PROTOCOL)
    assert protocol["frozen_before_test_evaluation"]
    assert protocol["test_used_for_selection"] is False
    assert protocol["methods"]["wsindy_fno"]["used"] is False
    assert protocol["methods"]["pysr"]["used"] is False
    assert protocol["burgers2d"]["scored"] is False
    assert protocol["reported"]["min_seed_success_rate"] == 0.8
    assert protocol["reported"]["noise"] == 0.01
    assert [item["n_side"] for item in protocol["lattices"]] == [8, 16, 32, 64]
    frac = float(protocol["methods"]["wsindy"]["test_function"]["center_stride_fraction"])
    wave = int(protocol["problems"]["ns2d"]["forcing"]["kolmogorov"]["wavenumber"])
    for n_side in (8, 16, 32, 64):
        assert stride_is_incommensurate(n_side, frac, wave)
    mask = mask_for_lattice(protocol, 8)
    assert mask.n_side == 8
    assert mask.n == 64
    rng = np.random.default_rng(0)
    truth = np.array([-1.0, -1.0, 0.5, 0.5, 1.0, 0.0, 0.0], dtype=np.float64)
    design = rng.normal(size=(80, truth.size))
    target = design @ truth
    names = ("u_wx", "v_wy", "w_xx", "w_yy", "force_y", "const", "force_x")
    coef = stlsq(design, target, 0.08, max_iter=10, column_normalize=True)
    assert exact_support(support_of(coef, names), ("u_wx", "v_wy", "w_xx", "w_yy", "force_y"))
    print("v052 smoke ok", protocol_sha256(PROTOCOL)[:12])
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
