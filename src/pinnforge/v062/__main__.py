"""CLI for the v0.6.2 sim-train / The Well-test inverse."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

PROTOCOL = Path("docs/v062/protocol.json")


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="pinnforge v062",
        description="v0.6.2 sim-train The Well-test shear-flow inverse.",
    )
    sub = parser.add_subparsers(dest="command", required=True)
    sub.add_parser("protocol-hash", help="Print the SHA-256 of the frozen protocol")
    val = sub.add_parser("validate", help="Compare the numpy solver to The Well")
    val.add_argument("--dt", type=float, default=0.002)
    val.add_argument("--traj", type=int, default=3)
    gen = sub.add_parser("generate", help="Generate continuous-Sc training trajectories")
    gen.add_argument("--workers", type=int, default=4)
    sub.add_parser("pack-well", help="Pack The Well late_9 test window")
    score = sub.add_parser("score", help="Train FNO seeds, then score the frozen methods")
    score.add_argument("--workers", type=int, default=None)
    score.add_argument("--checkpoint", default=None)
    sub.add_parser("report", help="Write docs/v062/report.md")
    sub.add_parser("smoke", help="Tiny synthetic fixture; no dataset download")
    return parser


def main(argv: list[str] | None = None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)
    if args.command == "protocol-hash":
        if not PROTOCOL.is_file():
            print("protocol not frozen")
            return 1
        from pinnforge.reference.spectral2d.dataset import protocol_sha256

        print(protocol_sha256(PROTOCOL))
        return 0
    if args.command == "validate":
        from pinnforge.v062.validate import run_validation, write_stop_report

        payload = run_validation(dt=float(args.dt), traj=int(args.traj))
        print(json.dumps({"overall_pass": payload["overall_pass"], "ic_pass": payload["ic_pass"], "window_pass": payload["window_pass"]}))
        if not payload["overall_pass"]:
            write_stop_report(payload)
            return 2
        return 0
    if args.command == "generate":
        from pinnforge.v062.generate import run_generate

        record = run_generate(n_workers=int(args.workers))
        print(json.dumps({"n_sim": record["n_sim"]}))
        return 0
    if args.command == "pack-well":
        from pinnforge.v062.study import run_pack_well

        record = run_pack_well()
        print(json.dumps({"n_instances": record["n_instances"]}))
        return 0
    if args.command == "score":
        from pinnforge.v062.study import run_score

        scores = run_score(
            workers=args.workers,
            checkpoint=None if args.checkpoint is None else Path(args.checkpoint),
        )
        print("overall", "PASS" if scores["bars"]["overall"] else "FAIL")
        return 0
    if args.command == "report":
        from pinnforge.v062.report import write_report

        print(write_report())
        return 0
    if args.command == "smoke":
        return _smoke()
    raise ValueError(args.command)


def _smoke() -> int:
    import numpy as np

    from pinnforge.v062.solver import init_standard_shear, rel_l2, solve_shear

    ux, uy = init_standard_shear(16, 32, n_shear=2, n_blobs=2, width=1.0)
    rolled = solve_shear(ux, uy, ux, nu=0.05, kappa=0.05, dt=0.01, t_final=0.05, save_dt=0.01)
    assert rolled["fields"].shape == (6, 4, 16, 32)
    assert np.isfinite(rolled["fields"]).all()
    assert rel_l2(ux, ux) == 0.0
    print("v062 smoke ok solver fixture")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
