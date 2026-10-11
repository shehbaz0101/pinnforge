"""CLI for the v0.4.1 2D viscosity inverse."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from pinnforge.reference.spectral2d.dataset import load_protocol, protocol_sha256

PROTOCOL = Path("docs/v041/protocol.json")


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="pinnforge v041",
        description="v0.4.1 2D viscosity inverse with a larger calibration split.",
    )
    sub = parser.add_subparsers(dest="command", required=True)
    sub.add_parser("protocol-hash", help="Print the SHA-256 of the frozen protocol")
    sub.add_parser("generate", help="Generate the 64x64 pilots")
    train = sub.add_parser("train", help="Train the 2D FNO arms")
    train.add_argument("--seeds", default=None, help="Comma-separated seeds (default: protocol seeds)")
    sub.add_parser("score", help="Calibrate, then score the frozen test split")
    sub.add_parser("report", help="Write docs/v041/report.md and figures")
    sub.add_parser("smoke", help="Tiny CPU smoke that does not score the study")
    return parser


def main(argv: list[str] | None = None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)
    if args.command == "protocol-hash":
        print(protocol_sha256(PROTOCOL))
        return 0
    if args.command == "generate":
        from pinnforge.v041.study import run_generate

        record = run_generate()
        print(json.dumps({"problems": list(record["problems"]), "protocol_sha256": record["protocol_sha256"]}))
        return 0
    if args.command == "train":
        from pinnforge.v041.study import run_train

        seeds = None if args.seeds is None else [int(item) for item in args.seeds.split(",")]
        run_train(seeds=seeds)
        return 0
    if args.command == "score":
        from pinnforge.v041.study import run_calibrate_and_test

        scores = run_calibrate_and_test()
        for problem, block in scores["problems"].items():
            print(problem, "pass" if block["bars"]["pass"] else "fail")
        return 0
    if args.command == "report":
        from pinnforge.v041.report import write_report

        path = write_report()
        print(path)
        return 0
    if args.command == "smoke":
        return _smoke()
    raise ValueError(args.command)


def _smoke() -> int:
    import numpy as np

    from pinnforge.reference.spectral2d.burgers import solve_burgers2d
    from pinnforge.reference.spectral2d.common import Grid2D
    from pinnforge.reference.spectral2d.initial import bandlimited_velocity
    from pinnforge.v04.sensors import SensorMask
    from pinnforge.v041.inverse import bracket_bounds, bracket_search

    protocol = load_protocol(PROTOCOL)
    assert protocol["frozen_before_test_evaluation"]
    assert int(protocol["splits"]["n_cal"]) >= 60
    assert protocol["pass_bars"]["in_distribution_error"]["limit"] == 0.05
    grid = Grid2D(8)
    rng = np.random.default_rng(0)
    u0 = bandlimited_velocity(grid, rng, max_mode=1)
    burgers = solve_burgers2d(u0, 0.04, dt=0.01, t_final=0.02, save_dt=0.01)
    assert np.isfinite(burgers.u).all()
    low, high = bracket_bounds(0.04, 0.2, 0.012, 0.072)
    assert low < 0.04 < high
    mask = SensorMask(n=8, n_side=4, burst_starts=(0, 1), burst_length=2, frame_dt=0.01)
    tiny = dict(protocol)
    tiny["time"] = {"dt": 0.01, "t_final": 0.02, "save_dt": 0.01}
    tiny["domain"] = {**protocol["domain"], "n": 8, "length": grid.length}
    fitted = bracket_search(
        burgers.u,
        mask,
        first_guess=0.04,
        q_log=0.2,
        n_local=3,
        problem="burgers2d",
        protocol=tiny,
        forcing="none",
    )
    assert np.isfinite(fitted["nu_hat"])
    from pinnforge.ml_import import require_torch
    from pinnforge.v04.fno2d import FNO2d, pack_inputs

    torch = require_torch()
    model = FNO2d(5, 4, width=4, modes=2, n_layers=1)
    pred = model(pack_inputs(torch.zeros(1, 2, 2, 8, 8), torch.tensor([0.0])))
    assert pred.shape == (1, 4, 8, 8)
    print("v041 smoke ok", protocol_sha256(PROTOCOL)[:12])
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
