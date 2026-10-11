"""CLI for the v0.4 2D viscosity inverse."""

from __future__ import annotations

import argparse
import json

from pinnforge.reference.spectral2d.dataset import load_protocol, protocol_sha256


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="pinnforge v04",
        description="2D Burgers and Navier-Stokes viscosity inverse study.",
    )
    sub = parser.add_subparsers(dest="command", required=True)
    sub.add_parser("protocol-hash", help="Print the SHA-256 of the frozen protocol")
    sub.add_parser("generate", help="Generate the 64x64 pilots")
    sub.add_parser("convergence", help="Run the solver refinement check")
    train = sub.add_parser("train", help="Train the 2D FNO arms")
    train.add_argument("--seeds", default=None, help="Comma-separated seeds (default: protocol seeds)")
    sub.add_parser("score", help="Validate, then score the frozen test split")
    sub.add_parser("report", help="Write docs/v04/report.md and figures")
    sub.add_parser("smoke", help="Tiny CPU smoke that does not score the study")
    return parser


def main(argv: list[str] | None = None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)
    if args.command == "protocol-hash":
        print(protocol_sha256())
        return 0
    if args.command == "generate":
        from pinnforge.v04.study import run_generate

        record = run_generate()
        print(json.dumps({"problems": list(record["problems"]), "protocol_sha256": record["protocol_sha256"]}))
        return 0
    if args.command == "convergence":
        from pinnforge.v04.study import run_convergence

        record = run_convergence()
        print(json.dumps({"pass": record["pass"], "gate": record["gate"]}))
        return 0
    if args.command == "train":
        from pinnforge.v04.study import run_train

        seeds = None if args.seeds is None else [int(item) for item in args.seeds.split(",")]
        run_train(seeds=seeds)
        return 0
    if args.command == "score":
        from pinnforge.v04.study import run_validate_and_test

        scores = run_validate_and_test()
        for problem, block in scores["problems"].items():
            print(problem, "pass" if block["bars"]["pass"] else "fail")
        return 0
    if args.command == "report":
        from pinnforge.v04.report import write_report

        path = write_report()
        print(path)
        return 0
    if args.command == "smoke":
        return _smoke()
    raise ValueError(args.command)


def _smoke() -> int:
    """N=8, two steps, untrained FNO, manufactured residual. No test scores."""

    import numpy as np

    from pinnforge.reference.spectral2d.burgers import solve_burgers2d
    from pinnforge.reference.spectral2d.common import Grid2D
    from pinnforge.reference.spectral2d.initial import bandlimited_velocity
    from pinnforge.reference.spectral2d.navier_stokes import solve_ns2d
    from pinnforge.v04.baselines import closed_form_ls
    from pinnforge.v04.sensors import SensorMask, lift_sensors, read_sensors

    protocol = load_protocol()
    assert protocol["frozen_before_test_evaluation"]
    grid = Grid2D(8)
    rng = np.random.default_rng(0)
    u0 = bandlimited_velocity(grid, rng, max_mode=1)
    burgers = solve_burgers2d(u0, 0.04, dt=0.01, t_final=0.02, save_dt=0.01)
    assert burgers.u.shape[0] == 3
    assert np.isfinite(burgers.u).all()
    omega0 = np.sin(grid.mesh()[0]) * np.cos(grid.mesh()[1])
    ns = solve_ns2d(omega0, 0.04, dt=0.01, t_final=0.02, save_dt=0.01, forcing="decaying")
    assert np.isfinite(ns.omega).all()
    mask = SensorMask(n=8, n_side=4, burst_starts=(0, 1), burst_length=2, frame_dt=0.01)
    sensors = read_sensors(burgers.u, mask)
    lifted = lift_sensors(sensors, mask)
    assert lifted.shape[-1] == 8
    estimate = closed_form_ls(
        burgers.u, mask, problem="burgers2d", dt=0.01, forcing="none", length=grid.length
    )
    assert np.isfinite(estimate) or np.isnan(estimate)
    from pinnforge.ml_import import require_torch
    from pinnforge.v04.fno2d import FNO2d, pack_inputs

    torch = require_torch()
    model = FNO2d(5, 4, width=4, modes=2, n_layers=1)
    frames = torch.zeros(1, 2, 2, 8, 8)
    nu = torch.tensor([0.0])
    pred = model(pack_inputs(frames, nu))
    assert pred.shape == (1, 4, 8, 8)
    print("v04 smoke ok", protocol_sha256()[:12])
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
