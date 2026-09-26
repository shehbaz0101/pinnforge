"""Commands for the periodic Burgers reference.

``estimate`` prints storage and, with ``--benchmark``, one measured
trajectory time. ``convergence`` writes the study under ``--output``.
``pilot`` writes a dataset directory. None of these commands train a
network or call the coordinate-PINN path.
"""

from __future__ import annotations

import argparse
import time
from pathlib import Path

from pinnforge.reference.numerical.dataset import PilotConfig, estimate_storage, generate_pilot
from pinnforge.reference.numerical.initial import draw_initial_condition
from pinnforge.reference.numerical.solver import solve
from pinnforge.reference.numerical.study import run_convergence


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(prog="python -m pinnforge.reference.numerical")
    subparsers = parser.add_subparsers(dest="command", required=True)
    estimate = subparsers.add_parser("estimate", help="Print storage for a pilot size")
    _add_pilot_args(estimate)
    estimate.add_argument(
        "--benchmark",
        action="store_true",
        help="Time one trajectory at the pilot resolution and extrapolate",
    )
    convergence = subparsers.add_parser("convergence", help="Run the convergence study")
    convergence.add_argument("--output", type=Path, default=Path("docs/stage2"))
    pilot = subparsers.add_parser("pilot", help="Generate a pilot dataset directory")
    _add_pilot_args(pilot)
    pilot.add_argument("--output", type=Path, required=True)
    args = parser.parse_args(argv)
    if args.command == "estimate":
        _estimate(args)
    elif args.command == "convergence":
        run_convergence(args.output)
    elif args.command == "pilot":
        manifest = generate_pilot(args.output, _pilot_config(args))
        print(
            f"wrote {args.output} ({len(manifest['instances'])} instances, "
            f"solver_config_sha256={manifest['solver_config_sha256']})"
        )
    else:
        raise AssertionError(args.command)


def _add_pilot_args(parser: argparse.ArgumentParser) -> None:
    parser.add_argument("--train", type=int, default=512)
    parser.add_argument("--val", type=int, default=128)
    parser.add_argument("--test", type=int, default=128)
    parser.add_argument("--n", type=int, default=256)
    parser.add_argument("--dt", type=float, default=1e-3)
    parser.add_argument("--save-dt", type=float, default=0.01)
    parser.add_argument("--t-final", type=float, default=1.0)
    parser.add_argument("--master-seed", type=int, default=20260926)
    parser.add_argument("--split-seed", type=int, default=20260926)
    parser.add_argument("--batch-size", type=int, default=8)


def _pilot_config(args: argparse.Namespace) -> PilotConfig:
    return PilotConfig(
        n_train=args.train,
        n_val=args.val,
        n_test=args.test,
        n=args.n,
        dt=args.dt,
        t_final=args.t_final,
        save_dt=args.save_dt,
        master_seed=args.master_seed,
        split_seed=args.split_seed,
        batch_size=args.batch_size,
    )


def _estimate(args: argparse.Namespace) -> None:
    config = _pilot_config(args)
    storage = estimate_storage(config)
    print(
        "instances={n_instances} times={n_times} n={n} "
        "bytes_per_trajectory={bytes_per_trajectory} field_mebibytes={field_mebibytes:.3f}".format(
            **storage
        )
    )
    if not args.benchmark:
        return
    initial = draw_initial_condition(config.master_seed, 0, config.n)
    started = time.perf_counter()
    solve(initial.u0, initial.nu, dt=config.dt, t_final=config.t_final, save_dt=config.save_dt)
    elapsed = time.perf_counter() - started
    print(
        f"one_trajectory_seconds={elapsed:.3f} "
        f"extrapolated_seconds={elapsed * config.total():.1f}"
    )


if __name__ == "__main__":
    main()
