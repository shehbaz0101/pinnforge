"""Command line interface.

``version`` prints the package version. ``equations`` lists registered
equation ids. ``sample`` draws a seeded collocation batch for a built-in
spec and prints counts and bounds. ``--output`` writes a JSON record to a
relative path inside the working directory. ``residual`` prints the mean
squared residual of a tiny untrained MLP; it imports torch and needs the
optional ``ml`` extra. None of these commands trains a network.
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

from pydantic import ValidationError

from pinnforge import __version__
from pinnforge.equations import registered_equations
from pinnforge.sampling import (
    SAMPLE_METHODS,
    SampleConfig,
    default_spec,
    format_summary,
    resolve_equation_id,
    resolve_output_path,
    sample_equation,
    write_sample_record,
)
from pinnforge.sampling.defaults import CLI_COUNT_DEFAULTS, EQUATION_ALIASES

_RESIDUAL_INTERIOR = 16
_RESIDUAL_HIDDEN = (8, 8)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="pinnforge",
        description="PINNForge physics-informed neural network sandbox.",
    )
    parser.add_argument("--version", action="version", version=f"pinnforge {__version__}")
    subparsers = parser.add_subparsers(dest="command", required=True)
    subparsers.add_parser("version", help="Print the package version")
    subparsers.add_parser("equations", help="List registered equation ids")
    sample = subparsers.add_parser(
        "sample",
        help="Draw a seeded collocation batch and print counts and bounds",
    )
    sample.add_argument(
        "--equation",
        required=True,
        choices=sorted(EQUATION_ALIASES),
        help="Equation name. harmonic, burgers, and poisson are the short names.",
    )
    sample.add_argument(
        "--n-interior",
        type=int,
        default=64,
        help="Interior collocation count (default: 64).",
    )
    sample.add_argument(
        "--n-ic",
        type=int,
        default=None,
        help="Initial-condition count (default: 16 for harmonic and Burgers, 0 for Poisson).",
    )
    sample.add_argument(
        "--n-bc",
        type=int,
        default=None,
        help="Boundary count (default: 0 for harmonic, 16 for Burgers and Poisson).",
    )
    sample.add_argument("--seed", type=int, default=0, help="Seed for numpy.random.default_rng.")
    sample.add_argument(
        "--method",
        choices=list(SAMPLE_METHODS),
        default="uniform",
        help="Sampler for free axes (default: uniform).",
    )
    sample.add_argument(
        "--output",
        type=Path,
        help=(
            "Write a JSON record to this relative path. It must end in .json and "
            "stay inside the working directory."
        ),
    )
    residual = subparsers.add_parser(
        "residual",
        help="Print residual MSE of a tiny untrained MLP (needs the ml extra)",
    )
    residual.add_argument(
        "--equation",
        required=True,
        choices=sorted(EQUATION_ALIASES),
        help="Equation name. harmonic, burgers, and poisson are the short names.",
    )
    residual.add_argument(
        "--seed",
        type=int,
        default=0,
        help="Seed for the sampler and the untrained network (default: 0).",
    )
    return parser


def main(argv: list[str] | None = None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)
    if args.command == "version":
        print(f"pinnforge {__version__}")
        return 0
    if args.command == "equations":
        for equation_id in registered_equations():
            print(equation_id)
        return 0
    if args.command == "sample":
        return _sample(parser, args)
    if args.command == "residual":
        return _residual(parser, args)
    parser.error(f"unknown command {args.command}")
    return 2


def _sample(parser: argparse.ArgumentParser, args: argparse.Namespace) -> int:
    try:
        equation_id = resolve_equation_id(args.equation)
        defaults = CLI_COUNT_DEFAULTS[equation_id]
        n_ic = defaults["n_ic"] if args.n_ic is None else args.n_ic
        n_bc = defaults["n_bc"] if args.n_bc is None else args.n_bc
        config = SampleConfig(
            n_interior=args.n_interior,
            n_ic=n_ic,
            n_bc=n_bc,
            seed=args.seed,
            method=args.method,
        )
        spec = default_spec(equation_id)
        output = None if args.output is None else resolve_output_path(args.output)
        batch = sample_equation(spec, config)
        if output is not None:
            write_sample_record(spec, config, batch, output)
    except ValidationError as exc:
        parser.error(_format_validation(exc))
    except ValueError as exc:
        parser.error(str(exc))
    else:
        print(format_summary(batch), end="")
        return 0
    return 2


def _residual(parser: argparse.ArgumentParser, args: argparse.Namespace) -> int:
    """Build an untrained MLP, sample interior points, and print residual MSE.

    The network is not optimized. Day 4 is the training loop. Torch is
    imported here so ``version``, ``equations``, and ``sample`` stay
    importable without the ``ml`` extra.
    """

    from pinnforge.ml_import import InstallHint

    try:
        import torch

        from pinnforge.models import mlp_from_spec
        from pinnforge.residuals import mean_squared_residual
    except InstallHint as exc:
        print(str(exc), file=sys.stderr)
        return 1
    except ModuleNotFoundError as exc:
        if exc.name != "torch":
            raise
        print(str(InstallHint()), file=sys.stderr)
        return 1
    try:
        equation_id = resolve_equation_id(args.equation)
        config = SampleConfig(
            n_interior=_RESIDUAL_INTERIOR,
            n_ic=0,
            n_bc=0,
            seed=args.seed,
            method="uniform",
        )
        spec = default_spec(equation_id)
        batch = sample_equation(spec, config)
        torch.manual_seed(args.seed)
        model = mlp_from_spec(spec, _RESIDUAL_HIDDEN)
        coords = torch.tensor(batch.interior, dtype=torch.float32)
        mse = mean_squared_residual(model, coords, spec)
    except ValidationError as exc:
        parser.error(_format_validation(exc))
    except ValueError as exc:
        parser.error(str(exc))
    else:
        print(f"equation: {spec.equation_id}")
        print(f"seed: {args.seed}")
        print(f"residual_mse: {float(mse.detach()):.8e}")
        return 0
    return 2


def _format_validation(exc: ValidationError) -> str:
    parts: list[str] = []
    for error in exc.errors():
        location = ".".join(str(item) for item in error["loc"])
        message = str(error["msg"])
        parts.append(f"{location}: {message}" if location else message)
    return "; ".join(parts) if parts else "invalid sample config"
