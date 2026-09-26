"""Command line interface.

``version`` prints the package version. ``equations`` lists registered
equation ids. ``sample`` draws a seeded collocation batch for a built-in
spec and prints counts and bounds. ``--output`` writes a JSON record to a
relative path inside the working directory. ``residual`` prints the mean
squared residual of a tiny untrained MLP. ``train`` runs Adam on that
residual plus the soft penalties and writes ``metrics.jsonl`` and a
checkpoint. ``residual`` and ``train`` import torch and need the optional
``ml`` extra.
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
    train = subparsers.add_parser(
        "train",
        help="Train a small MLP with Adam and write metrics and a checkpoint",
    )
    train.add_argument(
        "--equation",
        required=True,
        choices=sorted(EQUATION_ALIASES),
        help="Equation name. harmonic, burgers, and poisson are the short names.",
    )
    train.add_argument(
        "--epochs",
        type=int,
        default=None,
        help="Adam steps (default: 50). metrics.jsonl also records epoch 0.",
    )
    train.add_argument("--seed", type=int, default=None, help="Seed for the batch and the network (default: 0).")
    train.add_argument("--lr", type=float, default=None, help="Adam learning rate (default: 0.001).")
    train.add_argument(
        "--n-interior",
        type=int,
        default=None,
        help="Interior collocation count (default: 32).",
    )
    train.add_argument(
        "--n-ic",
        type=int,
        default=None,
        help="Initial-condition count (default: 16 for harmonic and Burgers, 0 for Poisson).",
    )
    train.add_argument(
        "--n-bc",
        type=int,
        default=None,
        help="Boundary count (default: 0 for harmonic, 16 for Burgers and Poisson).",
    )
    train.add_argument(
        "--hidden-widths",
        default=None,
        help="Comma-separated MLP widths (default: 16,16).",
    )
    train.add_argument("--w-pde", type=float, default=None, help="Weight on residual MSE (default: 1).")
    train.add_argument("--w-ic", type=float, default=None, help="Weight on the initial-condition penalty (default: 1).")
    train.add_argument("--w-bc", type=float, default=None, help="Weight on the Dirichlet penalty (default: 1).")
    train.add_argument(
        "--method",
        choices=list(SAMPLE_METHODS),
        default=None,
        help="Sampler for free axes (default: uniform).",
    )
    train.add_argument(
        "--checkpoint-dir",
        default=None,
        help="Relative directory for checkpoint.pt (default: checkpoints).",
    )
    train.add_argument(
        "--log-path",
        default=None,
        help="Relative metrics path ending in .jsonl (default: metrics.jsonl).",
    )
    train.add_argument(
        "--device",
        choices=("cpu",),
        default=None,
        help="Training device (default: cpu).",
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
    if args.command == "train":
        return _train(parser, args)
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

    The network is not optimized. ``pinnforge train`` is the training
    loop. Torch is imported here so ``version``, ``equations``, and
    ``sample`` stay importable without the ``ml`` extra.
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


def _train(parser: argparse.ArgumentParser, args: argparse.Namespace) -> int:
    """Train a built-in equation and print the final loss.

    Torch is imported here so ``version``, ``equations``, and ``sample``
    stay importable without the ``ml`` extra.
    """

    from pinnforge.ml_import import InstallHint

    try:
        from pinnforge.training import train_loop
    except InstallHint as exc:
        print(str(exc), file=sys.stderr)
        return 1
    except ModuleNotFoundError as exc:
        if exc.name != "torch":
            raise
        print(str(InstallHint()), file=sys.stderr)
        return 1
    try:
        config = _train_config(args)
        result = train_loop(config)
    except ValidationError as exc:
        parser.error(_format_validation(exc))
    except ValueError as exc:
        parser.error(str(exc))
    else:
        final = result.history[-1]
        print(f"equation: {config.equation_id}")
        print(f"seed: {config.seed}")
        print(f"epochs: {config.epochs}")
        print(f"device: {config.device}")
        checkpoint = (Path(config.checkpoint_dir) / "checkpoint.pt").as_posix()
        print(f"checkpoint: {checkpoint}")
        print(f"log: {Path(config.log_path).as_posix()}")
        print(f"loss: {final.loss:.8e}")
        print(f"loss_pde: {final.loss_pde:.8e}")
        print(f"loss_ic: {final.loss_ic:.8e}")
        print(f"loss_bc: {final.loss_bc:.8e}")
        print(f"lr: {final.lr:.8e}")
        return 0
    return 2


def _train_config(args: argparse.Namespace):
    """Build a :class:`~pinnforge.training.TrainConfig` from CLI flags.

    Omitted flags keep the TrainConfig defaults, including the
    per-equation initial and boundary counts.
    """

    from pinnforge.training import TrainConfig

    payload: dict[str, object] = {"equation_id": args.equation}
    optional = {
        "epochs": args.epochs,
        "seed": args.seed,
        "lr": args.lr,
        "n_interior": args.n_interior,
        "n_ic": args.n_ic,
        "n_bc": args.n_bc,
        "w_pde": args.w_pde,
        "w_ic": args.w_ic,
        "w_bc": args.w_bc,
        "method": args.method,
        "checkpoint_dir": args.checkpoint_dir,
        "log_path": args.log_path,
        "device": args.device,
    }
    for key, value in optional.items():
        if value is not None:
            payload[key] = value
    if args.hidden_widths is not None:
        payload["hidden_widths"] = _parse_hidden_widths(args.hidden_widths)
    return TrainConfig.model_validate(payload)


def _parse_hidden_widths(text: str) -> tuple[int, ...]:
    parts = [part.strip() for part in text.split(",")]
    if not parts or any(part == "" for part in parts):
        raise ValueError("hidden widths must be comma-separated positive integers")
    widths: list[int] = []
    for part in parts:
        if not part.isdigit():
            raise ValueError("hidden widths must be comma-separated positive integers")
        widths.append(int(part))
    return tuple(widths)


def _format_validation(exc: ValidationError) -> str:
    parts: list[str] = []
    for error in exc.errors():
        location = ".".join(str(item) for item in error["loc"])
        message = str(error["msg"])
        parts.append(f"{location}: {message}" if location else message)
    return "; ".join(parts) if parts else "invalid configuration"
