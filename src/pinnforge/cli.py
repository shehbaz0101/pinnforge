"""Command line interface.

``version`` prints the package version. ``equations`` lists registered
equation ids. ``sample`` draws a seeded collocation batch for a built-in
spec and prints counts and bounds. ``--output`` writes a JSON record to a
relative path inside the sandbox root. ``residual`` prints the mean
squared residual of a tiny untrained MLP. ``train`` runs Adam on that
residual plus the soft penalties and writes ``metrics.jsonl`` and a
checkpoint. ``eval`` loads a checkpoint and prints L2 and residual
metrics. ``residual``, ``train``, ``eval``, ``run``, and ``demo`` import torch
and need the optional ``ml`` extra. ``run --config`` trains and then
evaluates from a relative YAML or JSON experiment file. ``demo`` does
that for a checked-in sample under ``samples/configs/`` (harmonic by
default) and prints the loss and the reference error. The flag-based
``train`` and ``eval`` commands are unchanged. ``serve`` starts the
localhost HTTP API and needs the optional ``api`` extra. It binds to
``127.0.0.1`` and refuses ``0.0.0.0`` unless ``--allow-remote`` is set.
``POST /train``, ``POST /eval``, and ``POST /run`` share a rate limit.
``serve``, ``train``, ``eval``, ``run``, and ``demo`` refuse
non-loopback TCP connects.
"""

from __future__ import annotations

import argparse
import os
import sys
from collections.abc import Iterator
from contextlib import contextmanager
from pathlib import Path

from pydantic import ValidationError

from pinnforge import __version__
from pinnforge.api.bind import resolve_bind_host, resolve_port
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
_API_HINT = (
    "PINNForge's HTTP API needs the optional api extra. "
    'Install it with: pip install -e ".[api]"'
)
_API_MODULES = frozenset({"fastapi", "uvicorn", "starlette", "httpx"})


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="pinnforge",
        description="PINNForge physics-informed neural network sandbox.",
    )
    parser.add_argument("--version", action="version", version=f"pinnforge {__version__}")
    path_flags = argparse.ArgumentParser(add_help=False)
    path_flags.add_argument(
        "--data-root",
        default=None,
        help=(
            "Sandbox root for config, checkpoint, metrics, and JSON paths. "
            "Defaults to PINNFORGE_DATA_ROOT when that is set, otherwise the "
            "working directory. Paths must be relative to this directory."
        ),
    )
    subparsers = parser.add_subparsers(dest="command", required=True)
    subparsers.add_parser("version", help="Print the package version")
    subparsers.add_parser("equations", help="List registered equation ids")
    sample = subparsers.add_parser(
        "sample",
        parents=[path_flags],
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
        parents=[path_flags],
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
    train.add_argument(
        "--w-bc",
        type=float,
        default=None,
        help="Weight on the Dirichlet, Neumann, and periodic penalties (default: 1).",
    )
    train.add_argument(
        "--checkpoint-interval",
        type=int,
        default=None,
        help="Epochs between tagged checkpoints. Epoch 0 and the last epoch are always written (default: 1).",
    )
    train.add_argument(
        "--resume-from",
        default=None,
        help="Relative .pt checkpoint to resume. Loads weights, Adam state, and the torch RNG state.",
    )
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
    evaluate = subparsers.add_parser(
        "eval",
        parents=[path_flags],
        help="Score a checkpoint against a reference and the residual (needs the ml extra)",
    )
    evaluate.add_argument(
        "--checkpoint",
        required=True,
        type=Path,
        help="Relative checkpoint.pt path inside the working directory.",
    )
    evaluate.add_argument(
        "--equation",
        required=True,
        choices=sorted(EQUATION_ALIASES),
        help="Equation name. It must match the checkpoint. harmonic, burgers, and poisson are the short names.",
    )
    evaluate.add_argument(
        "--n-interior",
        type=int,
        default=None,
        help="Interior points for the eval batch (default: 64).",
    )
    evaluate.add_argument("--seed", type=int, default=None, help="Seed for the eval batch (default: 0).")
    evaluate.add_argument(
        "--bins",
        type=int,
        default=None,
        help="Histogram bins for |residual| (default: 10).",
    )
    evaluate.add_argument(
        "--write-json",
        type=Path,
        help=(
            "Write the eval record to this relative path. It must end in .json and "
            "stay inside the working directory."
        ),
    )
    run = subparsers.add_parser(
        "run",
        parents=[path_flags],
        help="Train then evaluate from a YAML or JSON experiment config",
    )
    run.add_argument(
        "--config",
        required=True,
        type=Path,
        help=(
            "Relative .yaml, .yml, or .json experiment file. It must stay inside "
            "the working directory. The file selects the equation and the train "
            "and eval settings."
        ),
    )
    demo = subparsers.add_parser(
        "demo",
        parents=[path_flags],
        help="Train a short CPU sample and print the reference error (needs the ml extra)",
    )
    demo.add_argument(
        "--equation",
        default="harmonic",
        choices=sorted(EQUATION_ALIASES),
        help=(
            "Sample to run (default: harmonic). harmonic uses "
            "samples/configs/harmonic.yaml, poisson uses samples/configs/poisson.json, "
            "and burgers uses samples/configs/burgers.yaml."
        ),
    )
    demo.add_argument(
        "--epochs",
        type=int,
        default=None,
        help="Replace the sample epoch count. Must be from 1 to 5. The sample file is not rewritten.",
    )
    serve = subparsers.add_parser(
        "serve",
        parents=[path_flags],
        help="Serve the localhost HTTP API (needs the api extra)",
    )
    serve.add_argument(
        "--host",
        default="127.0.0.1",
        help=(
            "Bind address (default: 127.0.0.1). 0.0.0.0 and :: are refused "
            "unless --allow-remote is set."
        ),
    )
    serve.add_argument(
        "--port",
        type=int,
        default=8000,
        help="TCP port (default: 8000).",
    )
    serve.add_argument(
        "--allow-remote",
        action="store_true",
        help="Allow binding 0.0.0.0 or ::. Off by default so the API stays on localhost.",
    )
    serve.add_argument(
        "--rate-limit",
        type=int,
        default=None,
        help=(
            "POST /train, POST /eval, and POST /run requests allowed per window "
            "for each client (default: 60, or PINNFORGE_RATE_LIMIT)."
        ),
    )
    serve.add_argument(
        "--rate-window",
        type=float,
        default=None,
        help=(
            "Rate-limit window in seconds (default: 60, or PINNFORGE_RATE_WINDOW_SECONDS)."
        ),
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
    if args.command == "eval":
        return _eval(parser, args)
    if args.command == "run":
        return _run(parser, args)
    if args.command == "demo":
        return _demo(parser, args)
    if args.command == "serve":
        return _serve(parser, args)
    parser.error(f"unknown command {args.command}")
    return 2


def _sample(parser: argparse.ArgumentParser, args: argparse.Namespace) -> int:
    try:
        with _data_root_env(args):
            return _sample_in_root(args)
    except ValidationError as exc:
        parser.error(_format_validation(exc))
    except ValueError as exc:
        parser.error(str(exc))
    return 2


def _sample_in_root(args: argparse.Namespace) -> int:
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
    print(format_summary(batch), end="")
    return 0


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

    Checkpoint and log paths are checked before torch is imported, so a
    path that leaves the sandbox root fails the same way with or without
    the ``ml`` extra. Torch is imported only after that check.
    """

    try:
        with _data_root_env(args):
            config = _train_config(args)
            _check_train_paths(config)
            _install_offline_guard()
            return _train_with_torch(config)
    except ValidationError as exc:
        parser.error(_format_validation(exc))
    except ValueError as exc:
        parser.error(str(exc))
    return 2


def _train_with_torch(config: object) -> int:
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
    result = train_loop(config)
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


def _eval(parser: argparse.ArgumentParser, args: argparse.Namespace) -> int:
    """Load a checkpoint and print L2 and residual metrics.

    The checkpoint and ``--write-json`` paths are checked before torch is
    imported, so a path that leaves the sandbox root fails the same way
    with or without the ``ml`` extra.
    """

    from pinnforge.specs.paths import resolve_inside_cwd

    try:
        with _data_root_env(args):
            resolve_inside_cwd(args.checkpoint, suffix=".pt", label="checkpoint")
            if args.write_json is not None:
                resolve_output_path(args.write_json)
            config = _eval_config(args)
            _install_offline_guard()
            return _eval_with_torch(args, config)
    except ValidationError as exc:
        parser.error(_format_validation(exc))
    except ValueError as exc:
        parser.error(str(exc))
    return 2


def _eval_with_torch(args: argparse.Namespace, config: object) -> int:
    from pinnforge.ml_import import InstallHint

    try:
        from pinnforge.evaluation import evaluate_checkpoint, write_eval_json
    except InstallHint as exc:
        print(str(exc), file=sys.stderr)
        return 1
    except ModuleNotFoundError as exc:
        if exc.name != "torch":
            raise
        print(str(InstallHint()), file=sys.stderr)
        return 1
    result = evaluate_checkpoint(args.checkpoint, config, equation=args.equation)
    if args.write_json is not None:
        write_eval_json(result, args.write_json)
    print(f"equation: {result.equation_id}")
    print(f"checkpoint: {result.checkpoint}")
    print(f"method: {result.method}")
    print(f"seed: {result.seed}")
    print(f"n_interior: {result.n_interior}")
    print(f"reference: {result.reference}")
    print(f"l2: {_format_metric(result.l2)}")
    print(f"relative_l2: {_format_metric(result.relative_l2)}")
    print(f"residual_mean_abs: {result.residual_mean_abs:.8e}")
    print(f"residual_max_abs: {result.residual_max_abs:.8e}")
    _print_condition_metrics(result)
    if args.write_json is not None:
        print(f"json: {Path(args.write_json).as_posix()}")
    return 0


def _run(parser: argparse.ArgumentParser, args: argparse.Namespace) -> int:
    """Train then evaluate the experiment in ``--config``.

    The config path and the checkpoint, log, and eval JSON paths inside
    it are checked before torch is imported, so a path that escapes the
    sandbox root fails the same way with or without the ``ml`` extra.
    """

    from pinnforge.experiments import load_experiment_config

    try:
        with _data_root_env(args):
            config = load_experiment_config(args.config)
            _install_offline_guard()
            return _run_with_torch(config)
    except ValidationError as exc:
        parser.error(_format_validation(exc))
    except ValueError as exc:
        parser.error(str(exc))
    return 2


def _run_with_torch(config: object) -> int:
    from pinnforge.ml_import import InstallHint

    try:
        from pinnforge.experiments.run import run_experiment
    except InstallHint as exc:
        print(str(exc), file=sys.stderr)
        return 1
    except ModuleNotFoundError as exc:
        if exc.name != "torch":
            raise
        print(str(InstallHint()), file=sys.stderr)
        return 1
    result = run_experiment(config)
    final = result.train.history[-1]
    print(f"equation: {config.equation.equation_id}")
    print(f"seed: {config.train.seed}")
    print(f"epochs: {config.train.epochs}")
    print(f"device: {config.train.device}")
    print(f"checkpoint: {result.evaluation.checkpoint}")
    print(f"log: {Path(config.train.log_path).as_posix()}")
    print(f"loss: {final.loss:.8e}")
    print(f"loss_pde: {final.loss_pde:.8e}")
    print(f"loss_ic: {final.loss_ic:.8e}")
    print(f"loss_bc: {final.loss_bc:.8e}")
    print(f"lr: {final.lr:.8e}")
    print(f"method: {result.evaluation.method}")
    print(f"eval_seed: {result.evaluation.seed}")
    print(f"n_interior: {result.evaluation.n_interior}")
    print(f"reference: {result.evaluation.reference}")
    print(f"l2: {_format_metric(result.evaluation.l2)}")
    print(f"relative_l2: {_format_metric(result.evaluation.relative_l2)}")
    print(f"residual_mean_abs: {result.evaluation.residual_mean_abs:.8e}")
    print(f"residual_max_abs: {result.evaluation.residual_max_abs:.8e}")
    _print_condition_metrics(result.evaluation)
    if result.eval_json is not None:
        print(f"json: {Path(config.eval_json).as_posix()}")
    return 0


def _demo(parser: argparse.ArgumentParser, args: argparse.Namespace) -> int:
    """Train a checked-in sample and print the loss and the reference error.

    The sample path is resolved inside the sandbox root before torch is
    imported. ``--epochs`` replaces the count in memory. The offline
    guard is installed before the training import, including when the
    ``ml`` extra is missing.
    """

    from pinnforge.demo import DemoDependencyError, execute_demo, load_demo_experiment

    try:
        with _data_root_env(args):
            config_path, config = load_demo_experiment(args.equation, epochs=args.epochs)
            _install_offline_guard()
            try:
                result = execute_demo(config_path, config)
            except DemoDependencyError as exc:
                print(str(exc), file=sys.stderr)
                return 1
    except ValidationError as exc:
        parser.error(_format_validation(exc))
    except ValueError as exc:
        parser.error(str(exc))
    else:
        print(result.summary, end="")
        return 0
    return 2


def _serve(parser: argparse.ArgumentParser, args: argparse.Namespace) -> int:
    """Bind the HTTP API. Torch is not imported.

    The host, data root, and rate limit are checked before uvicorn is
    imported. ``0.0.0.0`` and ``::`` need ``--allow-remote``. The offline
    guard is installed before the server accepts clients.
    """

    try:
        with _data_root_env(args), _rate_limit_env(args):
            return _serve_in_root(args)
    except ValueError as exc:
        parser.error(str(exc))
    return 2


def _serve_in_root(args: argparse.Namespace) -> int:
    host = resolve_bind_host(args.host, allow_remote=args.allow_remote)
    port = resolve_port(args.port)
    try:
        import uvicorn

        from pinnforge.api.app import create_app
    except ImportError as exc:
        if _missing_api_extra(exc):
            print(_API_HINT, file=sys.stderr)
            return 1
        raise
    _install_offline_guard()
    uvicorn.run(create_app(), host=host, port=port)
    return 0


def _missing_api_extra(exc: ImportError) -> bool:
    name = exc.name if isinstance(exc, ModuleNotFoundError) else None
    if not isinstance(name, str) or name == "":
        return False
    return name.split(".", 1)[0] in _API_MODULES


def _check_train_paths(config: object) -> None:
    """Reject checkpoint and metrics paths that leave the sandbox root."""

    from pinnforge.specs.paths import resolve_inside_cwd

    resolve_inside_cwd(config.checkpoint_dir, label="checkpoint_dir")
    resolve_inside_cwd(config.log_path, suffix=".jsonl", label="log_path")
    resume_from = getattr(config, "resume_from", None)
    if resume_from is not None:
        resolve_inside_cwd(resume_from, suffix=".pt", label="resume_from")


def _print_condition_metrics(result: object) -> None:
    print(f"max_abs_error: {_format_metric(getattr(result, 'max_abs_error', None))}")
    print(f"ic_error: {_format_metric(getattr(result, 'ic_error', None))}")
    print(f"bc_error: {_format_metric(getattr(result, 'bc_error', None))}")
    rng = getattr(result, "rng", None)
    name = rng.get("name") if isinstance(rng, dict) else None
    print(f"rng_stream: {name if isinstance(name, str) else 'unavailable'}")


def _install_offline_guard() -> None:
    from pinnforge.offline import install_offline_guard

    install_offline_guard()


@contextmanager
def _data_root_env(args: argparse.Namespace) -> Iterator[None]:
    """Point ``PINNFORGE_DATA_ROOT`` at ``--data-root`` for one command."""

    from pinnforge.specs.paths import DATA_ROOT_ENV, configure_sandbox_root

    raw = getattr(args, "data_root", None)
    if raw is None:
        yield
        return
    root = configure_sandbox_root(raw)
    previous = os.environ.get(DATA_ROOT_ENV)
    os.environ[DATA_ROOT_ENV] = str(root)
    try:
        yield
    finally:
        if previous is None:
            os.environ.pop(DATA_ROOT_ENV, None)
        else:
            os.environ[DATA_ROOT_ENV] = previous


@contextmanager
def _rate_limit_env(args: argparse.Namespace) -> Iterator[None]:
    """Apply ``--rate-limit`` and ``--rate-window`` for one serve process.

    The environment is restored when the server returns. A bad value
    raises :class:`ValueError` and still restores the previous variables.
    """

    from pinnforge.ratelimit import RATE_LIMIT_ENV, RATE_WINDOW_ENV, read_rate_settings

    previous: dict[str, str | None] = {}
    if args.rate_limit is not None:
        previous[RATE_LIMIT_ENV] = os.environ.get(RATE_LIMIT_ENV)
        os.environ[RATE_LIMIT_ENV] = str(args.rate_limit)
    if args.rate_window is not None:
        previous[RATE_WINDOW_ENV] = os.environ.get(RATE_WINDOW_ENV)
        os.environ[RATE_WINDOW_ENV] = str(float(args.rate_window))
    try:
        read_rate_settings()
        yield
    finally:
        for key, old in previous.items():
            if old is None:
                os.environ.pop(key, None)
            else:
                os.environ[key] = old


def _eval_config(args: argparse.Namespace):
    """Build an :class:`~pinnforge.specs.eval.EvalConfig` from CLI flags.

    Omitted flags keep the EvalConfig defaults. This import does not load torch.
    """

    from pinnforge.specs.eval import EvalConfig

    payload: dict[str, object] = {}
    if args.n_interior is not None:
        payload["n_interior"] = args.n_interior
    if args.seed is not None:
        payload["seed"] = args.seed
    if args.bins is not None:
        payload["bins"] = args.bins
    return EvalConfig.model_validate(payload)


def _format_metric(value: float | None) -> str:
    if value is None:
        return "unavailable"
    return f"{value:.8e}"


def _train_config(args: argparse.Namespace):
    """Build a :class:`~pinnforge.training.TrainConfig` from CLI flags.

    Omitted flags keep the TrainConfig defaults, including the
    per-equation initial and boundary counts.
    """

    from pinnforge.specs.train import TrainConfig

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
        "checkpoint_interval": args.checkpoint_interval,
        "resume_from": args.resume_from,
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
