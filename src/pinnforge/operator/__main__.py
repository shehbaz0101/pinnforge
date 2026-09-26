"""Train and evaluate the data-only Burgers FNO.

``train`` reads the train and validation splits only. ``eval`` scores one
split, defaulting to test. Both commands use the pilot manifest for the
instance split and for the training-set ``u`` mean and standard deviation.
Torch is imported only when a command actually trains or scores.

The same commands are available as ``pinnforge fno train`` and
``pinnforge fno eval``.
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

from pinnforge.operator.defaults import (
    DEFAULT_BATCH_SIZE,
    DEFAULT_EPOCHS,
    DEFAULT_LAYERS,
    DEFAULT_LR,
    DEFAULT_MODES,
    DEFAULT_SEED,
    DEFAULT_WIDTH,
)
from pinnforge.operator.windows import DEFAULT_INPUT_FRAMES, DEFAULT_OUTPUT_FRAMES, DEFAULT_STRIDE


def main(argv: list[str] | None = None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)
    try:
        if args.command == "train":
            return _train(args)
        if args.command == "eval":
            return _eval(args)
    except ValueError as exc:
        parser.error(str(exc))
    parser.error(f"unknown command {args.command}")
    return 2


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="pinnforge fno",
        description="Data-only 1D Fourier neural operator on Stage 2 Burgers windows.",
    )
    subparsers = parser.add_subparsers(dest="command", required=True)
    train = subparsers.add_parser("train", help="Train on the train split and select with validation relative L2")
    train.add_argument("--pilot", type=Path, default=Path("artifacts/burgers_pilot"))
    train.add_argument("--manifest", type=Path, default=Path("docs/stage2/pilot_manifest.json"))
    train.add_argument("--output", type=Path, default=Path("runs/stage3_fno"))
    train.add_argument("--epochs", type=int, default=DEFAULT_EPOCHS)
    train.add_argument("--batch-size", type=int, default=DEFAULT_BATCH_SIZE)
    train.add_argument("--lr", type=float, default=DEFAULT_LR)
    train.add_argument("--width", type=int, default=DEFAULT_WIDTH)
    train.add_argument("--modes", type=int, default=DEFAULT_MODES)
    train.add_argument("--layers", type=int, default=DEFAULT_LAYERS)
    train.add_argument("--input-frames", type=int, default=DEFAULT_INPUT_FRAMES)
    train.add_argument("--output-frames", type=int, default=DEFAULT_OUTPUT_FRAMES)
    train.add_argument("--stride", type=int, default=DEFAULT_STRIDE)
    train.add_argument("--seed", type=int, default=DEFAULT_SEED)
    train.add_argument(
        "--skip-field-hash",
        action="store_true",
        help="Do not check field_sha256 against the manifest. The reported run leaves this off.",
    )
    evaluate = subparsers.add_parser("eval", help="Score a checkpoint on one split (default: test)")
    evaluate.add_argument("--pilot", type=Path, default=Path("artifacts/burgers_pilot"))
    evaluate.add_argument("--manifest", type=Path, default=Path("docs/stage2/pilot_manifest.json"))
    evaluate.add_argument("--checkpoint", type=Path, required=True)
    evaluate.add_argument("--split", choices=("train", "val", "test"), default="test")
    evaluate.add_argument("--batch-size", type=int, default=DEFAULT_BATCH_SIZE)
    evaluate.add_argument("--output", type=Path, default=None, help="Eval JSON path. Default: beside the checkpoint.")
    evaluate.add_argument("--skip-field-hash", action="store_true")
    return parser


def _train(args: argparse.Namespace) -> int:
    train_from_paths = _import_trainer()
    result = train_from_paths(
        pilot_dir=args.pilot,
        manifest_path=args.manifest,
        output_dir=args.output,
        epochs=args.epochs,
        batch_size=args.batch_size,
        lr=args.lr,
        width=args.width,
        modes=args.modes,
        n_layers=args.layers,
        input_frames=args.input_frames,
        output_frames=args.output_frames,
        stride=args.stride,
        seed=args.seed,
        check_field_hash=not args.skip_field_hash,
        log=_print_epoch,
    )
    print(f"checkpoint: {result.checkpoint.as_posix()}")
    print(f"metrics: {result.metrics_path.as_posix()}")
    print(f"manifest: {result.manifest_path.as_posix()}")
    print(f"parameters: {result.parameter_count}")
    print(f"selected_epoch: {result.selected_epoch}")
    print(f"train_mse: {result.train_mse:.8e}")
    print(f"val_mse: {result.val_mse:.8e}")
    print(f"val_relative_l2: {result.val_relative_l2:.8e}")
    return 0


def _eval(args: argparse.Namespace) -> int:
    evaluate_checkpoint, load_fno_checkpoint, write_eval_json = _import_evaluator()
    loaded = load_fno_checkpoint(args.checkpoint)
    score = evaluate_checkpoint(
        args.checkpoint,
        args.pilot,
        args.manifest,
        split=args.split,
        batch_size=args.batch_size,
        check_field_hash=not args.skip_field_hash,
    )
    output = args.output
    if output is None:
        output = args.checkpoint.parent / f"eval_{args.split}.json"
    write_eval_json(score, output, checkpoint=args.checkpoint, epoch=loaded.epoch)
    print(f"split: {score.split}")
    print(f"checkpoint: {args.checkpoint.as_posix()}")
    print(f"epoch: {loaded.epoch}")
    print(f"n_instances: {score.n_instances}")
    print(f"n_windows: {score.n_windows}")
    print(f"normalized_mse: {score.normalized_mse:.8e}")
    print(f"mean_relative_l2: {score.mean_relative_l2:.8e}")
    print(f"median_relative_l2: {score.median_relative_l2:.8e}")
    print(f"pooled_relative_l2: {score.pooled_relative_l2:.8e}")
    print(f"persistence_mean_relative_l2: {score.persistence_mean_relative_l2:.8e}")
    print(f"json: {Path(output).as_posix()}")
    return 0


def _print_epoch(row: object) -> None:
    print(
        f"epoch: {row.epoch} train_mse: {row.train_mse:.8e} "
        f"val_mse: {row.val_mse:.8e} val_relative_l2: {row.val_relative_l2:.8e}",
        flush=True,
    )


def _import_trainer():
    from pinnforge.ml_import import InstallHint

    try:
        from pinnforge.operator.train import train_from_paths
    except InstallHint as exc:
        print(str(exc), file=sys.stderr)
        raise SystemExit(1) from exc
    except ModuleNotFoundError as exc:
        if exc.name != "torch":
            raise
        print(str(InstallHint()), file=sys.stderr)
        raise SystemExit(1) from exc
    return train_from_paths


def _import_evaluator():
    from pinnforge.ml_import import InstallHint

    try:
        from pinnforge.operator.checkpoint import load_fno_checkpoint
        from pinnforge.operator.evaluate import evaluate_checkpoint, write_eval_json
    except InstallHint as exc:
        print(str(exc), file=sys.stderr)
        raise SystemExit(1) from exc
    except ModuleNotFoundError as exc:
        if exc.name != "torch":
            raise
        print(str(InstallHint()), file=sys.stderr)
        raise SystemExit(1) from exc
    return evaluate_checkpoint, load_fno_checkpoint, write_eval_json


if __name__ == "__main__":
    raise SystemExit(main())
