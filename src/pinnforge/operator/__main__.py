"""Train and evaluate the Burgers FNO.

``train`` reads the train and validation splits only. ``eval`` scores one
split, defaulting to test. Both commands use the pilot manifest for the
instance split and for the training-set ``u`` mean and standard deviation.
``--loss data`` is the Stage 3 normalized MSE. ``--loss residual`` and
``--loss hybrid`` add the discrete Burgers residual. ``inverse`` recovers
each instance viscosity from the preregistered sparse sensors by
minimizing that residual. ``slices`` scores a data-only checkpoint on the
harder pilot's full test split, the frozen ``hard_ood`` cut, and the
complement. ``aggregate`` reduces those per-seed records. Torch is
imported only when a command actually trains, scores, or runs the Adam
viscosity check. ``aggregate`` does not import torch.

The same commands are available as ``pinnforge fno train``,
``pinnforge fno eval``, and ``pinnforge fno inverse``.
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
from pinnforge.operator.residual import (
    DEFAULT_FRAME_DT,
    LOSS_MODES,
    RESIDUAL_SCOPES,
    RESIDUAL_SPACES,
    LossConfig,
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
        if args.command == "inverse":
            return _inverse(args)
        if args.command == "slices":
            return _slices(args)
        if args.command == "aggregate":
            return _aggregate(args)
    except ValueError as exc:
        parser.error(str(exc))
    parser.error(f"unknown command {args.command}")
    return 2


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="pinnforge fno",
        description=(
            "1D Fourier neural operator on Burgers windows. "
            "The default pilot is the Stage 2 dataset. "
            "--pilot and --manifest also accept the Stage A harder pilot. "
            "The default loss is normalized data MSE. "
            "residual and hybrid add the discrete Burgers residual. "
            "inverse recovers scalar viscosity from preregistered sparse sensors. "
            "slices scores hard_ood on a data-only checkpoint."
        ),
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
        "--protocol",
        type=Path,
        default=None,
        help=(
            "Frozen training-protocol JSON. When set, the pilot, manifest, "
            "seed, window, architecture, epochs, and data-only loss must match it."
        ),
    )
    train.add_argument("--loss", choices=LOSS_MODES, default="data")
    train.add_argument(
        "--residual-weight",
        type=float,
        default=None,
        help="Positive weight on the residual MSE. Required for hybrid and rejected otherwise.",
    )
    train.add_argument("--residual-scope", choices=RESIDUAL_SCOPES, default="with_input")
    train.add_argument("--residual-space", choices=RESIDUAL_SPACES, default="physical")
    train.add_argument(
        "--dt",
        type=float,
        default=DEFAULT_FRAME_DT,
        help="Saved-frame spacing used by the central time difference (pilot save_dt is 0.01).",
    )
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
    evaluate.add_argument(
        "--residual-scope",
        choices=RESIDUAL_SCOPES,
        default=None,
        help="Override the checkpoint residual scope when scoring |R|. Default: the checkpoint value.",
    )
    evaluate.add_argument("--residual-space", choices=RESIDUAL_SPACES, default=None)
    evaluate.add_argument(
        "--dt",
        type=float,
        default=None,
        help="Override the checkpoint frame spacing when scoring |R|.",
    )
    evaluate.add_argument("--skip-field-hash", action="store_true")
    inverse = subparsers.add_parser(
        "inverse",
        help="Recover instance viscosity from the preregistered sparse sensors",
    )
    inverse.add_argument("--pilot", type=Path, default=Path("artifacts/burgers_pilot"))
    inverse.add_argument("--manifest", type=Path, default=Path("docs/stage2/pilot_manifest.json"))
    inverse.add_argument("--split", choices=("train", "val", "test"), default="test")
    inverse.add_argument(
        "--output",
        type=Path,
        default=None,
        help="JSON record. Default: runs/stage5_inverse/eval_<split>.json",
    )
    inverse.add_argument("--skip-field-hash", action="store_true")
    slices = subparsers.add_parser(
        "slices",
        help="Score full test, hard_ood, and the complement on a data-only checkpoint",
    )
    slices.add_argument("--pilot", type=Path, default=Path("artifacts/burgers_hard_pilot"))
    slices.add_argument("--manifest", type=Path, default=Path("docs/stage_a/pilot_manifest.json"))
    slices.add_argument("--protocol", type=Path, default=Path("docs/v02/pilot_protocol.json"))
    slices.add_argument(
        "--train-protocol",
        type=Path,
        default=None,
        help="When set, the checkpoint and the slice counts must match this frozen training protocol.",
    )
    slices.add_argument("--checkpoint", type=Path, required=True)
    slices.add_argument("--split", choices=("train", "val", "test"), default="test")
    slices.add_argument("--batch-size", type=int, default=DEFAULT_BATCH_SIZE)
    slices.add_argument("--output", type=Path, required=True)
    slices.add_argument("--skip-field-hash", action="store_true")
    aggregate = subparsers.add_parser(
        "aggregate",
        help="Mean and sample std of slice scores across the frozen seeds",
    )
    aggregate.add_argument("--protocol", type=Path, default=Path("docs/v02/stage_b_train_protocol.json"))
    aggregate.add_argument("--inputs", type=Path, nargs="+", required=True)
    aggregate.add_argument("--output", type=Path, required=True)
    return parser


def _train(args: argparse.Namespace) -> int:
    loss = _loss_from_train_args(args)
    if args.protocol is not None:
        from pinnforge.operator.slices import (
            assert_train_call_matches_protocol,
            load_train_protocol,
        )

        protocol = load_train_protocol(args.protocol)
        assert_train_call_matches_protocol(
            protocol,
            pilot=args.pilot,
            manifest=args.manifest,
            epochs=args.epochs,
            batch_size=args.batch_size,
            lr=args.lr,
            width=args.width,
            modes=args.modes,
            layers=args.layers,
            input_frames=args.input_frames,
            output_frames=args.output_frames,
            stride=args.stride,
            seed=args.seed,
            loss=args.loss,
            residual_weight=args.residual_weight,
        )
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
        loss=loss,
        check_field_hash=not args.skip_field_hash,
        protocol_path=args.protocol,
        log=_print_epoch,
    )
    print(f"checkpoint: {result.checkpoint.as_posix()}")
    print(f"metrics: {result.metrics_path.as_posix()}")
    print(f"manifest: {result.manifest_path.as_posix()}")
    print(f"parameters: {result.parameter_count}")
    print(f"selected_epoch: {result.selected_epoch}")
    print(f"loss: {loss.describe()}")
    print(f"wall_clock_seconds: {result.wall_clock_seconds:.4f}")
    print(f"train_mse: {result.train_mse:.8e}")
    print(f"val_mse: {result.val_mse:.8e}")
    print(f"val_relative_l2: {result.val_relative_l2:.8e}")
    return 0


def _eval(args: argparse.Namespace) -> int:
    evaluate_checkpoint, load_fno_checkpoint, write_eval_json = _import_evaluator()
    loaded = load_fno_checkpoint(args.checkpoint)
    residual = _loss_from_eval_args(args, loaded.loss)
    score = evaluate_checkpoint(
        args.checkpoint,
        args.pilot,
        args.manifest,
        split=args.split,
        batch_size=args.batch_size,
        check_field_hash=not args.skip_field_hash,
        residual=residual,
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
    print(f"mean_abs_residual: {score.mean_abs_residual:.8e}")
    print(f"residual_mse: {score.residual_mse:.8e}")
    print(f"target_mean_abs_residual: {score.target_mean_abs_residual:.8e}")
    print(
        "residual_definition: "
        f"scope={residual.residual_scope} space={residual.residual_space} dt={residual.dt:.8e}"
    )
    print(f"json: {Path(output).as_posix()}")
    return 0


def _inverse(args: argparse.Namespace) -> int:
    from pinnforge.operator.inverse import (
        ABLATION_OBSERVATION,
        PREREGISTERED_OBSERVATION,
        dense_reference_observation,
        load_recovery_split,
        score_trajectories,
        write_inverse_json,
    )

    trajectories, baseline = load_recovery_split(
        args.pilot,
        args.manifest,
        split=args.split,
        check_field_hash=not args.skip_field_hash,
    )
    primary = score_trajectories(trajectories, PREREGISTERED_OBSERVATION, baseline)
    ablation = score_trajectories(trajectories, ABLATION_OBSERVATION, baseline)
    dense_spec = dense_reference_observation(int(trajectories[0].u.shape[0]))
    dense = score_trajectories(trajectories, dense_spec, baseline)
    output = args.output
    if output is None:
        output = Path("runs/stage5_inverse") / f"eval_{args.split}.json"
    payload = {
        "format": "pinnforge.burgers_inverse.v1",
        "split": args.split,
        "n_instances": primary.n_instances,
        "baseline_nu": baseline,
        "baseline_source": "training-split mean viscosity in the pilot manifest",
        "objective": "least squares of the Stage 4 central Burgers residual in scalar nu",
        "primary_pattern": PREREGISTERED_OBSERVATION.to_dict(),
        "primary": primary.to_dict(include_instances=True),
        "ablation_pattern": ABLATION_OBSERVATION.to_dict(),
        "ablation": ablation.to_dict(include_instances=False),
        "dense_pattern": dense_spec.to_dict(),
        "dense_reference": dense.to_dict(include_instances=False),
    }
    write_inverse_json(payload, output)
    observed = PREREGISTERED_OBSERVATION.n_observed_frames() * PREREGISTERED_OBSERVATION.n_sensors
    print(f"split: {args.split}")
    print(f"pattern: {primary.pattern}")
    print(f"n_instances: {primary.n_instances}")
    print(f"n_observations_per_instance: {observed}")
    print(f"baseline_nu: {baseline:.16e}")
    _print_recovery("primary", primary)
    _print_recovery("ablation", ablation)
    _print_recovery("dense_reference", dense)
    print(f"json: {Path(output).as_posix()}")
    return 0


def _slices(args: argparse.Namespace) -> int:
    from pinnforge.operator.slices import (
        assert_checkpoint_matches_train_protocol,
        assert_manifest_slice_counts,
        load_data_protocol,
        load_train_protocol,
        write_json,
    )

    data_protocol = load_data_protocol(args.protocol)
    train_protocol = None
    if args.train_protocol is not None:
        train_protocol = load_train_protocol(args.train_protocol)
        if float(train_protocol["hard_ood"]["threshold_nu"]) != float(data_protocol["hard_ood"]["threshold_nu"]):
            raise ValueError("train protocol and data protocol thresholds differ")
    evaluate_slices, load_fno_checkpoint = _import_slice_evaluator()
    if train_protocol is not None:
        loaded = load_fno_checkpoint(args.checkpoint)
        assert_checkpoint_matches_train_protocol(
            {
                "loss_mode": loaded.loss.mode,
                "seed": loaded.seed,
                "width": loaded.model.width,
                "modes": loaded.model.modes,
                "layers": loaded.model.n_layers,
                "input_frames": loaded.spec.input_frames,
                "output_frames": loaded.spec.output_frames,
                "stride": loaded.spec.stride,
            },
            train_protocol,
        )
    payload = evaluate_slices(
        args.checkpoint,
        args.pilot,
        args.manifest,
        data_protocol,
        split=args.split,
        batch_size=args.batch_size,
        check_field_hash=not args.skip_field_hash,
    )
    if train_protocol is not None:
        assert_manifest_slice_counts(payload["slices"], train_protocol, args.split)
        payload["training_protocol"] = str(args.train_protocol)
    write_json(payload, args.output)
    print(f"split: {payload['split']}")
    print(f"seed: {payload['seed']}")
    print(f"selected_epoch: {payload['selected_epoch']}")
    print(f"threshold_nu: {payload['threshold_nu']:.16e}")
    print(f"loss_mode: {payload['loss_mode']}")
    for name in ("full_test", "hard_ood", "complement"):
        section = payload["slices"][name]
        rollout = payload["rollout"][name]
        print(
            f"{name}: n_instances={section['n_instances']} n_windows={section['n_windows']} "
            f"mean_relative_l2={section['mean_relative_l2']:.8e} "
            f"persistence_mean_relative_l2={section['persistence_mean_relative_l2']:.8e} "
            f"rollout_mean_instance_relative_l2={rollout['mean_instance_relative_l2']:.8e} "
            f"rollout_persistence_mean_instance_relative_l2="
            f"{rollout['persistence_mean_instance_relative_l2']:.8e}"
        )
    print(f"json: {Path(args.output).as_posix()}")
    return 0


def _aggregate(args: argparse.Namespace) -> int:
    import json

    from pinnforge.operator.slices import aggregate_slice_records, load_train_protocol, write_json

    protocol = load_train_protocol(args.protocol)
    records = []
    for path in args.inputs:
        payload = json.loads(Path(path).read_text(encoding="utf-8"))
        if not isinstance(payload, dict):
            raise ValueError(f"{path} must contain a JSON object")
        records.append(payload)
    summary = aggregate_slice_records(records, protocol)
    summary["training_protocol"] = str(args.protocol)
    write_json(summary, args.output)
    print(f"seeds: {summary['seeds']}")
    print(f"ddof: {summary['ddof']}")
    for name in ("full_test", "hard_ood", "complement"):
        metric = summary["slices"][name]["mean_relative_l2"]
        baseline = summary["slices"][name]["persistence_mean_relative_l2"]
        rollout = summary["rollout"][name]["mean_instance_relative_l2"]
        print(
            f"{name}_mean_relative_l2: {metric['mean']:.8e} ± {metric['std']:.8e} "
            f"persistence: {baseline['mean']:.8e} ± {baseline['std']:.8e} "
            f"rollout: {rollout['mean']:.8e} ± {rollout['std']:.8e}"
        )
    print(f"json: {Path(args.output).as_posix()}")
    return 0


def _print_recovery(label: str, score: object) -> None:
    print(f"{label}_mean_abs_error: {score.mean_abs_error:.16e}")
    print(f"{label}_median_abs_error: {score.median_abs_error:.16e}")
    print(f"{label}_max_abs_error: {score.max_abs_error:.16e}")
    print(f"{label}_mean_rel_error: {score.mean_rel_error:.16e}")
    print(f"{label}_median_rel_error: {score.median_rel_error:.16e}")
    print(f"{label}_max_rel_error: {score.max_rel_error:.16e}")
    correlation = score.correlation
    if correlation is None:
        print(f"{label}_correlation: null")
    else:
        print(f"{label}_correlation: {correlation:.16e}")
    print(f"{label}_n_failures: {score.n_failures}")
    print(f"{label}_n_nonpositive: {score.n_nonpositive}")
    print(f"{label}_n_worse_than_baseline: {score.n_worse_than_baseline}")
    print(f"{label}_baseline_mean_abs_error: {score.baseline_mean_abs_error:.16e}")
    print(f"{label}_baseline_mean_rel_error: {score.baseline_mean_rel_error:.16e}")
    print(f"{label}_mean_abs_residual: {score.mean_abs_residual:.16e}")
    print(f"{label}_mean_abs_residual_at_truth: {score.mean_abs_residual_at_truth:.16e}")
    print(f"{label}_mean_square_residual: {score.mean_square_residual:.16e}")
    print(f"{label}_mean_square_residual_at_truth: {score.mean_square_residual_at_truth:.16e}")


def _print_epoch(row: object) -> None:
    print(
        f"epoch: {row.epoch} train_mse: {row.train_mse:.8e} "
        f"train_objective: {row.train_objective:.8e} "
        f"val_mse: {row.val_mse:.8e} val_relative_l2: {row.val_relative_l2:.8e} "
        f"val_mean_abs_residual: {row.val_mean_abs_residual:.8e}",
        flush=True,
    )


def _loss_from_train_args(args: argparse.Namespace) -> LossConfig:
    weight = args.residual_weight
    if args.loss == "hybrid":
        if weight is None:
            raise ValueError("hybrid loss requires --residual-weight > 0")
    elif weight is not None:
        raise ValueError("--residual-weight is only valid with --loss hybrid")
    else:
        weight = 0.0
    return LossConfig(
        mode=args.loss,
        residual_weight=0.0 if weight is None else weight,
        residual_scope=args.residual_scope,
        residual_space=args.residual_space,
        dt=args.dt,
    )


def _loss_from_eval_args(args: argparse.Namespace, saved: LossConfig) -> LossConfig:
    """Residual stencil for scoring. Training mode is not reapplied."""

    return LossConfig(
        mode="data",
        residual_weight=0.0,
        residual_scope=saved.residual_scope if args.residual_scope is None else args.residual_scope,
        residual_space=saved.residual_space if args.residual_space is None else args.residual_space,
        dt=saved.dt if args.dt is None else args.dt,
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


def _import_slice_evaluator():
    from pinnforge.ml_import import InstallHint

    try:
        from pinnforge.operator.checkpoint import load_fno_checkpoint
        from pinnforge.operator.evaluate import evaluate_slices
    except InstallHint as exc:
        print(str(exc), file=sys.stderr)
        raise SystemExit(1) from exc
    except ModuleNotFoundError as exc:
        if exc.name != "torch":
            raise
        print(str(InstallHint()), file=sys.stderr)
        raise SystemExit(1) from exc
    return evaluate_slices, load_fno_checkpoint


if __name__ == "__main__":
    raise SystemExit(main())
