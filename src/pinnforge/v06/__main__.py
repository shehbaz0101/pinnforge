"""CLI for the v0.6 The Well shear_flow inverse."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from pinnforge.reference.spectral2d.dataset import load_protocol, protocol_sha256

PROTOCOL = Path("docs/v06/protocol.json")


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="pinnforge v06",
        description="v0.6 The Well shear-flow nu/kappa inverse.",
    )
    sub = parser.add_subparsers(dest="command", required=True)
    sub.add_parser("protocol-hash", help="Print the SHA-256 of the frozen protocol")
    sub.add_parser("download", help="Fetch the frozen Hugging Face file list anonymously")
    sub.add_parser("pack", help="Downsample the named window onto the coarse grid")
    score = sub.add_parser("score", help="Train FNO seeds, then score the frozen methods")
    score.add_argument("--workers", type=int, default=None)
    score.add_argument("--checkpoint", default=None)
    sub.add_parser("report", help="Write docs/v06/report.md")
    sub.add_parser("smoke", help="Tiny synthetic fixture; no dataset download")
    return parser


def main(argv: list[str] | None = None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)
    if args.command == "protocol-hash":
        print(protocol_sha256(PROTOCOL))
        return 0
    if args.command == "download":
        from pinnforge.v06.download import run_download

        record = run_download()
        print(json.dumps({"n_files": len(record["files"]), "protocol_sha256": record["protocol_sha256"]}))
        return 0
    if args.command == "pack":
        from pinnforge.v06.study import run_pack

        record = run_pack()
        print(json.dumps({"n_instances": record["n_instances"], "splits": {k: len(v) for k, v in record["splits"].items()}}))
        return 0
    if args.command == "score":
        from pinnforge.v06.study import run_score

        scores = run_score(
            workers=args.workers,
            checkpoint=None if args.checkpoint is None else Path(args.checkpoint),
        )
        print("overall", "PASS" if scores["bars"]["overall"] else "FAIL")
        return 0
    if args.command == "report":
        from pinnforge.v06.report import write_report

        print(write_report())
        return 0
    if args.command == "smoke":
        return _smoke()
    raise ValueError(args.command)


def _smoke() -> int:
    import numpy as np

    from pinnforge.v06.masks import mask_from_protocol
    from pinnforge.v06.synthetic import manufactured_pack
    from pinnforge.v06.wsindy import fit_coefficients

    protocol = load_protocol(PROTOCOL)
    assert protocol["frozen_before_test_evaluation"]
    assert protocol["test_used_for_selection"] is False
    assert protocol["methods"]["pysr"]["used"] is False
    assert int(protocol["splits"]["from_train_files"]["n_cal"]) >= 60
    tiny = json.loads(json.dumps(protocol))
    tiny["downsample"]["nx"] = 8
    tiny["downsample"]["ny"] = 16
    tiny["sensors"]["n_side_x"] = 4
    tiny["sensors"]["n_side_y"] = 8
    fields, nu, kappa = manufactured_pack(tiny, n_instances=1)
    mask = mask_from_protocol(tiny)
    hat = fit_coefficients(fields[0], protocol=tiny, mask=mask, full_field=True)
    assert np.isfinite(hat["kappa"])
    print("v06 smoke ok", protocol_sha256(PROTOCOL)[:12])
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
