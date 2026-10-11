"""Run ``python -m pinnforge.claims check``."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

from pinnforge.claims.check import check


def main(argv: list[str] | None = None, prog: str = "python -m pinnforge.claims") -> int:
    parser = argparse.ArgumentParser(prog=prog)
    subparsers = parser.add_subparsers(dest="command", required=True)
    check_parser = subparsers.add_parser(
        "check",
        help="Fail if a registered claim or an unregistered superiority phrase disagrees with evidence",
    )
    check_parser.add_argument(
        "--root",
        type=Path,
        default=None,
        help="Repository root. Defaults to the current working directory.",
    )
    check_parser.add_argument(
        "--registry",
        type=Path,
        default=None,
        help="Claims registry. Defaults to docs/claims.yaml under --root.",
    )
    args = parser.parse_args(argv)
    if args.command != "check":
        parser.error(f"unknown command {args.command}")
        return 2
    failures = check(args.root, args.registry)
    if failures:
        for item in failures:
            print(item.format(), file=sys.stderr)
        print(f"{len(failures)} claim check(s) failed", file=sys.stderr)
        return 1
    print("claim checks passed")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
