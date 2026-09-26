"""Command line interface.

``version`` prints the package version. ``equations`` lists the Day 1
equation ids. Neither command trains a network or imports torch.
"""

from __future__ import annotations

import argparse

from pinnforge import __version__
from pinnforge.equations import registered_equations


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="pinnforge",
        description="PINNForge physics-informed neural network sandbox.",
    )
    parser.add_argument("--version", action="version", version=f"pinnforge {__version__}")
    subparsers = parser.add_subparsers(dest="command", required=True)
    subparsers.add_parser("version", help="Print the package version")
    subparsers.add_parser("equations", help="List registered equation ids")
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
    parser.error(f"unknown command {args.command}")
    return 2
