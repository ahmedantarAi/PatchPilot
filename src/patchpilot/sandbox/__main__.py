"""CLI entrypoint for `python -m patchpilot.sandbox selftest`."""

from __future__ import annotations

import argparse
import sys

from patchpilot.sandbox.fixtures import run_selftest


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        prog="python -m patchpilot.sandbox",
        description="PatchPilot hardened sandbox CLI",
    )
    sub = parser.add_subparsers(dest="command", required=True)
    sub.add_parser("selftest", help="Run the 8 hostile fixtures, lock test, and deps install check")
    args = parser.parse_args(argv)
    if args.command == "selftest":
        return run_selftest()
    return 1


if __name__ == "__main__":
    raise SystemExit(main(sys.argv[1:]))
