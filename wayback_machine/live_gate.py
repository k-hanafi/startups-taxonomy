"""Paid archive commands stay off until the caller passes --live."""

from __future__ import annotations

import argparse
import sys

FROZEN_MESSAGE = (
    "This paid command is frozen. Pass --live to spend credits. "
    "Classify companies with: python -m two_pass_classifier"
)


def add_live_flag(parser: argparse.ArgumentParser) -> None:
    parser.add_argument(
        "--live",
        action="store_true",
        help="Run the paid command. Without this flag the command exits 2.",
    )


def require_live(enabled: bool) -> None:
    if enabled:
        return
    print(FROZEN_MESSAGE, file=sys.stderr)
    raise SystemExit(2)
