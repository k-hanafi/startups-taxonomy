#!/usr/bin/env python3
"""Stage C CLI: the paid Tavily extract over the frozen target list.

Resumable and safe to interrupt (Ctrl-C drains cleanly; re-running skips done
companies). For an overnight run, wrap in caffeinate OUTSIDE the sandbox:

    caffeinate -ims python3 wayback_machine/scripts/run_extract.py
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(PROJECT_ROOT))

from wayback_machine.config import (  # noqa: E402
    DEFAULT_BUDGET_CREDITS,
    DEFAULT_MAX_CONCURRENT_ROWS,
    ExtractConfig,
)
from wayback_machine.extract import run_extract  # noqa: E402
from wayback_machine.live_gate import add_live_flag, require_live  # noqa: E402
from wayback_machine.paths import SCRAPE_TARGETS_CSV  # noqa: E402


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    add_live_flag(parser)
    parser.add_argument("--targets", type=Path, default=SCRAPE_TARGETS_CSV)
    parser.add_argument("--max-companies", type=int, default=None,
                        help="Cap rows this run (smoke test / incremental).")
    parser.add_argument("--concurrency", type=int, default=DEFAULT_MAX_CONCURRENT_ROWS)
    parser.add_argument("--budget-credits", type=float, default=DEFAULT_BUDGET_CREDITS)
    parser.add_argument("--extract-depth", default="basic", choices=["basic", "advanced"])
    parser.add_argument("--extract-rpm", type=float, default=None,
                        help="Override RPM cap. 0 disables the limiter.")
    args = parser.parse_args(argv)
    require_live(args.live)

    if args.extract_depth == "advanced":
        print(
            "[extract] advanced depth bills 2 credits per 5 successes "
            "(basic is 1 per 5); budget estimates use the advanced rate.",
            file=sys.stderr,
        )

    report = run_extract(
        targets_csv=args.targets,
        config=ExtractConfig(extract_depth=args.extract_depth),
        budget_credits=args.budget_credits,
        max_companies=args.max_companies,
        max_concurrent_rows=args.concurrency,
        extract_rpm=args.extract_rpm,
    )
    print("\n" + report.format_report(), file=sys.stderr)


if __name__ == "__main__":
    main()
