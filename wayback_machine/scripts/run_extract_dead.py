#!/usr/bin/env python3
"""Stage C CLI (survivorship): the paid Tavily /extract over the dead target list.

One single-page extract per company on its pre-death Wayback snapshot — enough
evidence for the classifier, and far friendlier to the Internet Archive's playback
limits than the multi-page crawl (see wayback_machine/extract_dead.py).

Resumable and safe to interrupt (Ctrl-C drains at the next row boundary; re-running
skips finished companies and never re-bills them). For the full overnight run, wrap
in caffeinate OUTSIDE the Cursor sandbox:

    caffeinate -ims python3 wayback_machine/scripts/run_extract_dead.py

Smoke-test first with --max-companies 20.
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(PROJECT_ROOT))

from wayback_machine.config import (  # noqa: E402
    DEFAULT_BUDGET_CREDITS,
    DEFAULT_HEARTBEAT_EVERY,
    DEFAULT_MAX_OUTAGE_SECONDS,
    ExtractConfig,
)
from wayback_machine.extract_dead import (  # noqa: E402
    DEFAULT_EXTRACT_DEAD_CONCURRENCY,
    run_extract_dead,
)
from wayback_machine.live_gate import add_live_flag, require_live  # noqa: E402
from wayback_machine.paths import (  # noqa: E402
    CRAWL_DEAD_JSONL,
    CRAWL_STATE_DEAD_JSON,
    SCRAPE_PROCESSED_DEAD_CSV,
    SCRAPE_TARGETS_DEAD_CSV,
)


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    add_live_flag(parser)
    parser.add_argument("--targets", type=Path, default=SCRAPE_TARGETS_DEAD_CSV)
    parser.add_argument("--max-companies", type=int, default=None,
                        help="Cap rows this run (smoke test / incremental).")
    parser.add_argument("--concurrency", type=int, default=DEFAULT_EXTRACT_DEAD_CONCURRENCY)
    parser.add_argument("--budget-credits", type=float, default=DEFAULT_BUDGET_CREDITS)
    parser.add_argument("--extract-depth", default="basic", choices=["basic", "advanced"],
                        help="Tavily extract depth (advanced bills 2 credits/5, basic 1/5).")
    parser.add_argument("--extract-rpm", type=float, default=None,
                        help="Override extract RPM cap. 0 disables the limiter.")
    parser.add_argument("--max-outage-seconds", type=float, default=DEFAULT_MAX_OUTAGE_SECONDS,
                        help="Cap the transient-failure retry window (lower it for fast smoke tests).")
    parser.add_argument("--heartbeat-every", type=int, default=DEFAULT_HEARTBEAT_EVERY,
                        help="Log progress every N companies (0 disables).")
    parser.add_argument("--output-jsonl", type=Path, default=CRAWL_DEAD_JSONL)
    parser.add_argument("--processed", type=Path, default=SCRAPE_PROCESSED_DEAD_CSV)
    parser.add_argument("--state", type=Path, default=CRAWL_STATE_DEAD_JSON)
    args = parser.parse_args(argv)
    require_live(args.live)

    if args.extract_depth == "advanced":
        print(
            "[extract_dead] advanced depth bills 2 credits per 5 successes "
            "(basic is 1 per 5); budget estimates use the advanced rate.",
            file=sys.stderr,
        )

    report = run_extract_dead(
        targets_csv=args.targets,
        output_jsonl=args.output_jsonl,
        state_json=args.state,
        config=ExtractConfig(extract_depth=args.extract_depth),
        processed_csv=args.processed,
        budget_credits=args.budget_credits,
        max_companies=args.max_companies,
        max_concurrent_rows=args.concurrency,
        extract_rpm=args.extract_rpm,
        max_outage_seconds=args.max_outage_seconds,
        heartbeat_every=args.heartbeat_every,
    )
    print("\n" + report.format_report(), file=sys.stderr)


if __name__ == "__main__":
    main()
