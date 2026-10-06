#!/usr/bin/env python3
"""CLI adapter for the cost-controlled Tavily Crawl enrichment job."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

from .crawl import (
    DEFAULT_CLASSIFIER_INPUT_CSV,
    DEFAULT_CRAWL_RPM_HEADROOM,
    DEFAULT_CRAWL_STATE_JSON,
    DEFAULT_HEARTBEAT_EVERY,
    DEFAULT_HEARTBEAT_LOG,
    DEFAULT_MAX_CONCURRENT_ROWS,
    DEFAULT_MAX_OUTAGE_SECONDS,
    DEFAULT_OUTAGE_BACKOFF_MAX_SECONDS,
    DEFAULT_OUTAGE_BACKOFF_MIN_SECONDS,
    DEFAULT_RAW_RESULTS_JSONL,
    DEFAULT_RUN_MANIFEST_CSV,
    DEFAULT_TAVILY_PROCESSED_CSV,
    TAVILY_CRAWL_RPM_DOCUMENTED,
    TavilyCrawlConfig,
    run_tavily_crawl,
)
from .master_csv import DEFAULT_MASTER_CSV


def _path(value: str) -> Path:
    return Path(value).expanduser()


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="python -m tavily_crawler crawl",
        description="Run Tavily Crawl over master_csv.csv (skips website_alive=false rows).",
    )
    parser.add_argument(
        "--input",
        type=_path,
        dest="input",
        default=DEFAULT_MASTER_CSV,
        help="data/master_csv.csv (rows with website_alive=false are skipped).",
    )
    parser.add_argument("--output", type=_path, default=DEFAULT_RAW_RESULTS_JSONL,
                        help="Append-only raw Tavily JSONL (one record per company).")
    parser.add_argument("--processed-csv", type=_path, default=DEFAULT_TAVILY_PROCESSED_CSV,
                        dest="processed_csv",
                        help="Append-only processed output CSV (5 columns, written inline).")
    parser.add_argument("--classifier-input", type=_path, default=DEFAULT_CLASSIFIER_INPUT_CSV,
                        dest="classifier_input",
                        help="Joined master+evidence CSV written on clean completion.")
    parser.add_argument("--state", type=_path, default=DEFAULT_CRAWL_STATE_JSON)
    parser.add_argument("--budget-credits", type=float, default=100_000.0)
    parser.add_argument("--max-companies", type=int, default=None)
    parser.add_argument("--limit", type=int, default=5)
    parser.add_argument("--max-depth", type=int, default=2)
    parser.add_argument("--max-breadth", type=int, default=20)
    parser.add_argument("--chunks-per-source", type=int, default=4)
    parser.add_argument("--timeout", type=float, default=60.0)
    parser.add_argument("--allow-external", action="store_true")
    parser.add_argument(
        "--max-concurrent-rows",
        type=int,
        default=DEFAULT_MAX_CONCURRENT_ROWS,
        help="Number of worker threads crawling different companies in parallel. "
             f"Default {DEFAULT_MAX_CONCURRENT_ROWS}. Use 1 for strictly serial I/O.",
    )
    parser.add_argument(
        "--crawl-rpm",
        type=float,
        default=None,
        help="Hard cap on Tavily Crawl POSTs per minute (sliding window). "
             f"Default: {TAVILY_CRAWL_RPM_DOCUMENTED} × --crawl-rpm-headroom. "
             "Set to 0 to disable throttling (not recommended).",
        metavar="RPM",
    )
    parser.add_argument(
        "--crawl-rpm-headroom",
        type=float,
        default=DEFAULT_CRAWL_RPM_HEADROOM,
        help=f"Fraction of Tavily's documented crawl limit ({TAVILY_CRAWL_RPM_DOCUMENTED} RPM) to target "
             f"when --crawl-rpm is omitted. Default {DEFAULT_CRAWL_RPM_HEADROOM} (~"
             f"{int(TAVILY_CRAWL_RPM_DOCUMENTED * DEFAULT_CRAWL_RPM_HEADROOM)} RPM).",
    )
    parser.add_argument(
        "--heartbeat-every",
        type=int,
        default=DEFAULT_HEARTBEAT_EVERY,
        help="Emit one progress line per N finished rows. 0 disables.",
    )
    parser.add_argument(
        "--heartbeat-log",
        type=_path,
        default=DEFAULT_HEARTBEAT_LOG,
        help="Append heartbeat lines to this file in addition to stderr.",
    )
    parser.add_argument(
        "--manifest-csv",
        type=_path,
        default=DEFAULT_RUN_MANIFEST_CSV,
        help="Append one summary row per run to this CSV.",
    )
    parser.add_argument(
        "--max-outage-seconds",
        type=float,
        default=DEFAULT_MAX_OUTAGE_SECONDS,
        help="Per-row long-outage retry cap. After this much wall time of transient errors "
             "on the same row, give up and continue.",
    )
    parser.add_argument(
        "--outage-backoff-min-seconds",
        type=float,
        default=DEFAULT_OUTAGE_BACKOFF_MIN_SECONDS,
    )
    parser.add_argument(
        "--outage-backoff-max-seconds",
        type=float,
        default=DEFAULT_OUTAGE_BACKOFF_MAX_SECONDS,
    )
    parser.add_argument(
        "--min-free-disk-gb",
        type=float,
        default=5.0,
        help="Refuse to start if free disk at the output dir is below this. Use 0 to disable.",
    )
    parser.add_argument(
        "--skip-preflight-call",
        action="store_true",
        help="Skip the ~1-credit dry call that confirms the API key and request shape work.",
    )
    parser.add_argument(
        "--live",
        action="store_true",
        help="Run the paid crawl. Without this flag the command exits 2.",
    )
    return parser


def main(argv: list[str] | None = None) -> None:
    args = build_parser().parse_args(argv)
    if not args.live:
        print(
            "This paid command is frozen. Pass --live to spend credits. "
            "Classify companies with: python -m two_pass_classifier",
            file=sys.stderr,
        )
        raise SystemExit(2)
    config = TavilyCrawlConfig(
        limit=args.limit,
        max_depth=args.max_depth,
        max_breadth=args.max_breadth,
        chunks_per_source=args.chunks_per_source,
        timeout=args.timeout,
        allow_external=args.allow_external,
    )
    report = run_tavily_crawl(
        queue_csv=args.input,
        output_jsonl=args.output,
        state_json=args.state,
        config=config,
        budget_credits=args.budget_credits,
        max_companies=args.max_companies,
        processed_csv=args.processed_csv,
        classifier_input_csv=args.classifier_input,
        heartbeat_every=args.heartbeat_every,
        heartbeat_log=args.heartbeat_log,
        manifest_csv=args.manifest_csv,
        max_outage_seconds=args.max_outage_seconds,
        outage_backoff_min_seconds=args.outage_backoff_min_seconds,
        outage_backoff_max_seconds=args.outage_backoff_max_seconds,
        min_free_disk_gb=args.min_free_disk_gb,
        preflight_dry_call=not args.skip_preflight_call,
        max_concurrent_rows=args.max_concurrent_rows,
        crawl_rpm=args.crawl_rpm,
        crawl_rpm_headroom=args.crawl_rpm_headroom,
    )
    print(report.format_report())
    print(f"  Raw results JSONL:        {args.output}")
    print(f"  Processed output CSV:     {args.processed_csv}")
    print(f"  Classifier input CSV:     {args.classifier_input}")
    print(f"  Crawl state JSON:         {args.state}")
    print(f"  Heartbeat log:            {args.heartbeat_log}")
    print(f"  Run manifest:             {args.manifest_csv}")


if __name__ == "__main__":
    main()
