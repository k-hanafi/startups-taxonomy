#!/usr/bin/env python3
"""Offline breakdown of the dead-cohort crawl: WHY each company has no evidence.

Reads ``crawl_dead.jsonl`` with the stdlib only (no API calls, no OPENAI/TAVILY
key) and prints a table: total rows, successes by method, and the no-evidence rows
grouped by ``failure_reason``. The point is to separate recoverable infrastructure
problems (``rate_limited`` / ``transient_error`` / ``network_error`` — a resume
will re-attempt these) from a permanent property of the company
(``no_archive_content`` — the Archive genuinely has nothing usable).

Deliberately imports neither the crawl engine nor any paid client. It uses
``wayback_machine.paths`` for the default JSONL location.

Usage:

    python3 wayback_machine/scripts/summarize_crawl_failures.py
    python3 wayback_machine/scripts/summarize_crawl_failures.py --jsonl path/to.jsonl
"""

from __future__ import annotations

import argparse
import json
import sys
from collections import Counter
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(PROJECT_ROOT))

from wayback_machine.paths import CRAWL_DEAD_JSONL  # noqa: E402

SUCCESS_STATUSES = ("success", "success_fallback", "success_extract_fallback")
TERMINAL_EMPTY_STATUS = "empty_results"


def _failure_reason(row: dict) -> str:
    """Failure bucket for a non-success row; mirrors crawl_dead._row_failure_reason."""
    reason = str(row.get("failure_reason", "")).strip()
    if reason:
        return reason
    status = str(row.get("status", "")).strip()
    if status == TERMINAL_EMPTY_STATUS:
        return "legacy_empty"
    return status or "unknown"


def summarize(jsonl_path: Path) -> dict:
    """Single stdlib pass over the JSONL → counts by method and failure reason."""
    total = 0
    successes: Counter[str] = Counter()
    thin = 0
    failures: Counter[str] = Counter()
    retryable_failures = 0

    if not jsonl_path.exists():
        return {
            "total": 0, "successes": {}, "thin_evidence": 0,
            "failures": {}, "retryable_failures": 0,
        }

    with jsonl_path.open(encoding="utf-8") as fh:
        for line in fh:
            if not line.strip():
                continue
            try:
                row = json.loads(line)
            except json.JSONDecodeError:
                continue
            total += 1
            status = str(row.get("status", ""))
            if status in SUCCESS_STATUSES:
                successes[status] += 1
            elif status == "thin_evidence":
                thin += 1
            else:
                failures[_failure_reason(row)] += 1
                if row.get("retryable") is True:
                    retryable_failures += 1

    return {
        "total": total,
        "successes": dict(successes),
        "thin_evidence": thin,
        "failures": dict(failures),
        "retryable_failures": retryable_failures,
    }


def _pct(n: int, total: int) -> str:
    return f"{(n / total * 100.0):5.1f}%" if total else "  n/a"


def format_report(summary: dict) -> str:
    total = summary["total"]
    succ_total = sum(summary["successes"].values())
    fail_total = sum(summary["failures"].values())

    lines = [
        "DEAD-COHORT CRAWL FAILURE BREAKDOWN",
        f"  Rows recorded:        {total:,}",
        f"  Successes:            {succ_total:,} ({_pct(succ_total, total)})",
    ]
    for status in SUCCESS_STATUSES:
        n = summary["successes"].get(status, 0)
        if n:
            lines.append(f"    {status:<26} {n:,} ({_pct(n, total)})")
    lines.append(f"  Thin evidence:        {summary['thin_evidence']:,} "
                 f"({_pct(summary['thin_evidence'], total)})")
    lines.append(f"  No-evidence rows:     {fail_total:,} ({_pct(fail_total, total)})")
    for reason, n in sorted(summary["failures"].items(), key=lambda kv: (-kv[1], kv[0])):
        lines.append(f"    {reason:<26} {n:,} ({_pct(n, total)})")
    lines.append(f"  Retryable (resume):   {summary['retryable_failures']:,} "
                 f"({_pct(summary['retryable_failures'], total)})")
    return "\n".join(lines)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--jsonl", type=Path, default=CRAWL_DEAD_JSONL,
                        help="Path to crawl_dead.jsonl (default: the survivorship output).")
    args = parser.parse_args()
    print(format_report(summarize(args.jsonl)))


if __name__ == "__main__":
    main()
