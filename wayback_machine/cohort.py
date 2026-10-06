"""Vendored column contracts and cohort helpers.

Holds the contracts from ``tavily_crawler/master_csv.py`` that must agree
exactly, plus the two Wayback-specific helpers for reading a coverage probe row
and turning it into a raw archive snapshot URL.
"""

from __future__ import annotations

from urllib.parse import urlparse

from .config import COHORT_FOUNDED_CUTOFF, SNAPSHOT_HOST, SNAPSHOT_SUFFIX

# ---------------------------------------------------------------------------
# Vendored schema (must match tavily_crawler/master_csv.py exactly)
# ---------------------------------------------------------------------------

# Static company metadata in master_csv.csv (11 fields, pre-crawl).
MASTER_CSV_COLUMNS = [
    "org_uuid",
    "name",
    "homepage_url",
    "short_description",
    "Long description",
    "category_list",
    "category_groups_list",
    "founded_date",
    "employee_count",
    "total_funding_usd",
    "website_alive",
]

# Classifier input columns. The 2023 output must keep this order. Only
# website_evidence differs across evidence strands.
CLASSIFIER_INPUT_COLUMNS = MASTER_CSV_COLUMNS + [
    "website_pages_used",
    "website_evidence",
]

# Columns produced by probe_coverage.py (the coverage_*.csv contract).
COVERAGE_FIELDS = [
    "org_uuid", "name", "homepage_url", "founded_date", "host",
    "has_2023", "closest_ts", "days_from_target", "n_window_captures",
    "has_any_ever", "status",
]


def is_valid_homepage_url(value: object) -> bool:
    """Return True for crawlable HTTP(S) homepage URLs."""
    raw = str(value).strip() if value is not None else ""
    if not raw or raw.lower() in {"nan", "none", "nat"}:
        return False
    parsed = urlparse(raw)
    return parsed.scheme in {"http", "https"} and bool(parsed.netloc)


# ---------------------------------------------------------------------------
# Wayback-specific helpers
# ---------------------------------------------------------------------------


def is_retrievable(row: dict[str, str]) -> bool:
    """True if a coverage row has a usable March-2023 capture to scrape."""
    return (
        row.get("status") == "ok"
        and row.get("has_2023") == "True"
        and bool((row.get("closest_ts") or "").strip())
    )


def existed_by(founded_date: str, cutoff: str = COHORT_FOUNDED_CUTOFF) -> bool:
    """True if the company was founded on or before ``cutoff`` (YYYY-MM, inclusive).

    Our ``founded_date`` is YYYY-MM, so a lexicographic prefix compare is correct
    and also degrades sanely for year-only values. Blank founding is treated as
    "did not exist" (conservative); the current cohort has none.
    """
    fd = (founded_date or "").strip()
    if not fd:
        return False
    return fd[:7] <= cutoff


def build_snapshot_url(timestamp: str, homepage_url: str) -> str:
    """Build the raw (toolbar-free) Wayback URL for a capture.

    The ``id_`` suffix after the 14-digit timestamp is what makes the Archive
    return the original page bytes instead of its wrapped viewer. Wayback
    resolves to the nearest capture if this exact pair shifts slightly.
    """
    url = (homepage_url or "").strip()
    if "://" not in url:
        url = "https://" + url
    return f"{SNAPSHOT_HOST}/web/{timestamp}{SNAPSHOT_SUFFIX}/{url}"
