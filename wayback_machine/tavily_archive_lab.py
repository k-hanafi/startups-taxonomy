"""Interactive lab for running the EXACT modern Tavily pipeline on archived sites.

This is the reusable engine behind the demo notebook (``notebooks/tavily_archive_lab.ipynb``)
and, later, the per-company scrape worker for the production dead-cohort pipeline.
It deliberately reuses the live pipeline's crawl config + functions + evidence
cleaner unmodified, so what we validate here is exactly what production will run.

The flow for one dead company:

    death_coverage.csv row
        -> archive_url(homepage, closest_ts, "if_")   # pre-death snapshot, iframe mode
        -> the modern TavilyCrawlConfig + crawl call   # 5 pages, depth 2, same instructions
        -> rewrite each archived page URL back to its origin
        -> compact_tavily_response()                   # same cleaner the live cohort used
        -> format_user_message()                       # production user-message text

Heavy imports (the extract endpoint) are deferred into the methods that need
them, so ``import`` and ``candidates()`` work offline with no API keys present.
This lab no longer classifies. Production classification is
``python -m two_pass_classifier``.
"""

from __future__ import annotations

import hashlib
import os
import re
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import pandas as pd

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

# python.org macOS installs can miss a usable system CA bundle. The notebook's
# Tavily calls go through stdlib urllib, so point OpenSSL at certifi's bundled
# roots before the urllib helpers are imported/called.
try:
    import certifi
except ImportError:
    certifi = None
else:
    os.environ.setdefault("SSL_CERT_FILE", certifi.where())
    os.environ.setdefault("REQUESTS_CA_BUNDLE", certifi.where())

from tavily_crawler.crawl import (
    TAVILY_CRAWL_ENDPOINT,
    TavilyCrawlConfig,
    _api_key,
    _call_tavily_crawl_with_retries,
    _fallback_config,
    _has_usable_results,
    extract_usage_credits,
)
from tavily_crawler.website_evidence import compact_tavily_response

from wayback_machine.cdx import to_host
from wayback_machine.cohort import MASTER_CSV_COLUMNS

DEATH_COVERAGE_CSV = PROJECT_ROOT / "wayback_machine" / "data" / "death_coverage.csv"
MASTER_CSV = PROJECT_ROOT / "data" / "master_csv.csv"
PRODUCTION_CSV = PROJECT_ROOT / "outputs" / "production_csvs" / "production_classifications.csv"

CANDIDATE_COLUMNS = [
    "name", "website_alive", "n_captures", "lifespan_days", "founded_date",
    "closest_ts", "death_ts", "target_url", "latest_url", "earliest_url",
    "homepage_url", "org_uuid",
]

# Hosts that are shared platforms, not a company's own crawlable site: their
# capture history belongs to the platform (e.g. airtable.com had 2,618 captures
# spanning decades for a 2022 company). Filtered from candidates by default so
# the demo table leads with real per-company domains.
_SHARED_HOSTS = frozenset({
    "airtable.com", "notion.so", "google.com", "docs.google.com",
    "sites.google.com", "medium.com", "substack.com", "linktr.ee", "carrd.co",
    "github.com", "angel.co", "itsmyurls.com",
})
_SHARED_HOST_SUFFIXES = (
    ".notion.site", ".wixsite.com", ".github.io", ".webflow.io",
    ".framer.website", ".substack.com", ".myshopify.com",
)

_ARCHIVE_PREFIX_RE = re.compile(r"^https?://web\.archive\.org/web/\d+[a-z_]*/")
# Non-anchored variant for stripping archive prefixes from inline links in content.
_INLINE_ARCHIVE_RE = re.compile(r"https?://web\.archive\.org/web/\d+[a-z_]*/")

# Wayback serves if_ (iframe-mode) pages with an injected capture banner that
# Tavily extracts as text. The live + 2023 cohorts have no such chrome, so we
# strip it in this archive-only layer to keep recovered evidence comparable.
# High-precision line matchers (run against each stripped line of raw_content).
_WB_MONTHS = r"Jan|Feb|Mar|Apr|May|Jun|Jul|Aug|Sep|Oct|Nov|Dec"
_WAYBACK_CHROME_LINE_RES = (
    re.compile(r"^\d+\s+captures$", re.I),
    re.compile(r"^\[\d+\s+captures\]\(.*\)$", re.I),
    re.compile(r"^about this capture$", re.I),
    re.compile(r"^collected by$", re.I),
    re.compile(r"^timestamps$", re.I),
    re.compile(r"^collection:\s", re.I),
    re.compile(r"^web crawl data from common crawl\.?$", re.I),
    re.compile(r"^(?:the\s+)?wayback machine\b.*$", re.I),
    re.compile(rf"^\d{{1,2}}\s+(?:{_WB_MONTHS})\w*\s+\d{{4}}\s*[-\u2013]\s*"
               rf"\d{{1,2}}\s+(?:{_WB_MONTHS})\w*\s+\d{{4}}$", re.I),
    re.compile(r"^\|[\s|]*$"),                                  # empty calendar grid
    re.compile(r"^\|\s*-{2,}.*$"),                              # md table separator
    re.compile(rf"^\|(?:\s*(?:{_WB_MONTHS})\s*\|)+$", re.I),    # month row
    re.compile(r"^\|(?:\s*\d{1,4}\s*\|)+$"),                    # day / year row(s)
)


def _strip_wayback_chrome(text: str) -> str:
    """Drop Wayback capture-banner lines and rewrite inline archive links to origin."""
    kept = [
        line for line in (text or "").split("\n")
        if not any(rx.match(line.strip()) for rx in _WAYBACK_CHROME_LINE_RES)
    ]
    cleaned = _INLINE_ARCHIVE_RE.sub("", "\n".join(kept))
    # Tavily sometimes normalizes embedded origins as ``https:/host`` (one slash).
    return re.sub(r"(https?):/([^/])", r"\1://\2", cleaned)


def _is_shared_host(host: str) -> bool:
    host = (host or "").lower()
    return host in _SHARED_HOSTS or host.endswith(_SHARED_HOST_SUFFIXES)


# ---------------------------------------------------------------------------
# Archive URL helpers
# ---------------------------------------------------------------------------


def archive_url(homepage_url: str, timestamp: str, modifier: str = "if_") -> str:
    """Build a Wayback URL for ``homepage_url`` at ``timestamp``.

    ``modifier`` selects how the archive serves the capture:
      - ``""``   : the human viewer (Wayback toolbar + rewritten links).
      - ``"if_"``: iframe mode — toolbar stripped, links still rewritten to stay
                   inside the archive. Required for crawling (links don't escape).
      - ``"id_"``: identity mode — raw bytes, links point at the dead live domain.
                   Good for a single-page extract; a crawler would escape on it.
    """
    url = (homepage_url or "").strip()
    if url and "://" not in url:
        url = "https://" + url
    return f"https://web.archive.org/web/{timestamp}{modifier}/{url}"


def _unwrap_archive_url(url: str) -> str:
    """Strip the ``web.archive.org/web/<ts><modifier>/`` prefix back to the origin URL."""
    origin = _ARCHIVE_PREFIX_RE.sub("", url or "")
    # Tavily sometimes normalizes Wayback-captured URLs as ``https:/host`` (one
    # slash) instead of ``https://host``. Normalize before host checks and before
    # the classifier sees page URLs.
    origin = re.sub(r"^(https?):/([^/])", r"\1://\2", origin)
    return origin


def _scope_for(homepage_url: str) -> tuple[tuple[str, ...], tuple[str, ...]]:
    """Per-company crawl scope: keep the crawl on this company's archived pages.

    On the archive every page lives under ``web.archive.org``, so the modern
    ``allow_external=False`` confinement (which kept the live crawl on the start
    domain) no longer isolates one company. We re-create it with a path regex that
    matches only archived URLs embedding this host, plus a domain pin to the archive.
    """
    host = to_host(homepage_url)
    if not host:
        return (), ()
    escaped = re.escape(host)
    # Tavily/Wayback can represent embedded origin URLs as either
    # ``https://host`` or ``https:/host`` in crawled paths, so accept one or more
    # slashes after the scheme.
    select_paths = (rf"/web/\d+[a-z_]*/https?:/+(?:www\.)?{escaped}.*",)
    select_domains = (r"^web\.archive\.org$",)
    return select_paths, select_domains


@dataclass(frozen=True)
class _ScopedCrawlConfig(TavilyCrawlConfig):
    """The modern crawl config + per-company archive scope.

    Subclasses (rather than edits) ``TavilyCrawlConfig`` so every base default
    (limit/depth/breadth/instructions/exclude_paths) is byte-identical to the live
    run; we only append ``select_paths``/``select_domains`` to the request body.
    ``dataclasses.replace`` (used by the empty-instructions fallback) preserves both.
    """

    select_paths: tuple[str, ...] = ()
    select_domains: tuple[str, ...] = ()

    def request_payload(self, url: str) -> dict[str, Any]:
        payload = super().request_payload(url)
        if self.select_paths:
            payload["select_paths"] = list(self.select_paths)
        if self.select_domains:
            payload["select_domains"] = list(self.select_domains)
        return payload


# ---------------------------------------------------------------------------
# Evidence + diagnostics
# ---------------------------------------------------------------------------


def _rewrite_to_origin(response: dict[str, Any]) -> dict[str, Any]:
    """Return a response whose page URLs are the origin URLs, not archive URLs.

    The cleaner labels pages and fills ``website_pages_used`` straight from each
    result's ``url``. Rewriting archive URLs back to their origin makes the
    recovered evidence format identical to the live cohort's.
    """
    pages = response.get("results")
    if not isinstance(pages, list):
        return {"results": []}
    rewritten = [
        {
            **p,
            "url": _unwrap_archive_url(str(p.get("url", ""))),
            "raw_content": _strip_wayback_chrome(str(p.get("raw_content", ""))),
        }
        for p in pages
        if isinstance(p, dict)
    ]
    return {"results": rewritten}


def clean_evidence(response: dict[str, Any]) -> tuple[str, str]:
    """(`website_pages_used`, `website_evidence`) via the live cleaner, origin URLs."""
    return compact_tavily_response(_rewrite_to_origin(response))


def diagnose(response: dict[str, Any], homepage_url: str) -> dict[str, Any]:
    """Quantify whether the crawl stayed on the company and how much it recovered."""
    results = response.get("results") if isinstance(response, dict) else None
    results = results if isinstance(results, list) else []
    host = to_host(homepage_url)

    page_urls: list[str] = []
    on_domain = 0
    total_chars = 0
    for page in results:
        if not isinstance(page, dict):
            continue
        origin = _unwrap_archive_url(str(page.get("url", "")))
        page_urls.append(origin)
        origin_host = to_host(origin)
        if origin_host and host and (
            origin_host == host
            or origin_host.endswith("." + host)
            or host.endswith("." + origin_host)
        ):
            on_domain += 1
        total_chars += len(str(page.get("raw_content", "")))

    n = len(page_urls)
    return {
        "pages": n,
        "on_domain_pages": on_domain,
        "off_domain_pages": n - on_domain,
        "on_domain_pct": round(on_domain / n * 100, 1) if n else 0.0,
        "total_raw_chars": total_chars,
        "credits": extract_usage_credits(response),
        "page_urls": page_urls,
    }


# ---------------------------------------------------------------------------
# Result containers
# ---------------------------------------------------------------------------


@dataclass
class Company:
    """One selectable dead company, resolved from a ``death_coverage.csv`` row."""

    index: int
    org_uuid: str
    name: str
    homepage_url: str
    founded_date: str
    website_alive: str
    n_captures: int
    lifespan_days: int
    closest_ts: str
    death_ts: str
    browse_url: str  # human Wayback viewer (toolbar) — open this to eyeball the site


@dataclass
class ScrapeResult:
    """Everything ``scrape(i)`` produced, ready for the notebook to render."""

    company: Company
    method: str
    modifier: str
    scope: bool
    start_url: str
    tavily_requests: list[dict[str, Any]]
    response: dict[str, Any]
    pages_used: str
    evidence: str
    diagnostics: dict[str, Any]
    classifier_input: str
    verdict: dict[str, Any] | None      # always None; classification left this lab
    baseline: dict[str, Any] | None     # stored metadata-only production verdict
    fallback_used: bool
    error: str | None = None


# ---------------------------------------------------------------------------
# The lab
# ---------------------------------------------------------------------------


class ArchiveLab:
    """Stateful, index-driven harness: ``candidates()`` -> ``inspect(i)`` -> ``scrape(i)``.

    Holds the ranked candidate table for the session and an in-memory cache keyed
    by ``(org_uuid, config signature)`` so re-selecting a company never re-spends.
    """

    def __init__(
        self,
        death_coverage_csv: str | Path = DEATH_COVERAGE_CSV,
        master_csv: str | Path = MASTER_CSV,
        production_csv: str | Path = PRODUCTION_CSV,
    ) -> None:
        self.death_coverage_csv = Path(death_coverage_csv)
        self.master_csv = Path(master_csv)
        self.production_csv = Path(production_csv)
        self._current: pd.DataFrame | None = None
        self._master: dict[str, dict[str, str]] | None = None
        self._production: dict[str, dict[str, str]] | None = None
        self._cache: dict[tuple, ScrapeResult] = {}

    # -- candidate selection ------------------------------------------------

    def candidates(
        self,
        n: int = 25,
        *,
        alive: bool | None = False,
        require_pre_death: bool = True,
        exclude_shared_hosts: bool = True,
    ) -> pd.DataFrame:
        """Rank resolvable companies by richness and store the table for ``i`` lookups.

        Defaults target the demo's story — companies *gone from the live web* that we
        have a genuine pre-death capture for:
          - ``alive=False`` keeps only dead sites (``alive=True`` only live; ``None`` both).
          - ``require_pre_death`` keeps only rows whose chosen capture predates death.
          - ``exclude_shared_hosts`` drops shared-platform homepages (see ``_SHARED_HOSTS``).

        Ranks by ``n_captures`` (how often the archive saw the site — a popularity/
        substance proxy), then ``lifespan_days`` as a tiebreak. Note ``lifespan_days``
        is the archive span of the *domain* and can predate the company if the domain
        was reused, so it is intentionally secondary. Reads the CSV defensively since
        the probe may still be appending.
        """
        df = pd.read_csv(
            self.death_coverage_csv,
            dtype=str,
            keep_default_na=False,
            on_bad_lines="skip",
        )
        sel = df[df["status"] == "ok"].copy()
        sel = sel[sel["closest_ts"].str.strip() != ""]
        if alive is True:
            sel = sel[sel["website_alive"] == "true"]
        elif alive is False:
            sel = sel[sel["website_alive"] == "false"]
        if require_pre_death:
            sel = sel[sel["has_pre_death_snapshot"] == "True"]
        if exclude_shared_hosts:
            sel = sel[~sel["host"].map(_is_shared_host)]
        for col in ("n_captures", "lifespan_days"):
            sel[col] = pd.to_numeric(sel[col], errors="coerce").fillna(0).astype(int)
        sel = sel.sort_values(
            ["n_captures", "lifespan_days"], ascending=False, kind="stable"
        )
        sel = sel.drop_duplicates(subset="org_uuid", keep="first")
        table = sel.head(n)[CANDIDATE_COLUMNS].reset_index(drop=True)
        self._current = table
        return table

    def _row(self, i: int) -> pd.Series:
        if self._current is None:
            self.candidates()
        assert self._current is not None
        if i < 0 or i >= len(self._current):
            raise IndexError(f"i={i} out of range 0..{len(self._current) - 1}; call candidates() to refresh")
        return self._current.iloc[i]

    def inspect(self, i: int) -> Company:
        """Resolve company ``i`` for free — no Tavily call, no spend."""
        row = self._row(i)
        homepage = str(row["homepage_url"])
        closest_ts = str(row["closest_ts"])
        browse = str(row.get("target_url") or "").strip() or archive_url(homepage, closest_ts, "")
        return Company(
            index=i,
            org_uuid=str(row["org_uuid"]),
            name=str(row["name"]),
            homepage_url=homepage,
            founded_date=str(row.get("founded_date", "")),
            website_alive=str(row.get("website_alive", "")),
            n_captures=int(row.get("n_captures", 0) or 0),
            lifespan_days=int(row.get("lifespan_days", 0) or 0),
            closest_ts=closest_ts,
            death_ts=str(row.get("death_ts", "")),
            browse_url=browse,
        )

    # -- the deliberate paid step ------------------------------------------

    def scrape(
        self,
        i: int,
        *,
        method: str = "crawl",
        modifier: str | None = None,
        scope: bool = True,
        limit: int | None = None,
        max_depth: int | None = None,
        instructions: str | None = None,
        classify: bool = False,
        force: bool = False,
    ) -> ScrapeResult:
        """Run Tavily on company ``i``'s pre-death snapshot and clean the evidence.

        ``classify=True`` is refused. The batch classifier that used to score
        this evidence has been removed.

        ``method="crawl"`` runs the exact modern multi-page crawl (default ``if_``,
        scoped to the company). ``method="extract"`` runs the single-page PIVOT
        alternative (default ``id_``). Results are cached per (company, config).
        """
        if classify:
            raise RuntimeError(
                "Historical classification ran on the retired batch classifier. "
                "Production classification is python -m two_pass_classifier."
            )
        company = self.inspect(i)
        modifier = modifier or ("id_" if method == "extract" else "if_")
        key = self._cache_key(company.org_uuid, method, modifier, scope, limit, max_depth, instructions, classify)
        if not force and key in self._cache:
            return self._cache[key]

        start_url = archive_url(company.homepage_url, company.closest_ts, modifier)
        error: str | None = None
        response: dict[str, Any] = {"results": []}
        tavily_requests = self._planned_requests(
            method, start_url, company.homepage_url, scope, limit, max_depth, instructions,
        )
        fallback_used = False

        try:
            if method == "extract":
                response, tavily_requests = self._extract(start_url)
            else:
                response, fallback_used, tavily_requests = self._crawl(
                    start_url, company.homepage_url, scope, limit, max_depth, instructions,
                )
        except Exception as exc:  # noqa: BLE001 — surface any Tavily/network failure to the panel
            error = f"{type(exc).__name__}: {exc}"

        pages_used, evidence = ("", "")
        diagnostics = diagnose(response, company.homepage_url)
        classifier_input = ""
        verdict: dict[str, Any] | None = None
        if error is None:
            pages_used, evidence = clean_evidence(response)
            classifier_input = self._format_input(company.org_uuid, pages_used, evidence)

        result = ScrapeResult(
            company=company,
            method=method,
            modifier=modifier,
            scope=scope,
            start_url=start_url,
            tavily_requests=tavily_requests,
            response=response,
            pages_used=pages_used,
            evidence=evidence,
            diagnostics=diagnostics,
            classifier_input=classifier_input,
            verdict=verdict,
            baseline=self.baseline_verdict(company.org_uuid),
            fallback_used=fallback_used,
            error=error,
        )
        if error is None:
            self._cache[key] = result
        return result

    # -- Tavily callers (reuse the exact live functions) -------------------

    def _planned_requests(
        self,
        method: str,
        start_url: str,
        homepage_url: str,
        scope: bool,
        limit: int | None,
        max_depth: int | None,
        instructions: str | None,
    ) -> list[dict[str, Any]]:
        """Request payloads known before the network call.

        Used so the notebook can still show the attempted Tavily request if the
        HTTP call itself fails. Crawl fallback payloads are appended by ``_crawl``
        only when the primary response is empty and fallback is actually attempted.
        """
        if method == "extract":
            from wayback_machine.config import TAVILY_EXTRACT_ENDPOINT, ExtractConfig

            cfg = ExtractConfig()
            return [{
                "label": "extract request",
                "endpoint": TAVILY_EXTRACT_ENDPOINT,
                "payload": cfg.request_payload(start_url),
            }]

        cfg = self._build_config(homepage_url, scope, limit, max_depth, instructions)
        return [{
            "label": "primary crawl request",
            "endpoint": TAVILY_CRAWL_ENDPOINT,
            "payload": cfg.request_payload(start_url),
        }]

    def _crawl(
        self,
        start_url: str,
        homepage_url: str,
        scope: bool,
        limit: int | None,
        max_depth: int | None,
        instructions: str | None,
    ) -> tuple[dict[str, Any], bool, list[dict[str, Any]]]:
        cfg = self._build_config(homepage_url, scope, limit, max_depth, instructions)
        requests = [{
            "label": "primary crawl request",
            "endpoint": TAVILY_CRAWL_ENDPOINT,
            "payload": cfg.request_payload(start_url),
        }]
        api_key = _api_key()
        response = _call_tavily_crawl_with_retries(start_url, cfg, api_key)
        if _has_usable_results(response):
            return response, False, requests
        # Same empty-result fallback the live runner uses: drop the LLM instructions.
        fallback_cfg = _fallback_config(cfg)
        requests.append({
            "label": "empty-result fallback crawl request",
            "endpoint": TAVILY_CRAWL_ENDPOINT,
            "payload": fallback_cfg.request_payload(start_url),
        })
        response = _call_tavily_crawl_with_retries(start_url, fallback_cfg, api_key)
        return response, True, requests

    def _build_config(
        self,
        homepage_url: str,
        scope: bool,
        limit: int | None,
        max_depth: int | None,
        instructions: str | None,
    ) -> TavilyCrawlConfig:
        overrides: dict[str, Any] = {}
        if limit is not None:
            overrides["limit"] = limit
        if max_depth is not None:
            overrides["max_depth"] = max_depth
        if instructions is not None:
            overrides["instructions"] = instructions
        if scope:
            select_paths, select_domains = _scope_for(homepage_url)
            return _ScopedCrawlConfig(select_paths=select_paths, select_domains=select_domains, **overrides)
        return TavilyCrawlConfig(**overrides)

    def _extract(self, start_url: str) -> tuple[dict[str, Any], list[dict[str, Any]]]:
        from wayback_machine.config import TAVILY_EXTRACT_ENDPOINT, ExtractConfig
        from wayback_machine.extract import call_tavily_extract

        cfg = ExtractConfig()
        requests = [{
            "label": "extract request",
            "endpoint": TAVILY_EXTRACT_ENDPOINT,
            "payload": cfg.request_payload(start_url),
        }]
        return call_tavily_extract(start_url, cfg, _api_key()), requests

    def _format_input(self, org_uuid: str, pages_used: str, evidence: str) -> str:
        from two_pass_classifier.formatter import format_user_message

        row = {
            **self._metadata(org_uuid),
            "website_pages_used": pages_used,
            "website_evidence": evidence,
        }
        return format_user_message(row)

    def baseline_verdict(self, org_uuid: str) -> dict[str, Any] | None:
        """The company's existing row in production_classifications.csv (metadata-only)."""
        return self._production_index().get(org_uuid)

    # -- lazy data loaders --------------------------------------------------

    def _metadata(self, org_uuid: str) -> dict[str, str]:
        meta = self._master_index().get(org_uuid)
        if meta is not None:
            return meta
        # Fall back to the thin candidate row so classification still runs.
        row = self._current[self._current["org_uuid"] == org_uuid] if self._current is not None else None
        base = {col: "" for col in MASTER_CSV_COLUMNS}
        if row is not None and not row.empty:
            r = row.iloc[0]
            for col in ("org_uuid", "name", "homepage_url", "founded_date", "website_alive"):
                if col in r:
                    base[col] = str(r[col])
        return base

    def _master_index(self) -> dict[str, dict[str, str]]:
        if self._master is None:
            df = pd.read_csv(self.master_csv, dtype=str, keep_default_na=False, on_bad_lines="skip")
            self._master = {str(r["org_uuid"]): r.to_dict() for _, r in df.iterrows()}
        return self._master

    def _production_index(self) -> dict[str, dict[str, str]]:
        if self._production is None:
            df = pd.read_csv(self.production_csv, dtype=str, keep_default_na=False, on_bad_lines="skip")
            self._production = {str(r["CompanyID"]): r.to_dict() for _, r in df.iterrows()}
        return self._production

    @staticmethod
    def _cache_key(
        org_uuid: str,
        method: str,
        modifier: str,
        scope: bool,
        limit: int | None,
        max_depth: int | None,
        instructions: str | None,
        classify: bool,
    ) -> tuple:
        instr = "default" if instructions is None else hashlib.sha1(instructions.encode()).hexdigest()[:10]
        return (org_uuid, method, modifier, bool(scope), limit, max_depth, instr, bool(classify))


def _main() -> None:
    """Small smoke test for direct execution.

    The real demo UI is the notebook. This entrypoint exists so running
    ``python wayback_machine/tavily_archive_lab.py`` confirms the lab can load
    and shows a few candidate indices without spending Tavily/OpenAI credits.
    """
    lab = ArchiveLab()
    table = lab.candidates(10)
    print("ArchiveLab loaded. Top inspect-first candidates:")
    print(
        table[
            ["name", "website_alive", "n_captures", "lifespan_days", "closest_ts", "homepage_url"]
        ].to_string()
    )
    if not table.empty:
        company = lab.inspect(0)
        print("\nTry this in the notebook:")
        print("  show(0)")
        print("  scrape(0)")
        print(f"\nWayback browser link for row 0:\n  {company.browse_url}")


if __name__ == "__main__":
    _main()
