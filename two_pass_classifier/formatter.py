"""Format company rows into the user message both production passes share."""

from __future__ import annotations

import re
from datetime import datetime
from typing import Any, Mapping

from .cohort import compute_cohort
from .manifest import ManifestRow

FORMATTER_VERSION = 2

# Emergency guard for pathological website-enriched inputs. Normal Tavily inputs
# should stay well below this; it mainly protects request size if a page
# extraction unexpectedly returns very large content.
MAX_USER_MESSAGE_CHARS: int = 100_000


def _extract_year(date_str: str) -> str:
    """Pull a 4-digit year from Crunchbase date strings like '01nov2016'."""
    cleaned = str(date_str).strip()
    if not cleaned or cleaned.lower() in ("nan", "none", "nat"):
        return "Unknown"
    for i in range(len(cleaned) - 3):
        chunk = cleaned[i : i + 4]
        if chunk.isdigit() and 1900 <= int(chunk) <= 2100:
            return chunk
    return cleaned


def _normalize_founded_date(date_str: Any) -> str:
    """Normalize founded date strings to one prompt field.

    Accepts YYYY-MM (from master_csv.csv), full Crunchbase strings like
    ``01jan2020``, ISO dates, and bare years. Returns YYYY-MM when a month is
    available, YYYY when only a year can be recovered, and Unknown otherwise.
    """
    cleaned = _clean(date_str)
    if not cleaned:
        return "Unknown"

    # Already normalized to YYYY-MM by build_master_csv.py
    if re.fullmatch(r"\d{4}-\d{2}", cleaned):
        return cleaned

    for fmt in ("%d%b%Y", "%d-%b-%y", "%d-%b-%Y", "%Y-%m-%d", "%m/%d/%Y"):
        try:
            return datetime.strptime(cleaned, fmt).strftime("%Y-%m")
        except ValueError:
            pass

    return _extract_year(cleaned)


def _clean(value: Any) -> str:
    """Convert a value to a stripped string. Treat NaN/None as blank."""
    s = str(value).strip() if value is not None else ""
    if s.lower() in ("nan", "none", "nat"):
        return ""
    return s


def _merge_keywords(row: dict[str, Any]) -> str:
    """Combine category_list and category_groups_list into one Keywords field.

    The model benefits from both the specific tags (category_list) and the
    broader groupings (category_groups_list) to make classification decisions.
    """
    cats = _clean(row.get("category_list", ""))
    groups = _clean(row.get("category_groups_list", ""))
    if cats and groups:
        return f"{cats}, {groups}"
    return cats or groups


def _resource_context(row: dict[str, Any]) -> str:
    """Format scale/resource fields that help RAD confidence without dominating."""
    fields = [
        ("EmployeeCount", row.get("employee_count", "")),
        ("TotalFundingUSD", row.get("total_funding_usd", "")),
    ]
    parts = [f"{label}: {_clean(value)}" for label, value in fields if _clean(value)]
    return "; ".join(parts)


def format_user_message(row: dict[str, Any]) -> str:
    """Convert one CSV row into the user message string.

    Args:
        row: Dictionary whose keys are raw CSV column names
             (org_uuid, name, short_description, Long description,
              category_list, category_groups_list, founded_date), plus optional
              employee_count, total_funding_usd, website_evidence, etc.

    Returns:
        Formatted text block matching the prompt's INPUT FORMAT section.
    """
    cid = _clean(row.get("org_uuid", ""))
    cname = _clean(row.get("name", ""))
    short = _clean(row.get("short_description", ""))
    long = _clean(row.get("Long description", ""))
    keywords = _merge_keywords(row)
    founded_date = _normalize_founded_date(row.get("founded_date", ""))
    resource_context = _resource_context(row)
    website_pages = _clean(row.get("website_pages_used", ""))
    website_evidence = _clean(row.get("website_evidence", ""))

    parts = [
        f"CompanyID: {cid}",
        f"CompanyName: {cname}",
        f"Short Description: {short}",
    ]

    if not long:
        parts.append("Long Description: [not available]")
    else:
        parts.append(f"Long Description: {long}")

    parts.append(f"Keywords: {keywords}")
    parts.append(f"FoundedDate: {founded_date}")

    if resource_context:
        parts.append(f"Resource Context: {resource_context}")

    if website_evidence:
        if website_pages:
            parts.append(f"Website Pages Used: {website_pages}")
        parts.append(f"Website Evidence:\n{website_evidence}")

    message = "\n".join(parts)

    if len(message) > MAX_USER_MESSAGE_CHARS:
        message = message[:MAX_USER_MESSAGE_CHARS] + "\n[truncated]"

    return message


def build_custom_id(org_uuid: str) -> str:
    """Create a deterministic custom_id for result matching.

    The custom_id is the sole key for matching async results to their original
    inputs. Response order is not guaranteed.
    """
    sanitized = _clean(org_uuid).replace(" ", "-")
    if not sanitized:
        raise ValueError("Cannot build custom_id from blank org_uuid")
    return f"startup-{sanitized}"


def format_input_message(
    row: ManifestRow | Mapping[str, Any],
) -> str:
    """Format the shared classifier inputs through the production package."""
    return format_user_message(_row_inputs(row))


def format_pass_a_message(
    row: ManifestRow | Mapping[str, Any],
) -> str:
    """Format the binary gate input without website page metadata."""
    inputs = _row_inputs(row)
    inputs["website_pages_used"] = ""
    return format_input_message(inputs)


def format_pass_b_message(
    row: ManifestRow | Mapping[str, Any],
    verdict: int,
) -> str:
    """Format the full input plus the fixed family and code-owned cohort."""
    if verdict not in (0, 1):
        raise ValueError(f"verdict must be 0 or 1, got {verdict!r}")
    inputs = _row_inputs(row)
    if isinstance(row, ManifestRow):
        cohort = row.cohort
    else:
        cohort = str(row.get("cohort") or compute_cohort(inputs.get("founded_date")))
    return (
        format_input_message(inputs)
        + f"\nPriorBinaryVerdict: {verdict}"
        + f"\nCohort: {cohort}"
    )


def _row_inputs(row: ManifestRow | Mapping[str, Any]) -> dict[str, Any]:
    if isinstance(row, ManifestRow):
        return dict(row.inputs)
    inputs = row.get("inputs")
    if isinstance(inputs, Mapping):
        return dict(inputs)
    return dict(row)
