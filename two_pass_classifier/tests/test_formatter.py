"""Tests for the CSV row formatter and custom_id builder.

Verifies that format_user_message produces the exact expected output from
a known input row, handles missing fields gracefully, and that
build_custom_id never produces collisions from NaN data.
"""

import math

import pytest

from two_pass_classifier.formatter import (
    MAX_USER_MESSAGE_CHARS,
    _clean,
    _extract_year,
    _merge_keywords,
    _normalize_founded_date,
    build_custom_id,
    format_user_message,
)


# -- _clean --------------------------------------------------------------------


class TestClean:
    def test_strips_whitespace(self):
        assert _clean("  hello  ") == "hello"

    def test_nan_becomes_blank(self):
        assert _clean("nan") == ""
        assert _clean("NaN") == ""
        assert _clean(float("nan")) == ""

    def test_none_becomes_blank(self):
        assert _clean(None) == ""
        assert _clean("None") == ""

    def test_normal_string_passes_through(self):
        assert _clean("Acme Corp") == "Acme Corp"


# -- _extract_year -------------------------------------------------------------


class TestExtractYear:
    def test_crunchbase_format(self):
        assert _extract_year("01nov2016") == "2016"

    def test_iso_date(self):
        assert _extract_year("2023-01-15") == "2023"

    def test_nan_returns_unknown(self):
        assert _extract_year("nan") == "Unknown"
        assert _extract_year("NaT") == "Unknown"

    def test_empty_returns_unknown(self):
        assert _extract_year("") == "Unknown"


# -- _normalize_founded_date ----------------------------------------------------


class TestNormalizeFoundedDate:
    def test_crunchbase_compact_format(self):
        assert _normalize_founded_date("01jan2020") == "2020-01"

    def test_master_hyphen_format(self):
        assert _normalize_founded_date("01-Jan-24") == "2024-01"

    def test_iso_format(self):
        assert _normalize_founded_date("2023-03-14") == "2023-03"

    def test_year_month_passthrough(self):
        assert _normalize_founded_date("2022-04") == "2022-04"

    def test_year_only_fallback(self):
        assert _normalize_founded_date("founded in 2022") == "2022"

    def test_missing_returns_unknown(self):
        assert _normalize_founded_date("") == "Unknown"


# -- _merge_keywords -----------------------------------------------------------


class TestMergeKeywords:
    def test_both_present(self):
        row = {"category_list": "AI,ML", "category_groups_list": "Software"}
        assert _merge_keywords(row) == "AI,ML, Software"

    def test_only_category_list(self):
        row = {"category_list": "AI,ML", "category_groups_list": float("nan")}
        assert _merge_keywords(row) == "AI,ML"

    def test_both_missing(self):
        row = {"category_list": "", "category_groups_list": ""}
        assert _merge_keywords(row) == ""


# -- format_user_message -------------------------------------------------------


_SAMPLE_ROW = {
    "org_uuid": "abc-123",
    "name": "TestCo",
    "short_description": "A test company",
    "Long description": "Longer description of the test company",
    "category_list": "AI,Software",
    "category_groups_list": "Technology",
    "founded_date": "01jan2020",
}


class TestFormatUserMessage:
    def test_all_fields_present(self):
        msg = format_user_message(_SAMPLE_ROW)
        assert "CompanyID: abc-123" in msg
        assert "CompanyName: TestCo" in msg
        assert "Short Description: A test company" in msg
        assert "Long Description: Longer description" in msg
        assert "Keywords: AI,Software, Technology" in msg
        assert "FoundedDate: 2020-01" in msg
        assert "YearFounded:" not in msg

    def test_missing_long_description(self):
        row = {**_SAMPLE_ROW, "Long description": float("nan")}
        msg = format_user_message(row)
        assert "Long Description: [not available]" in msg

    def test_truncation(self):
        row = {**_SAMPLE_ROW, "Long description": "x" * (MAX_USER_MESSAGE_CHARS + 1_000)}
        msg = format_user_message(row)
        assert len(msg) <= MAX_USER_MESSAGE_CHARS + 20  # +20 for "[truncated]\n"
        assert "[truncated]" in msg

    def test_field_order(self):
        msg = format_user_message(_SAMPLE_ROW)
        lines = msg.strip().split("\n")
        assert lines[0].startswith("CompanyID:")
        assert lines[1].startswith("CompanyName:")
        assert lines[2].startswith("Short Description:")
        assert lines[3].startswith("Long Description:")
        assert lines[4].startswith("Keywords:")
        assert lines[5].startswith("FoundedDate:")

    def test_optional_resource_context(self):
        row = {
            **_SAMPLE_ROW,
            "employee_count": "1-10",
            "total_funding_usd": "5000000",
        }
        msg = format_user_message(row)
        assert "Resource Context:" in msg
        assert "EmployeeCount: 1-10" in msg
        assert "TotalFundingUSD: 5000000" in msg

    def test_optional_website_evidence(self):
        row = {
            **_SAMPLE_ROW,
            "website_pages_used": "https://example.com/product",
            "website_evidence": "[Page 1: product]\nURL: https://example.com/product\nAI product evidence",
        }
        msg = format_user_message(row)
        assert "Website Pages Used: https://example.com/product" in msg
        assert "Website Evidence:" in msg
        assert "AI product evidence" in msg


# -- build_custom_id ----------------------------------------------------------


class TestBuildCustomId:
    def test_basic(self):
        assert build_custom_id("abc-123") == "startup-abc-123"

    def test_strips_whitespace(self):
        assert build_custom_id("  abc-123  ") == "startup-abc-123"

    def test_replaces_spaces(self):
        assert build_custom_id("abc 123") == "startup-abc-123"

    def test_rejects_blank(self):
        with pytest.raises(ValueError, match="blank org_uuid"):
            build_custom_id("")

    def test_rejects_nan_string(self):
        with pytest.raises(ValueError, match="blank org_uuid"):
            build_custom_id("nan")

    def test_rejects_none_string(self):
        with pytest.raises(ValueError, match="blank org_uuid"):
            build_custom_id("None")
