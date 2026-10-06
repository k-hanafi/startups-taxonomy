from __future__ import annotations

from two_pass_classifier import cohort, confidence, config, request_builder
from two_pass_classifier.manifest import ManifestRow
from two_pass_classifier.request_builder import (
    RequestSettings,
    build_pass_a_request,
    build_pass_b_request,
    pass_a_request_fingerprint,
    pass_a_request_identity,
    request_fingerprint,
    request_identity,
)


def _row() -> ManifestRow:
    return ManifestRow(
        company_id="company-1",
        company_name="Company One",
        cohort="GENAI-ERA",
        company_alive="yes",
        website_snapshot_date="2026-05-04",
        evidence_source="live",
        source_row_number=2,
        input_hash="input-hash",
        inputs={
            "org_uuid": "company-1",
            "name": "Company One",
            "short_description": "AI workflow",
            "Long description": "A longer description",
            "category_list": "Artificial Intelligence",
            "category_groups_list": "Software",
            "founded_date": "2024-01",
            "employee_count": "1-10",
            "total_funding_usd": "100",
            "website_pages_used": "https://company.test",
            "website_evidence": "We build an AI workflow.",
        },
    )


def test_production_requests_use_locked_schemas_efforts_and_cache_routes():
    settings = RequestSettings(model="gpt-5.6-luna", pass_b_effort="medium")
    request_a = build_pass_a_request(_row(), settings)
    request_b0 = build_pass_b_request(_row(), 0, settings)
    request_b1 = build_pass_b_request(_row(), 1, settings)

    assert request_a["reasoning"] == {"effort": "none"}
    assert request_a["top_logprobs"] == config.PASS_A_TOP_LOGPROBS
    assert request_a["include"] == list(config.LOGPROB_INCLUDE)
    assert list(request_a["text"]["format"]["schema"]["properties"])[0] == "ai_native"
    assert "Website Pages Used" not in request_a["input"]

    assert request_b0["reasoning"] == {"effort": "medium"}
    assert request_b0["max_output_tokens"] == config.PASS_B_MAX_OUTPUT_TOKENS["medium"]
    assert "top_logprobs" not in request_b0
    assert request_b0["prompt_cache_key"] != request_b1["prompt_cache_key"]
    assert len(
        {
            request_a["prompt_cache_key"],
            request_b0["prompt_cache_key"],
            request_b1["prompt_cache_key"],
        }
    ) == 3
    assert request_b1["input"].endswith(
        "PriorBinaryVerdict: 1\nCohort: GENAI-ERA"
    )


def test_output_cap_changes_request_fingerprint():
    baseline = RequestSettings(
        model="gpt-5.4-nano",
        pass_b_effort="low",
    )
    changed_a = RequestSettings(
        model="gpt-5.4-nano",
        pass_b_effort="low",
        pass_a_max_output_tokens=baseline.pass_a_max_output_tokens + 1,
    )
    changed_b = RequestSettings(
        model="gpt-5.4-nano",
        pass_b_effort="low",
        pass_b_max_output_tokens=int(baseline.pass_b_max_output_tokens or 0) + 1,
    )

    assert request_fingerprint(baseline) != request_fingerprint(changed_a)
    assert request_fingerprint(baseline) != request_fingerprint(changed_b)


def test_raw_input_mapping_uses_production_cohort_and_formatter():
    row = _row()
    settings = RequestSettings(
        model="gpt-5.4-mini",
        pass_b_effort="low",
    )

    manifest_request = build_pass_b_request(row, 1, settings)
    raw_request = build_pass_b_request(row.inputs, 1, settings)

    assert raw_request == manifest_request
    assert raw_request["input"].endswith(
        "PriorBinaryVerdict: 1\nCohort: GENAI-ERA"
    )


def test_pass_a_bank_identity_is_production_owned_and_effort_independent():
    low = RequestSettings(
        model="gpt-5.6-luna",
        pass_b_effort="low",
    )
    high = RequestSettings(
        model="gpt-5.6-luna",
        pass_b_effort="high",
    )

    assert pass_a_request_identity(low) == pass_a_request_identity(high)
    assert pass_a_request_fingerprint(low) == pass_a_request_fingerprint(high)
    assert request_identity(low)["module_source_sha256"][
        "two_pass_formatter"
    ] == (
        pass_a_request_identity(low)["module_source_sha256"][
            "two_pass_formatter"
        ]
    )


def test_formatter_helper_source_drift_changes_full_and_pass_a_fingerprints(
    monkeypatch,
):
    settings = RequestSettings(model="gpt-5.4-nano")
    baseline_full = request_fingerprint(settings)
    baseline_pass_a = pass_a_request_fingerprint(settings)
    original = request_builder._module_source_bytes

    def drifted(module):
        source = original(module)
        if module.__name__ == "two_pass_classifier.formatter":
            return source + b"\n# helper-only semantic drift\n"
        return source

    monkeypatch.setattr(request_builder, "_module_source_bytes", drifted)

    assert request_fingerprint(settings) != baseline_full
    assert pass_a_request_fingerprint(settings) != baseline_pass_a


def test_cohort_policy_drift_changes_full_fingerprint(monkeypatch):
    settings = RequestSettings(model="gpt-5.4-nano")
    baseline_full = request_fingerprint(settings)
    baseline_pass_a = pass_a_request_fingerprint(settings)

    monkeypatch.setattr(
        cohort,
        "COHORT_BOUNDARY_DAY",
        cohort.COHORT_BOUNDARY_DAY + 1,
    )

    assert request_fingerprint(settings) != baseline_full
    assert pass_a_request_fingerprint(settings) == baseline_pass_a


def test_confidence_semantics_drift_changes_full_and_pass_a_fingerprints(
    monkeypatch,
):
    settings = RequestSettings(model="gpt-5.4-nano")
    baseline_full = request_fingerprint(settings)
    baseline_pass_a = pass_a_request_fingerprint(settings)

    monkeypatch.setattr(
        confidence,
        "MAX_CENSORED_INTERVAL_WIDTH",
        confidence.MAX_CENSORED_INTERVAL_WIDTH + 0.01,
    )

    assert request_fingerprint(settings) != baseline_full
    assert pass_a_request_fingerprint(settings) != baseline_pass_a
