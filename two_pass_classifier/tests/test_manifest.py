from __future__ import annotations

import csv
import hashlib
import json
from pathlib import Path

import pytest

from two_pass_classifier import manifest as manifest_module
from two_pass_classifier.input_contract import MODEL_INPUT_COLUMNS, SOURCE_COLUMNS
from two_pass_classifier.manifest import (
    ManifestCollisionError,
    ManifestValidationError,
    build_manifest,
    load_manifest,
    write_manifest,
)
from two_pass_classifier.paths import (
    DEFAULT_DEAD_INPUT,
    DEFAULT_DEAD_SCRAPE_PROCESSED,
    DEFAULT_LIVE_INPUT,
    DEFAULT_LIVE_RAW_RESULTS,
)


def test_source_columns_match_existing_classifier_input_contract():
    from tavily_crawler.master_csv import (
        CLASSIFIER_INPUT_COLUMNS as CRAWLER_INPUT_COLUMNS,
    )
    from wayback_machine.cohort import (
        CLASSIFIER_INPUT_COLUMNS as WAYBACK_INPUT_COLUMNS,
    )

    assert list(SOURCE_COLUMNS) == list(CRAWLER_INPUT_COLUMNS)
    assert list(SOURCE_COLUMNS) == list(WAYBACK_INPUT_COLUMNS)


def _row(company_id: str, evidence: str, **overrides: str) -> dict[str, str]:
    row = {column: "" for column in SOURCE_COLUMNS}
    row.update(
        {
            "org_uuid": company_id,
            "name": f"Company {company_id}",
            "homepage_url": f"https://{company_id}.example",
            "short_description": "  leading and trailing  ",
            "Long description": 'Quotes "and", commas\nand newlines',
            "category_list": "AI, Software",
            "category_groups_list": "Software",
            "founded_date": "2023-03-14",
            "employee_count": "11-50",
            "total_funding_usd": "1000000",
            "website_alive": "True",
            "website_pages_used": "https://example.test/a|https://example.test/b",
            "website_evidence": evidence,
        }
    )
    row.update(overrides)
    return row


def _write_csv(path: Path, rows: list[dict[str, str]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(SOURCE_COLUMNS))
        writer.writeheader()
        writer.writerows(rows)


def _write_live_raw(
    path: Path,
    rows: list[tuple[str, str]] | list[dict[str, object]],
) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="") as handle:
        for item in rows:
            if isinstance(item, tuple):
                org_uuid, requested_at = item
                payload = {
                    "org_uuid": org_uuid,
                    "ok": True,
                    "requested_at": requested_at,
                }
            else:
                payload = item
            handle.write(json.dumps(payload, ensure_ascii=False) + "\n")


def _write_dead_scrape(path: Path, rows: list[tuple[str, str]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(
            handle,
            fieldnames=[
                "org_uuid",
                "name",
                "homepage_url",
                "snapshot_ts",
                "website_pages_used",
                "website_evidence",
            ],
        )
        writer.writeheader()
        for org_uuid, snapshot_ts in rows:
            writer.writerow(
                {
                    "org_uuid": org_uuid,
                    "name": f"Company {org_uuid}",
                    "homepage_url": f"https://{org_uuid}.example",
                    "snapshot_ts": snapshot_ts,
                    "website_pages_used": "https://example.test",
                    "website_evidence": "archive",
                }
            )


def _build(
    tmp_path: Path,
    live_rows: list[dict[str, str]],
    dead_rows: list[dict[str, str]],
    *,
    live_dates: list[tuple[str, str]] | None = None,
    dead_dates: list[tuple[str, str]] | None = None,
):
    live = tmp_path / "live.csv"
    dead = tmp_path / "dead.csv"
    live_raw = tmp_path / "raw_results.jsonl"
    dead_scrape = tmp_path / "scrape_processed_dead.csv"
    _write_csv(live, live_rows)
    _write_csv(dead, dead_rows)

    if live_dates is None:
        live_dates = [
            (row["org_uuid"], "2026-05-04T17:12:06.086815+00:00")
            for row in live_rows
            if row["website_evidence"].strip()
        ]
    if dead_dates is None:
        dead_dates = [
            (row["org_uuid"], "20240902223056")
            for row in dead_rows
            if row["website_evidence"].strip()
        ]
    _write_live_raw(live_raw, live_dates)
    _write_dead_scrape(dead_scrape, dead_dates)
    return build_manifest(
        live,
        dead,
        live_raw_results=live_raw,
        dead_scrape_processed=dead_scrape,
    )


def test_manifest_preserves_values_hashes_and_measured_counts(tmp_path):
    evidence = "  Exact café evidence,\r\nwith punctuation | and quotes \"kept\".  "
    manifest = _build(
        tmp_path,
        [_row("live-1", evidence), _row("live-blank", " \t")],
        [_row("dead-1", "Recovered archive evidence")],
        live_dates=[("live-1", "2026-05-04T17:12:06.086815+00:00")],
        dead_dates=[("dead-1", "20240902223056")],
    )

    assert manifest.row_count == 2
    assert manifest.source_counts == {"live": 1, "dead": 1}
    assert [row.company_id for row in manifest.rows] == ["live-1", "dead-1"]
    assert manifest.rows[0].inputs["website_evidence"] == evidence
    assert manifest.rows[0].inputs["short_description"] == "  leading and trailing  "
    assert manifest.rows[0].cohort == "GENAI-ERA"
    assert manifest.rows[0].company_alive == "yes"
    assert manifest.rows[0].website_snapshot_date == "2026-05-04"
    assert manifest.rows[1].company_alive == "no"
    assert manifest.rows[1].website_snapshot_date == "2024-09-02"
    assert len(manifest.rows[0].input_hash) == 64
    assert len(manifest.rows_sha256) == 64
    assert len(manifest.manifest_sha256) == 64

    live = tmp_path / "live.csv"
    expected_live_hash = hashlib.sha256(live.read_bytes()).hexdigest()
    assert manifest.sources[0].file_sha256 == expected_live_hash
    assert manifest.sources[0].input_row_count == 2
    assert manifest.sources[0].included_row_count == 1

    artifact = write_manifest(manifest, tmp_path / "manifests")
    loaded = load_manifest(artifact)
    assert loaded == manifest
    assert loaded.rows[0].inputs["website_evidence"] == evidence
    assert loaded.rows[0].company_alive == "yes"
    assert loaded.rows[1].website_snapshot_date == "2024-09-02"


def test_input_hash_changes_only_for_model_visible_values(tmp_path):
    dead_rows = [_row("dead-1", "dead evidence")]
    dead_dates = [("dead-1", "20240902223056")]

    live_a = tmp_path / "live-a.csv"
    live_b = tmp_path / "live-b.csv"
    live_c = tmp_path / "live-c.csv"
    live_raw = tmp_path / "raw_results.jsonl"
    dead = tmp_path / "dead.csv"
    dead_scrape = tmp_path / "scrape_processed_dead.csv"
    _write_csv(dead, dead_rows)
    _write_dead_scrape(dead_scrape, dead_dates)
    _write_live_raw(live_raw, [("live-1", "2026-05-04T17:12:06.086815+00:00")])
    _write_csv(live_a, [_row("live-1", "same", homepage_url="https://one.example")])
    _write_csv(live_b, [_row("live-1", "same", homepage_url="https://two.example")])
    _write_csv(live_c, [_row("live-1", "changed")])

    hash_a = build_manifest(
        live_a,
        dead,
        live_raw_results=live_raw,
        dead_scrape_processed=dead_scrape,
    ).rows[0].input_hash
    assert (
        build_manifest(
            live_b,
            dead,
            live_raw_results=live_raw,
            dead_scrape_processed=dead_scrape,
        ).rows[0].input_hash
        == hash_a
    )
    assert (
        build_manifest(
            live_c,
            dead,
            live_raw_results=live_raw,
            dead_scrape_processed=dead_scrape,
        ).rows[0].input_hash
        != hash_a
    )


def test_provenance_fields_do_not_enter_model_input_hash(tmp_path):
    live_rows = [_row("live-1", "same evidence")]
    dead_rows = [_row("dead-1", "dead evidence")]
    first = _build(
        tmp_path / "a",
        live_rows,
        dead_rows,
        live_dates=[("live-1", "2026-05-04T17:12:06.086815+00:00")],
        dead_dates=[("dead-1", "20240902223056")],
    )
    second = _build(
        tmp_path / "b",
        live_rows,
        dead_rows,
        live_dates=[("live-1", "2026-06-01T00:00:00+00:00")],
        dead_dates=[("dead-1", "20200101120000")],
    )

    assert first.rows[0].input_hash == second.rows[0].input_hash
    assert first.rows[1].input_hash == second.rows[1].input_hash
    assert first.rows[0].website_snapshot_date != second.rows[0].website_snapshot_date
    assert first.rows[1].website_snapshot_date != second.rows[1].website_snapshot_date
    assert "company_alive" not in MODEL_INPUT_COLUMNS
    assert "website_snapshot_date" not in MODEL_INPUT_COLUMNS
    assert "evidence_source" not in MODEL_INPUT_COLUMNS
    assert "website_alive" in SOURCE_COLUMNS
    assert "website_alive" not in MODEL_INPUT_COLUMNS


def test_missing_snapshot_date_join_fails_loudly(tmp_path):
    with pytest.raises(ManifestValidationError, match="no website_snapshot_date join"):
        _build(
            tmp_path,
            [_row("live-1", "evidence")],
            [_row("dead-1", "archive")],
            live_dates=[],
            dead_dates=[("dead-1", "20240902223056")],
        )


def test_live_date_uses_requested_at_date_portion(tmp_path):
    manifest = _build(
        tmp_path,
        [_row("live-1", "evidence")],
        [_row("dead-1", "archive")],
        live_dates=[("live-1", "2026-05-04T17:12:06.086815+00:00")],
        dead_dates=[("dead-1", "20230314125959")],
    )
    assert manifest.rows[0].website_snapshot_date == "2026-05-04"
    assert manifest.rows[1].website_snapshot_date == "2023-03-14"


def test_live_join_ignores_non_ok_raw_rows(tmp_path):
    live = tmp_path / "live.csv"
    dead = tmp_path / "dead.csv"
    live_raw = tmp_path / "raw_results.jsonl"
    dead_scrape = tmp_path / "scrape_processed_dead.csv"
    _write_csv(live, [_row("live-1", "evidence")])
    _write_csv(dead, [_row("dead-1", "archive")])
    _write_dead_scrape(dead_scrape, [("dead-1", "20240902223056")])
    _write_live_raw(
        live_raw,
        [
            {
                "org_uuid": "live-1",
                "ok": False,
                "requested_at": "2020-01-01T00:00:00+00:00",
            },
            {
                "org_uuid": "live-1",
                "ok": True,
                "requested_at": "2026-05-04T17:12:06.086815+00:00",
            },
        ],
    )
    manifest = build_manifest(
        live,
        dead,
        live_raw_results=live_raw,
        dead_scrape_processed=dead_scrape,
    )
    assert manifest.rows[0].website_snapshot_date == "2026-05-04"


def test_live_join_keeps_last_successful_requested_at(tmp_path):
    live = tmp_path / "live.csv"
    dead = tmp_path / "dead.csv"
    live_raw = tmp_path / "raw_results.jsonl"
    dead_scrape = tmp_path / "scrape_processed_dead.csv"
    _write_csv(live, [_row("live-1", "evidence")])
    _write_csv(dead, [_row("dead-1", "archive")])
    _write_dead_scrape(dead_scrape, [("dead-1", "20240902223056")])
    _write_live_raw(
        live_raw,
        [
            {
                "org_uuid": "live-1",
                "ok": True,
                "requested_at": "2020-01-01T00:00:00+00:00",
            },
            {
                "org_uuid": "live-1",
                "ok": True,
                "requested_at": "2026-05-04T17:12:06.086815+00:00",
            },
        ],
    )
    manifest = build_manifest(
        live,
        dead,
        live_raw_results=live_raw,
        dead_scrape_processed=dead_scrape,
    )
    assert manifest.rows[0].website_snapshot_date == "2026-05-04"


@pytest.mark.parametrize("source", ["live", "dead"])
def test_duplicate_ids_fail(tmp_path, source):
    duplicate_rows = [_row("same", "one"), _row("same", "two")]
    live_rows = duplicate_rows if source == "live" else [_row("live", "one")]
    dead_rows = duplicate_rows if source == "dead" else [_row("dead", "one")]
    with pytest.raises(ManifestValidationError, match="duplicate company_id"):
        _build(tmp_path, live_rows, dead_rows)


def test_live_dead_overlap_fails(tmp_path):
    with pytest.raises(ManifestValidationError, match="overlap"):
        _build(
            tmp_path,
            [_row("overlap", "live")],
            [_row("overlap", "dead")],
        )


def test_dead_blank_evidence_fails_while_live_blank_is_filtered(tmp_path):
    with pytest.raises(ManifestValidationError, match="blank website_evidence"):
        _build(
            tmp_path,
            [_row("live-blank", "\n\t"), _row("live-good", "evidence")],
            [_row("dead-blank", "  ")],
        )

    manifest = _build(
        tmp_path / "ok",
        [_row("live-blank", "\n\t"), _row("live-good", "evidence")],
        [_row("dead-good", "archive")],
    )
    assert [row.company_id for row in manifest.rows] == ["live-good", "dead-good"]
    assert all(row.inputs["website_evidence"].strip() for row in manifest.rows)


def test_blank_company_id_fails(tmp_path):
    with pytest.raises(ManifestValidationError, match="blank org_uuid"):
        _build(
            tmp_path,
            [_row(" ", "evidence")],
            [_row("dead", "archive")],
        )


def test_immutable_write_is_idempotent_and_never_overwrites(tmp_path):
    manifest = _build(
        tmp_path,
        [_row("live", "evidence")],
        [_row("dead", "archive")],
    )
    output_dir = tmp_path / "manifests"

    first = write_manifest(manifest, output_dir)
    original = first.read_bytes()
    assert write_manifest(manifest, output_dir) == first
    assert first.read_bytes() == original

    alternate = _build(
        tmp_path / "alternate",
        [_row("alternate", "different evidence")],
        [_row("dead", "archive")],
    )
    alternate_path = write_manifest(alternate, tmp_path / "alternate-manifests")
    first.write_bytes(alternate_path.read_bytes())
    different = first.read_bytes()
    with pytest.raises(ManifestCollisionError, match="refusing to overwrite"):
        write_manifest(manifest, output_dir)
    assert first.read_bytes() == different


def test_partial_target_is_replaced_by_valid_atomic_publication(tmp_path):
    manifest = _build(
        tmp_path,
        [_row("live", "evidence")],
        [_row("dead", "archive")],
    )
    output_dir = tmp_path / "manifests"
    output_dir.mkdir()
    target = output_dir / f"manifest_{manifest.manifest_sha256}.jsonl"
    target.write_text('{"record_type":"manifest"', encoding="utf-8")

    artifact = write_manifest(manifest, output_dir)

    assert artifact == target
    assert load_manifest(artifact) == manifest
    assert list(output_dir.glob(".*.tmp")) == []


def test_failed_publication_cleans_temp_and_retry_succeeds(
    tmp_path,
    monkeypatch,
):
    manifest = _build(
        tmp_path,
        [_row("live", "evidence")],
        [_row("dead", "archive")],
    )
    output_dir = tmp_path / "manifests"
    target = output_dir / f"manifest_{manifest.manifest_sha256}.jsonl"
    real_replace = manifest_module.os.replace

    def interrupted_replace(source, destination):
        raise OSError("simulated publication interruption")

    monkeypatch.setattr(manifest_module.os, "replace", interrupted_replace)
    with pytest.raises(OSError, match="simulated publication interruption"):
        write_manifest(manifest, output_dir)

    assert not target.exists()
    assert list(output_dir.glob(".*.tmp")) == []

    monkeypatch.setattr(manifest_module.os, "replace", real_replace)
    assert load_manifest(write_manifest(manifest, output_dir)) == manifest


def test_local_production_artifacts_join_when_present():
    required = (
        DEFAULT_LIVE_INPUT,
        DEFAULT_DEAD_INPUT,
        DEFAULT_LIVE_RAW_RESULTS,
        DEFAULT_DEAD_SCRAPE_PROCESSED,
    )
    if not all(path.is_file() for path in required):
        pytest.skip("local production crawl/archive artifacts are not present")

    manifest = build_manifest()
    assert manifest.row_count == 37746
    assert manifest.source_counts == {"live": 22032, "dead": 15714}
    assert all(row.company_alive in {"yes", "no"} for row in manifest.rows)
    assert all(
        len(row.website_snapshot_date) == 10 and row.website_snapshot_date[4] == "-"
        for row in manifest.rows
    )
    assert {row.company_alive for row in manifest.rows if row.evidence_source == "live"} == {
        "yes"
    }
    assert {row.company_alive for row in manifest.rows if row.evidence_source == "dead"} == {
        "no"
    }
