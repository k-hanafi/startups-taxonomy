from __future__ import annotations

import asyncio
import csv
import io
import json
from datetime import UTC, datetime, timedelta
from pathlib import Path
from typing import Any

import pytest
from rich.console import Console

from two_pass_classifier import cli, request_builder, workflow
from two_pass_classifier.input_contract import SOURCE_COLUMNS
from two_pass_classifier.journal import JOURNAL_VERSION, RunLock
from two_pass_classifier.manifest import build_manifest, load_manifest, write_manifest
from two_pass_classifier.request_builder import (
    RequestSettings,
    request_fingerprint,
    request_identity,
)
from two_pass_classifier.runner import ProductionRunner
from two_pass_classifier.workflow import (
    build_run_metadata,
    load_run_context,
    select_smoke_manifest,
)

from .test_manifest import _write_dead_scrape, _write_live_raw
from .test_runner import (
    _FakeClient,
    _FakeHTTPError,
    _pass_a_response,
    _pass_b_response,
    _runner_settings,
)


@pytest.fixture
def registry(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> dict[str, Path]:
    runs = tmp_path / "runs"
    manifests = tmp_path / "manifests"
    monkeypatch.setattr(workflow, "RUNS_DIR", runs)
    monkeypatch.setattr(workflow, "MANIFESTS_DIR", manifests)
    return {"runs": runs, "manifests": manifests}


def _row(company_id: str, evidence: str) -> dict[str, str]:
    row = {column: "" for column in SOURCE_COLUMNS}
    row.update(
        {
            "org_uuid": company_id,
            "name": f"Company {company_id}",
            "homepage_url": f"https://{company_id}.example",
            "short_description": "AI workflow",
            "Long description": "A detailed product description.",
            "category_list": "Artificial Intelligence",
            "category_groups_list": "Software",
            "founded_date": "2024-01",
            "employee_count": "1-10",
            "total_funding_usd": "100",
            "website_alive": "True",
            "website_pages_used": f"https://{company_id}.example/about",
            "website_evidence": evidence,
        }
    )
    return row


def _write_csv(path: Path, rows: list[dict[str, str]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(SOURCE_COLUMNS))
        writer.writeheader()
        writer.writerows(rows)


def _manifest_artifact(
    root: Path,
    *,
    live_count: int,
    dead_count: int,
) -> tuple[Path, Any]:
    live = root / "live.csv"
    dead = root / "dead.csv"
    live_rows = [
        _row(f"live-{index}", "Live evidence " * (index + 1))
        for index in range(live_count)
    ]
    dead_rows = [
        _row(f"dead-{index}", "Archive evidence " * (index + 1))
        for index in range(dead_count)
    ]
    _write_csv(live, live_rows)
    _write_csv(dead, dead_rows)
    live_raw = root / "raw_results.jsonl"
    dead_scrape = root / "scrape_processed_dead.csv"
    _write_live_raw(
        live_raw,
        [
            (row["org_uuid"], "2026-05-04T17:12:06.086815+00:00")
            for row in live_rows
        ],
    )
    _write_dead_scrape(
        dead_scrape,
        [(row["org_uuid"], "20240902223056") for row in dead_rows],
    )
    manifest = build_manifest(
        live,
        dead,
        live_raw_results=live_raw,
        dead_scrape_processed=dead_scrape,
    )
    artifact = write_manifest(manifest, root / "manifests")
    return artifact, manifest


def _console() -> tuple[Console, io.StringIO]:
    stream = io.StringIO()
    return (
        Console(
            file=stream,
            color_system=None,
            force_terminal=False,
            width=240,
        ),
        stream,
    )


def _invoke(
    argv: list[str],
    *,
    client_factory: Any = None,
) -> tuple[int, str]:
    console, stream = _console()
    code = cli.main(
        argv,
        client_factory=client_factory,
        console=console,
    )
    return code, stream.getvalue()


def _successful_fake_client() -> _FakeClient:
    counter = 0

    async def handler(kwargs: dict[str, Any]) -> Any:
        nonlocal counter
        counter += 1
        if "top_logprobs" in kwargs:
            return _pass_a_response(
                verdict=1,
                response_id=f"a-{counter}",
            )
        return _pass_b_response(
            verdict=1,
            response_id=f"b-{counter}",
        )

    return _FakeClient(handler)


def test_parser_defaults_allowlist_and_help_for_every_command(capsys):
    parser = cli.build_parser()
    parsed = parser.parse_args(["cost-preview"])
    assert parsed.model == "gpt-5.6-luna"
    assert parsed.effort == "low"

    with pytest.raises(SystemExit) as invalid:
        parser.parse_args(["cost-preview", "--model", "gpt-unknown"])
    assert invalid.value.code == 2

    commands = (
        "build-manifest",
        "cost-preview",
        "smoke",
        "run",
        "status",
        "resume",
        "retry",
    )
    for command in commands:
        with pytest.raises(SystemExit) as help_exit:
            parser.parse_args([command, "--help"])
        assert help_exit.value.code == 0
    output = capsys.readouterr().out
    for command in commands:
        assert command in output


def test_offline_cost_preview_needs_no_key_and_uses_manifest_rows(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
):
    artifact, manifest = _manifest_artifact(
        tmp_path,
        live_count=2,
        dead_count=1,
    )
    monkeypatch.delenv("OPENAI_API_KEY", raising=False)

    code, output = _invoke(
        ["cost-preview", "--manifest", str(artifact), "--model", "gpt-5.4-nano"]
    )

    assert code == 0
    assert f"{manifest.row_count}" in output
    assert "No API calls were made" in output
    assert "One-attempt cap projection" in output
    assert "retries or later resumes can exceed it" in output
    assert "ceiling" not in output.lower()
    assert "Batch" not in output


def test_paid_confirmation_blocks_before_client_or_run_creation(
    tmp_path: Path,
    registry: dict[str, Path],
    monkeypatch: pytest.MonkeyPatch,
):
    artifact, _ = _manifest_artifact(
        tmp_path / "source",
        live_count=5,
        dead_count=5,
    )
    monkeypatch.setattr(cli.Confirm, "ask", lambda *args, **kwargs: False)

    def forbidden_factory(api_key: str | None) -> Any:
        raise AssertionError("declined paid action must not construct a client")

    code, output = _invoke(
        [
            "smoke",
            "--manifest",
            str(artifact),
            "--run-id",
            "declined-smoke",
        ],
        client_factory=forbidden_factory,
    )

    assert code == 0
    assert "Cancelled before creating a run" in output
    assert not (registry["runs"] / "declined-smoke").exists()


def test_paid_command_loads_key_lazily_after_confirmation(
    tmp_path: Path,
    registry: dict[str, Path],
    monkeypatch: pytest.MonkeyPatch,
):
    artifact, _ = _manifest_artifact(
        tmp_path / "source",
        live_count=5,
        dead_count=5,
    )
    monkeypatch.delenv("OPENAI_API_KEY", raising=False)
    monkeypatch.setattr(
        "two_pass_classifier.api_key.dotenv_values",
        lambda path: {},
    )

    code, output = _invoke(
        [
            "smoke",
            "--manifest",
            str(artifact),
            "--run-id",
            "missing-key",
            "--yes",
        ]
    )

    assert code != 0
    assert "OPENAI_API_KEY is missing" in output
    assert not (registry["runs"] / "missing-key").exists()


def test_smoke_selection_is_deterministic_balanced_and_label_free(tmp_path: Path):
    _, manifest = _manifest_artifact(
        tmp_path,
        live_count=7,
        dead_count=7,
    )

    first = select_smoke_manifest(manifest)
    second = select_smoke_manifest(manifest)

    assert first.row_count == 10
    assert first.source_counts == {"live": 5, "dead": 5}
    assert [row.company_id for row in first.rows] == [
        row.company_id for row in second.rows
    ]
    assert all("ai_native" not in row.inputs for row in first.rows)


def test_fake_client_smoke_then_full_run_uses_separate_outputs(
    tmp_path: Path,
    registry: dict[str, Path],
):
    artifact, manifest = _manifest_artifact(
        tmp_path / "source",
        live_count=6,
        dead_count=6,
    )
    smoke_client = _successful_fake_client()
    smoke_code, smoke_output = _invoke(
        [
            "smoke",
            "--manifest",
            str(artifact),
            "--model",
            "gpt-5.4-nano",
            "--effort",
            "low",
            "--run-id",
            "smoke-e2e",
            "--yes",
        ],
        client_factory=lambda api_key: smoke_client,
    )
    assert smoke_code == 0, smoke_output
    smoke_context = load_run_context("smoke-e2e")
    assert smoke_context.manifest.row_count == 10
    assert (
        smoke_context.run_config["parent_manifest_sha256"]
        == manifest.manifest_sha256
    )
    assert smoke_context.run_config["ai_family_stratified"] is False
    assert len(smoke_client.responses.calls) == 20

    full_client = _successful_fake_client()
    run_code, run_output = _invoke(
        [
            "run",
            "--manifest",
            str(artifact),
            "--model",
            "gpt-5.4-nano",
            "--effort",
            "low",
            "--run-id",
            "full-e2e",
            "--yes",
        ],
        client_factory=lambda api_key: full_client,
    )
    assert run_code == 0, run_output
    full_context = load_run_context("full-e2e")
    assert full_context.manifest.row_count == 12
    assert len(full_client.responses.calls) == 24
    assert smoke_context.paths.final_csv != full_context.paths.final_csv
    with full_context.paths.final_csv.open(
        encoding="utf-8",
        newline="",
    ) as handle:
        assert len(list(csv.DictReader(handle))) == 12
    assert full_context.paths.final_csv.is_file()
    assert smoke_context.paths.final_csv.is_file()
    assert full_context.paths.run_dir == registry["runs"] / "full-e2e"


def test_run_refuses_missing_smoke_and_existing_run_id(
    tmp_path: Path,
    registry: dict[str, Path],
):
    artifact, _ = _manifest_artifact(
        tmp_path / "source",
        live_count=5,
        dead_count=5,
    )

    def forbidden_factory(api_key: str | None) -> Any:
        raise AssertionError("smoke gate failure must happen before client creation")

    missing_code, missing_output = _invoke(
        [
            "run",
            "--manifest",
            str(artifact),
            "--model",
            "gpt-5.4-nano",
            "--run-id",
            "missing-smoke",
            "--yes",
        ],
        client_factory=forbidden_factory,
    )
    assert missing_code != 0
    assert "no successful 10-company smoke matches" in missing_output

    smoke_client = _successful_fake_client()
    smoke_code, _ = _invoke(
        [
            "smoke",
            "--manifest",
            str(artifact),
            "--model",
            "gpt-5.4-nano",
            "--run-id",
            "smoke-for-existing",
            "--yes",
        ],
        client_factory=lambda api_key: smoke_client,
    )
    assert smoke_code == 0
    existing = registry["runs"] / "already-there"
    existing.mkdir(parents=True)
    orphan_code, orphan_output = _invoke(
        [
            "run",
            "--manifest",
            str(artifact),
            "--model",
            "gpt-5.4-nano",
            "--run-id",
            "already-there",
            "--yes",
        ],
        client_factory=forbidden_factory,
    )
    assert orphan_code != 0
    assert "no journal yet" in orphan_output.lower()
    assert "resume already-there" not in orphan_output

    (existing / "events.jsonl").write_text(
        "{}\n",
        encoding="utf-8",
    )
    existing_code, existing_output = _invoke(
        [
            "run",
            "--manifest",
            str(artifact),
            "--model",
            "gpt-5.4-nano",
            "--run-id",
            "already-there",
            "--yes",
        ],
        client_factory=forbidden_factory,
    )
    assert existing_code != 0
    assert "continue it with" in existing_output.lower()
    assert "resume already-there" in existing_output


def test_run_reports_small_manifest_instead_of_missing_smoke(
    tmp_path: Path,
    registry: dict[str, Path],
):
    del registry
    artifact, _ = _manifest_artifact(
        tmp_path / "source",
        live_count=2,
        dead_count=1,
    )
    code, output = _invoke(
        [
            "run",
            "--manifest",
            str(artifact),
            "--model",
            "gpt-5.4-nano",
            "--run-id",
            "too-small",
            "--yes",
        ],
        client_factory=lambda api_key: pytest.fail(
            "small manifests must fail before client creation"
        ),
    )
    assert code != 0
    assert "smoke requires 10 companies" in output
    assert "no successful 10-company smoke matches" not in output


def test_run_refuses_stale_smoke_fingerprint(
    tmp_path: Path,
    registry: dict[str, Path],
):
    del registry
    artifact, _ = _manifest_artifact(
        tmp_path / "source",
        live_count=5,
        dead_count=5,
    )
    smoke_client = _successful_fake_client()
    smoke_code, _ = _invoke(
        [
            "smoke",
            "--manifest",
            str(artifact),
            "--model",
            "gpt-5.4-nano",
            "--effort",
            "low",
            "--run-id",
            "low-smoke",
            "--yes",
        ],
        client_factory=lambda api_key: smoke_client,
    )
    assert smoke_code == 0

    code, output = _invoke(
        [
            "run",
            "--manifest",
            str(artifact),
            "--model",
            "gpt-5.4-nano",
            "--effort",
            "high",
            "--run-id",
            "high-run",
            "--yes",
        ],
        client_factory=lambda api_key: pytest.fail(
            "stale smoke must fail before client creation"
        ),
    )
    assert code != 0
    assert "no successful 10-company smoke matches" in output


def test_run_refuses_smoke_with_wrong_selection_ids(
    tmp_path: Path,
    registry: dict[str, Path],
):
    artifact, _ = _manifest_artifact(
        tmp_path / "source",
        live_count=5,
        dead_count=5,
    )
    smoke_code, _ = _invoke(
        [
            "smoke",
            "--manifest",
            str(artifact),
            "--model",
            "gpt-5.4-nano",
            "--run-id",
            "selection-smoke",
            "--yes",
        ],
        client_factory=lambda api_key: _successful_fake_client(),
    )
    assert smoke_code == 0

    journal = registry["runs"] / "selection-smoke" / "events.jsonl"
    lines = journal.read_text(encoding="utf-8").splitlines()
    header = json.loads(lines[0])
    recorded = list(header["run_config"]["selection_company_ids"])
    assert recorded
    header["run_config"]["selection_company_ids"] = list(reversed(recorded))
    journal.write_text(
        "\n".join([json.dumps(header, sort_keys=True), *lines[1:]]) + "\n",
        encoding="utf-8",
    )

    code, output = _invoke(
        [
            "run",
            "--manifest",
            str(artifact),
            "--model",
            "gpt-5.4-nano",
            "--run-id",
            "blocked-by-selection",
            "--yes",
        ],
        client_factory=lambda api_key: pytest.fail(
            "wrong smoke selection must fail before client creation"
        ),
    )
    assert code != 0
    assert "no successful 10-company smoke matches" in output


def test_formatter_helper_drift_invalidates_smoke_approval(
    tmp_path: Path,
    registry: dict[str, Path],
    monkeypatch: pytest.MonkeyPatch,
):
    del registry
    artifact, _ = _manifest_artifact(
        tmp_path / "source",
        live_count=5,
        dead_count=5,
    )
    smoke_code, smoke_output = _invoke(
        [
            "smoke",
            "--manifest",
            str(artifact),
            "--model",
            "gpt-5.4-nano",
            "--run-id",
            "helper-smoke",
            "--yes",
        ],
        client_factory=lambda api_key: _successful_fake_client(),
    )
    assert smoke_code == 0, smoke_output
    original = request_builder._module_source_bytes

    def drifted(module):
        source = original(module)
        if module.__name__ == "two_pass_classifier.formatter":
            return source + b"\n# changed helper\n"
        return source

    monkeypatch.setattr(request_builder, "_module_source_bytes", drifted)
    code, output = _invoke(
        [
            "run",
            "--manifest",
            str(artifact),
            "--model",
            "gpt-5.4-nano",
            "--run-id",
            "helper-run",
            "--yes",
        ],
        client_factory=lambda api_key: pytest.fail(
            "fingerprint mismatch must fail before client creation"
        ),
    )

    assert code != 0
    assert "no successful 10-company smoke matches" in output


def test_formatter_helper_drift_rejects_resume_before_api_client(
    tmp_path: Path,
    registry: dict[str, Path],
    monkeypatch: pytest.MonkeyPatch,
):
    artifact, manifest = _manifest_artifact(
        tmp_path / "source",
        live_count=1,
        dead_count=0,
    )
    settings = RequestSettings(model="gpt-5.4-nano", pass_b_effort="low")
    run_id = "helper-resume"
    run_path = registry["runs"] / run_id
    metadata = build_run_metadata(
        kind="full",
        run_id=run_id,
        manifest_path=artifact,
        manifest=manifest,
        settings=settings,
    )
    runner = ProductionRunner(
        manifest=manifest,
        run_dir=run_path,
        client=_successful_fake_client(),
        settings=_runner_settings(requests=settings),
        run_metadata=metadata,
        install_signal_handlers=False,
    )
    asyncio.run(runner.run())
    original = request_builder._module_source_bytes

    def drifted(module):
        source = original(module)
        if module.__name__ == "two_pass_classifier.formatter":
            return source + b"\n# changed helper\n"
        return source

    monkeypatch.setattr(request_builder, "_module_source_bytes", drifted)
    code, output = _invoke(
        ["resume", run_id, "--yes"],
        client_factory=lambda api_key: pytest.fail(
            "resume mismatch must fail before client creation"
        ),
    )

    assert code != 0
    assert "do not match this run" in output


def test_resume_uses_locked_settings_and_skips_durable_pass_a(
    tmp_path: Path,
    registry: dict[str, Path],
):
    artifact, manifest = _manifest_artifact(
        tmp_path / "source",
        live_count=1,
        dead_count=0,
    )
    settings = RequestSettings(model="gpt-5.4-nano", pass_b_effort="low")
    run_id = "partial-resume"
    run_path = registry["runs"] / run_id
    metadata = build_run_metadata(
        kind="full",
        run_id=run_id,
        manifest_path=artifact,
        manifest=manifest,
        settings=settings,
    )
    holder: dict[str, ProductionRunner] = {}

    async def first_handler(kwargs: dict[str, Any]) -> Any:
        holder["runner"].request_shutdown()
        return _pass_a_response(response_id="partial-a")

    first = ProductionRunner(
        manifest=manifest,
        run_dir=run_path,
        client=_FakeClient(first_handler),
        settings=_runner_settings(requests=settings),
        run_metadata=metadata,
        install_signal_handlers=False,
    )
    holder["runner"] = first
    first_result = asyncio.run(first.run())
    assert first_result.pass_a_checkpoint_count == 1
    assert first_result.completed_count == 0

    async def second_handler(kwargs: dict[str, Any]) -> Any:
        assert "top_logprobs" not in kwargs
        return _pass_b_response(response_id="resumed-b")

    second_client = _FakeClient(second_handler)
    code, output = _invoke(
        ["resume", run_id, "--yes"],
        client_factory=lambda api_key: second_client,
    )

    assert code == 0, output
    assert len(second_client.responses.calls) == 1
    assert load_run_context(run_id).paths.final_csv.is_file()
    with pytest.raises(SystemExit):
        cli.build_parser().parse_args(
            ["resume", run_id, "--model", "gpt-5.4-mini"]
        )


def test_retry_appends_events_without_mutating_history(
    tmp_path: Path,
    registry: dict[str, Path],
):
    artifact, manifest = _manifest_artifact(
        tmp_path / "source",
        live_count=1,
        dead_count=0,
    )
    settings = RequestSettings(model="gpt-5.4-nano", pass_b_effort="low")
    run_id = "retry-run"
    run_path = registry["runs"] / run_id
    metadata = build_run_metadata(
        kind="full",
        run_id=run_id,
        manifest_path=artifact,
        manifest=manifest,
        settings=settings,
    )

    async def handler(kwargs: dict[str, Any]) -> Any:
        raise _FakeHTTPError(429, "too many requests", {"retry-after": "0"})

    runner = ProductionRunner(
        manifest=manifest,
        run_dir=run_path,
        client=_FakeClient(handler),
        settings=_runner_settings(requests=settings, attempts=1),
        run_metadata=metadata,
        install_signal_handlers=False,
    )
    result = asyncio.run(runner.run())
    assert result.all_complete is False
    before = runner.paths.journal.read_bytes()
    status_code, status_json = _invoke(["status", run_id, "--json"])
    status_payload = json.loads(status_json)
    assert status_code == 0
    assert status_payload["retryable_failures"]["total"] == 1
    assert status_payload["retryable_failures"]["by_stage_and_reason"] == {
        "pass_a": {"rate_limit": 1}
    }
    assert status_payload["terminal_failures"]["total"] == 0

    code, output = _invoke(["retry", run_id])

    after = runner.paths.journal.read_bytes()
    assert code == 0
    assert after.startswith(before)
    assert len(after) > len(before)
    assert "resume retry-run" in output
    context = load_run_context(run_id)
    assert len(context.state.retry_requests) == 1
    assert context.state.latest_errors == {}


def test_retry_stage_filter_exits_error_when_other_stage_failures_remain(
    tmp_path: Path,
    registry: dict[str, Path],
):
    artifact, manifest = _manifest_artifact(
        tmp_path / "source",
        live_count=1,
        dead_count=0,
    )
    settings = RequestSettings(model="gpt-5.4-nano", pass_b_effort="low")
    run_id = "retry-stage-miss"
    run_path = registry["runs"] / run_id
    metadata = build_run_metadata(
        kind="full",
        run_id=run_id,
        manifest_path=artifact,
        manifest=manifest,
        settings=settings,
    )

    async def handler(kwargs: dict[str, Any]) -> Any:
        raise _FakeHTTPError(429, "too many requests", {"retry-after": "0"})

    runner = ProductionRunner(
        manifest=manifest,
        run_dir=run_path,
        client=_FakeClient(handler),
        settings=_runner_settings(requests=settings, attempts=1),
        run_metadata=metadata,
        install_signal_handlers=False,
    )
    result = asyncio.run(runner.run())
    assert result.all_complete is False

    code, output = _invoke(["retry", run_id, "--stage", "pass_b"])

    assert code != 0
    assert "No active retriable failures matched the requested stage." in output
    status_code, status_json = _invoke(["status", run_id, "--json"])
    status_payload = json.loads(status_json)
    assert status_code == 0
    assert status_payload["retryable_failures"]["total"] == 1
    assert status_payload["retryable_failures"]["by_stage_and_reason"] == {
        "pass_a": {"rate_limit": 1}
    }


def test_status_elapsed_uses_wall_clock_before_first_finish(
    tmp_path: Path,
    registry: dict[str, Path],
):
    artifact, manifest = _manifest_artifact(
        tmp_path / "source",
        live_count=1,
        dead_count=0,
    )
    settings = RequestSettings(model="gpt-5.4-nano", pass_b_effort="low")
    run_id = "in-flight-elapsed"
    run_path = registry["runs"] / run_id
    run_manifest = write_manifest(manifest, run_path / "inputs")
    metadata = build_run_metadata(
        kind="full",
        run_id=run_id,
        manifest_path=run_manifest,
        manifest=manifest,
        settings=settings,
        parent_manifest_path=artifact,
        parent_manifest=manifest,
    )
    fingerprint = request_fingerprint(settings)
    created_at = datetime.now(UTC) - timedelta(seconds=45)
    header = {
        "event_type": "run_started",
        "event_id": "header-event",
        "journal_version": JOURNAL_VERSION,
        "created_at": created_at.isoformat(),
        "manifest_sha256": manifest.manifest_sha256,
        "manifest_rows_sha256": manifest.rows_sha256,
        "manifest_row_count": manifest.row_count,
        "request_fingerprint": fingerprint,
        "request_identity": request_identity(settings),
        "run_config": metadata,
    }
    (run_path / "events.jsonl").write_text(
        json.dumps(header, separators=(",", ":")) + "\n",
        encoding="utf-8",
    )

    code, raw = _invoke(["status", run_id, "--json"])
    payload = json.loads(raw)

    assert code == 0
    assert payload["elapsed_seconds"] is not None
    assert payload["elapsed_seconds"] >= 40
    assert payload["complete"] == 0


def test_status_human_and_json_are_offline(
    tmp_path: Path,
    registry: dict[str, Path],
    monkeypatch: pytest.MonkeyPatch,
):
    artifact, manifest = _manifest_artifact(
        tmp_path / "source",
        live_count=5,
        dead_count=5,
    )
    client = _successful_fake_client()
    code, _ = _invoke(
        [
            "smoke",
            "--manifest",
            str(artifact),
            "--model",
            "gpt-5.4-nano",
            "--run-id",
            "status-smoke",
            "--yes",
        ],
        client_factory=lambda api_key: client,
    )
    assert code == 0
    monkeypatch.delenv("OPENAI_API_KEY", raising=False)

    human_code, human = _invoke(["status", "status-smoke"])
    json_code, raw_json = _invoke(["status", "status-smoke", "--json"])
    payload = json.loads(raw_json)

    assert human_code == json_code == 0
    assert "Offline run status" in human
    assert "Measured cost" in human
    assert "RPM utilization" in human
    assert payload["manifest_total"] == 10
    assert payload["complete"] == 10
    assert payload["canonical_output_ready"] is True
    assert payload["manifest_sha256"] != manifest.manifest_sha256


def test_status_is_read_only_and_available_while_runner_lock_is_held(
    tmp_path: Path,
    registry: dict[str, Path],
):
    artifact, manifest = _manifest_artifact(
        tmp_path / "source",
        live_count=1,
        dead_count=0,
    )
    settings = RequestSettings(model="gpt-5.4-nano", pass_b_effort="low")
    run_id = "live-status"
    run_path = registry["runs"] / run_id
    metadata = build_run_metadata(
        kind="full",
        run_id=run_id,
        manifest_path=artifact,
        manifest=manifest,
        settings=settings,
    )
    runner = ProductionRunner(
        manifest=manifest,
        run_dir=run_path,
        client=_successful_fake_client(),
        settings=_runner_settings(requests=settings),
        run_metadata=metadata,
        install_signal_handlers=False,
    )
    asyncio.run(runner.run())
    original = runner.paths.journal.read_bytes() + b'{"event_type":"pass_a'
    runner.paths.journal.write_bytes(original)

    with RunLock(runner.paths.lock):
        code, raw_json = _invoke(["status", run_id, "--json"])

    assert code == 0
    assert json.loads(raw_json)["complete"] == 1
    assert runner.paths.journal.read_bytes() == original


@pytest.mark.parametrize("damage", ("missing", "stale"))
def test_resume_rebuilds_complete_outputs_without_creating_api_client(
    tmp_path: Path,
    registry: dict[str, Path],
    damage: str,
):
    artifact, manifest = _manifest_artifact(
        tmp_path / "source",
        live_count=1,
        dead_count=0,
    )
    settings = RequestSettings(model="gpt-5.4-nano", pass_b_effort="low")
    run_id = "complete-output-repair"
    run_path = registry["runs"] / run_id
    metadata = build_run_metadata(
        kind="full",
        run_id=run_id,
        manifest_path=artifact,
        manifest=manifest,
        settings=settings,
    )
    runner = ProductionRunner(
        manifest=manifest,
        run_dir=run_path,
        client=_successful_fake_client(),
        settings=_runner_settings(requests=settings),
        run_metadata=metadata,
        install_signal_handlers=False,
    )
    asyncio.run(runner.run())
    expected = runner.paths.final_csv.read_bytes()
    if damage == "missing":
        runner.paths.final_csv.unlink()
    else:
        runner.paths.final_csv.write_text("stale\n", encoding="utf-8")
    runner.paths.in_progress_csv.write_text("stale\n", encoding="utf-8")

    def forbidden_factory(api_key: str | None) -> Any:
        raise AssertionError("complete journal repair must remain offline")

    code, output = _invoke(
        ["resume", run_id, "--yes"],
        client_factory=forbidden_factory,
    )

    assert code == 0, output
    assert "already complete" in output
    assert runner.paths.final_csv.read_bytes() == expected
    assert runner.paths.in_progress_csv.read_bytes() == expected
