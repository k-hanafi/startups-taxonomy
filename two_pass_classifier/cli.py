"""Beginner-friendly production workflow for the two-pass classifier."""

from __future__ import annotations

import argparse
import asyncio
import inspect
import json
import sys
from pathlib import Path
from typing import Any, Callable, Mapping

from rich.console import Console
from rich.panel import Panel
from rich.prompt import Confirm
from rich.table import Table

from . import config
from .api_key import MissingAPIKeyError, load_real_api_key
from .costing import CostPreview, estimate_manifest_cost
from .journal import (
    JournalCorruptionError,
    ResumeMismatchError,
    RunLockedError,
    append_retry_events,
)
from .manifest import (
    Manifest,
    ManifestCollisionError,
    ManifestValidationError,
    build_manifest,
    load_manifest,
    write_manifest,
)
from .paths import (
    DEFAULT_DEAD_INPUT,
    DEFAULT_LIVE_INPUT,
    MANIFESTS_DIR,
    PROJECT_ROOT,
)
from .request_builder import RequestSettings
from .runner import (
    ProductionRunner,
    RunnerSettings,
    StageFailed,
    create_async_openai_client,
)
from .status import RunStatus, build_run_status
from .workflow import (
    WorkflowError,
    build_run_metadata,
    find_matching_smoke,
    load_run_context,
    make_run_id,
    new_run_dir,
    repair_run_artifacts,
    resolve_manifest_path,
    select_smoke_manifest,
)

ClientFactory = Callable[[str | None], Any]
EXIT_ERROR = 1
EXIT_USAGE = 2


class CLIError(ValueError):
    """An actionable end-user workflow error."""


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="python -m two_pass_classifier",
        description=(
            "Build, price, validate, run, and resume the production two-pass "
            "startup classifier. Offline commands never need an API key."
        ),
        epilog=(
            "Recommended order: build-manifest, cost-preview, smoke, run. "
            "Use status at any time, resume after interruption, and retry to "
            "record an explicit retry request for retriable failures."
        ),
    )
    subcommands = parser.add_subparsers(dest="command", required=True)

    manifest_parser = subcommands.add_parser(
        "build-manifest",
        help="Validate live and archive inputs, then write an immutable manifest.",
        description=(
            "Combine the live and archive classifier inputs into one "
            "content-addressed manifest. This command is offline."
        ),
    )
    manifest_parser.add_argument(
        "--live",
        default=str(DEFAULT_LIVE_INPUT),
        metavar="CSV",
        help="Live classifier input CSV (default: repository live output).",
    )
    manifest_parser.add_argument(
        "--dead",
        default=str(DEFAULT_DEAD_INPUT),
        metavar="CSV",
        help="Recovered archive classifier input CSV (default: repository archive output).",
    )
    manifest_parser.add_argument(
        "--output-dir",
        default=str(MANIFESTS_DIR),
        metavar="DIR",
        help="Manifest directory (default: outputs/two_pass_classifier/manifests).",
    )

    preview_parser = subcommands.add_parser(
        "cost-preview",
        help="Count production input tokens and estimate normal Responses cost.",
        description=(
            "Count the selected manifest with the production prompts, schemas, "
            "and formatter. This command makes no API calls."
        ),
    )
    _add_manifest_argument(preview_parser)
    _add_semantic_arguments(preview_parser)

    smoke_parser = subcommands.add_parser(
        "smoke",
        help="Run the exact production requests on 10 deterministic companies.",
        description=(
            "Select exactly 10 companies from the full manifest, balanced by "
            "live and archive evidence when possible, then run both passes."
        ),
    )
    _add_manifest_argument(smoke_parser)
    _add_semantic_arguments(smoke_parser)
    _add_new_run_arguments(smoke_parser)

    run_parser = subcommands.add_parser(
        "run",
        help="Create a new full production run after a matching smoke succeeds.",
        description=(
            "Start a new run over the complete manifest. A successful matching "
            "10-company smoke is required before any paid request can start."
        ),
    )
    _add_manifest_argument(run_parser)
    _add_semantic_arguments(run_parser)
    _add_new_run_arguments(run_parser)

    status_parser = subcommands.add_parser(
        "status",
        help="Report one run from local journal data only.",
        description=(
            "Show progress, failures, measured usage and cost, throughput, ETA, "
            "rate utilization, concurrency, and output paths. No API calls."
        ),
    )
    status_parser.add_argument(
        "run_id",
        metavar="RUN_ID",
        help="Run directory name under outputs/two_pass_classifier/runs.",
    )
    status_parser.add_argument(
        "--json",
        action="store_true",
        help="Print machine-readable JSON instead of Rich tables.",
    )

    resume_parser = subcommands.add_parser(
        "resume",
        help="Continue only missing or retriable work with locked semantics.",
        description=(
            "Load the immutable model, effort, prompts, schemas, formatter, and "
            "output caps from the journal, then continue unfinished work."
        ),
    )
    resume_parser.add_argument("run_id", metavar="RUN_ID", help="Run to continue.")
    _add_paid_confirmation(resume_parser)
    resume_parser.add_argument(
        "--max-attempts",
        type=_positive_int,
        default=config.MAX_REQUEST_ATTEMPTS,
        metavar="N",
        help=(
            "Advanced operational override: physical attempts per request "
            f"during this continuation (default: {config.MAX_REQUEST_ATTEMPTS})."
        ),
    )

    retry_parser = subcommands.add_parser(
        "retry",
        help="Append retry events for retriable failures without deleting history.",
        description=(
            "Record explicit retry requests for active retriable failures. "
            "Completed rows and terminal failures are never reset."
        ),
    )
    retry_parser.add_argument("run_id", metavar="RUN_ID", help="Run to update.")
    retry_parser.add_argument(
        "--stage",
        choices=("pass_a", "pass_b"),
        default=None,
        help="Limit retry events to one stage (default: both stages).",
    )
    retry_parser.add_argument(
        "--continue",
        dest="continue_run",
        action="store_true",
        help="Continue the paid run immediately after appending retry events.",
    )
    _add_paid_confirmation(retry_parser)
    retry_parser.add_argument(
        "--max-attempts",
        type=_positive_int,
        default=config.MAX_REQUEST_ATTEMPTS,
        metavar="N",
        help=(
            "Advanced operational override used only with --continue "
            f"(default: {config.MAX_REQUEST_ATTEMPTS})."
        ),
    )
    return parser


def main(
    argv: list[str] | None = None,
    *,
    client_factory: ClientFactory | None = None,
    console: Console | None = None,
) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)
    output = console or Console()
    handlers = {
        "build-manifest": _cmd_build_manifest,
        "cost-preview": _cmd_cost_preview,
        "smoke": _cmd_smoke,
        "run": _cmd_run,
        "status": _cmd_status,
        "resume": _cmd_resume,
        "retry": _cmd_retry,
    }
    try:
        return handlers[args.command](
            args,
            console=output,
            client_factory=client_factory,
        )
    except KeyboardInterrupt:
        output.print("\n[yellow]Stopped by user.[/yellow]")
        return 130
    except (
        CLIError,
        WorkflowError,
        ManifestValidationError,
        ManifestCollisionError,
        JournalCorruptionError,
        ResumeMismatchError,
        RunLockedError,
        FileNotFoundError,
        ValueError,
    ) as exc:
        output.print(f"[bold red]Error:[/bold red] {exc}")
        return EXIT_USAGE
    except StageFailed as exc:
        output.print(f"[bold red]Paid run blocked:[/bold red] {exc}")
        output.print(
            "Inspect the durable local state with "
            f"'python -m two_pass_classifier status {args.run_id}'."
        )
        return EXIT_ERROR


def _cmd_build_manifest(
    args: argparse.Namespace,
    *,
    console: Console,
    client_factory: ClientFactory | None,
) -> int:
    del client_factory
    live = _project_path(args.live)
    dead = _project_path(args.dead)
    output_dir = _project_path(args.output_dir)
    manifest = build_manifest(live, dead)
    expected_path = output_dir / f"manifest_{manifest.manifest_sha256}.jsonl"
    reused = expected_path.exists()
    artifact = write_manifest(manifest, output_dir)

    table = Table(title="Immutable production manifest")
    table.add_column("Source")
    table.add_column("Input rows", justify="right")
    table.add_column("Included", justify="right")
    table.add_column("Source SHA-256")
    for source in manifest.sources:
        table.add_row(
            source.evidence_source,
            f"{source.input_row_count:,}",
            f"{source.included_row_count:,}",
            source.file_sha256,
        )
    console.print(table)
    console.print(f"[bold]Total rows:[/bold] {manifest.row_count:,}")
    console.print(f"[bold]Rows SHA-256:[/bold] {manifest.rows_sha256}")
    console.print(f"[bold]Manifest SHA-256:[/bold] {manifest.manifest_sha256}")
    console.print(f"[bold]Output:[/bold] {artifact}")
    console.print(
        "[green]Reused identical immutable output.[/green]"
        if reused
        else "[green]Wrote new immutable output.[/green]"
    )
    console.print("No API calls were made.")
    return 0


def _cmd_cost_preview(
    args: argparse.Namespace,
    *,
    console: Console,
    client_factory: ClientFactory | None,
) -> int:
    del client_factory
    manifest_path, manifest = _load_selected_manifest(args.manifest)
    settings = _settings_from_args(args)
    preview = estimate_manifest_cost(manifest, settings)
    _print_configuration(
        console,
        settings=settings,
        rows=manifest.row_count,
        title="Offline cost configuration",
        manifest_path=manifest_path,
    )
    _print_cost_preview(console, preview)
    console.print(
        "[green]No API calls were made.[/green] The estimate uses normal "
        "Responses list prices, assumes no cache savings, and applies no "
        "discounted processing rate."
    )
    return 0


def _cmd_smoke(
    args: argparse.Namespace,
    *,
    console: Console,
    client_factory: ClientFactory | None,
) -> int:
    parent_path, parent = _load_selected_manifest(args.manifest)
    settings = _settings_from_args(args)
    smoke_manifest = select_smoke_manifest(parent)
    run_id = args.run_id or make_run_id(
        "smoke",
        settings.model,
        settings.pass_b_effort,
    )
    run_path = new_run_dir(run_id)
    args.run_id = run_id
    preview = estimate_manifest_cost(smoke_manifest, settings)

    _print_configuration(
        console,
        settings=settings,
        rows=smoke_manifest.row_count,
        title="Paid smoke configuration",
        manifest_path=parent_path,
        run_id=run_id,
    )
    source_counts = smoke_manifest.source_counts
    console.print(
        "Selection: "
        f"{source_counts.get('live', 0)} live, "
        f"{source_counts.get('dead', 0)} archive. "
        "AI family is not a stratum because the manifest contains no labels."
    )
    _print_cost_preview(console, preview)
    if not _confirm_paid(console, yes=args.yes, action="Start this paid smoke?"):
        console.print("[yellow]Cancelled before creating a run.[/yellow]")
        return 0

    client = _make_paid_client(client_factory)
    smoke_manifest_path = write_manifest(
        smoke_manifest,
        run_path / "inputs",
    )
    metadata = build_run_metadata(
        kind="smoke",
        run_id=run_id,
        manifest_path=smoke_manifest_path,
        manifest=smoke_manifest,
        settings=settings,
        parent_manifest_path=parent_path,
        parent_manifest=parent,
    )
    result = _run_runner(
        manifest=smoke_manifest,
        run_dir=run_path,
        settings=settings,
        metadata=metadata,
        client=client,
    )
    _print_run_result(console, run_id, result)
    return _canonical_exit_code(run_id, console)


def _cmd_run(
    args: argparse.Namespace,
    *,
    console: Console,
    client_factory: ClientFactory | None,
) -> int:
    manifest_path, manifest = _load_selected_manifest(args.manifest)
    settings = _settings_from_args(args)
    smoke = find_matching_smoke(
        parent_manifest=manifest,
        settings=settings,
    )
    if smoke is None:
        raise CLIError(
            "no successful 10-company smoke matches this manifest, model, "
            "Pass B effort, prompts, schemas, formatter, and output caps. Run "
            f"'python -m two_pass_classifier smoke --manifest "
            f"{_command_path(manifest_path)} --model {settings.model} "
            f"--effort {settings.pass_b_effort}' first"
        )

    run_id = args.run_id or make_run_id(
        "run",
        settings.model,
        settings.pass_b_effort,
    )
    run_path = new_run_dir(run_id)
    args.run_id = run_id
    preview = estimate_manifest_cost(manifest, settings)
    _print_configuration(
        console,
        settings=settings,
        rows=manifest.row_count,
        title="Paid full-run configuration",
        manifest_path=manifest_path,
        run_id=run_id,
    )
    console.print(
        f"[green]Smoke gate passed:[/green] {smoke.run_id} "
        f"({smoke.completed_at})"
    )
    _print_cost_preview(console, preview)
    if not _confirm_paid(
        console,
        yes=args.yes,
        action="Start this new paid full run?",
    ):
        console.print("[yellow]Cancelled before creating a run.[/yellow]")
        return 0

    client = _make_paid_client(client_factory)
    metadata = build_run_metadata(
        kind="full",
        run_id=run_id,
        manifest_path=manifest_path,
        manifest=manifest,
        settings=settings,
    )
    result = _run_runner(
        manifest=manifest,
        run_dir=run_path,
        settings=settings,
        metadata=metadata,
        client=client,
    )
    _print_run_result(console, run_id, result)
    return _canonical_exit_code(run_id, console)


def _cmd_status(
    args: argparse.Namespace,
    *,
    console: Console,
    client_factory: ClientFactory | None,
) -> int:
    del client_factory
    status = build_run_status(load_run_context(args.run_id))
    if args.json:
        console.file.write(
            json.dumps(status.to_dict(), indent=2, sort_keys=True) + "\n"
        )
        console.file.flush()
    else:
        _print_human_status(console, status)
    return 0


def _cmd_resume(
    args: argparse.Namespace,
    *,
    console: Console,
    client_factory: ClientFactory | None,
) -> int:
    context = load_run_context(args.run_id)
    repair_run_artifacts(context)
    context = load_run_context(args.run_id)
    return _resume_context(
        context,
        console=console,
        client_factory=client_factory,
        yes=args.yes,
        max_attempts=args.max_attempts,
    )


def _cmd_retry(
    args: argparse.Namespace,
    *,
    console: Console,
    client_factory: ClientFactory | None,
) -> int:
    context = load_run_context(args.run_id)
    events = append_retry_events(
        context.paths.run_dir,
        manifest=context.manifest,
        stage=args.stage,
    )
    console.print(
        f"[green]Appended {len(events):,} retry event(s)[/green] to "
        f"{context.paths.journal}. No history was deleted."
    )
    if not events:
        status = build_run_status(context)
        console.print("No active retriable failures matched the requested stage.")
        if status.retryable_failures.total or status.terminal_failures.total:
            return EXIT_ERROR
        return 0
    if not args.continue_run:
        console.print(
            "Continue with: "
            f"[bold]python -m two_pass_classifier resume {args.run_id}[/bold]"
        )
        refreshed = build_run_status(load_run_context(args.run_id))
        return EXIT_ERROR if refreshed.terminal_failures.total else 0
    refreshed_context = load_run_context(args.run_id)
    return _resume_context(
        refreshed_context,
        console=console,
        client_factory=client_factory,
        yes=args.yes,
        max_attempts=args.max_attempts,
    )


def _resume_context(
    context: Any,
    *,
    console: Console,
    client_factory: ClientFactory | None,
    yes: bool,
    max_attempts: int,
) -> int:
    status = build_run_status(context)
    if status.canonical_output_ready:
        console.print(
            f"[green]Run {context.run_id} is already complete.[/green] "
            f"Final output: {context.paths.final_csv}"
        )
        return 0
    if status.remaining_runnable == 0:
        _print_human_status(console, status)
        console.print(
            "[bold red]No runnable work remains.[/bold red] Terminal failures "
            "cannot be reset by retry."
        )
        return EXIT_ERROR

    pass_a_ids = {
        row.company_id
        for row in context.manifest.rows
        if row.company_id not in context.state.completed
        and row.company_id not in context.state.pass_a
        and (
            row.company_id not in context.state.latest_errors
            or bool(context.state.latest_errors[row.company_id].get("retriable"))
        )
    }
    pass_b_families: dict[str, int | None] = {}
    for row in context.manifest.rows:
        if row.company_id in context.state.completed:
            continue
        error = context.state.latest_errors.get(row.company_id)
        if error is not None and not error.get("retriable"):
            continue
        pass_a = context.state.pass_a.get(row.company_id)
        pass_b_families[row.company_id] = (
            int(pass_a["normalized"]["ai_native"])
            if pass_a is not None
            else None
        )
    preview = estimate_manifest_cost(
        context.manifest,
        context.settings,
        pass_a_company_ids=pass_a_ids,
        pass_b_families=pass_b_families,
    )
    _print_configuration(
        console,
        settings=context.settings,
        rows=status.remaining_runnable,
        title="Paid resume configuration",
        manifest_path=context.manifest_path,
        run_id=context.run_id,
    )
    console.print(
        f"Will continue {status.remaining_runnable:,} missing or retriable "
        f"company task(s). Completed rows skipped: {status.complete:,}."
    )
    _print_cost_preview(console, preview)
    if not _confirm_paid(
        console,
        yes=yes,
        action="Continue this paid run?",
    ):
        console.print("[yellow]Cancelled. Journal state is unchanged.[/yellow]")
        return 0

    client = _make_paid_client(client_factory)
    result = _run_runner(
        manifest=context.manifest,
        run_dir=context.paths.run_dir,
        settings=context.settings,
        metadata=context.run_config,
        client=client,
        max_attempts=max_attempts,
    )
    _print_run_result(console, context.run_id, result)
    return _canonical_exit_code(context.run_id, console)


def _run_runner(
    *,
    manifest: Manifest,
    run_dir: Path,
    settings: RequestSettings,
    metadata: Mapping[str, Any],
    client: Any,
    max_attempts: int = config.MAX_REQUEST_ATTEMPTS,
) -> Any:
    runner = ProductionRunner(
        manifest=manifest,
        run_dir=run_dir,
        client=client,
        settings=RunnerSettings(
            requests=settings,
            max_request_attempts=max_attempts,
        ),
        run_metadata=metadata,
    )

    async def execute() -> Any:
        try:
            return await runner.run()
        finally:
            close = getattr(client, "close", None)
            if close is not None:
                closed = close()
                if inspect.isawaitable(closed):
                    await closed

    return asyncio.run(execute())


def _canonical_exit_code(run_id: str, console: Console) -> int:
    context = load_run_context(run_id)
    status = build_run_status(context)
    _print_human_status(console, status)
    if status.canonical_output_ready and not status.terminal_failures.total:
        return 0
    if status.terminal_failures.total:
        console.print(
            "[bold red]Run ended with unresolved terminal failures.[/bold red]"
        )
    else:
        console.print(
            "[bold yellow]Run is incomplete.[/bold yellow] Continue with "
            f"'python -m two_pass_classifier resume {run_id}'."
        )
    return EXIT_ERROR


def _print_run_result(console: Console, run_id: str, result: Any) -> None:
    console.print(
        f"[bold]Run {run_id} stopped:[/bold] {result.stopped}; "
        f"complete: {result.completed_count:,}/{result.manifest_row_count:,}."
    )


def _print_configuration(
    console: Console,
    *,
    settings: RequestSettings,
    rows: int,
    title: str,
    manifest_path: Path,
    run_id: str | None = None,
) -> None:
    lines = [
        f"[bold]Model:[/bold] {settings.model}",
        f"[bold]Pass A reasoning:[/bold] {config.PASS_A_EFFORT} (fixed)",
        f"[bold]Pass B reasoning:[/bold] {settings.pass_b_effort}",
        (
            f"[bold]Output caps:[/bold] Pass A "
            f"{settings.pass_a_max_output_tokens:,}, Pass B "
            f"{int(settings.pass_b_max_output_tokens or 0):,}"
        ),
        f"[bold]Companies in this action:[/bold] {rows:,}",
        f"[bold]Manifest:[/bold] {manifest_path}",
    ]
    if run_id is not None:
        lines.insert(0, f"[bold]Run ID:[/bold] {run_id}")
    console.print(Panel("\n".join(lines), title=title, border_style="cyan"))


def _print_cost_preview(console: Console, preview: CostPreview) -> None:
    table = Table(title="Cost preview")
    table.add_column("Stage")
    table.add_column("Requests", justify="right")
    table.add_column("Counted input tokens", justify="right")
    table.add_column("Planning output", justify="right")
    table.add_column("One-attempt cap output", justify="right")
    table.add_column("Planning cost", justify="right")
    table.add_column("One-attempt cap cost", justify="right")
    for label, stage in (("Pass A", preview.pass_a), ("Pass B", preview.pass_b)):
        table.add_row(
            label,
            f"{stage.request_count:,}",
            _number_range(stage.input_tokens_min, stage.input_tokens_max),
            f"{stage.estimated_output_tokens:,}",
            f"{stage.one_attempt_cap_tokens:,}",
            _money_range(stage.estimated_cost_min, stage.estimated_cost_max),
            _money_range(
                stage.one_attempt_cap_cost_min,
                stage.one_attempt_cap_cost_max,
            ),
        )
    console.print(table)
    if preview.unknown_family_count:
        console.print(
            "Pass B input family scenarios: "
            f"0-series {preview.pass_b_input_if_family_0:,} tokens; "
            f"1-series {preview.pass_b_input_if_family_1:,} tokens. "
            f"Unknown family before Pass A: {preview.unknown_family_count:,}."
        )
    else:
        console.print(
            "Measured Pass A family split for remaining Pass B work: "
            f"0-series {preview.known_family_counts[0]:,}; "
            f"1-series {preview.known_family_counts[1]:,}."
        )
    console.print(
        f"[bold]Full planning total:[/bold] "
        f"{_money_range(preview.estimated_total_min, preview.estimated_total_max)}"
    )
    console.print(
        f"[bold]One-attempt cap projection:[/bold] "
        f"{_money_range(preview.one_attempt_cap_total_min, preview.one_attempt_cap_total_max)}"
    )
    console.print(
        "Assumptions: counted production prompt, schema, formatter, and row "
        "tokens plus a fixed framing allowance; Pass A output is provisional "
        "pending the aligned paid eval; reasoning tokens are billed as output; "
        "cache savings are treated as zero. The cap projection assumes one "
        "attempt per planned request, so retries or later resumes can exceed it."
    )


def _print_human_status(console: Console, status: RunStatus) -> None:
    console.print(
        Panel(
            "\n".join(
                (
                    f"[bold]Run ID:[/bold] {status.run_id}",
                    f"[bold]Kind:[/bold] {status.kind}",
                    f"[bold]Model:[/bold] {status.model}",
                    f"[bold]Pass A reasoning:[/bold] {status.pass_a_effort}",
                    f"[bold]Pass B reasoning:[/bold] {status.pass_b_effort}",
                    f"[bold]Manifest rows:[/bold] {status.manifest_total:,}",
                )
            ),
            title="Offline run status",
            border_style="cyan",
        )
    )
    progress = Table(title="Progress")
    progress.add_column("Untouched", justify="right")
    progress.add_column("Pass A only", justify="right")
    progress.add_column("Complete", justify="right")
    progress.add_column("Runnable now", justify="right")
    progress.add_row(
        f"{status.untouched:,}",
        f"{status.pass_a_only:,}",
        f"{status.complete:,}",
        f"{status.remaining_runnable:,}",
    )
    console.print(progress)

    failures = Table(title="Unresolved failures")
    failures.add_column("Disposition")
    failures.add_column("Total", justify="right")
    failures.add_column("By stage and reason")
    failures.add_row(
        "Retryable",
        f"{status.retryable_failures.total:,}",
        _failure_text(status.retryable_failures.by_stage_and_reason),
    )
    failures.add_row(
        "Terminal",
        f"{status.terminal_failures.total:,}",
        _failure_text(status.terminal_failures.by_stage_and_reason),
    )
    console.print(failures)

    usage = status.usage
    operations = Table(title="Measured operation")
    operations.add_column("Metric")
    operations.add_column("Value", justify="right")
    operations.add_row("Physical requests", f"{usage['physical_requests']:,}")
    operations.add_row("Input tokens", f"{usage['input_tokens']:,}")
    operations.add_row(
        "Cached input tokens",
        f"{usage['cached_input_tokens']:,}",
    )
    operations.add_row("Output tokens", f"{usage['output_tokens']:,}")
    operations.add_row("Reasoning tokens", f"{usage['reasoning_tokens']:,}")
    operations.add_row("Measured cost", f"${usage['cost_usd']:,.4f}")
    operations.add_row(
        "Requests missing usage",
        f"{usage['requests_missing_usage']:,}",
    )
    operations.add_row(
        "Throughput",
        (
            f"{status.throughput_companies_per_hour:,.1f} companies/hour "
            "(active time)"
            if status.throughput_companies_per_hour is not None
            else "not available"
        ),
    )
    operations.add_row(
        "ETA",
        _duration(status.eta_seconds),
    )
    operations.add_row(
        "Last concurrency limit",
        _optional_number(status.concurrency.get("last_recorded_limit")),
    )
    operations.add_row(
        "Active at last event",
        _optional_number(status.concurrency.get("active_at_last_event")),
    )
    operations.add_row(
        "RPM utilization",
        _percentage(status.rate_utilization.get("rpm")),
    )
    operations.add_row(
        "TPM utilization",
        _percentage(status.rate_utilization.get("tpm")),
    )
    console.print(operations)

    outputs = Table(title="Output paths")
    outputs.add_column("Artifact")
    outputs.add_column("Path")
    for label, path in status.output_paths.items():
        outputs.add_row(label, path)
    console.print(outputs)
    if status.canonical_output_ready:
        console.print("[green]Canonical classifications.csv is complete.[/green]")
    else:
        console.print(
            "[yellow]Canonical classifications.csv is withheld until every "
            "manifest row is complete.[/yellow]"
        )


def _make_paid_client(client_factory: ClientFactory | None) -> Any:
    if client_factory is not None:
        return client_factory(None)
    return create_async_openai_client(api_key=_load_real_api_key())


def _load_real_api_key() -> str:
    try:
        return load_real_api_key()
    except MissingAPIKeyError as exc:
        raise CLIError(str(exc)) from exc


def _confirm_paid(console: Console, *, yes: bool, action: str) -> bool:
    if yes:
        console.print("[yellow]Confirmation skipped because --yes was provided.[/yellow]")
        return True
    return Confirm.ask(action, console=console, default=False)


def _load_selected_manifest(value: str | None) -> tuple[Path, Manifest]:
    path = resolve_manifest_path(value)
    return path, load_manifest(path)


def _settings_from_args(args: argparse.Namespace) -> RequestSettings:
    return RequestSettings(
        model=args.model,
        pass_b_effort=args.effort,
    )


def _add_manifest_argument(parser: argparse.ArgumentParser) -> None:
    parser.add_argument(
        "--manifest",
        default=None,
        metavar="JSONL",
        help=(
            "Full immutable manifest. If omitted, use the newest valid "
            "manifest under outputs/two_pass_classifier/manifests."
        ),
    )


def _add_semantic_arguments(parser: argparse.ArgumentParser) -> None:
    parser.add_argument(
        "--model",
        choices=config.SUPPORTED_MODELS,
        default=config.DEFAULT_MODEL,
        help=f"Model override (default: {config.DEFAULT_MODEL}).",
    )
    parser.add_argument(
        "--effort",
        "--effort-b",
        dest="effort",
        choices=config.SUPPORTED_PASS_B_EFFORTS,
        default=config.DEFAULT_PASS_B_EFFORT,
        help=(
            "Pass B reasoning effort override "
            f"(default: {config.DEFAULT_PASS_B_EFFORT}). Pass A stays none."
        ),
    )


def _add_new_run_arguments(parser: argparse.ArgumentParser) -> None:
    parser.add_argument(
        "--run-id",
        default=None,
        metavar="RUN_ID",
        help="Readable run ID override (default: timestamp plus random suffix).",
    )
    _add_paid_confirmation(parser)


def _add_paid_confirmation(parser: argparse.ArgumentParser) -> None:
    parser.add_argument(
        "--yes",
        "-y",
        action="store_true",
        help="Skip the interactive paid-action confirmation.",
    )


def _project_path(value: str | Path) -> Path:
    path = Path(value).expanduser()
    if not path.is_absolute():
        path = PROJECT_ROOT / path
    return path.resolve()


def _command_path(path: Path) -> str:
    try:
        return str(path.resolve().relative_to(PROJECT_ROOT))
    except ValueError:
        return str(path)


def _positive_int(value: str) -> int:
    parsed = int(value)
    if parsed < 1:
        raise argparse.ArgumentTypeError("must be a positive integer")
    return parsed


def _number_range(low: int, high: int) -> str:
    return f"{low:,}" if low == high else f"{low:,} to {high:,}"


def _money_range(low: float, high: float) -> str:
    return f"${low:,.2f}" if abs(low - high) < 0.005 else f"${low:,.2f} to ${high:,.2f}"


def _duration(seconds: float | None) -> str:
    if seconds is None:
        return "not available"
    remaining = max(0, round(seconds))
    hours, remainder = divmod(remaining, 3_600)
    minutes, secs = divmod(remainder, 60)
    if hours:
        return f"{hours}h {minutes}m"
    if minutes:
        return f"{minutes}m {secs}s"
    return f"{secs}s"


def _optional_number(value: Any) -> str:
    return f"{int(value):,}" if isinstance(value, (int, float)) else "not recorded"


def _percentage(value: Any) -> str:
    return f"{float(value) * 100:.1f}%" if isinstance(value, (int, float)) else "not recorded"


def _failure_text(value: Mapping[str, Mapping[str, int]]) -> str:
    parts = [
        f"{stage}: "
        + ", ".join(f"{reason}={count}" for reason, count in reasons.items())
        for stage, reasons in value.items()
    ]
    return "; ".join(parts) if parts else "none"


if __name__ == "__main__":
    sys.exit(main())
