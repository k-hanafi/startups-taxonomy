#!/usr/bin/env python3
"""CLI entry point for the legacy V1 single-pass classification pipeline.

Uses the OpenAI Responses API (`POST /v1/responses`) for batch jobs and for
the ``test`` command (structured JSON via ``text.format``).

Each subcommand does exactly one thing and reads state.json to know where
to start. --dry-run on prepare and run prints the full cost plan without
touching the API.

Usage:
    python -m single_pass_classifier prepare  [--dry-run] [--data path/to/input.csv]
    python -m single_pass_classifier submit   [--concurrency 50]
    python -m single_pass_classifier status
    python -m single_pass_classifier download
    python -m single_pass_classifier retry
    python -m single_pass_classifier merge    [--output path]
    python -m single_pass_classifier run      [--dry-run] [--concurrency 50]
"""

from __future__ import annotations

import argparse
import logging
import sys
from pathlib import Path

import pandas as pd

from .builder import build_batch_files, load_system_prompt
from .config import DEFAULT_BATCH_SIZE, DEFAULT_MODEL, ESTIMATED_TOKENS_PER_REQUEST
from .downloader import collect_failed_custom_ids, download_completed
from .formatter import build_custom_id, format_user_message
from .logger import setup_logging
from .merger import DEFAULT_OUTPUT_PATH, print_report
from .monitor import print_status, submit_and_monitor
from .paths import DEFAULT_CLASSIFIER_INPUT_CSV
from .state import BatchRecord, PipelineState
from .submitter import BillingLimitError
from .tokens import estimate_cost

logger = logging.getLogger(__name__)

_PROJECT_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_DATA_CSV = DEFAULT_CLASSIFIER_INPUT_CSV


def _resolve_data(args: argparse.Namespace) -> Path:
    """Return the dataset CSV path from --data or the default."""
    raw = getattr(args, "data", None)
    if raw:
        p = Path(raw)
        return p if p.is_absolute() else _PROJECT_ROOT / p
    return DEFAULT_DATA_CSV


# -- Subcommand handlers -------------------------------------------------------


def _cmd_prepare(args: argparse.Namespace) -> None:
    """Build JSONL batch files from the dataset. With --dry-run, print cost only."""
    setup_logging()

    data_csv = _resolve_data(args)
    row_slice = _parse_rows(args.rows)
    df = pd.read_csv(data_csv)
    if row_slice is not None:
        df = df.iloc[row_slice]

    user_messages = [format_user_message(row._asdict()) for row in df.itertuples(index=False)]
    system_prompt = load_system_prompt()

    estimate = estimate_cost(system_prompt, user_messages, args.model, args.batch_size)
    print(estimate.format_report())

    if args.dry_run:
        return

    files = build_batch_files(df, model=args.model, batch_size=args.batch_size)

    state = PipelineState.load()
    state.run_id = ""
    state.model = args.model
    state.total_companies = len(df)
    # Clear stale batch records and token totals so re-runs start clean.
    state.batches = {}
    state.total_prompt_tokens = 0
    state.total_completion_tokens = 0
    state.total_cached_tokens = 0

    for idx, fpath in enumerate(files, start=1):
        row_start = (idx - 1) * args.batch_size
        row_end = min(row_start + args.batch_size - 1, len(df) - 1)
        key = fpath.stem
        state.batches[key] = BatchRecord(
            batch_number=idx,
            file_path=str(fpath),
            row_range=f"{row_start}-{row_end}",
            estimated_tokens=args.batch_size * ESTIMATED_TOKENS_PER_REQUEST,
        )

    state.save()
    logger.info(
        "Prepared %d batch files. Run 'python -m single_pass_classifier submit' next.",
        len(files),
    )


def _cmd_submit(args: argparse.Namespace) -> None:
    """Submit pending batches and monitor until all complete."""
    setup_logging()
    state = PipelineState.load()

    if not state.batches:
        logger.error(
            "No batches prepared. Run 'python -m single_pass_classifier prepare' first."
        )
        sys.exit(1)

    submit_and_monitor(
        state,
        concurrency=args.concurrency,
        model=args.model,
        batch_size=args.batch_size,
    )


def _cmd_status(args: argparse.Namespace) -> None:
    """Print a one-shot status table of all tracked batches."""
    setup_logging()
    state = PipelineState.load()
    print_status(state)


def _cmd_download(args: argparse.Namespace) -> None:
    """Download results for all completed batches."""
    setup_logging()
    state = PipelineState.load()
    download_completed(state)


def _download_error_files(state: PipelineState) -> None:
    """Download error JSONL files for batches that had failures."""
    from .downloader import ERRORS_DIR, _download_file
    from .submitter import get_client

    batches_needing_download = [
        rec for rec in state.completed_batches() + state.failed_batches()
        if rec.error_file_id
        and not (ERRORS_DIR / f"batch_{rec.batch_number:04d}_errors.jsonl").exists()
    ]

    if not batches_needing_download:
        return

    client = get_client()
    for rec in batches_needing_download:
        dest = ERRORS_DIR / f"batch_{rec.batch_number:04d}_errors.jsonl"
        _download_file(client, rec.error_file_id, dest)


def _cmd_retry(args: argparse.Namespace) -> None:
    """Collect failed custom_ids from error files and re-submit as new batches.

    Downloads error files if not already present, extracts the failed
    custom_ids, pulls matching request lines from the original JSONL
    batch files, and splits into properly-sized batches.
    """
    setup_logging()
    import json
    from .builder import OUTPUT_DIR
    from .downloader import ERRORS_DIR

    state = PipelineState.load()

    logger.info("Downloading error files for batches with failures...")
    _download_error_files(state)

    failed_ids = collect_failed_custom_ids(state)
    if not failed_ids:
        logger.info("No failed requests to retry.")
        return

    failed_id_set = set(failed_ids)
    logger.info("Found %d unique failed requests. Extracting from original JSONL files...", len(failed_id_set))

    retry_lines: list[str] = []
    for rec in sorted(state.batches.values(), key=lambda b: b.batch_number):
        if rec.failed_count == 0 or rec.file_path.endswith("retry_batch.jsonl"):
            continue
        fpath = Path(rec.file_path)
        if not fpath.exists():
            logger.warning("Original batch file missing: %s", fpath)
            continue
        with open(fpath, encoding="utf-8") as f:
            for line in f:
                obj = json.loads(line)
                if obj.get("custom_id") in failed_id_set:
                    retry_lines.append(line.rstrip("\n"))

    if not retry_lines:
        logger.warning("Could not extract any matching requests from original JSONL files.")
        return

    logger.info("Extracted %d request lines for retry.", len(retry_lines))

    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    batch_size = args.batch_size
    next_num = max((b.batch_number for b in state.batches.values()), default=0) + 1
    written_files: list[Path] = []

    for chunk_start in range(0, len(retry_lines), batch_size):
        chunk = retry_lines[chunk_start : chunk_start + batch_size]
        chunk_idx = chunk_start // batch_size + 1
        file_path = OUTPUT_DIR / f"retry_batch_{chunk_idx:04d}.jsonl"

        with open(file_path, "w", encoding="utf-8") as f:
            f.write("\n".join(chunk) + "\n")

        batch_num = next_num + chunk_idx - 1
        state.batches[file_path.stem] = BatchRecord(
            batch_number=batch_num,
            file_path=str(file_path),
            row_range=f"retry-{len(chunk)}",
            estimated_tokens=len(chunk) * ESTIMATED_TOKENS_PER_REQUEST,
        )
        written_files.append(file_path)
        logger.info("Wrote %s  (%d requests)", file_path.name, len(chunk))

    state.save()
    logger.info(
        "Created %d retry batch file(s) with %d total requests. "
        "Run 'python -m single_pass_classifier submit'.",
        len(written_files), len(retry_lines),
    )


def _cmd_merge(args: argparse.Namespace) -> None:
    """Print the classification distribution and cost report."""
    setup_logging()
    state = PipelineState.load()
    output_path = Path(args.output) if args.output else DEFAULT_OUTPUT_PATH
    print_report(state, output_path)


def _cmd_test(args: argparse.Namespace) -> None:
    """Classify a single company synchronously using flex processing.

    Flex tier is priced at batch API rates with prompt caching on top.
    Closes the prompt iteration feedback loop from $100 per batch to
    near-free. Falls back to standard processing on 429 Resource
    Unavailable.
    """
    setup_logging()

    data_csv = _resolve_data(args)
    df = pd.read_csv(data_csv)

    if args.company_id:
        match = df[df["org_uuid"] == args.company_id]
    elif args.company_name:
        match = df[df["name"].str.contains(args.company_name, case=False, na=False)]
    else:
        logger.error("Provide --company-id or --company-name.")
        sys.exit(1)

    if match.empty:
        logger.error("No matching company found.")
        sys.exit(1)

    row = match.iloc[0]
    row_dict = row.to_dict()
    user_msg = format_user_message(row_dict)

    from .builder import _openai_strict_schema, load_system_prompt, responses_text_format_json_schema
    from .config import MAX_OUTPUT_TOKENS, PROMPT_CACHE_KEY
    from .submitter import get_client

    client = get_client()
    system_prompt = load_system_prompt()
    schema = _openai_strict_schema()

    logger.info("Testing classification for: %s", row_dict.get("name", ""))

    import json
    from rich.console import Console
    from rich.panel import Panel
    from tenacity import retry, stop_after_attempt, wait_fixed

    @retry(stop=stop_after_attempt(2), wait=wait_fixed(2), reraise=True)
    def _call(tier: str) -> dict:
        response = client.responses.create(
            model=args.model,
            instructions=system_prompt,
            input=user_msg,
            prompt_cache_key=PROMPT_CACHE_KEY,
            max_output_tokens=MAX_OUTPUT_TOKENS,
            store=False,
            text=responses_text_format_json_schema(schema),
            service_tier=tier,
        )
        return json.loads(response.output_text)

    try:
        result = _call("flex")
        tier_used = "flex"
    except Exception:
        logger.warning("Flex tier unavailable. Falling back to auto.")
        result = _call("auto")
        tier_used = "auto"

    from .schema import ClassificationResult
    validated = ClassificationResult.model_validate(result)

    console = Console()
    console.print()
    console.print(Panel(
        "\n".join(f"[bold]{k}[/]: {v}" for k, v in validated.model_dump().items()),
        title=f"Classification Result (service_tier={tier_used})",
        expand=False,
    ))
    console.print()


def _cmd_run(args: argparse.Namespace) -> None:
    """Full pipeline: prepare, submit, download, then print report."""
    setup_logging()

    args_ns = argparse.Namespace(**vars(args))
    _cmd_prepare(args_ns)

    if args.dry_run:
        return

    _cmd_submit(args_ns)
    _cmd_download(args_ns)
    print_report(PipelineState.load(), DEFAULT_OUTPUT_PATH)


# -- Argument parsing -----------------------------------------------------------


def _parse_rows(rows_str: str | None) -> slice | None:
    """Parse a row range string like '0:50000' into a slice."""
    if not rows_str:
        return None
    parts = rows_str.split(":")
    if len(parts) != 2:
        raise argparse.ArgumentTypeError(f"Invalid row range: {rows_str}. Use start:end.")
    return slice(int(parts[0]), int(parts[1]))


def _add_common_args(parser: argparse.ArgumentParser) -> None:
    """Add --model, --batch-size, and --data to a subcommand parser."""
    parser.add_argument(
        "--model", default=DEFAULT_MODEL,
        help=f"OpenAI model name (default: {DEFAULT_MODEL})",
    )
    parser.add_argument(
        "--batch-size", type=int, default=DEFAULT_BATCH_SIZE, dest="batch_size",
        help=f"Requests per JSONL file (default: {DEFAULT_BATCH_SIZE})",
    )
    parser.add_argument(
        "--data", default=None,
        help="Path to input CSV (default: outputs/tavilycrawl/processed/classifier_input.csv)",
    )


def build_parser() -> argparse.ArgumentParser:
    """Build the CLI argument parser with all subcommands."""
    parser = argparse.ArgumentParser(
        prog="python -m single_pass_classifier",
        description="Legacy V1 single-pass AI-native classifier via OpenAI Batch API",
    )
    subs = parser.add_subparsers(dest="command", required=True)

    # prepare
    p = subs.add_parser(
        "prepare",
        help="Build JSONL batch files from the dataset and show cost estimate",
    )
    _add_common_args(p)
    p.add_argument("--dry-run", action="store_true", dest="dry_run",
                   help="Print cost breakdown only. No files written, no API calls.")
    p.add_argument("--rows", default=None,
                   help="Row range to process, e.g. '0:50000'")
    p.set_defaults(func=_cmd_prepare)

    # submit
    p = subs.add_parser(
        "submit",
        help="Submit pending batches and monitor until all complete",
    )
    _add_common_args(p)
    p.add_argument("--concurrency", type=int, default=50,
                   help="Max batches in-flight simultaneously (default: 50 — effectively all)")
    p.set_defaults(func=_cmd_submit)

    # status
    p = subs.add_parser("status", help="Print status of all tracked batches")
    p.set_defaults(func=_cmd_status)

    # download
    p = subs.add_parser("download", help="Download results for completed batches")
    p.set_defaults(func=_cmd_download)

    # retry
    p = subs.add_parser(
        "retry",
        help="Re-submit failed requests from error files as a new batch",
    )
    _add_common_args(p)
    p.set_defaults(func=_cmd_retry)

    # merge
    p = subs.add_parser(
        "merge",
        help="Merge batch outputs into final CSV and print distribution report",
    )
    p.add_argument("--output", default=None,
                   help="Output CSV path (default: outputs/production_csvs/production_classifications.csv)")
    p.set_defaults(func=_cmd_merge)

    # test
    p = subs.add_parser(
        "test",
        help="Classify one company synchronously using flex pricing for prompt iteration",
    )
    _add_common_args(p)
    p.add_argument("--company-id", default=None, dest="company_id",
                   help="org_uuid of the company to test")
    p.add_argument("--company-name", default=None, dest="company_name",
                   help="Partial name match (case-insensitive)")
    p.set_defaults(func=_cmd_test)

    # run
    p = subs.add_parser(
        "run",
        help="Full pipeline: prepare -> submit -> download -> report",
    )
    _add_common_args(p)
    p.add_argument("--dry-run", action="store_true", dest="dry_run",
                   help="Run prepare in dry-run mode only")
    p.add_argument("--rows", default=None,
                   help="Row range to process, e.g. '0:50000'")
    p.add_argument("--concurrency", type=int, default=50,
                   help="Max batches in-flight simultaneously (default: 50 — effectively all)")
    p.add_argument("--output", default=None)
    p.set_defaults(func=_cmd_run)

    return parser


def main() -> None:
    args = build_parser().parse_args()
    try:
        args.func(args)
    except BillingLimitError:
        sys.exit(2)


if __name__ == "__main__":
    main()
