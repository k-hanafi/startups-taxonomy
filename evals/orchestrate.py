"""End-to-end paid matrix orchestrator: ``python -m evals run-evals``.

Every invocation runs from scratch: rebuilds Pass A banks, mints new cell
run ids, scores, then builds the dashboard. Phase 1 = 3 banks in parallel;
phase 2 = 9 Pass B cells in parallel. Progress is read from disk
(predictions.jsonl / scored.json), not scraped from child stdout. Child
logs land in each run's ``run.log``.
"""

from __future__ import annotations

import datetime as _dt
import json
import logging
import subprocess
import sys
import time
import webbrowser
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Callable, Literal

from rich.console import Console
from rich.live import Live
from rich.table import Table
from rich.text import Text
from two_pass_classifier import config as production_config

from evals import config as cfg
from evals.classification import matrix_cells
from evals.cost_preview import print_matrix_preview
from evals.jsonl_io import iter_jsonl
from evals.paths import (
    EVAL_INSTANCES_DIR,
    PROJECT_ROOT,
    RUNS_DIR,
    pass_a_bank_run_id,
    run_config_path,
    run_dir,
    run_predictions_path,
    run_scored_path,
)
from evals.runner import load_golden_rows

logger = logging.getLogger(__name__)

JobKind = Literal["pass_a", "cell", "dashboard"]
JobStatus = Literal["pending", "running", "done", "failed", "skipped"]

_STATUS_STYLE = {
    "pending": ("dim", "·"),
    "running": ("yellow", "…"),
    "done": ("green", "✓"),
    "failed": ("red", "✗"),
    "skipped": ("green", "✓"),
}


@dataclass
class Job:
    """One checklist row supervised by the orchestrator."""

    key: str
    label: str
    kind: JobKind
    model: str | None = None
    effort_b: str | None = None
    run_id: str | None = None
    status: JobStatus = "pending"
    rows_done: int = 0
    rows_total: int = 0
    spend_usd: float = 0.0
    started_at: float | None = None
    finished_at: float | None = None
    returncode: int | None = None
    error: str | None = None
    process: subprocess.Popen[Any] | None = field(default=None, repr=False)
    # File handle for child stdout/stderr (Popen.stdout is None when a file
    # object is passed, so we keep our own reference to close it).
    log_handle: Any = field(default=None, repr=False)
    # Cell jobs run classify then score; track which stage is active.
    cell_stage: Literal["classify", "score"] | None = None


def require_openai_key() -> None:
    """Refuse with a plain-English message if the API key is missing or a placeholder.

    Uses the same env-or-``keys/openai.env`` check as production paid commands.
    """
    from two_pass_classifier.api_key import MissingAPIKeyError, load_real_api_key

    try:
        load_real_api_key(PROJECT_ROOT)
    except MissingAPIKeyError as exc:
        env_file = PROJECT_ROOT / "keys" / "openai.env"
        raise SystemExit(
            f"{exc}\n"
            f"  1. Create {env_file} (git ignores this file).\n"
            "  2. Put one line in it: OPENAI_API_KEY=sk-...\n"
            "  3. Re-run: python -m evals run-evals\n"
            "No need to export the key in your shell. The harness loads the file."
        ) from exc


def open_dashboard_index() -> Path:
    """Open the instance archive index in the default browser. Returns the path.

    Rebuilds the index first so deleted smoke pages cannot leave a dead link.
    """
    from evals.instances import load_registry, sync_index

    index_path = sync_index(EVAL_INSTANCES_DIR)
    entries = load_registry(EVAL_INSTANCES_DIR)
    pages = list(EVAL_INSTANCES_DIR.glob("eval_instance_*.html"))
    if not entries and not pages:
        raise SystemExit(
            f"No saved eval instances under {EVAL_INSTANCES_DIR}.\n"
            "Run a paid sweep first:\n"
            "  python3 -m evals run-evals\n"
            "That builds the dashboard and refreshes the index "
            "(newest run at the top)."
        )
    webbrowser.open(index_path.resolve().as_uri())
    print(f"Opened {index_path}")
    return index_path


def _elapsed_s(job: Job, now: float | None = None) -> float:
    if job.started_at is None:
        return 0.0
    end = job.finished_at if job.finished_at is not None else (now or time.monotonic())
    return max(0.0, end - job.started_at)


def _format_elapsed(seconds: float) -> str:
    s = int(seconds)
    if s < 60:
        return f"{s}s"
    m, rem = divmod(s, 60)
    if m < 60:
        return f"{m}m{rem:02d}s"
    h, m = divmod(m, 60)
    return f"{h}h{m:02d}m"


def spend_from_predictions(
    run_id: str,
    model: str,
    *,
    prefixes: tuple[str, ...] = ("a", "b"),
) -> float:
    """Sum billed USD from token fields already written into predictions.jsonl.

    Pass ``prefixes=("a",)`` for Pass A banks and ``prefixes=("b",)`` for
    Pass B cells so banked Pass A usage is not double-counted in the footer.
    """
    path = run_predictions_path(run_id)
    if not path.exists():
        return 0.0
    pricing = production_config.require_model_pricing(model)
    total = 0.0
    for rec in iter_jsonl(path, tolerate_truncated_final=True):
        for prefix in prefixes:
            inp = rec.get(f"{prefix}_input_tokens") or 0
            out = rec.get(f"{prefix}_output_tokens") or 0
            cached = min(
                int(rec.get(f"{prefix}_cached_tokens") or 0),
                int(inp),
            )
            uncached = int(inp) - cached
            total += (
                (uncached / 1e6) * pricing["input"]
                + (cached / 1e6) * pricing["cached_input"]
                + (out / 1e6) * pricing["output"]
            )
    return total


def count_completed_predictions(run_id: str) -> int:
    path = run_predictions_path(run_id)
    if not path.exists():
        return 0
    n = 0
    for rec in iter_jsonl(path, tolerate_truncated_final=True):
        if rec.get("status") == "completed":
            n += 1
    return n


def cell_already_scored(run_id: str, expected_n: int) -> bool:
    """True only when this run was scored for exactly *expected_n* rows.

    Exact match (not ≥) so a finished full matrix cannot be mistaken for a
    ``--limit N`` smoke, and vice versa.
    """
    path = run_scored_path(run_id)
    if not path.exists():
        return False
    try:
        data = json.loads(path.read_text(encoding="utf-8"))
    except json.JSONDecodeError:
        return False
    if int(data.get("n_scored") or 0) != expected_n:
        return False
    cfg_path = run_config_path(run_id)
    if cfg_path.exists():
        try:
            conf = json.loads(cfg_path.read_text(encoding="utf-8"))
        except json.JSONDecodeError:
            return False
        if conf.get("n_rows") is not None and int(conf["n_rows"]) != expected_n:
            return False
    return True


def mint_cell_run_id(
    model: str,
    effort_b: str,
    *,
    repeat: int = 1,
    date: str | None = None,
) -> str:
    """Mint an unused classification run_id (never reuses a prior cell dir)."""
    day = date or _dt.date.today().isoformat()
    r = repeat
    while True:
        run_id = f"{day}_classification_{model}_{effort_b}_r{r}"
        if not (RUNS_DIR / run_id).exists():
            return run_id
        r += 1


def build_job_plan(
    n_rows: int,
    *,
    date: str | None = None,
    repeat: int = 1,
) -> list[Job]:
    """Build the checklist rows (3 Pass A + 9 cells + dashboard).

    Every call mints fresh cell run ids so ``run-evals`` always pays for a
    new matrix instead of resuming scored dirs from a prior invocation.
    """
    jobs: list[Job] = []
    for model in cfg.EVAL_MODELS:
        short = model.split("-")[-1]
        jobs.append(
            Job(
                key=f"pass_a:{model}",
                label=f"Pass A bank ({short})",
                kind="pass_a",
                model=model,
                run_id=pass_a_bank_run_id(model),
                rows_total=n_rows,
            )
        )
    for model, effort in matrix_cells():
        short = model.split("-")[-1]
        run_id = mint_cell_run_id(
            model, effort, repeat=repeat, date=date,
        )
        jobs.append(
            Job(
                key=f"cell:{model}:{effort}",
                label=f"{short} / {effort}",
                kind="cell",
                model=model,
                effort_b=effort,
                run_id=run_id,
                rows_total=n_rows,
            )
        )
    jobs.append(
        Job(
            key="dashboard",
            label="Build dashboard + archive index",
            kind="dashboard",
            rows_total=1,
        )
    )
    return jobs


def refresh_job_progress(job: Job) -> None:
    """Update rows_done / spend from disk artifacts for a job."""
    if job.kind == "dashboard":
        if job.status == "done":
            job.rows_done = 1
        return
    if job.run_id is None or job.model is None:
        return
    job.rows_done = count_completed_predictions(job.run_id)
    if job.kind == "pass_a":
        job.spend_usd = spend_from_predictions(
            job.run_id, job.model, prefixes=("a",),
        )
    elif job.kind == "cell":
        # Pass B only: Pass A tokens were already billed on the bank job.
        job.spend_usd = spend_from_predictions(
            job.run_id, job.model, prefixes=("b",),
        )
    if job.kind == "cell" and job.status in ("done", "skipped"):
        # Prefer scored.json coverage once scoring finished.
        if cell_already_scored(job.run_id, job.rows_total):
            job.rows_done = job.rows_total


def build_status_table(jobs: list[Job], *, now: float | None = None) -> Table:
    """Pure Rich table for the live checklist (unit-testable without a TTY)."""
    table = Table(
        title="Eval matrix progress",
        show_header=True,
        header_style="bold",
        expand=True,
    )
    table.add_column("Step", ratio=3)
    table.add_column("Status", width=10)
    table.add_column("Rows", justify="right", width=12)
    table.add_column("Elapsed", justify="right", width=10)
    table.add_column("Spend", justify="right", width=10)

    clock = now if now is not None else time.monotonic()
    for job in jobs:
        style, mark = _STATUS_STYLE[job.status]
        status_text = Text(f"{mark} {job.status}", style=style)
        rows = (
            f"{job.rows_done}/{job.rows_total}"
            if job.kind != "dashboard"
            else ("1/1" if job.status in ("done", "skipped") else "0/1")
        )
        spend = f"${job.spend_usd:.4f}" if job.kind != "dashboard" else "—"
        table.add_row(
            job.label,
            status_text,
            rows,
            _format_elapsed(_elapsed_s(job, clock)),
            spend,
        )

    total_spend = sum(j.spend_usd for j in jobs if j.kind != "dashboard")
    done = sum(1 for j in jobs if j.status in ("done", "skipped"))
    failed = sum(1 for j in jobs if j.status == "failed")
    footer = (
        f"{done}/{len(jobs)} steps green · "
        f"{failed} failed · "
        f"spend so far ${total_spend:.4f}"
    )
    table.caption = footer
    return table


def phase1_complete(jobs: list[Job]) -> bool:
    """True when every Pass A bank is done or skipped."""
    phase1 = [j for j in jobs if j.kind == "pass_a"]
    return all(j.status in ("done", "skipped") for j in phase1) and bool(phase1)


def phase1_failed(jobs: list[Job]) -> bool:
    return any(j.status == "failed" for j in jobs if j.kind == "pass_a")


def cells_all_terminal(jobs: list[Job]) -> bool:
    cells = [j for j in jobs if j.kind == "cell"]
    return all(j.status in ("done", "skipped", "failed") for j in cells)


def any_cell_failed(jobs: list[Job]) -> bool:
    return any(j.status == "failed" for j in jobs if j.kind == "cell")


def should_block_dashboard(jobs: list[Job]) -> bool:
    """Dashboard must not build from a partial/failed matrix."""
    return phase1_failed(jobs) or any_cell_failed(jobs) or not all(
        j.status in ("done", "skipped")
        for j in jobs
        if j.kind in ("pass_a", "cell")
    )


def _python_cmd() -> list[str]:
    return [sys.executable, "-m", "evals"]


def _spawn(cmd: list[str], log_path: Path) -> tuple[subprocess.Popen[Any], Any]:
    log_path.parent.mkdir(parents=True, exist_ok=True)
    log_f = log_path.open("a", encoding="utf-8")
    log_f.write(f"\n--- $ {' '.join(cmd)}\n")
    log_f.flush()
    proc = subprocess.Popen(
        cmd,
        stdout=log_f,
        stderr=subprocess.STDOUT,
        text=True,
        cwd=str(PROJECT_ROOT),
    )
    return proc, log_f


def _close_log(job: Job) -> None:
    handle = job.log_handle
    job.log_handle = None
    if handle is None:
        return
    try:
        handle.close()
    except Exception:
        pass


def _mark_running(job: Job) -> None:
    job.status = "running"
    job.started_at = time.monotonic()
    job.finished_at = None
    job.error = None


def _mark_done(job: Job) -> None:
    job.status = "done"
    job.finished_at = time.monotonic()
    job.returncode = 0
    refresh_job_progress(job)


def _mark_failed(job: Job, rc: int, detail: str | None = None) -> None:
    job.status = "failed"
    job.finished_at = time.monotonic()
    job.returncode = rc
    job.error = detail or f"exit {rc}"
    refresh_job_progress(job)


def _mark_skipped(job: Job) -> None:
    job.status = "skipped"
    job.started_at = job.started_at or time.monotonic()
    job.finished_at = time.monotonic()
    job.returncode = 0
    refresh_job_progress(job)
    if job.kind != "dashboard":
        job.rows_done = job.rows_total


def _start_pass_a(job: Job, limit: int | None) -> None:
    assert job.model is not None and job.run_id is not None
    # Always rebuild the stable bank: run-evals is a full paid matrix, not a
    # resume of yesterday's Pass A.
    cmd = _python_cmd() + ["bank-pass-a", "--model", job.model, "--rerun"]
    if limit is not None:
        cmd += ["--limit", str(limit)]
    log_path = run_dir(job.run_id) / "run.log"
    _mark_running(job)
    job.process, job.log_handle = _spawn(cmd, log_path)


def _start_cell_classify(job: Job, limit: int | None) -> None:
    assert job.model and job.effort_b and job.run_id
    cmd = _python_cmd() + [
        "run-classification",
        "--model", job.model,
        "--effort-b", job.effort_b,
        "--require-matrix-cell",
        "--run-id", job.run_id,
    ]
    if limit is not None:
        cmd += ["--limit", str(limit)]
    log_path = run_dir(job.run_id) / "run.log"
    _mark_running(job)
    job.cell_stage = "classify"
    job.process, job.log_handle = _spawn(cmd, log_path)


def _start_cell_score(job: Job) -> None:
    assert job.run_id
    # allow-partial / allow-missing: mini/luna often return a one-sided
    # {0,1} top_logprobs pool even at depth 5, so calibration may cover
    # only a subset (or none). Accuracy scoring must still finish.
    cmd = _python_cmd() + [
        "score", job.run_id,
        "--confidence-from-raw",
        "--allow-partial-confidence",
        "--allow-missing-confidence",
    ]
    log_path = run_dir(job.run_id) / "run.log"
    job.cell_stage = "score"
    # Keep status=running; do not reset started_at (elapsed covers classify+score).
    _close_log(job)
    job.process, job.log_handle = _spawn(cmd, log_path)


def cell_needs_robustness_refresh(run_id: str) -> bool:
    """True when scored.json is missing the valid_mass robustness block.

    Cells skipped as already-scored can predate a later scoring path that
    writes ``robustness.valid_mass``. Re-score is offline and cheap.
    """
    path = run_scored_path(run_id)
    if not path.exists():
        return False
    try:
        data = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, ValueError, json.JSONDecodeError):
        return True
    rob = data.get("robustness") if isinstance(data, dict) else None
    return not isinstance(rob, dict) or not rob.get("valid_mass")


def refresh_cell_scores_for_dashboard(jobs: list[Job]) -> None:
    """Offline re-score of finished cells so robustness blocks are current."""
    for job in jobs:
        if job.kind != "cell" or job.status not in ("done", "skipped"):
            continue
        if not job.run_id:
            continue
        if not cell_needs_robustness_refresh(job.run_id):
            continue
        cmd = _python_cmd() + [
            "score", job.run_id,
            "--confidence-from-raw",
            "--allow-partial-confidence",
            "--allow-missing-confidence",
        ]
        log_path = run_dir(job.run_id) / "run.log"
        log_path.parent.mkdir(parents=True, exist_ok=True)
        with log_path.open("a", encoding="utf-8") as log_f:
            log_f.write(f"\n--- $ {' '.join(cmd)}  # robustness refresh\n")
            log_f.flush()
            subprocess.run(
                cmd,
                stdout=log_f,
                stderr=subprocess.STDOUT,
                text=True,
                cwd=str(PROJECT_ROOT),
                check=False,
            )


def _start_dashboard(job: Job, cell_run_ids: list[str]) -> None:
    cmd = _python_cmd() + ["dashboard", "--runs", *cell_run_ids]
    log_path = PROJECT_ROOT / "evals" / "runs" / "_orchestrate_dashboard.log"
    _mark_running(job)
    job.process, job.log_handle = _spawn(cmd, log_path)


def _handle_finished_ok(job: Job) -> None:
    """Advance a job whose current subprocess exited 0."""
    if job.kind == "pass_a":
        _mark_done(job)
        return
    if job.kind == "dashboard":
        _mark_done(job)
        job.rows_done = 1
        return
    # Cell: classify -> score -> done.
    if job.cell_stage == "classify":
        _start_cell_score(job)
        return
    _mark_done(job)


def _handle_finished(job: Job, rc: int) -> None:
    """Advance or fail a job after its subprocess exits."""
    if rc != 0:
        _mark_failed(job, rc)
        return
    _handle_finished_ok(job)


def run_evals(
    *,
    yes: bool = False,
    limit: int | None = None,
    console: Console | None = None,
    confirm_fn: Callable[[str], str] | None = None,
    poll_seconds: float = 1.0,
) -> int:
    """Run the full paid matrix from scratch. Returns process exit code.

    Every invocation rebuilds Pass A banks and mints new cell run ids so
    prior scored dirs are never reused. Re-paying is intentional: the
    command is a full matrix, not a resume helper.
    """
    require_openai_key()

    rows = load_golden_rows()
    if limit is not None:
        if limit < 1:
            raise SystemExit(f"--limit must be a positive row cap, got {limit}")
        rows = rows[:limit]
    n_rows = len(rows)

    estimate = print_matrix_preview(rows)
    print()
    if not yes:
        ask = confirm_fn or input
        answer = ask(
            f"Proceed with paid run (~${estimate.total_cost:.4f})? [y/N] "
        ).strip().lower()
        if answer not in {"y", "yes"}:
            print("Aborted. No API calls made.")
            return 1

    jobs = build_job_plan(n_rows)

    con = console or Console()
    # Launch phase-1 banks that still need work.
    for job in jobs:
        if job.kind == "pass_a" and job.status == "pending":
            _start_pass_a(job, limit)

    phase2_started = False
    dashboard_started = False
    exit_code = 0

    with Live(
        build_status_table(jobs),
        console=con,
        refresh_per_second=4,
        transient=False,
    ) as live:
        while True:
            for job in jobs:
                if job.process is None:
                    continue
                finished = job.process.poll() is not None
                if not finished:
                    refresh_job_progress(job)
                    continue
                rc = job.process.wait()
                job.process = None
                _close_log(job)
                _handle_finished(job, rc)

            # Start phase 2 once all banks are green.
            if not phase2_started and phase1_complete(jobs):
                phase2_started = True
                for job in jobs:
                    if job.kind == "cell" and job.status == "pending":
                        _start_cell_classify(job, limit)
            elif not phase2_started and phase1_failed(jobs):
                # Banks crashed: mark remaining pending cells failed.
                for job in jobs:
                    if job.status == "pending":
                        _mark_failed(
                            job, 1, "blocked: Pass A bank failed",
                        )
                exit_code = 1

            # Dashboard when every cell is terminal and none failed.
            if (
                phase2_started
                and cells_all_terminal(jobs)
                and not dashboard_started
            ):
                dash = next(j for j in jobs if j.kind == "dashboard")
                if should_block_dashboard(jobs):
                    _mark_failed(
                        dash, 1,
                        "blocked: one or more cells failed or incomplete",
                    )
                    dashboard_started = True
                    exit_code = 1
                else:
                    cell_ids = [
                        j.run_id for j in jobs
                        if j.kind == "cell" and j.run_id
                    ]
                    refresh_cell_scores_for_dashboard(jobs)
                    dashboard_started = True
                    _start_dashboard(dash, cell_ids)

            live.update(build_status_table(jobs))

            # Terminal: every job done/skipped/failed and no live processes.
            if all(
                j.status in ("done", "skipped", "failed") for j in jobs
            ) and all(j.process is None for j in jobs):
                break
            time.sleep(poll_seconds)

    live_final = build_status_table(jobs)
    con.print(live_final)

    if any(j.status == "failed" for j in jobs):
        exit_code = 1
        con.print(
            "[red]Eval matrix unfinished.[/red] "
            "Re-run [bold]python -m evals run-evals[/bold] to start a "
            "fresh full matrix (prior cells are not resumed)."
        )
        failed = [j for j in jobs if j.status == "failed"]
        for j in failed:
            log_hint = ""
            if j.run_id:
                log_hint = f" (log: {run_dir(j.run_id) / 'run.log'})"
            con.print(f"  ✗ {j.label}: {j.error or 'failed'}{log_hint}")
        return exit_code

    con.print(
        "[green]Eval matrix complete.[/green] "
        "Open the archive index with "
        "[bold]python -m evals open-dashboard[/bold]."
    )
    return 0
