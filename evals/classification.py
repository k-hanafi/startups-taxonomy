"""Stage 5: Pass A/B classification runner over the golden set.

Pass A (internal reasoning off, logprobs on) answers the binary "AI-native or
not?" and records its explicit analysis. The decision field remains first so
the confidence signal stays on a structurally located 0/1 token. Pass B
(reasoning enabled, no logprobs) assigns the fine-grained subclass,
hard-constrained by the response schema to the family Pass A chose, plus RAD
when the family is AI-native. Cohort never touches an LLM: it is a pure
function of founded_date.

Empirical basis (see the split-reasoning plan): logprobs and reasoning are
mutually exclusive per request, binary accuracy survives without reasoning
(93% vs Fable either way), and 10-way subclass accuracy does not (41% vs 66%).

Uses tenacity retries, per-row resume keyed by custom_id, production request
fingerprints, and raw responses banked per pass. A row is complete only when
both passes succeed and validate against the production schemas.

Pass B effort sweeps bank Pass A once per model by default into
``evals/runs/pass_a_banks/<model>/``. Later efforts auto-reuse that bank so
effort deltas are not confounded by resampling the gate. Escape hatches:
``--rerun-pass-a`` (invalidate / rebuild) and ``--pass-a-from`` (pin a
historical run). Without a bank, a crash between passes still re-runs Pass A
on resume of the cell run.
"""

from __future__ import annotations

import datetime
import json
import logging
import shutil
import time
from types import SimpleNamespace
from typing import Any, Optional

from openai import OpenAI

from two_pass_classifier import config as production_config
from two_pass_classifier.formatter import build_custom_id
from two_pass_classifier.cohort import compute_cohort
from two_pass_classifier.confidence import (
    BinaryConfidenceUnavailable,
    LogprobExtractionError,
    extract_binary_confidence,
)
from two_pass_classifier.exporter import stable_source_union
from two_pass_classifier.request_builder import (
    RequestSettings,
    build_pass_a_request,
    build_pass_b_request,
    pass_a_request_fingerprint,
    pass_a_request_identity,
    request_fingerprint,
    request_identity,
)
from two_pass_classifier.schema import (
    PassAResult,
    PassBAINativeResult,
    PassBNotAINativeResult,
)

from evals import config as cfg
from evals.jsonl_io import append_jsonl, iter_jsonl
from evals.paths import (
    pass_a_bank_run_id,
    run_config_path,
    run_dir,
    run_predictions_path,
    run_raw_dir,
)
from evals.runner import (
    _RETRIABLE,
    _completed_custom_ids,
    _git_commit,
    load_golden_rows,
)

logger = logging.getLogger(__name__)


def _openai_client() -> OpenAI:
    """Open a paid client. Key lookup happens here, not at import time."""
    from two_pass_classifier.api_key import load_real_api_key

    return OpenAI(api_key=load_real_api_key())


BinaryResult = PassAResult
SubclassResultAI = PassBAINativeResult
SubclassResultNot = PassBNotAINativeResult


def request_settings(model: str, effort_b: str) -> RequestSettings:
    """Build the exact production request settings for one eval cell."""
    return RequestSettings(model=model, pass_b_effort=effort_b)


def pass_a_kwargs(row: dict[str, Any], model: str) -> dict[str, Any]:
    """Thin adapter to the production Pass A request builder."""
    settings = request_settings(model, production_config.DEFAULT_PASS_B_EFFORT)
    return build_pass_a_request(row, settings)


def pass_b_kwargs(
    row: dict[str, Any],
    verdict: int,
    model: str,
    effort_b: str,
) -> dict[str, Any]:
    """Thin adapter to the production Pass B request builder."""
    return build_pass_b_request(
        row,
        verdict,
        request_settings(model, effort_b),
    )


def production_request_metadata(model: str, effort_b: str) -> dict[str, Any]:
    """Return production-owned request identities for an eval run."""
    settings = request_settings(model, effort_b)
    return {
        "semantic_request_fingerprint": request_fingerprint(settings),
        "request_identity": request_identity(settings),
        "pass_a_request_fingerprint": pass_a_request_fingerprint(settings),
        "pass_a_request_identity": pass_a_request_identity(settings),
    }


# ---------------------------------------------------------------------------
# Record assembly
# ---------------------------------------------------------------------------

def _usage_fields(resp: Any, prefix: str) -> dict[str, Any]:
    from evals.usage import cached_tokens_from_usage

    usage = getattr(resp, "usage", None)
    reasoning = None
    if usage is not None and getattr(usage, "output_tokens_details", None) is not None:
        reasoning = getattr(usage.output_tokens_details, "reasoning_tokens", None)
    return {
        f"{prefix}_input_tokens": getattr(usage, "input_tokens", None) if usage else None,
        f"{prefix}_output_tokens": getattr(usage, "output_tokens", None) if usage else None,
        f"{prefix}_reasoning_tokens": reasoning,
        # 0 when usage/details absent; same semantics as single-pass runner.
        f"{prefix}_cached_tokens": cached_tokens_from_usage(usage),
    }


def _parse_output(resp: Any) -> Optional[dict[str, Any]]:
    text = getattr(resp, "output_text", "") or ""
    if not text:
        return None
    try:
        return json.loads(text)
    except json.JSONDecodeError:
        return None


def _validated_output(
    resp: Any,
    result_cls: type[PassAResult]
    | type[PassBAINativeResult]
    | type[PassBNotAINativeResult],
) -> Optional[dict[str, Any]]:
    parsed = _parse_output(resp)
    if parsed is None:
        return None
    try:
        return result_cls.model_validate(parsed).model_dump(mode="json")
    except (TypeError, ValueError):
        return None


def _sampled_label_confidence(
    resp: Any,
    verdict: int,
) -> tuple[float | None, str]:
    raw = resp.model_dump()
    try:
        confidence = extract_binary_confidence(raw)
    except BinaryConfidenceUnavailable:
        return None, "unavailable"
    except LogprobExtractionError:
        return None, "invalid"
    if confidence.ai_native != verdict:
        return None, "verdict_mismatch"
    return confidence.sampled_probability, "available"


def _validated_pass_a(
    resp: Any,
) -> tuple[Optional[dict[str, Any]], float | None, str | None]:
    parsed = _validated_output(resp, PassAResult)
    if parsed is None:
        return None, None, None
    confidence, status = _sampled_label_confidence(
        resp,
        int(parsed["ai_native"]),
    )
    if status in {"invalid", "verdict_mismatch"}:
        return None, None, status
    return parsed, confidence, status


def assemble_record(custom_id: str, org_uuid: str, model: str, effort_b: str,
                    cohort: str, resp_a: Any, resp_b: Any,
                    latency_a_s: float | None = None,
                    latency_b_s: float | None = None) -> dict[str, Any]:
    """One predictions.jsonl line combining both passes.

    status is 'completed' only when both passes completed AND parsed; resume
    treats anything else as unfinished and re-runs the row.
    """
    a, confidence, confidence_status = _validated_pass_a(resp_a)
    verdict = a.get("ai_native") if a else None
    result_b_cls = (
        PassBAINativeResult if verdict == 1 else PassBNotAINativeResult
    )
    b = (
        _validated_output(resp_b, result_b_cls)
        if resp_b is not None and verdict in (0, 1)
        else None
    )

    status_a = getattr(resp_a, "status", None)
    status_b = getattr(resp_b, "status", None) if resp_b is not None else None
    ok = status_a == "completed" and status_b == "completed" and a is not None and b is not None

    if ok:
        status = "completed"
    else:
        # Never let a failed row carry status="completed" (resume would skip it
        # forever): an API-completed pass whose JSON did not parse, or a row
        # where Pass B never ran, must remain retryable.
        status = status_b or status_a or "failed"
        if status == "completed":
            status = "parse_failed"

    pass_a_sources = list(a.get("sources_used") or []) if a else []
    pass_b_sources = list(b.get("sources_used") or []) if b else []
    sources_used = (
        json.loads(stable_source_union(pass_a_sources, pass_b_sources))
        if a and b
        else []
    )
    record: dict[str, Any] = {
        "custom_id": custom_id,
        "org_uuid": org_uuid,
        "model": model,
        "effort_a": production_config.PASS_A_EFFORT,
        "effort_b": effort_b,
        "status": status,
        "status_a": status_a,
        "status_b": status_b,
        "ai_native": verdict,
        "subclass": b.get("subclass") if b else None,
        # RAD is structural for the zero family: not a model opinion.
        "rad_score": (b.get("rad_score") if verdict == 1 else "RAD-NA") if b else None,
        "cohort": cohort,
        "ai_native_confidence": confidence,
        "confidence_extraction_status": confidence_status,
        "subclass_confidence": b.get("subclass_confidence") if b else None,
        "rad_confidence": (
            b.get("rad_confidence") if verdict == 1 and b else None
        ),
        "sources_used": sources_used,
        "ai_native_reasoning": a.get("ai_native_reasoning") if a else None,
        "ai_native_critique": a.get("ai_native_critique") if a else None,
        "pass_a_sources_used": pass_a_sources,
        "subclass_reasoning": b.get("subclass_reasoning") if b else None,
        "rad_reasoning": b.get("rad_reasoning") if b else None,
        "pass_b_sources_used": pass_b_sources,
        "subclass_critique": b.get("subclass_critique") if b else None,
        "rad_critique": (
            b.get("rad_critique") if verdict == 1 and b else None
        ),
    }
    record.update(_usage_fields(resp_a, "a"))
    record["a_latency_s"] = latency_a_s
    if resp_b is not None:
        record.update(_usage_fields(resp_b, "b"))
        record["b_latency_s"] = latency_b_s
    # Flat totals under the single-pass field names so the scorer reads one
    # field for both run shapes. A row missing Pass B has no meaningful
    # end-to-end latency (it will be re-run), so the total stays None.
    if latency_a_s is not None and latency_b_s is not None:
        record["latency_s"] = round(latency_a_s + latency_b_s, 3)
    else:
        record["latency_s"] = None
    # Cached tokens always sum A+B (0 when a pass is missing).
    record["cached_tokens"] = int(record.get("a_cached_tokens") or 0) + int(
        record.get("b_cached_tokens") or 0
    )
    return record


# ---------------------------------------------------------------------------
# The run engine
# ---------------------------------------------------------------------------

def make_run_id(model: str, effort_b: str, repeat: int) -> str:
    date = datetime.date.today().isoformat()
    return f"{date}_classification_{model}_{effort_b}_r{repeat}"


@_RETRIABLE
def _create(client: OpenAI, kwargs: dict[str, Any]) -> Any:
    return client.responses.create(**kwargs)


# Resume refuses to mix changed prompts/schemas/model into an existing run.
_RESUME_INVARIANTS = (
    "model", "effort_b", "repeat", "n_rows",
    "semantic_request_fingerprint", "pass_a_request_fingerprint",
)


def _ensure_config(run_id: str, model: str, effort_b: str, repeat: int,
                   n_rows: int,
                   pass_a_bank_run_id: str | None = None) -> None:
    settings = request_settings(model, effort_b)
    run_config = {
        "run_id": run_id,
        "kind": "classification",
        "model": model,
        "effort_a": production_config.PASS_A_EFFORT,
        "effort_b": effort_b,
        "repeat": repeat,
        "n_rows": n_rows,
        "top_logprobs": production_config.PASS_A_TOP_LOGPROBS,
        "pass_a_max_output_tokens": settings.pass_a_max_output_tokens,
        "pass_b_max_output_tokens": settings.pass_b_max_output_tokens,
        "pass_a_cache_key": production_config.PASS_A_CACHE_KEY,
        "pass_b_cache_keys": dict(production_config.PASS_B_CACHE_KEYS),
        "pass_a_bank_run_id": pass_a_bank_run_id,
        "git_commit": _git_commit(),
        "created_utc": datetime.datetime.now(datetime.timezone.utc).isoformat(),
        **production_request_metadata(model, effort_b),
    }
    path = run_config_path(run_id)
    if not path.exists():
        path.write_text(json.dumps(run_config, indent=2), encoding="utf-8")
        return
    prior = json.loads(path.read_text(encoding="utf-8"))
    mismatched = [
        key
        for key in _RESUME_INVARIANTS
        if prior.get(key) != run_config[key]
    ]
    if mismatched:
        raise SystemExit(
            f"Cannot resume classification run {run_id}: {mismatched} changed since it "
            "started. Start a fresh run with a new --run-id."
        )
    if prior.get("pass_a_bank_run_id") != pass_a_bank_run_id:
        raise SystemExit(
            f"Cannot resume classification run {run_id}: pass_a_bank_run_id changed "
            f"({prior.get('pass_a_bank_run_id')!r} -> {pass_a_bank_run_id!r})."
        )


def _index_banked_pass_a(bank_run_id: str) -> dict[str, dict[str, Any]]:
    """Load completed Pass A rows that have raw on disk (empty if none).

    Soft index for resume / coverage checks: does not raise on empty or
    partial banks. ``load_pass_a_bank`` wraps this and fails loudly when
    the bank is empty or missing raw for a completed prediction row.
    """
    preds_path = run_predictions_path(bank_run_id)
    if not preds_path.exists():
        return {}
    bank_config: dict[str, Any] = {}
    config_path = run_config_path(bank_run_id)
    if config_path.exists():
        bank_config = json.loads(config_path.read_text(encoding="utf-8"))

    by_cid: dict[str, dict[str, Any]] = {}
    for rec in iter_jsonl(preds_path, tolerate_truncated_final=True):
        if rec.get("status") != "completed":
            continue
        cid = rec.get("custom_id")
        verdict = rec.get("ai_native")
        if not cid or verdict not in (0, 1):
            continue
        by_cid[cid] = rec

    raw_root = run_raw_dir(bank_run_id)
    out: dict[str, dict[str, Any]] = {}
    for cid, rec in by_cid.items():
        raw_path = raw_root / f"{cid}_a.json"
        if not raw_path.exists():
            continue
        raw_a = json.loads(raw_path.read_text(encoding="utf-8"))
        validated, _, _ = _validated_pass_a(_BankedPassAResponse(raw_a))
        if (
            validated is None
            or validated["ai_native"] != int(rec["ai_native"])
        ):
            continue
        out[cid] = {
            "ai_native": int(validated["ai_native"]),
            "raw_a": raw_a,
            "record": rec,
            "bank_model": bank_config.get("model") or rec.get("model"),
        }
    return out


def load_pass_a_bank(bank_run_id: str) -> dict[str, dict[str, Any]]:
    """Load banked Pass A verdicts + raw payloads keyed by custom_id.

    Returns custom_id -> {ai_native, raw_a (dict), usage latency fields from
    the banked prediction record}. Raises SystemExit if the bank is missing
    completed Pass A rows or raw ``*_a.json`` files.
    """
    _assert_bank_identity(bank_run_id)
    preds_path = run_predictions_path(bank_run_id)
    if not preds_path.exists():
        raise SystemExit(
            f"Pass A bank {bank_run_id!r} has no predictions at {preds_path}"
        )

    # Detect completed preds that lack raw (hard fail) vs soft index gaps.
    completed_cids: list[str] = []
    for rec in iter_jsonl(preds_path, tolerate_truncated_final=True):
        if rec.get("status") != "completed":
            continue
        cid = rec.get("custom_id")
        if cid and rec.get("ai_native") in (0, 1):
            completed_cids.append(cid)

    if not completed_cids:
        raise SystemExit(
            f"Pass A bank {bank_run_id!r} has no completed rows with ai_native "
            "in {{0,1}}"
        )

    out = _index_banked_pass_a(bank_run_id)
    missing = [cid for cid in completed_cids if cid not in out]
    if missing:
        raw_root = run_raw_dir(bank_run_id)
        invalid = [
            cid for cid in missing if (raw_root / f"{cid}_a.json").exists()
        ]
        missing_raw = [cid for cid in missing if cid not in invalid]
        if invalid:
            raise SystemExit(
                f"Pass A bank {bank_run_id!r} has raw Pass A that no longer "
                f"validates against production PassAResult for "
                f"{len(invalid)} row(s), e.g. {invalid[0]}_a.json. "
                "Re-bank Pass A with --rerun-pass-a."
            )
        raise SystemExit(
            f"Pass A bank {bank_run_id!r} missing raw Pass A files for "
            f"{len(missing_raw)} row(s), e.g. {missing_raw[0]}_a.json "
            "(raw/ is machine-local; copy it with the bank run)."
        )
    return out


def pass_a_bank_covers(
    bank_run_id: str,
    custom_ids: list[str],
    *,
    index: dict[str, dict[str, Any]] | None = None,
) -> bool:
    """True when bank_run_id has completed Pass A + raw for every custom_id.

    Pass ``index`` to avoid rescanning the bank JSONL (load once, reuse).
    """
    have = index if index is not None else _index_banked_pass_a(bank_run_id)
    return all(cid in have for cid in custom_ids)


def clear_pass_a_bank(model: str) -> str:
    """Remove the stable per-model Pass A bank directory. Returns bank run_id."""
    bank_id = pass_a_bank_run_id(model)
    path = run_dir(bank_id)
    if path.exists():
        shutil.rmtree(path)
    return bank_id


def _ensure_pass_a_bank_config(bank_id: str, model: str, n_rows: int) -> None:
    """Write or validate the stable Pass A bank config snapshot."""
    metadata = production_request_metadata(
        model,
        production_config.DEFAULT_PASS_B_EFFORT,
    )
    settings = request_settings(
        model,
        production_config.DEFAULT_PASS_B_EFFORT,
    )
    bank_config = {
        "run_id": bank_id,
        "kind": "pass_a_bank",
        "model": model,
        "effort_a": production_config.PASS_A_EFFORT,
        "n_rows": n_rows,
        "top_logprobs": production_config.PASS_A_TOP_LOGPROBS,
        "pass_a_max_output_tokens": settings.pass_a_max_output_tokens,
        "pass_a_cache_key": production_config.PASS_A_CACHE_KEY,
        "git_commit": _git_commit(),
        "created_utc": datetime.datetime.now(datetime.timezone.utc).isoformat(),
        "pass_a_request_fingerprint": metadata["pass_a_request_fingerprint"],
        "pass_a_request_identity": metadata["pass_a_request_identity"],
    }
    path = run_config_path(bank_id)
    if not path.exists():
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(json.dumps(bank_config, indent=2), encoding="utf-8")
        return
    prior = json.loads(path.read_text(encoding="utf-8"))
    for key in ("model", "n_rows", "pass_a_request_fingerprint"):
        if prior.get(key) != bank_config[key]:
            raise SystemExit(
                f"Cannot extend Pass A bank {bank_id!r}: {key} changed "
                f"({prior.get(key)!r} -> {bank_config[key]!r}). "
                "Pass --rerun-pass-a "
                "to rebuild the bank."
            )


def _persist_pass_a_bank_row(
    bank_id: str,
    model: str,
    cid: str,
    org_uuid: str,
    resp_a: Any,
    latency_a_s: float | None,
    raw_a: dict[str, Any],
    *,
    index: dict[str, dict[str, Any]],
) -> None:
    """Append one completed Pass A row into the stable bank (idempotent skip).

    ``index`` is the in-memory bank map (updated after a successful append)
    so callers never rescan a growing JSONL.
    """
    if cid in index:
        return
    a, _, _ = _validated_pass_a(resp_a)
    if a is None or a.get("ai_native") not in (0, 1):
        return
    run_raw_dir(bank_id).mkdir(parents=True, exist_ok=True)
    (run_raw_dir(bank_id) / f"{cid}_a.json").write_text(
        json.dumps(raw_a, ensure_ascii=False), encoding="utf-8"
    )
    record = {
        "custom_id": cid,
        "org_uuid": org_uuid,
        "model": model,
        "status": "completed",
        "ai_native": a["ai_native"],
        "a_latency_s": latency_a_s,
        **_usage_fields(resp_a, "a"),
    }
    append_jsonl(run_predictions_path(bank_id), record)
    bank_config: dict[str, Any] = {}
    config_path = run_config_path(bank_id)
    if config_path.exists():
        bank_config = json.loads(config_path.read_text(encoding="utf-8"))
    index[cid] = {
        "ai_native": int(a["ai_native"]),
        "raw_a": raw_a,
        "record": record,
        "bank_model": bank_config.get("model") or model,
    }


def _assert_bank_identity(
    bank_id: str,
    expected_model: str | None = None,
) -> None:
    """Refuse any bank not built from the current production Pass A."""
    path = run_config_path(bank_id)
    if not path.exists():
        raise SystemExit(
            f"Pass A bank {bank_id!r} has no config.json. Historical banks "
            "without a production fingerprint cannot be reused."
        )
    prior = json.loads(path.read_text(encoding="utf-8"))
    model = prior.get("model")
    if not model:
        raise SystemExit(
            f"Pass A bank {bank_id!r} has no model in config.json and cannot "
            "prove production compatibility."
        )
    if expected_model is not None and model != expected_model:
        raise SystemExit(
            f"Pass A bank {bank_id!r} was run with model {model!r}, but this "
            f"run requested {expected_model!r}. Bank Pass A once per model."
        )
    try:
        expected = production_request_metadata(
            str(model),
            production_config.DEFAULT_PASS_B_EFFORT,
        )["pass_a_request_fingerprint"]
    except ValueError as exc:
        raise SystemExit(str(exc)) from exc
    actual = prior.get("pass_a_request_fingerprint")
    if actual != expected:
        raise SystemExit(
            f"Pass A bank {bank_id!r} has an incompatible or missing "
            "production Pass A fingerprint. Rebuild it with --rerun-pass-a. "
            "Historical paid artifacts remain readable but cannot be reused."
        )


def resolve_pass_a_source(
    model: str,
    custom_ids: list[str],
    *,
    pass_a_from: str | None = None,
    rerun_pass_a: bool = False,
) -> tuple[str, dict[str, dict[str, Any]] | None, bool]:
    """Decide which Pass A bank to use.

    Returns (bank_id, bank_or_None, creating).
    bank is loaded when reusing; None when this run must call Pass A and
    persist into the stable bank. creating is True only for the stable-bank
    write path (not for --pass-a-from pins).

    Side-effect free: never deletes or writes a bank. Callers that honor
    ``--rerun-pass-a`` must clear the bank themselves after any dry-run gate.
    """
    if pass_a_from and rerun_pass_a:
        raise SystemExit(
            "Pass only one of --pass-a-from and --rerun-pass-a "
            "(cannot pin a historical bank and rebuild at once)."
        )
    if pass_a_from:
        _assert_bank_identity(pass_a_from, model)
        bank = load_pass_a_bank(pass_a_from)
        missing = [cid for cid in custom_ids if cid not in bank]
        if missing:
            raise SystemExit(
                f"Pass A bank {pass_a_from!r} has no completed row for "
                f"{missing[0]}; refuse partial reuse (science confound)."
            )
        return pass_a_from, bank, False

    bank_id = pass_a_bank_run_id(model)
    if rerun_pass_a:
        # Side-effect free here: caller clears the bank only when not dry-run.
        return bank_id, None, True

    if (
        run_config_path(bank_id).exists()
        or run_predictions_path(bank_id).exists()
    ):
        _assert_bank_identity(bank_id, model)
    if pass_a_bank_covers(bank_id, custom_ids):
        bank = load_pass_a_bank(bank_id)
        return bank_id, bank, False

    return bank_id, None, True


class _BankedPassAResponse:
    """Duck-typed Responses API object assembled from a banked raw dump."""

    def __init__(self, raw: dict[str, Any]):
        self._raw = raw
        self.status = raw.get("status", "completed")
        self.output_text = ""
        for item in raw.get("output") or []:
            if item.get("type") != "message":
                continue
            for content in item.get("content") or []:
                if content.get("type") == "output_text" and content.get("text"):
                    self.output_text = content["text"]
                    break
        usage = raw.get("usage") or {}
        details = usage.get("output_tokens_details") or {}
        cached_details = usage.get("input_tokens_details") or {}
        self.usage = SimpleNamespace(
            input_tokens=usage.get("input_tokens"),
            output_tokens=usage.get("output_tokens"),
            output_tokens_details=SimpleNamespace(
                reasoning_tokens=details.get("reasoning_tokens"),
            ),
            input_tokens_details=SimpleNamespace(
                cached_tokens=cached_details.get("cached_tokens"),
            ),
        )

    def model_dump(self) -> dict[str, Any]:
        return self._raw


def run_classification(model: str = cfg.DEFAULT_MODEL,
                 effort_b: str = cfg.DEFAULT_PASS_B_EFFORT,
                 repeat: int = 1,
                 limit: int | None = None,
                 dry_run: bool = False,
                 run_id: str | None = None,
                 pass_a_from: str | None = None,
                 rerun_pass_a: bool = False,
                 reuse_pass_a_from: str | None = None) -> str:
    """Run Pass A + Pass B over the golden set. Returns the run_id.

    Pass A is banked once per model under ``pass_a_banks/<model>/`` and
    reused by default for later Pass B efforts. ``--rerun-pass-a`` rebuilds
    that bank. ``--pass-a-from`` pins a historical bank (override).
    ``reuse_pass_a_from`` is a deprecated alias for ``pass_a_from``.
    """
    if limit is not None and limit < 1:
        raise ValueError(f"--limit must be a positive row cap, got {limit}")
    request_settings(model, effort_b)

    if reuse_pass_a_from and pass_a_from:
        raise SystemExit(
            "Pass only one of --pass-a-from and --reuse-pass-a-from "
            "(the latter is a deprecated alias)."
        )
    if reuse_pass_a_from:
        logger.warning(
            "--reuse-pass-a-from is deprecated. Pass A banks auto-reuse per "
            "model by default. Use --pass-a-from only to pin a historical bank."
        )
        pass_a_from = reuse_pass_a_from

    rows = load_golden_rows()
    if limit is not None:
        rows = rows[:limit]

    run_id = run_id or make_run_id(model, effort_b, repeat)
    custom_ids = [build_custom_id(r["org_uuid"]) for r in rows]

    bank_id, bank, creating_bank = resolve_pass_a_source(
        model, custom_ids,
        pass_a_from=pass_a_from,
        rerun_pass_a=rerun_pass_a,
    )

    if dry_run:
        _print_dry_run(
            rows, model, effort_b, run_id,
            pass_a_bank_id=bank_id if bank is not None else None,
        )
        return run_id

    # Disk mutation only after dry-run returns (resolve stays side-effect free).
    if rerun_pass_a:
        clear_pass_a_bank(model)

    run_dir(run_id).mkdir(parents=True, exist_ok=True)
    run_raw_dir(run_id).mkdir(parents=True, exist_ok=True)
    predictions_path = run_predictions_path(run_id)
    _ensure_config(
        run_id, model, effort_b, repeat, len(rows),
        pass_a_bank_run_id=bank_id,
    )

    # In-memory Pass A bank index: load once, update after each append.
    bank_index: dict[str, dict[str, Any]] = {}
    if creating_bank:
        run_dir(bank_id).mkdir(parents=True, exist_ok=True)
        run_raw_dir(bank_id).mkdir(parents=True, exist_ok=True)
        _ensure_pass_a_bank_config(bank_id, model, len(rows))
        bank_index = _index_banked_pass_a(bank_id)

    done = _completed_custom_ids(predictions_path)
    if done:
        logger.info("Resuming %s: %d rows already complete", run_id, len(done))

    client = _openai_client()
    todo = [r for r in rows if build_custom_id(r["org_uuid"]) not in done]
    if bank is not None:
        logger.info(
            "Two-pass run %s: %d rows (%s, B effort=%s, Pass A reused from %s)",
            run_id, len(todo), model, effort_b, bank_id,
        )
    else:
        logger.info(
            "Two-pass run %s: %d rows (%s, B effort=%s, banking Pass A to %s)",
            run_id, len(todo), model, effort_b, bank_id,
        )

    for i, row in enumerate(todo, start=1):
        cid = build_custom_id(row["org_uuid"])
        cohort = compute_cohort(row.get("founded_date", ""))

        if bank is not None:
            banked = bank.get(cid)
            if banked is None:
                raise SystemExit(
                    f"Pass A bank {bank_id!r} has no completed row "
                    f"for {cid}; refuse partial reuse (science confound)."
                )
            resp_a = _BankedPassAResponse(banked["raw_a"])
            latency_a_s = banked["record"].get("a_latency_s")
            (run_raw_dir(run_id) / f"{cid}_a.json").write_text(
                json.dumps(banked["raw_a"], ensure_ascii=False), encoding="utf-8"
            )
            a = {"ai_native": banked["ai_native"]}
        else:
            # Prefer a row already in the (partial) stable bank on resume.
            if cid in bank_index:
                banked = bank_index[cid]
                resp_a = _BankedPassAResponse(banked["raw_a"])
                latency_a_s = banked["record"].get("a_latency_s")
                raw_a = banked["raw_a"]
            else:
                # Wall-clock latency around each API call; retry backoff is
                # included, so this is the honest per-pass cost a production
                # caller would feel.
                started_a = time.monotonic()
                resp_a = _create(client, pass_a_kwargs(row, model))
                latency_a_s = round(time.monotonic() - started_a, 3)
                raw_a = resp_a.model_dump()
            (run_raw_dir(run_id) / f"{cid}_a.json").write_text(
                json.dumps(
                    raw_a if isinstance(raw_a, dict) else resp_a.model_dump(),
                    ensure_ascii=False,
                ),
                encoding="utf-8",
            )
            a, _, _ = _validated_pass_a(resp_a)
            if creating_bank and a is not None and a.get("ai_native") in (0, 1):
                _persist_pass_a_bank_row(
                    bank_id, model, cid, row["org_uuid"], resp_a, latency_a_s,
                    raw_a if isinstance(raw_a, dict) else resp_a.model_dump(),
                    index=bank_index,
                )

        resp_b = None
        latency_b_s = None
        if a is not None and a.get("ai_native") in (0, 1):
            started_b = time.monotonic()
            resp_b = _create(
                client,
                pass_b_kwargs(row, a["ai_native"], model, effort_b),
            )
            latency_b_s = round(time.monotonic() - started_b, 3)
            (run_raw_dir(run_id) / f"{cid}_b.json").write_text(
                json.dumps(resp_b.model_dump(), ensure_ascii=False), encoding="utf-8"
            )
        else:
            logger.warning("Pass A gave no usable verdict for %s (status=%s); "
                           "row will retry on resume", cid, getattr(resp_a, "status", None))

        record = assemble_record(
            cid, row["org_uuid"], model, effort_b, cohort, resp_a, resp_b,
            latency_a_s, latency_b_s,
        )
        record["pass_a_bank_run_id"] = bank_id
        append_jsonl(predictions_path, record)

        logger.info("  [%d/%d] %s -> A=%s B=%s/%s (%s)",
                    i, len(todo), str(row.get("name", ""))[:24],
                    record.get("ai_native"), record.get("subclass"),
                    record.get("rad_score"), record.get("status"))

    logger.info("Classification run %s complete: %s", run_id, predictions_path)
    return run_id


def bank_pass_a(
    model: str,
    limit: int | None = None,
    dry_run: bool = False,
    *,
    rerun: bool = False,
) -> str:
    """Bank Pass A for *model* only (no Pass B). Returns the bank run_id.

    Idempotent by default: skips custom_ids already completed in the stable
    bank so a re-run resumes. Used by ``run-evals`` phase 1 so all three
    models can bank in parallel before the 9 Pass B cells start. Pass
    ``rerun=True`` (CLI ``--rerun``) to delete the bank and rebuild it.
    """
    if limit is not None and limit < 1:
        raise ValueError(f"--limit must be a positive row cap, got {limit}")

    rows = load_golden_rows()
    if limit is not None:
        rows = rows[:limit]

    bank_id = pass_a_bank_run_id(model)
    request_settings(model, production_config.DEFAULT_PASS_B_EFFORT)
    custom_ids = [build_custom_id(r["org_uuid"]) for r in rows]

    if dry_run:
        from evals.cost_preview import estimate_pass_a

        est = estimate_pass_a(model, rows)
        logger.info(
            "DRY RUN Pass A bank %s: model=%s rows=%d "
            "~$%.4f (in ~%d + out ~%d)%s",
            bank_id, model, est.n_rows, est.est_total_cost,
            est.est_input_tokens, est.est_output_tokens,
            " [would rebuild]" if rerun else "",
        )
        return bank_id

    if rerun:
        clear_pass_a_bank(model)

    run_dir(bank_id).mkdir(parents=True, exist_ok=True)
    run_raw_dir(bank_id).mkdir(parents=True, exist_ok=True)
    _ensure_pass_a_bank_config(bank_id, model, len(rows))
    bank_index = _index_banked_pass_a(bank_id)

    todo = [
        r for r in rows
        if build_custom_id(r["org_uuid"]) not in bank_index
    ]
    if not todo and pass_a_bank_covers(bank_id, custom_ids, index=bank_index):
        logger.info(
            "Pass A bank %s already covers %d rows; nothing to do",
            bank_id, len(rows),
        )
        return bank_id

    client = _openai_client()
    logger.info(
        "Banking Pass A to %s: %d rows remaining (%s)",
        bank_id, len(todo), model,
    )
    for i, row in enumerate(todo, start=1):
        cid = build_custom_id(row["org_uuid"])
        started_a = time.monotonic()
        resp_a = _create(client, pass_a_kwargs(row, model))
        latency_a_s = round(time.monotonic() - started_a, 3)
        raw_a = resp_a.model_dump()
        a, _, _ = _validated_pass_a(resp_a)
        if a is not None and a.get("ai_native") in (0, 1):
            _persist_pass_a_bank_row(
                bank_id, model, cid, row["org_uuid"], resp_a, latency_a_s,
                raw_a, index=bank_index,
            )
            logger.info(
                "  [%d/%d] %s -> A=%s",
                i, len(todo), str(row.get("name", ""))[:24], a["ai_native"],
            )
        else:
            logger.warning(
                "Pass A gave no usable verdict for %s (status=%s); "
                "row will retry on resume",
                cid, getattr(resp_a, "status", None),
            )

    if not pass_a_bank_covers(bank_id, custom_ids):
        # Re-read disk: bank_index can disagree with pass_a_bank_covers if a
        # row landed on disk outside this process's index.
        on_disk = _index_banked_pass_a(bank_id)
        missing = [cid for cid in custom_ids if cid not in on_disk]
        example = missing[0] if missing else custom_ids[0]
        raise SystemExit(
            f"Pass A bank {bank_id!r} incomplete after run: "
            f"{len(missing)} row(s) still missing (e.g. {example}). "
            "Re-run bank-pass-a to resume."
        )
    logger.info("Pass A bank %s complete: %d rows", bank_id, len(rows))
    return bank_id


def matrix_cells() -> list[tuple[str, str]]:
    """Locked (model, Pass B effort) pairs in screen order."""
    return [
        (model, effort)
        for model in cfg.EVAL_MODELS
        for effort in cfg.MATRIX_PASS_B_EFFORTS
    ]


def validate_matrix_cell(model: str, effort_b: str) -> None:
    """Refuse unknown models/efforts so a typo cannot create an off-matrix paid run."""
    if model not in cfg.EVAL_MODELS:
        raise SystemExit(
            f"Unknown matrix model {model!r}. Locked EVAL_MODELS = "
            f"{cfg.EVAL_MODELS}"
        )
    if effort_b not in cfg.MATRIX_PASS_B_EFFORTS:
        raise SystemExit(
            f"Unknown matrix Pass B effort {effort_b!r}. Locked "
            f"MATRIX_PASS_B_EFFORTS = {cfg.MATRIX_PASS_B_EFFORTS}"
        )


def _print_dry_run(rows: list[dict[str, Any]], model: str,
                   effort_b: str, run_id: str,
                   pass_a_bank_id: str | None = None) -> None:
    # Single cost formula shared with ``python -m evals cost-preview``.
    from evals.cost_preview import estimate_cell

    include_pass_a = pass_a_bank_id is None
    est = estimate_cell(
        model, effort_b, rows, include_pass_a=include_pass_a
    )
    if pass_a_bank_id:
        logger.info(
            "DRY RUN %s (Pass A reused from %s; input+output for Pass B only)",
            run_id, pass_a_bank_id,
        )
    else:
        logger.info("DRY RUN %s (Pass A + Pass B; no bank yet)", run_id)
    logger.info("  model=%s pass A effort=%s, pass B effort=%s, rows=%d",
                model, production_config.PASS_A_EFFORT, effort_b, est.n_rows)
    logger.info(
        "  est input tokens ~%d (~$%.4f) + rough output/reasoning ~%d (~$%.4f) "
        "→ total ~$%.4f (output estimate is order-of-magnitude only)",
        est.est_input_tokens, est.est_input_cost,
        est.est_output_tokens, est.est_output_cost,
        est.est_total_cost,
    )
    if effort_b in ("medium", "high"):
        logger.info(
            "  WARNING: Pass B effort=%s can dominate spend via reasoning "
            "tokens; treat the output estimate as a floor, not a cap.",
            effort_b,
        )
    logger.info(
        "  production request fingerprint: %s",
        production_request_metadata(model, effort_b)[
            "semantic_request_fingerprint"
        ],
    )
