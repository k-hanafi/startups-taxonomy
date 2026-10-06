"""Normalize Responses API usage fields for eval run records.

Reads the cached-token field the Responses API returns, so eval cost math
and production cost math see the same signal.
Missing cache details are treated as 0 (not null): the API omits the
field when nothing was cached, and inventing a production hit rate would
be worse than under-counting.
"""

from __future__ import annotations

from typing import Any


def cached_tokens_from_usage(usage: Any) -> int:
    """Return ``cached_tokens`` from a Responses (or Chat) usage object.

    Offline-safe: ``None`` usage or missing details → ``0``.
    """
    if usage is None:
        return 0

    # Responses API (SDK object or dict): input_tokens_details.cached_tokens
    details = _get(usage, "input_tokens_details")
    if details is not None:
        return int(_get(details, "cached_tokens") or 0)

    # Chat Completions fallback: prompt_tokens_details.cached_tokens
    prompt_details = _get(usage, "prompt_tokens_details")
    if prompt_details is not None:
        return int(_get(prompt_details, "cached_tokens") or 0)

    return 0


def token_totals(record: dict[str, Any]) -> dict[str, int]:
    """input/output/reasoning/cached totals for one prediction record.

    Single-pass records use flat fields; classification records use a_/b_
    prefixes (summed). Missing fields default to 0. Callers that need
    "was cached_tokens recorded at all?" use
    ``evals.cost_extrapolate._records_have_cached_field`` separately.
    """
    if "a_input_tokens" in record:
        keys = {
            "input": ("a_input_tokens", "b_input_tokens"),
            "output": ("a_output_tokens", "b_output_tokens"),
            "reasoning": ("a_reasoning_tokens", "b_reasoning_tokens"),
            "cached": ("a_cached_tokens", "b_cached_tokens"),
        }
        return {
            kind: sum(int(record.get(k) or 0) for k in fields)
            for kind, fields in keys.items()
        }
    return {
        "input": int(record.get("input_tokens") or 0),
        "output": int(record.get("output_tokens") or 0),
        "reasoning": int(record.get("reasoning_tokens") or 0),
        "cached": int(record.get("cached_tokens") or 0),
    }


def _get(obj: Any, key: str) -> Any:
    if isinstance(obj, dict):
        return obj.get(key)
    return getattr(obj, key, None)
