"""Build stable production Responses API requests and fingerprints."""

from __future__ import annotations

import hashlib
import inspect
import json
import sys
from dataclasses import dataclass
from functools import lru_cache
from math import ceil
from pathlib import Path
from types import ModuleType
from typing import Any, Mapping

from pydantic import BaseModel

from . import (
    cohort as cohort_module,
    confidence as confidence_module,
    config,
    formatter as formatter_module,
    prompts as prompts_module,
    schema as schema_module,
)
from .formatter import (
    FORMATTER_VERSION,
    MAX_USER_MESSAGE_CHARS,
    format_input_message,
    format_pass_a_message,
    format_pass_b_message,
)
from .manifest import ManifestRow
from .prompts import load_pass_a_prompt, load_pass_b_prompt
from .schema import (
    PassAResult,
    PassBAINativeResult,
    PassBNotAINativeResult,
    strict_schema,
)


@dataclass(frozen=True, slots=True)
class RequestSettings:
    model: str = config.DEFAULT_MODEL
    pass_b_effort: str = config.DEFAULT_PASS_B_EFFORT
    pass_a_max_output_tokens: int = config.PASS_A_MAX_OUTPUT_TOKENS
    pass_b_max_output_tokens: int | None = None

    def __post_init__(self) -> None:
        if self.model not in config.SUPPORTED_MODELS:
            raise ValueError(
                f"unsupported model {self.model!r}; choose from "
                f"{config.SUPPORTED_MODELS}"
            )
        if self.pass_b_effort not in config.SUPPORTED_PASS_B_EFFORTS:
            raise ValueError(
                f"unsupported Pass B effort {self.pass_b_effort!r}; choose from "
                f"{config.SUPPORTED_PASS_B_EFFORTS}"
            )
        if self.pass_a_max_output_tokens < 1:
            raise ValueError("Pass A output cap must be positive")
        selected_cap = self.pass_b_max_output_tokens
        if selected_cap is None:
            selected_cap = config.PASS_B_MAX_OUTPUT_TOKENS[self.pass_b_effort]
            object.__setattr__(self, "pass_b_max_output_tokens", selected_cap)
        if selected_cap < 1:
            raise ValueError("Pass B output cap must be positive")


def build_pass_a_request(
    row: ManifestRow | Mapping[str, Any],
    settings: RequestSettings,
) -> dict[str, Any]:
    """Build the logprob-enabled binary gate request."""
    return {
        "model": settings.model,
        "instructions": load_pass_a_prompt(),
        "input": format_pass_a_message(row),
        "prompt_cache_key": config.PASS_A_CACHE_KEY,
        "max_output_tokens": settings.pass_a_max_output_tokens,
        "store": False,
        "text": _text_format(PassAResult),
        "reasoning": {"effort": config.PASS_A_EFFORT},
        "top_logprobs": config.PASS_A_TOP_LOGPROBS,
        "include": list(config.LOGPROB_INCLUDE),
    }


def build_pass_b_request(
    row: ManifestRow | Mapping[str, Any],
    verdict: int,
    settings: RequestSettings,
) -> dict[str, Any]:
    """Build the fixed-family reasoning request."""
    if verdict not in (0, 1):
        raise ValueError(f"verdict must be 0 or 1, got {verdict!r}")
    result_model = PassBAINativeResult if verdict == 1 else PassBNotAINativeResult
    return {
        "model": settings.model,
        "instructions": load_pass_b_prompt(verdict),
        "input": format_pass_b_message(row, verdict),
        "prompt_cache_key": config.PASS_B_CACHE_KEYS[verdict],
        "max_output_tokens": settings.pass_b_max_output_tokens,
        "store": False,
        "text": _text_format(result_model),
        "reasoning": {"effort": settings.pass_b_effort},
    }


def cache_route_for_pass(stage: str, family: int | None = None) -> str:
    if stage == "pass_a":
        return config.PASS_A_CACHE_KEY
    if stage == "pass_b" and family in (0, 1):
        return config.PASS_B_CACHE_KEYS[int(family)]
    raise ValueError(f"invalid cache route stage={stage!r}, family={family!r}")


def estimate_input_tokens(request: dict[str, Any]) -> int:
    """Estimate model-visible input and schema tokens for rate reservation."""
    stable = {
        "text": request.get("text"),
        "reasoning": request.get("reasoning"),
        "include": request.get("include"),
        "top_logprobs": request.get("top_logprobs"),
    }
    stable_payload = json.dumps(
        stable,
        ensure_ascii=False,
        separators=(",", ":"),
        sort_keys=True,
    )
    byte_count = sum(
        len(str(value or "").encode("utf-8"))
        for value in (
            request.get("instructions"),
            request.get("input"),
            stable_payload,
        )
    )
    return config.INPUT_TOKEN_ESTIMATE_FIXED_OVERHEAD + max(
        1,
        ceil(byte_count / config.INPUT_TOKEN_ESTIMATE_BYTES_PER_TOKEN),
    )


def request_identity(settings: RequestSettings) -> dict[str, Any]:
    """Return every stable input that defines provider request behavior."""
    schemas = {
        "pass_a": strict_schema(PassAResult),
        "pass_b_family_0": strict_schema(PassBNotAINativeResult),
        "pass_b_family_1": strict_schema(PassBAINativeResult),
    }
    semantic_modules = {
        "request_builder": sys.modules[__name__],
        "two_pass_formatter": formatter_module,
        "cohort": cohort_module,
        "schema": schema_module,
        "prompt_loader": prompts_module,
        "confidence": confidence_module,
    }
    return {
        "fingerprint_version": config.REQUEST_FINGERPRINT_VERSION,
        "model": settings.model,
        "pass_a_effort": config.PASS_A_EFFORT,
        "pass_b_effort": settings.pass_b_effort,
        "pass_a_top_logprobs": config.PASS_A_TOP_LOGPROBS,
        "logprob_include": list(config.LOGPROB_INCLUDE),
        "pass_a_cache_key": config.PASS_A_CACHE_KEY,
        "pass_b_cache_keys": dict(config.PASS_B_CACHE_KEYS),
        "pass_a_max_output_tokens": settings.pass_a_max_output_tokens,
        "pass_b_max_output_tokens": settings.pass_b_max_output_tokens,
        "provisional_pass_b_caps": dict(config.PASS_B_MAX_OUTPUT_TOKENS),
        "prompt_sha256": {
            "pass_a": _sha256_text(load_pass_a_prompt()),
            "pass_b_family_0": _sha256_text(load_pass_b_prompt(0)),
            "pass_b_family_1": _sha256_text(load_pass_b_prompt(1)),
        },
        "schema_sha256": {
            name: _sha256_json(schema) for name, schema in schemas.items()
        },
        "module_source_sha256": {
            name: _module_source_sha256(module)
            for name, module in semantic_modules.items()
        },
        "runtime_constants": {
            "formatter_version": FORMATTER_VERSION,
            "max_user_message_chars": MAX_USER_MESSAGE_CHARS,
            "cohort_boundary": {
                "year": cohort_module.COHORT_BOUNDARY_YEAR,
                "month": cohort_module.COHORT_BOUNDARY_MONTH,
                "day": cohort_module.COHORT_BOUNDARY_DAY,
                "month_only_boundary_is_genai": True,
                "missing_or_unparseable_is_pre_genai": True,
            },
            "confidence": {
                "decision_key": confidence_module.DECISION_KEY,
                "masked_sentinel_logprob": (
                    confidence_module.MASKED_SENTINEL_LOGPROB
                ),
                "max_censored_interval_width": (
                    confidence_module.MAX_CENSORED_INTERVAL_WIDTH
                ),
            },
        },
    }


def pass_a_request_identity(settings: RequestSettings) -> dict[str, Any]:
    """Return only the production settings that define a Pass A bank."""
    identity = request_identity(settings)
    return {
        "fingerprint_version": identity["fingerprint_version"],
        "model": identity["model"],
        "pass_a_effort": identity["pass_a_effort"],
        "pass_a_top_logprobs": identity["pass_a_top_logprobs"],
        "logprob_include": identity["logprob_include"],
        "pass_a_cache_key": identity["pass_a_cache_key"],
        "pass_a_max_output_tokens": identity["pass_a_max_output_tokens"],
        "prompt_sha256": identity["prompt_sha256"]["pass_a"],
        "schema_sha256": identity["schema_sha256"]["pass_a"],
        "module_source_sha256": {
            name: digest
            for name, digest in identity["module_source_sha256"].items()
            if name != "cohort"
        },
        "runtime_constants": {
            "formatter_version": identity["runtime_constants"][
                "formatter_version"
            ],
            "max_user_message_chars": identity["runtime_constants"][
                "max_user_message_chars"
            ],
            "confidence": identity["runtime_constants"]["confidence"],
        },
    }


def pass_a_request_fingerprint(settings: RequestSettings) -> str:
    """Hash the exact production Pass A identity for safe bank reuse."""
    return _sha256_json(pass_a_request_identity(settings))


def request_fingerprint(settings: RequestSettings) -> str:
    return _sha256_json(request_identity(settings))


@lru_cache(maxsize=3)
def _text_format(model_cls: type[BaseModel]) -> dict[str, Any]:
    return {
        "format": {
            "type": "json_schema",
            "name": model_cls.__name__,
            "strict": True,
            "schema": strict_schema(model_cls),
        }
    }


def _sha256_text(value: str) -> str:
    return hashlib.sha256(value.encode("utf-8")).hexdigest()


def _module_source_sha256(module: ModuleType) -> str:
    return hashlib.sha256(_module_source_bytes(module)).hexdigest()


def _module_source_bytes(module: ModuleType) -> bytes:
    source_path = inspect.getsourcefile(module)
    if source_path is None:
        raise RuntimeError(f"cannot locate semantic source for {module.__name__}")
    return Path(source_path).read_bytes()


def _sha256_json(value: Any) -> str:
    payload = json.dumps(
        value,
        ensure_ascii=False,
        separators=(",", ":"),
        sort_keys=True,
    ).encode("utf-8")
    return hashlib.sha256(payload).hexdigest()
