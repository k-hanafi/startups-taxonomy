"""Offline tests for production-aligned eval classification."""

from __future__ import annotations

import json
import math
from pathlib import Path
from types import SimpleNamespace

import pytest

from evals import classification
from evals.paths import pass_a_bank_run_id
from two_pass_classifier import config as production_config


@pytest.fixture
def sample_row() -> dict[str, str]:
    return {
        "org_uuid": "abc123-uuid",
        "name": "Acme AI",
        "short_description": "An AI thing.",
        "Long description": "A longer description.",
        "category_list": "Artificial Intelligence (AI)",
        "category_groups_list": "Software",
        "founded_date": "2023-05",
        "employee_count": "1-10",
        "total_funding_usd": "1000000",
        "website_pages_used": "https://acme.ai/",
        "website_evidence": "We build AI agents.",
    }


def _entry(
    token: str,
    logprob: float,
    top: list[tuple[str, float]] | None = None,
) -> dict:
    return {
        "token": token,
        "bytes": list(token.encode()),
        "logprob": logprob,
        "top_logprobs": [
            {
                "token": candidate,
                "bytes": list(candidate.encode()),
                "logprob": candidate_logprob,
            }
            for candidate, candidate_logprob in (
                [(token, logprob), *(top or [])]
            )
        ],
    }


def _response(
    payload: dict | None,
    *,
    status: str = "completed",
    chosen_probability: float | None = None,
    reasoning_tokens: int = 0,
    cached_tokens: int = 0,
):
    text = (
        json.dumps(payload, ensure_ascii=False, separators=(",", ":"))
        if payload is not None
        else ""
    )
    raw: dict = {
        "status": status,
        "usage": {
            "input_tokens": 100,
            "output_tokens": 50,
            "output_tokens_details": {
                "reasoning_tokens": reasoning_tokens,
            },
            "input_tokens_details": {
                "cached_tokens": cached_tokens,
            },
        },
        "output": [],
    }
    if text:
        content: dict = {"type": "output_text", "text": text}
        if chosen_probability is not None:
            marker = '"ai_native":'
            digit_index = text.index(marker) + len(marker)
            digit = text[digit_index]
            opponent = "0" if digit == "1" else "1"
            content["logprobs"] = [
                _entry(text[:digit_index], 0.0),
                _entry(
                    digit,
                    math.log(chosen_probability),
                    [(opponent, math.log(1.0 - chosen_probability))],
                ),
                _entry(text[digit_index + 1 :], 0.0),
            ]
        raw["output"] = [
            {
                "type": "message",
                "content": [content],
            }
        ]
    usage = raw["usage"]
    response = SimpleNamespace(
        status=status,
        output_text=text,
        usage=SimpleNamespace(
            input_tokens=usage["input_tokens"],
            output_tokens=usage["output_tokens"],
            output_tokens_details=SimpleNamespace(
                reasoning_tokens=reasoning_tokens
            ),
            input_tokens_details=SimpleNamespace(
                cached_tokens=cached_tokens
            ),
        ),
    )
    response.model_dump = lambda: raw
    return response


def _pass_a_payload(verdict: int = 1) -> dict:
    return {
        "ai_native": verdict,
        "ai_native_reasoning": "The product mechanism depends on AI.",
        "sources_used": ["website_evidence", "short_description"],
        "ai_native_critique": "The available evidence is direct and specific.",
    }


def _pass_b_ai_payload() -> dict:
    return {
        "subclass": "1E",
        "rad_score": "RAD-M",
        "subclass_confidence": 4,
        "rad_confidence": 3,
        "subclass_reasoning": "The product applies AI in one vertical.",
        "rad_reasoning": "It combines provider dependence with domain assets.",
        "sources_used": ["website_evidence", "resource_context"],
        "subclass_critique": "The vertical is clear.",
        "rad_critique": "Model ownership is not fully documented.",
    }


def _pass_b_non_ai_payload() -> dict:
    return {
        "subclass": "0B",
        "subclass_confidence": 5,
        "subclass_reasoning": "AI augments an existing software product.",
        "rad_reasoning": "RAD is structurally not applicable.",
        "sources_used": ["website_evidence"],
        "subclass_critique": "The underlying product predates the AI feature.",
    }


def _patch_run_paths(monkeypatch, tmp_path):
    monkeypatch.setattr(
        classification,
        "run_dir",
        lambda run_id: tmp_path / "runs" / run_id,
    )
    monkeypatch.setattr(
        classification,
        "run_raw_dir",
        lambda run_id: tmp_path / "runs" / run_id / "raw",
    )
    monkeypatch.setattr(
        classification,
        "run_predictions_path",
        lambda run_id: tmp_path / "runs" / run_id / "predictions.jsonl",
    )
    monkeypatch.setattr(
        classification,
        "run_config_path",
        lambda run_id: tmp_path / "runs" / run_id / "config.json",
    )


def _write_valid_bank(
    tmp_path: Path,
    monkeypatch,
    *,
    model: str = "gpt-5.4-nano",
    verdict: int = 1,
) -> str:
    _patch_run_paths(monkeypatch, tmp_path)
    bank_id = pass_a_bank_run_id(model)
    bank_dir = tmp_path / "runs" / bank_id
    raw_dir = bank_dir / "raw"
    raw_dir.mkdir(parents=True)
    classification._ensure_pass_a_bank_config(bank_id, model, 1)
    raw = _response(
        _pass_a_payload(verdict),
        chosen_probability=0.8,
    ).model_dump()
    (raw_dir / "startup-u1_a.json").write_text(
        json.dumps(raw),
        encoding="utf-8",
    )
    (bank_dir / "predictions.jsonl").write_text(
        json.dumps(
            {
                "custom_id": "startup-u1",
                "org_uuid": "u1",
                "model": model,
                "status": "completed",
                "ai_native": verdict,
                "a_latency_s": 0.5,
                "a_input_tokens": 100,
                "a_output_tokens": 50,
                "a_reasoning_tokens": 0,
                "a_cached_tokens": 0,
            }
        )
        + "\n",
        encoding="utf-8",
    )
    return bank_id


def _golden_row() -> list[dict[str, str]]:
    return [
        {
            "org_uuid": "u1",
            "name": "Acme",
            "short_description": "AI workflow",
            "Long description": "A longer description",
            "category_list": "Artificial Intelligence",
            "category_groups_list": "Software",
            "founded_date": "2024-01",
            "employee_count": "1-10",
            "total_funding_usd": "1",
            "website_pages_used": "https://acme.test/",
            "website_evidence": "We build AI.",
        }
    ]


def test_matrix_and_defaults_come_from_production():
    assert classification.cfg.DEFAULT_MODEL == production_config.DEFAULT_MODEL
    assert classification.cfg.DEFAULT_PASS_B_EFFORT == (
        production_config.DEFAULT_PASS_B_EFFORT
    )
    assert classification.cfg.EVAL_MODELS == production_config.SUPPORTED_MODELS
    assert classification.cfg.MATRIX_PASS_B_EFFORTS == (
        production_config.SUPPORTED_PASS_B_EFFORTS
    )
    assert len(classification.matrix_cells()) == 9


def test_assemble_record_retains_production_audit_fields():
    record = classification.assemble_record(
        "startup-u1",
        "u1",
        "gpt-5.4-nano",
        "low",
        "GENAI-ERA",
        _response(
            _pass_a_payload(),
            chosen_probability=0.8,
            cached_tokens=10,
        ),
        _response(
            _pass_b_ai_payload(),
            reasoning_tokens=500,
            cached_tokens=40,
        ),
        latency_a_s=0.8,
        latency_b_s=12.4,
    )

    assert record["status"] == "completed"
    assert record["ai_native_confidence"] == pytest.approx(0.8)
    assert record["subclass_confidence"] == 4
    assert record["rad_confidence"] == 3
    assert record["sources_used"] == [
        "website_evidence",
        "short_description",
        "resource_context",
    ]
    for field in (
        "ai_native_reasoning",
        "subclass_reasoning",
        "rad_reasoning",
        "ai_native_critique",
        "subclass_critique",
        "rad_critique",
    ):
        assert record[field]
    assert record["cached_tokens"] == 50
    assert record["latency_s"] == pytest.approx(13.2)


def test_non_ai_record_has_structural_rad_fields():
    record = classification.assemble_record(
        "startup-u1",
        "u1",
        "gpt-5.6-luna",
        "low",
        "PRE-GENAI",
        _response(
            _pass_a_payload(0),
            chosen_probability=0.9,
        ),
        _response(_pass_b_non_ai_payload()),
    )

    assert record["rad_score"] == "RAD-NA"
    assert record["rad_confidence"] is None
    assert record["rad_reasoning"]
    assert record["rad_critique"] is None


def test_schema_invalid_output_remains_retryable():
    pass_a = _response({"ai_native": 1}, chosen_probability=0.8)
    record = classification.assemble_record(
        "startup-u1",
        "u1",
        "gpt-5.4-nano",
        "low",
        "PRE-GENAI",
        pass_a,
        None,
    )

    assert record["status"] != "completed"


def test_old_run_config_fingerprint_is_rejected(tmp_path, monkeypatch):
    _patch_run_paths(monkeypatch, tmp_path)
    run = tmp_path / "runs" / "old-run"
    run.mkdir(parents=True)
    (run / "config.json").write_text(
        json.dumps(
            {
                "model": "gpt-5.4-nano",
                "effort_b": "low",
                "repeat": 1,
                "n_rows": 1,
                "prompt_a_sha256": "old",
            }
        ),
        encoding="utf-8",
    )

    with pytest.raises(SystemExit, match="fingerprint"):
        classification._ensure_config(
            "old-run",
            "gpt-5.4-nano",
            "low",
            repeat=1,
            n_rows=1,
        )


def test_load_pass_a_bank_reports_invalid_raw_not_missing_file(
    tmp_path,
    monkeypatch,
):
    bank_id = _write_valid_bank(tmp_path, monkeypatch)
    raw_path = tmp_path / "runs" / bank_id / "raw" / "startup-u1_a.json"
    invalid = _response({"ai_native": 1}, chosen_probability=0.8).model_dump()
    raw_path.write_text(json.dumps(invalid), encoding="utf-8")

    with pytest.raises(SystemExit, match="no longer validates") as exc:
        classification.load_pass_a_bank(bank_id)

    message = str(exc.value)
    assert "startup-u1_a.json" in message
    assert "copy it with the bank run" not in message
    assert "--rerun-pass-a" in message


def test_load_pass_a_bank_still_reports_missing_raw(tmp_path, monkeypatch):
    bank_id = _write_valid_bank(tmp_path, monkeypatch)
    (
        tmp_path / "runs" / bank_id / "raw" / "startup-u1_a.json"
    ).unlink()

    with pytest.raises(SystemExit, match="missing raw Pass A files") as exc:
        classification.load_pass_a_bank(bank_id)

    assert "copy it with the bank run" in str(exc.value)


def test_old_pass_a_bank_fingerprint_is_rejected(tmp_path, monkeypatch):
    _patch_run_paths(monkeypatch, tmp_path)
    bank_id = pass_a_bank_run_id("gpt-5.4-nano")
    bank = tmp_path / "runs" / bank_id
    bank.mkdir(parents=True)
    (bank / "config.json").write_text(
        json.dumps(
            {
                "kind": "pass_a_bank",
                "model": "gpt-5.4-nano",
                "prompt_a_sha256": "old",
            }
        ),
        encoding="utf-8",
    )

    with pytest.raises(SystemExit, match="incompatible or missing"):
        classification.resolve_pass_a_source(
            "gpt-5.4-nano",
            ["startup-u1"],
        )


def test_valid_bank_is_reused_for_pass_b_only(
    tmp_path,
    monkeypatch,
):
    model = "gpt-5.4-nano"
    bank_id = _write_valid_bank(tmp_path, monkeypatch, model=model)
    monkeypatch.setattr(classification, "load_golden_rows", _golden_row)
    monkeypatch.setattr(classification, "_openai_client", lambda: object())
    calls: list[dict] = []

    def fake_create(client, kwargs):
        calls.append(kwargs)
        return _response(_pass_b_ai_payload())

    monkeypatch.setattr(classification, "_create", fake_create)

    run_id = classification.run_classification(
        model=model,
        effort_b="low",
        run_id="aligned-cell",
    )

    assert run_id == "aligned-cell"
    assert len(calls) == 1
    assert "top_logprobs" not in calls[0]
    record = json.loads(
        (
            tmp_path
            / "runs"
            / run_id
            / "predictions.jsonl"
        ).read_text(encoding="utf-8")
    )
    assert record["pass_a_bank_run_id"] == bank_id
    assert record["ai_native_confidence"] == pytest.approx(0.8)
    assert record["subclass"] == "1E"


def test_dry_run_rerun_does_not_delete_historical_bank(
    tmp_path,
    monkeypatch,
):
    model = "gpt-5.4-nano"
    bank_id = _write_valid_bank(tmp_path, monkeypatch, model=model)
    monkeypatch.setattr(classification, "load_golden_rows", _golden_row)

    classification.run_classification(
        model=model,
        effort_b="low",
        run_id="dry-run",
        dry_run=True,
        rerun_pass_a=True,
    )

    assert (
        tmp_path / "runs" / bank_id / "predictions.jsonl"
    ).exists()


def test_unknown_model_refused_before_any_api_call():
    with pytest.raises(ValueError, match="unsupported model"):
        classification.run_classification(
            model="gpt-not-real",
            dry_run=True,
            limit=1,
        )
