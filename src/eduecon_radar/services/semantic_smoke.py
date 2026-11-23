"""Explicit, bounded diagnostics and single-paper live semantic smoke support."""

from __future__ import annotations

import json
import logging
import re
import sqlite3
import traceback
from dataclasses import dataclass
from pathlib import Path
from time import monotonic
from urllib.parse import urlsplit

from eduecon_radar.config import Settings
from eduecon_radar.db import Database
from eduecon_radar.domain.semantic_judgment import (
    DeepSeekSemanticAdapter,
    SemanticJudgment,
    _validate_identity,
    _validate_no_abstract_safety,
)
from eduecon_radar.domain.semantic_models import FastSemanticResult
from eduecon_radar.domain.semantic_prompting import (
    semantic_input_fingerprint,
)
from eduecon_radar.domain.semantic_storage import DomainSemanticRepository
from eduecon_radar.domain.storage import DomainEvidenceRepository
from eduecon_radar.services.deepseek import DeepSeekClient
from eduecon_radar.services.runtime_bundle import (
    RuntimeBundleResolver,
    RuntimeModelIdentity,
)
from eduecon_radar.services.semantic_screening import load_semantic_paper_input

logger = logging.getLogger(__name__)


@dataclass(frozen=True)
class SemanticFinalizationError(RuntimeError):
    stage: str
    error_type: str

    def __str__(self) -> str:
        return f"semantic_finalization_failed:{self.stage}:{self.error_type}"


def semantic_settings_diagnostic(settings: Settings) -> dict[str, object]:
    parsed = urlsplit(settings.deepseek_base_url)
    return {
        "key_configured": bool(settings.deepseek_api_key),
        "base_url_host": parsed.hostname or "invalid",
        "fast_model": settings.deepseek_model_fast,
        "strong_model": settings.deepseek_model_strong,
        "timeout_seconds": settings.deepseek_timeout_seconds,
        "max_retries": settings.deepseek_max_retries,
    }


def diagnose_semantic_runs(database_path: Path, *, paper_limit: int = 1) -> dict[str, object]:
    uri = database_path.resolve().as_uri() + "?mode=ro"
    with sqlite3.connect(uri, uri=True) as connection:
        connection.row_factory = sqlite3.Row
        rows = connection.execute(
            "SELECT run_role,succeeded,failure_type,failure_message,model_name,"
            "model_provider,prompt_version,engine_version,created_at "
            "FROM domain_semantic_runs WHERE succeeded=0 ORDER BY id DESC LIMIT ?",
            (paper_limit,),
        ).fetchall()
    samples = [
        {
            "role": row["run_role"],
            "succeeded": bool(row["succeeded"]),
            "error_type": row["failure_type"],
            "error_summary": _redact_error(row["failure_message"]),
            "model_name": row["model_name"],
            "model_provider": row["model_provider"],
            "prompt_version": row["prompt_version"],
            "engine_version": row["engine_version"],
            "http_status": None,
            "request_id": None,
            "response_body": "not_stored",
            "created_at": row["created_at"],
        }
        for row in rows
    ]
    return {"failed_run_samples": samples, "sample_count": len(samples)}


def run_fast_semantic_smoke(
    *,
    database: Database,
    settings: Settings,
    paper_id: int,
    debug: bool = False,
    allow_unreleased_bundle: bool = False,
) -> dict[str, object]:
    """Makes one explicitly requested Fast request; never invokes Strong or reviews."""
    if not settings.deepseek_api_key:
        raise ValueError("DEEPSEEK_API_KEY is not configured")
    evidence_run = DomainEvidenceRepository(database).latest_for_paper(paper_id)
    if evidence_run is None:
        raise ValueError("paper has no domain evidence run")
    paper = load_semantic_paper_input(database, paper_id, evidence_run)
    project_root = (
        settings.domain_config_dir.resolve().parents[2]
        if allow_unreleased_bundle
        else settings.project_root
    )
    runtime = RuntimeBundleResolver(project_root).resolve(
        mode="development" if allow_unreleased_bundle else "production",
        allow_unreleased_bundle=allow_unreleased_bundle,
        development_models=(
            RuntimeModelIdentity(
                provider="deepseek-compatible",
                fast=settings.deepseek_model_fast,
                strong=settings.deepseek_model_strong,
                translation=settings.deepseek_model_fast,
            )
            if allow_unreleased_bundle
            else None
        ),
    )
    config = runtime.semantic_config
    if runtime.unreleased_override:
        config = config.model_copy(
            update={"fast": config.fast.model_copy(update={"model": runtime.models.fast})}
        )
    prompts = runtime.prompt_builder
    client = DeepSeekClient(
        api_key=settings.deepseek_api_key,
        base_url=settings.deepseek_base_url,
        timeout_seconds=settings.deepseek_timeout_seconds,
        max_attempts=settings.deepseek_max_retries,
    )
    adapter = DeepSeekSemanticAdapter(client)
    prompt = prompts.build_fast(paper)
    fingerprint = semantic_input_fingerprint(
        paper=paper,
        prompt=prompt,
        model_role="fast",
        model_name=config.fast.model,
        engine_version=config.version,
        escalation_policy_version=config.escalation.version,
    )
    repository = DomainSemanticRepository(database)
    existing = repository.successful_by_fingerprint(fingerprint, "fast")
    if existing:
        return {
            "paper_id": paper_id,
            "status": "already_succeeded",
            "run_id": int(existing["id"]),
            "diagnostic": _diagnostic_payload(client),
            "runtime_bundle": runtime.audit_summary(),
        }
    started = monotonic()
    stage = "response_received"
    try:
        response = adapter.request_json(
            model=config.fast.model,
            system_prompt=prompt.system_prompt,
            user_prompt=prompt.user_prompt,
            schema=FastSemanticResult,
        )
        stage = "response_parsed"
        result = FastSemanticResult.model_validate(response.data)
        stage = "schema_validated"
        _validate_identity(
            result.paper_id,
            response.model_name,
            paper,
            expected_model_name=config.fast.model,
        )
        result = result.model_copy(update={"model_name": response.model_name})
        _validate_no_abstract_safety(paper, result, config)
        judgment = SemanticJudgment(
            result=result,
            prompt=prompt,
            input_tokens=response.input_tokens,
            output_tokens=response.output_tokens,
            estimated_cost=response.estimated_cost,
        )
        # Validate JSON-compatible serialization before opening the storage transaction.
        _stable_result_json(judgment.result)
        stage = "result_serialized"
        stage = "storage_started"
        stored = repository.save_success(
            paper=paper,
            role="fast",
            fingerprint=fingerprint,
            prompt=judgment.prompt,
            model_provider=config.fast.provider,
            model_name=judgment.result.model_name,
            engine_version=config.version,
            escalation_policy_version=config.escalation.version,
            result=judgment.result,
            input_tokens=judgment.input_tokens,
            output_tokens=judgment.output_tokens,
            estimated_cost=judgment.estimated_cost,
        )
        stage = "storage_completed"
        stage = "transaction_committed"
        return {
            "paper_id": paper_id,
            "status": "succeeded",
            "run_id": stored.id,
            "decision": judgment.result.decision,
            "elapsed_ms": int((monotonic() - started) * 1000),
            "diagnostic": _diagnostic_payload(client),
            "stage": "cli_result_built",
            "runtime_bundle": runtime.audit_summary(),
        }
    except Exception as exc:
        logger.exception(
            "semantic smoke finalization failed at stage=%s paper_id=%s", stage, paper_id
        )
        finalization_error = SemanticFinalizationError(stage=stage, error_type=type(exc).__name__)
        stored = repository.save_failure(
            paper=paper,
            role="fast",
            fingerprint=fingerprint,
            prompt=prompt,
            model_provider=config.fast.provider,
            model_name=config.fast.model,
            engine_version=config.version,
            escalation_policy_version=config.escalation.version,
            error=finalization_error,
        )
        result: dict[str, object] = {
            "paper_id": paper_id,
            "status": "failed",
            "run_id": stored.id,
            "elapsed_ms": int((monotonic() - started) * 1000),
            "diagnostic": _failure_diagnostic(client, finalization_error),
            "stage": stage,
            "error_type": type(exc).__name__,
        }
        if debug:
            result["debug_traceback"] = _safe_traceback(exc)
        return result


def _diagnostic_payload(client: DeepSeekClient) -> dict[str, object]:
    item = client.last_diagnostic
    if item is None:
        return {
            "http_status": None,
            "request_id": None,
            "json_parseable": False,
            "schema_valid": False,
        }
    return {
        "http_status": item.http_status,
        "request_id": item.request_id,
        "elapsed_ms": item.elapsed_ms,
        "json_parseable": item.json_parseable,
        "schema_valid": item.schema_valid,
        "error_type": item.error_type,
    }


def _failure_diagnostic(
    client: DeepSeekClient, error: SemanticFinalizationError
) -> dict[str, object]:
    payload = _diagnostic_payload(client)
    payload["error_type"] = error.error_type
    payload["finalization_stage"] = error.stage
    return payload


def _stable_result_json(result: FastSemanticResult) -> str:
    return json.dumps(
        result.model_dump(mode="json"), ensure_ascii=False, sort_keys=True, separators=(",", ":")
    )


def _safe_traceback(exc: Exception) -> dict[str, object]:
    frames = traceback.extract_tb(exc.__traceback__)
    return {
        "exception_type": type(exc).__name__,
        "frames": [f"{frame.filename}:{frame.lineno}:{frame.name}" for frame in frames],
    }


def _redact_error(value: object) -> str:
    text = str(value)
    if "validation errors" in text or "Invalid structured AI response" in text:
        return "schema_validation_failed"
    status = re.search(r"\b(401|403|429|5\d\d)\b", text)
    if status:
        return f"http_{status.group(1)}"
    if "timeout" in text.lower():
        return "timeout"
    text = re.sub(r"(?i)(authorization\s*[:=]\s*)([^\s,;]+)", r"\1[REDACTED]", text)
    text = re.sub(r"\bsk-[A-Za-z0-9_-]+\b", "[REDACTED]", text)
    return "unexpected_error"
