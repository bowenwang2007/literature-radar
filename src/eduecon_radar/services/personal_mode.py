# --- not part of the product core: see docs/PRODUCT_CORE.md ---
# LEGACY (task pipeline) + product sync shim. The retired
# ``PersonalWorkflowService`` task chain is kept for historical tests and is
# never imported by ``ui/``. Retrieval moved to ``services/journal_sync.py``.

from __future__ import annotations

import csv
import io
import json
import logging
import re
from collections.abc import Mapping, Sequence
from dataclasses import asdict, dataclass
from datetime import date
from pathlib import Path
from typing import Literal, Protocol

from eduecon_radar.db import Database, utcnow
from eduecon_radar.schemas import PersonalStrongDecision
from eduecon_radar.services.abstract_enrichment import (
    AbstractEnrichmentService,
    EnrichmentStats,
)
from eduecon_radar.services.deepseek import DeepSeekClient, DeepSeekError, DeepSeekResponse
from eduecon_radar.services.deterministic_screening import (
    SCREENING_PROMPT_VERSION,
    DeterministicScreeningService,
)
from eduecon_radar.services.journal_registry import RegistryJournal
from eduecon_radar.services.journal_sync import (
    JournalSyncResult as JournalSyncResult,
)
from eduecon_radar.services.journal_sync import (
    JournalSyncService as JournalSyncService,
)
from eduecon_radar.services.journal_sync import (
    PersonalJournalSyncService as PersonalJournalSyncService,
)
from eduecon_radar.services.journal_sync import (
    _existing_paper_state as _existing_paper_state,
)
from eduecon_radar.services.journal_sync import (
    _paper_key as _paper_key,
)
from eduecon_radar.services.journal_sync import (
    _record_initial_abstract_provenance as _record_initial_abstract_provenance,
)
from eduecon_radar.services.journal_sync import (
    _record_journal_match as _record_journal_match,
)
from eduecon_radar.services.journal_sync import (
    _record_matches_journal as _record_matches_journal,
)
from eduecon_radar.services.journal_sync import (
    _safe_error as _safe_error,
)
from eduecon_radar.services.journal_sync import (
    _safe_normalize_issn as _safe_normalize_issn,
)
from eduecon_radar.services.manual_abstract import (
    save_manual_abstract as save_manual_abstract,
)
from eduecon_radar.services.product_access import (
    local_request_allowed as local_request_allowed,
)
from eduecon_radar.services.task_review import TaskReviewInput, save_task_reviews
from eduecon_radar.services.translation import (
    TranslationParameters,
    TranslationService,
    TranslationStats,
)
from eduecon_radar.services.workflow import create_workflow_batch

LOGGER = logging.getLogger(__name__)
PERSONAL_STATE_VERSION = "7A-1"
PERSONAL_PROMPT_VERSION = "2026-07-31-personal-strong-v1"
PERSONAL_BATCH_SIZES = (1, 3, 5)
PERSONAL_ITEM_STATUSES = {
    "pending",
    "processing",
    "abstract_ready",
    "needs_retry",
    "needs_manual_abstract",
    "classification_ready",
    "classified",
    "failed",
}
HUMAN_DECISIONS = {"related", "irrelevant", "uncertain", "undecided"}


def personal_mode_allowed(enabled: bool, requested_mode: str | None, host: str | None) -> bool:
    """Backward-compatible wrapper; the local app no longer reads a mode switch."""
    return (
        bool(enabled)
        and requested_mode in {None, "", "personal"}
        and local_request_allowed(host)
    )


def personal_session_key(name: str) -> str:
    normalized = name.strip()
    if not normalized:
        raise ValueError("Personal session key name is required")
    return f"personal_{normalized}"


def personal_review_widget_key(task_id: int, paper_id: int, field_name: str) -> str:
    field = field_name.strip()
    if not field:
        raise ValueError("Personal review field name is required")
    return f"personal_review_{task_id}_{paper_id}_{field}"


def personal_review_save_enabled(
    pending_changes: int,
    *,
    confirmed: bool,
    saving: bool,
) -> bool:
    return pending_changes > 0 and confirmed and not saving


@dataclass(frozen=True)
class PersonalTaskCreationResult:
    task_id: int
    paper_ids: tuple[int, ...]
    journal_results: tuple[JournalSyncResult, ...]


@dataclass(frozen=True)
class PersonalTitleTranslationResult:
    considered: int
    translated: int
    pending: int
    failed: int
    ai_calls: int
    input_tokens: int
    output_tokens: int
    failure_reasons: tuple[str, ...]


@dataclass(frozen=True)
class PersonalTaskSnapshot:
    task_id: int
    state: dict[str, object]
    papers: tuple[dict[str, object], ...]
    counts: dict[str, int]


@dataclass(frozen=True)
class PersonalReviewDraft:
    decision: str
    note: str | None = None


def create_personal_task(
    database: Database,
    *,
    journal_results: tuple[JournalSyncResult, ...],
    selected_journals: tuple[RegistryJournal, ...],
    since: date,
    until: date,
    per_journal_limit: int,
    max_task_papers: int | None = None,
) -> PersonalTaskCreationResult:
    if max_task_papers is not None and not 1 <= max_task_papers <= 500:
        raise ValueError("Personal task paper limit must be between 1 and 500")
    matched = tuple(
        dict.fromkeys(
            paper_id
            for result in journal_results
            for paper_id in (result.matched_paper_ids or result.paper_ids)
        )
    )
    paper_ids = matched[:max_task_papers] if max_task_papers is not None else matched
    if not paper_ids:
        raise ValueError("No papers were retrieved for the selected scope")
    source_config = {
        "runtime_mode": "personal_local",
        "journals": [journal.journal_key for journal in selected_journals],
        "journal_issns": {
            journal.journal_key: list(journal.issns) for journal in selected_journals
        },
        "since": since.isoformat(),
        "until": until.isoformat(),
        "per_journal_limit": per_journal_limit,
        "max_task_papers": max_task_papers,
        "matched_paper_ids": list(paper_ids),
    }
    task_id = create_workflow_batch(
        database,
        name=f"本次同步 {utcnow()[:10]}",
        source_type="sync",
        paper_ids=paper_ids,
        source_config=source_config,
    )
    state = {
        "personal_local": {
            "schema_version": PERSONAL_STATE_VERSION,
            "stop_requested": False,
            "items": {
                str(paper_id): {
                    "status": "pending",
                    "stage": "pending",
                    "updated_at": utcnow(),
                    "attempts": 0,
                }
                for paper_id in paper_ids
            },
            "journal_results": [
                {
                    **asdict(result),
                    "query_errors": list(result.query_errors),
                    "paper_ids": list(result.paper_ids),
                }
                for result in journal_results
            ],
            "gold_candidates": [],
        }
    }
    with database.connect() as connection:
        connection.execute(
            """UPDATE workflow_batches
            SET status='ready',result_summary_json=?,updated_at=? WHERE id=?""",
            (json.dumps(state, ensure_ascii=False, sort_keys=True), utcnow(), task_id),
        )
    return PersonalTaskCreationResult(task_id, paper_ids, journal_results)


def translate_personal_task_titles(
    database: Database,
    task_id: int,
    translation_service: TranslationService | None,
) -> PersonalTitleTranslationResult:
    """Translate all missing task titles without entering abstract or AI screening."""
    state = load_personal_state(database, task_id)
    paper_ids = tuple(int(value) for value in state["items"])
    stats = TranslationStats()
    if translation_service is not None:
        if hasattr(translation_service, "run_batch_titles"):
            batch_result = translation_service.run_batch_titles(  # type: ignore[attr-defined]
                paper_ids,
                task_id=task_id,
            )
            stats = TranslationStats(
                ai_calls=batch_result.ai_calls,
                input_tokens=batch_result.input_tokens,
                output_tokens=batch_result.output_tokens,
                failure_reasons=list(batch_result.failure_reasons),
            )
        else:
            stats = translation_service.run(
                TranslationParameters(
                    review_status=None,
                    limit=max(1, len(paper_ids)),
                    missing_only=True,
                    paper_ids=paper_ids,
                    translate_titles=True,
                    translate_abstracts=False,
                    task_id=task_id,
                )
            )
    placeholders = ",".join("?" for _ in paper_ids)
    with database.connect() as connection:
        rows = connection.execute(
            f"""SELECT p.id,pt.chinese_title,pt.title_translation_status,
            pt.title_translation_source FROM papers p
            LEFT JOIN paper_translations pt ON pt.paper_id=p.id
            WHERE p.id IN ({placeholders})""",
            paper_ids,
        ).fetchall()
    translated = pending = failed = 0
    rows_by_id = {int(row["id"]): row for row in rows}
    now = utcnow()
    for paper_id in paper_ids:
        row = rows_by_id[paper_id]
        if row["chinese_title"]:
            status = (
                "manual_title_preserved"
                if row["title_translation_source"] == "human_edited"
                else "translated"
            )
            reason = None
            translated += 1
        elif translation_service is None:
            status = "title_translation_pending"
            reason = "deepseek_api_key_missing"
            pending += 1
        elif row["title_translation_status"] == "failed":
            status = "title_translation_retry"
            reason = "translation_validation_or_provider_error"
            failed += 1
        else:
            status = "title_translation_pending"
            reason = "translation_not_completed"
            pending += 1
        item = dict(state["items"][str(paper_id)])
        item["title_translation"] = {
            "status": status,
            "reason": reason,
            "source": row["title_translation_source"],
            "updated_at": now,
        }
        state["items"][str(paper_id)] = item
    state["last_title_translation"] = {
        "considered": len(paper_ids),
        "translated": translated,
        "pending": pending,
        "failed": failed,
        "ai_calls": stats.ai_calls,
        "input_tokens": stats.input_tokens,
        "output_tokens": stats.output_tokens,
        "failure_reasons": list(stats.failure_reasons),
        "updated_at": now,
    }
    _refresh_journal_processing_summary(state)
    _save_personal_state(database, task_id, state)
    return PersonalTitleTranslationResult(
        considered=len(paper_ids),
        translated=translated,
        pending=pending,
        failed=failed,
        ai_calls=stats.ai_calls,
        input_tokens=stats.input_tokens,
        output_tokens=stats.output_tokens,
        failure_reasons=tuple(stats.failure_reasons),
    )


def enrich_personal_task_abstracts(
    database: Database,
    task_id: int,
    enrichment_service: AbstractEnrichmentService,
    *,
    paper_ids: tuple[int, ...] | None = None,
) -> EnrichmentStats:
    """Run the common provider chain and persist explicit per-paper terminal states."""
    state = load_personal_state(database, task_id)
    task_paper_ids = tuple(int(value) for value in state["items"])
    paper_ids = tuple(dict.fromkeys(paper_ids or task_paper_ids))
    if not set(paper_ids).issubset(task_paper_ids):
        raise ValueError("Abstract retrieval paper does not belong to the task")
    stats = enrichment_service.enrich_papers(
        paper_ids, workflow_batch_id=task_id
    )
    placeholders = ",".join("?" for _ in paper_ids)
    with database.connect() as connection:
        rows = connection.execute(
            f"SELECT id,abstract,source_metadata_json FROM papers "
            f"WHERE id IN ({placeholders})",
            paper_ids,
        ).fetchall()
    now = utcnow()
    for row in rows:
        paper_id = int(row["id"])
        metadata = json.loads(row["source_metadata_json"] or "{}")
        audit = metadata.get("abstract_enrichment")
        audit = dict(audit) if isinstance(audit, dict) else {}
        if row["abstract"]:
            status = "abstract_ready"
        elif audit.get("retryable") or audit.get("execution_state") in {
            "pending",
            "retryable_error",
        }:
            status = "needs_retry"
        elif audit:
            status = "needs_manual_abstract"
        else:
            status = "failed"
        item = dict(state["items"][str(paper_id)])
        item["abstract_retrieval"] = {
            "status": status,
            "final_reason": audit.get("final_reason"),
            "attempts": list(audit.get("attempts") or []),
            "retryable": bool(audit.get("retryable")),
            "next_retry_at": next(
                (
                    attempt.get("next_retry_at")
                    for attempt in reversed(list(audit.get("attempts") or []))
                    if attempt.get("next_retry_at")
                ),
                None,
            ),
            "updated_at": now,
        }
        state["items"][str(paper_id)] = item
    state["last_abstract_retrieval"] = {
        "total": stats.total,
        "enriched": stats.enriched,
        "failed": stats.failed,
        "provider_queries": stats.provider_queries,
        "cache_hits": stats.cache_hits,
        "by_source": dict(stats.by_source),
        "by_result_state": dict(stats.by_result_state),
        "updated_at": now,
    }
    _refresh_journal_processing_summary(state)
    _save_personal_state(database, task_id, state)
    return stats


class PersonalClassifier(Protocol):
    model_name: str

    def classify(self, paper: dict[str, object]) -> DeepSeekResponse: ...


class PersonalStrongClassifier:
    def __init__(
        self,
        client: DeepSeekClient,
        *,
        model_name: str,
        prompts_dir: Path,
    ) -> None:
        self.client = client
        self.model_name = model_name
        self.prompts_dir = prompts_dir

    def classify(self, paper: dict[str, object]) -> DeepSeekResponse:
        abstract = str(paper.get("abstract") or "").strip()
        if not abstract:
            raise ValueError("needs_abstract")
        system = (self.prompts_dir / "personal_relevance_system.txt").read_text(
            encoding="utf-8"
        )
        template = (self.prompts_dir / "personal_relevance_user.txt").read_text(
            encoding="utf-8"
        )
        response = self.client.request_json(
            model=self.model_name,
            system_prompt=system,
            user_prompt=template.format(
                title=paper.get("title") or "",
                abstract=abstract,
                journal=paper.get("journal") or "",
                year=paper.get("year") or "",
            ),
            schema=PersonalStrongDecision,
        )
        decision = response.data
        assert isinstance(decision, PersonalStrongDecision)
        if decision.decision == "relevant":
            direct = _has_direct_evidence(
                decision.evidence,
                f"{paper.get('title') or ''}\n{abstract}",
            )
            if (
                decision.education_centrality != "central"
                or decision.economic_relevance != "substantive"
                or not direct
            ):
                raise DeepSeekError(
                    "Relevant decision failed the personal-mode evidence invariant",
                    raw_response=decision.model_dump_json(),
                    model_name=response.model_name,
                    validation_error="relevant_requires_central_substantive_direct_evidence",
                )
        return response


@dataclass(frozen=True)
class PersonalBatchBudget:
    max_ai_calls: int = 5
    max_total_tokens: int = 16000
    max_cost_microunits: int = 0

    def __post_init__(self) -> None:
        if self.max_ai_calls < 0 or self.max_total_tokens < 0:
            raise ValueError("Personal budget values cannot be negative")


@dataclass(frozen=True)
class PersonalBatchResult:
    selected: tuple[int, ...]
    completed: tuple[int, ...]
    failed: tuple[int, ...]
    remaining: tuple[int, ...]
    ai_calls: int
    input_tokens: int
    output_tokens: int
    budget_exhausted: bool


class PersonalWorkflowService:
    def __init__(
        self,
        database: Database,
        *,
        enrichment_service: AbstractEnrichmentService,
        translation_service: TranslationService,
        classifier: PersonalClassifier,
        screening_service: DeterministicScreeningService | None = None,
    ) -> None:
        self.database = database
        self.enrichment_service = enrichment_service
        self.translation_service = translation_service
        self.classifier = classifier
        self.screening_service = screening_service

    def process_full_chain(
        self,
        task_id: int,
        *,
        paper_ids: tuple[int, ...] | None = None,
    ) -> dict[str, object]:
        """One-click chain: abstract fetch -> abstract translation -> screening."""
        state = load_personal_state(self.database, task_id)
        task_paper_ids = tuple(int(value) for value in state["items"])
        selected = tuple(
            dict.fromkeys(paper_ids or task_paper_ids)
        )
        if not set(selected).issubset(task_paper_ids):
            raise ValueError("Paper does not belong to the personal task")
        result: dict[str, object] = {
            "completed": [],
            "needs_abstract": [],
            "needs_retry": [],
            "ai_calls": 0,
            "input_tokens": 0,
            "output_tokens": 0,
        }
        for paper_id in selected:
            paper = _load_personal_paper(self.database, task_id, paper_id)
            if not paper["abstract"]:
                enrich_personal_task_abstracts(
                    self.database,
                    task_id,
                    self.enrichment_service,
                    paper_ids=(paper_id,),
                )
                paper = _load_personal_paper(self.database, task_id, paper_id)
            if not paper["abstract"]:
                audit = dict(paper.get("abstract_enrichment") or {})
                status = (
                    "needs_retry"
                    if audit.get("retryable")
                    or audit.get("execution_state") == "pending"
                    else "needs_manual_abstract"
                )
                _update_personal_item(
                    self.database,
                    task_id,
                    paper_id,
                    status=status,
                    stage="awaiting_abstract_retrieval",
                    classification={
                        "decision": "needs_abstract",
                        "education_centrality": "none",
                        "economic_relevance": "none",
                        "confidence": "low",
                        "evidence": [],
                        "reason": "Original English abstract unavailable; no AI judgment made.",
                    },
                )
                result["needs_abstract"].append(paper_id)
                continue
            abstract_stats = self.translation_service.run(
                TranslationParameters(
                    review_status=None,
                    limit=1,
                    missing_only=True,
                    paper_ids=(paper_id,),
                    translate_titles=False,
                    translate_abstracts=True,
                    task_id=task_id,
                )
            )
            _merge_usage(result, abstract_stats)  # type: ignore[arg-type]
            if abstract_stats.failed:
                _update_personal_item(
                    self.database,
                    task_id,
                    paper_id,
                    status="needs_retry",
                    stage="abstract_translation",
                    error_type="translation_failed",
                    error_message="; ".join(abstract_stats.failure_reasons)[:500],
                )
                result["needs_retry"].append(paper_id)
                continue
            if self.screening_service is None:
                item_result = self._process_one(task_id, paper_id)
                if item_result["status"] == "classified":
                    result["completed"].append(paper_id)
                else:
                    result["needs_retry"].append(paper_id)
                _merge_usage(result, item_result)  # type: ignore[arg-type]
                continue
            outcome = self.screening_service.screen_paper(
                paper_id, task_id=task_id
            )
            _merge_usage(result, outcome)
            status = (
                "classified"
                if outcome.status == "classified"
                else "needs_retry"
                if outcome.status == "needs_retry"
                else outcome.status
            )
            _update_personal_item(
                self.database,
                task_id,
                paper_id,
                status=status,
                stage=(
                    "complete"
                    if outcome.status == "classified"
                    else "strong_classification"
                ),
                classification=outcome.classification,
                model_name=self.screening_service.model,
                prompt_version=SCREENING_PROMPT_VERSION,
                classified_at=utcnow() if outcome.status == "classified" else None,
            )
            if outcome.status == "classified":
                result["completed"].append(paper_id)
            elif outcome.status == "needs_retry":
                result["needs_retry"].append(paper_id)
            else:
                result["needs_abstract"].append(paper_id)
        return result

    def process_next_batch(
        self,
        task_id: int,
        *,
        batch_size: Literal[1, 3, 5] = 3,
        budget: PersonalBatchBudget | None = None,
    ) -> PersonalBatchResult:
        if batch_size not in PERSONAL_BATCH_SIZES:
            raise ValueError("Personal batch size must be 1, 3, or 5")
        state = load_personal_state(self.database, task_id)
        items = state["items"]
        with self.database.connect() as connection:
            ready_ids = {
                int(row["paper_id"])
                for row in connection.execute(
                    """SELECT wbi.paper_id FROM workflow_batch_items wbi
                    JOIN papers p ON p.id=wbi.paper_id
                    WHERE wbi.batch_id=? AND p.abstract IS NOT NULL
                    AND TRIM(p.abstract)!=''""",
                    (task_id,),
                ).fetchall()
            }
        selected = tuple(
            int(paper_id)
            for paper_id, item in items.items()
            if int(paper_id) in ready_ids
            and item.get("status") in {"pending", "processing", "needs_retry"}
        )[:batch_size]
        selected_budget = budget or PersonalBatchBudget(max_ai_calls=batch_size * 3)
        completed: list[int] = []
        failed: list[int] = []
        ai_calls = input_tokens = output_tokens = 0
        budget_exhausted = False
        for paper_id in selected:
            if load_personal_state(self.database, task_id).get("stop_requested"):
                break
            if ai_calls >= selected_budget.max_ai_calls:
                budget_exhausted = True
                break
            try:
                item_result = self._process_one(task_id, paper_id)
                ai_calls += int(item_result.get("ai_calls") or 0)
                input_tokens += int(item_result.get("input_tokens") or 0)
                output_tokens += int(item_result.get("output_tokens") or 0)
                if input_tokens + output_tokens > selected_budget.max_total_tokens:
                    budget_exhausted = True
                if ai_calls >= selected_budget.max_ai_calls:
                    budget_exhausted = True
                if item_result["status"] == "classified":
                    completed.append(paper_id)
                else:
                    failed.append(paper_id)
            except Exception as exc:
                LOGGER.exception("Personal processing failed for paper %s", paper_id)
                _update_personal_item(
                    self.database,
                    task_id,
                    paper_id,
                    status="failed",
                    stage="unexpected_error",
                    error_type=type(exc).__name__,
                    error_message=_safe_error(exc),
                )
                failed.append(paper_id)
            if budget_exhausted:
                break
        remaining = tuple(
            int(paper_id)
            for paper_id, item in load_personal_state(self.database, task_id)[
                "items"
            ].items()
            if int(paper_id) in ready_ids
            and item.get("status") in {"pending", "processing", "needs_retry"}
        )
        result = PersonalBatchResult(
            selected=selected,
            completed=tuple(completed),
            failed=tuple(failed),
            remaining=remaining,
            ai_calls=ai_calls,
            input_tokens=input_tokens,
            output_tokens=output_tokens,
            budget_exhausted=budget_exhausted,
        )
        final_state = load_personal_state(self.database, task_id)
        final_state["last_batch_usage"] = {
            "selected": list(result.selected),
            "completed": list(result.completed),
            "failed": list(result.failed),
            "ai_calls": result.ai_calls,
            "input_tokens": result.input_tokens,
            "output_tokens": result.output_tokens,
            "cost_microunits": 0,
            "budget_exhausted": result.budget_exhausted,
            "recorded_at": utcnow(),
        }
        _save_personal_state(self.database, task_id, final_state)
        return result

    def process_paper(self, task_id: int, paper_id: int) -> dict[str, object]:
        return self._process_one(task_id, paper_id)

    def _process_one(self, task_id: int, paper_id: int) -> dict[str, object]:
        current = load_personal_state(self.database, task_id)["items"].get(str(paper_id))
        if current is None:
            raise ValueError(f"Paper {paper_id} is not in personal task {task_id}")
        _update_personal_item(
            self.database,
            task_id,
            paper_id,
            status="processing",
            stage="abstract_translation",
            attempts=int(current.get("attempts") or 0) + 1,
            error_type=None,
            error_message=None,
        )
        usage = {"ai_calls": 0, "input_tokens": 0, "output_tokens": 0}
        paper = _load_personal_paper(self.database, task_id, paper_id)
        if not paper["abstract"]:
            audit = dict(paper.get("abstract_enrichment") or {})
            status = (
                "needs_retry"
                if audit.get("retryable")
                or audit.get("execution_state") == "pending"
                else "needs_manual_abstract"
            )
            result = {
                "status": status,
                "stage": "awaiting_abstract_retrieval",
                "abstract_reason": audit.get("final_reason")
                or "original_abstract_unavailable",
                "next_retry_at": audit.get("next_retry_at"),
                "provider_attempts": list(audit.get("attempts") or []),
                "classification": {
                    "decision": "needs_abstract",
                    "education_centrality": "none",
                    "economic_relevance": "none",
                    "confidence": "low",
                    "evidence": [],
                    "reason": "Original English abstract unavailable; no AI judgment made.",
                },
                **usage,
            }
            _update_personal_item(self.database, task_id, paper_id, **result)
            return result
        _update_personal_item(
            self.database,
            task_id,
            paper_id,
            status="abstract_ready",
            stage="abstract_translation",
        )
        abstract_stats = self.translation_service.run(
            TranslationParameters(
                review_status=None,
                limit=1,
                missing_only=True,
                paper_ids=(paper_id,),
                translate_titles=False,
                translate_abstracts=True,
                task_id=task_id,
            )
        )
        _merge_usage(usage, abstract_stats)
        if abstract_stats.failed:
            result = {
                "status": "needs_retry",
                "stage": "abstract_translation",
                "error_type": "translation_failed",
                "error_message": "; ".join(abstract_stats.failure_reasons)[:500],
                **usage,
            }
            _update_personal_item(self.database, task_id, paper_id, **result)
            return result
        _update_personal_item(
            self.database,
            task_id,
            paper_id,
            status="classification_ready",
            stage="strong_classification",
        )
        try:
            response = self.classifier.classify(paper)
        except DeepSeekError as exc:
            result = {
                "status": "needs_retry",
                "stage": "strong_classification",
                "error_type": "structured_output_error",
                "error_message": _safe_error(exc),
                "raw_ai_response": exc.raw_response,
                "validation_error": exc.validation_error,
                **usage,
            }
            _update_personal_item(self.database, task_id, paper_id, **result)
            return result
        usage["ai_calls"] += 1
        usage["input_tokens"] += response.input_tokens
        usage["output_tokens"] += response.output_tokens
        decision = response.data
        assert isinstance(decision, PersonalStrongDecision)
        result = {
            "status": "classified",
            "stage": "complete",
            "classification": decision.model_dump(mode="json"),
            "model_name": response.model_name,
            "prompt_version": PERSONAL_PROMPT_VERSION,
            "classified_at": utcnow(),
            **usage,
        }
        _update_personal_item(self.database, task_id, paper_id, **result)
        return result


def save_and_process_manual_abstract(
    workflow: PersonalWorkflowService,
    task_id: int,
    paper_id: int,
    abstract: str,
    source_url: str | None = None,
) -> dict[str, object]:
    saved = save_manual_abstract(workflow.database, paper_id, abstract, source_url)
    if saved.get("error"):
        return saved
    return workflow.process_paper(task_id, paper_id)


def request_personal_stop(database: Database, task_id: int) -> None:
    state = load_personal_state(database, task_id)
    state["stop_requested"] = True
    _save_personal_state(database, task_id, state)


def retry_personal_paper(
    workflow: PersonalWorkflowService, task_id: int, paper_id: int
) -> dict[str, object]:
    item = load_personal_state(workflow.database, task_id)["items"].get(str(paper_id))
    if not item:
        raise ValueError(f"Paper {paper_id} is not in personal task {task_id}")
    return workflow.process_paper(task_id, paper_id)


def retranslate_personal_paper_abstract(
    workflow: PersonalWorkflowService, task_id: int, paper_id: int
) -> TranslationStats:
    """Re-run the Chinese abstract translation for one paper (machine only)."""
    stats = workflow.translation_service.run(
        TranslationParameters(
            review_status=None,
            limit=1,
            missing_only=False,
            paper_ids=(paper_id,),
            translate_titles=False,
            translate_abstracts=True,
            task_id=task_id,
        )
    )
    if stats.failed:
        raise RuntimeError(
            "Abstract translation failed: " + "; ".join(stats.failure_reasons)[:500]
        )
    return stats


def rescreen_personal_paper(
    workflow: PersonalWorkflowService, task_id: int, paper_id: int
) -> dict[str, object]:
    """Force a fresh deterministic screening run for one paper."""
    if workflow.screening_service is None:
        raise ValueError("Deterministic screening service is not configured")
    outcome = workflow.screening_service.screen_paper(
        paper_id, task_id=task_id, force=True
    )
    status = (
        "classified"
        if outcome.status == "classified"
        else "needs_retry"
        if outcome.status == "needs_retry"
        else outcome.status
    )
    _update_personal_item(
        workflow.database,
        task_id,
        paper_id,
        status=status,
        stage="complete" if outcome.status == "classified" else "strong_classification",
        classification=outcome.classification,
        model_name=workflow.screening_service.model,
        prompt_version=SCREENING_PROMPT_VERSION,
        classified_at=utcnow() if outcome.status == "classified" else None,
    )
    return outcome.classification


def clear_personal_stop(database: Database, task_id: int) -> None:
    state = load_personal_state(database, task_id)
    state["stop_requested"] = False
    _save_personal_state(database, task_id, state)


def list_personal_task_papers(
    database: Database,
    task_id: int,
    *,
    state: Mapping[str, object] | None = None,
) -> list[dict[str, object]]:
    resolved_state = dict(state) if state is not None else load_personal_state(database, task_id)
    state_items = resolved_state.get("items")
    if not isinstance(state_items, dict):
        raise ValueError(f"Task {task_id} has invalid personal item state")
    with database.connect() as connection:
        rows = connection.execute(
            """SELECT p.id,p.title,p.abstract,p.authors_json,p.doi,p.source_url,
            p.publisher,
            p.source_metadata_json,
            COALESCE(p.issue_date,p.online_date) publication_date,j.name journal,
            pt.chinese_title,pt.chinese_abstract,pt.title_translation_status,
            pt.title_translation_source,pt.abstract_translation_status,
            pt.abstract_translation_source,pt.model_name translation_model_name,
            pt.prompt_version translation_prompt_version,
            tr.review_status,tr.final_decision,tr.reviewer_notes
            FROM workflow_batch_items wbi
            JOIN papers p ON p.id=wbi.paper_id
            JOIN journals j ON j.id=p.journal_id
            LEFT JOIN paper_translations pt ON pt.paper_id=p.id
            LEFT JOIN workflow_task_reviews tr
              ON tr.task_id=wbi.batch_id AND tr.paper_id=wbi.paper_id
            WHERE wbi.batch_id=? ORDER BY wbi.position,p.id""",
            (task_id,),
        ).fetchall()
    result = []
    for row in rows:
        item = dict(row)
        item["authors"] = json.loads(item.pop("authors_json") or "[]")
        metadata = json.loads(item.pop("source_metadata_json") or "{}")
        item["abstract_source"] = metadata.get("abstract_source") or (
            "crossref" if item.get("abstract") else None
        )
        item["abstract_provenance"] = metadata.get("abstract_provenance")
        item["abstract_enrichment"] = metadata.get("abstract_enrichment")
        item["metadata_response_had_abstract"] = bool(metadata.get("abstract"))
        item["processing"] = dict(state_items.get(str(item["id"])) or {})
        item["human_decision"] = _human_decision(item)
        result.append(item)
    return result


def personal_task_snapshot(database: Database, task_id: int) -> PersonalTaskSnapshot:
    """Load one immutable UI/export snapshot with two fixed database reads."""
    state = load_personal_state(database, task_id)
    papers = tuple(list_personal_task_papers(database, task_id, state=state))
    counts = personal_snapshot_counts(papers)
    return PersonalTaskSnapshot(
        task_id=task_id,
        state=state,
        papers=papers,
        counts=counts,
    )


def personal_snapshot_counts(
    papers: Sequence[Mapping[str, object]],
) -> dict[str, int]:
    counts = {
        "retrieved": len(papers),
        "pending_processing": 0,
        "missing_abstract": 0,
        "needs_retry": 0,
        "review_pending": 0,
        "completed": 0,
        "english_abstracts": 0,
        "chinese_abstracts": 0,
        "automatic_judgments": 0,
        "human_decisions": 0,
    }
    for paper in papers:
        processing = paper.get("processing")
        processing = processing if isinstance(processing, dict) else {}
        status = str(processing.get("status") or "pending")
        abstract_retrieval = processing.get("abstract_retrieval")
        abstract_retrieval = (
            abstract_retrieval if isinstance(abstract_retrieval, dict) else {}
        )
        human_decision = str(paper.get("human_decision") or "undecided")
        classification = processing.get("classification")
        classification = classification if isinstance(classification, dict) else {}
        if human_decision == "undecided":
            counts["pending_processing"] += 1
        if not str(paper.get("abstract") or "").strip():
            counts["missing_abstract"] += 1
        if status in {"needs_retry", "failed"} or abstract_retrieval.get(
            "status"
        ) in {"needs_retry", "failed"}:
            counts["needs_retry"] += 1
        if status == "classified" and human_decision == "undecided":
            counts["review_pending"] += 1
        if human_decision != "undecided":
            counts["completed"] += 1
            counts["human_decisions"] += 1
        if str(paper.get("abstract") or "").strip():
            counts["english_abstracts"] += 1
        if str(paper.get("chinese_abstract") or "").strip():
            counts["chinese_abstracts"] += 1
        if (
            status == "classified"
            and classification.get("decision")
            in {"relevant", "irrelevant", "uncertain"}
        ):
            counts["automatic_judgments"] += 1
    return counts


def list_personal_tasks(database: Database) -> list[dict[str, object]]:
    with database.connect() as connection:
        rows = connection.execute(
            """SELECT id,name,status,created_at,updated_at,source_config_json,
            result_summary_json
            FROM workflow_batches ORDER BY updated_at DESC,id DESC"""
        ).fetchall()
    tasks = []
    for row in rows:
        source = json.loads(row["source_config_json"] or "{}")
        if source.get("runtime_mode") != "personal_local":
            continue
        item = dict(row)
        item["source_config"] = source
        item["result_summary"] = json.loads(item.pop("result_summary_json") or "{}")
        tasks.append(item)
    return tasks


def save_personal_human_decision(
    database: Database,
    task_id: int,
    paper_id: int,
    *,
    decision: str,
    note: str | None = None,
) -> bool:
    changed = save_personal_human_decisions(
        database,
        task_id,
        {paper_id: PersonalReviewDraft(decision=decision, note=note)},
    )
    return changed == (paper_id,)


def personal_review_changes(
    papers: Sequence[Mapping[str, object]],
    drafts: Mapping[int, PersonalReviewDraft],
) -> dict[int, PersonalReviewDraft]:
    """Return only normalized, explicit human edits from one immutable snapshot."""
    originals = {int(paper["id"]): paper for paper in papers}
    changes: dict[int, PersonalReviewDraft] = {}
    for paper_id, draft in drafts.items():
        if paper_id not in originals:
            raise ValueError(f"Paper {paper_id} does not belong to the personal task")
        decision = _normalize_human_decision(draft.decision)
        note = _normalize_note(draft.note)
        original = originals[paper_id]
        original_decision = _normalize_human_decision(
            original.get("human_decision")
        )
        original_note = _normalize_note(original.get("reviewer_notes"))
        if decision == "undecided":
            continue
        if (decision, note) != (original_decision, original_note):
            changes[paper_id] = PersonalReviewDraft(decision=decision, note=note)
    return changes


def save_personal_human_decisions(
    database: Database,
    task_id: int,
    drafts: Mapping[int, PersonalReviewDraft],
    *,
    snapshot: PersonalTaskSnapshot | None = None,
) -> tuple[int, ...]:
    current = snapshot or personal_task_snapshot(database, task_id)
    if current.task_id != task_id:
        raise ValueError("Personal review snapshot belongs to another task")
    changes = personal_review_changes(current.papers, drafts)
    if not changes:
        return ()
    mapping = {
        "related": ("approved", "include"),
        "irrelevant": ("rejected", "exclude"),
        "uncertain": ("needs_revision", None),
    }
    values = {}
    for paper_id, draft in changes.items():
        review_status, final_decision = mapping[draft.decision]
        values[paper_id] = TaskReviewInput(
            review_status=review_status,
            final_decision=final_decision,
            reviewer_notes=draft.note,
            source="v05_human",
        )
    save_task_reviews(database, task_id, values)
    state = load_personal_state(database, task_id)
    papers_by_id = {int(paper["id"]): paper for paper in current.papers}
    gold_candidates = state.setdefault("gold_candidates", [])
    for paper_id, draft in changes.items():
        item = dict(state["items"].get(str(paper_id)) or {})
        paper = papers_by_id[paper_id]
        gold_candidates.append(
            {
                "paper_id": paper_id,
                "user_label": draft.decision,
                "automatic_judgment": item.get("classification"),
                "english_abstract_available": bool(paper.get("abstract")),
                "model_name": item.get("model_name"),
                "prompt_version": item.get("prompt_version"),
                "evidence": dict(item.get("classification") or {}).get(
                    "evidence", []
                ),
                "user_note": draft.note,
                "annotated_at": utcnow(),
                "claim": "benchmark_candidate_only",
            }
        )
    _save_personal_state(database, task_id, state)
    return tuple(changes)


def export_personal_task_csv(
    database: Database,
    task_id: int,
    *,
    papers: Sequence[Mapping[str, object]] | None = None,
) -> str:
    """Personal-only export with Phase 7A AI assistance clearly separated."""
    fields = [
        "paper_id",
        "doi",
        "original_title",
        "chinese_title",
        "authors",
        "journal",
        "publication_date",
        "original_abstract",
        "abstract_source",
        "chinese_abstract",
        "abstract_translation_source",
        "source_url",
        "automatic_decision",
        "education_centrality",
        "economic_relevance",
        "confidence",
        "direct_evidence",
        "ai_reason",
        "human_decision",
        "human_note",
    ]
    output = io.StringIO(newline="")
    writer = csv.DictWriter(output, fieldnames=fields)
    writer.writeheader()
    selected = papers if papers is not None else list_personal_task_papers(database, task_id)
    for paper in selected:
        classification = dict(paper["processing"].get("classification") or {})
        writer.writerow(
            {
                "paper_id": paper["id"],
                "doi": paper["doi"] or "",
                "original_title": paper["title"],
                "chinese_title": paper["chinese_title"] or "",
                "authors": "; ".join(paper["authors"]),
                "journal": paper["journal"],
                "publication_date": paper["publication_date"] or "",
                "original_abstract": paper["abstract"] or "",
                "abstract_source": paper.get("abstract_source") or "",
                "chinese_abstract": paper["chinese_abstract"] or "",
                "abstract_translation_source": paper.get(
                    "abstract_translation_source"
                )
                or "",
                "source_url": paper["source_url"] or "",
                "automatic_decision": classification.get("decision") or "",
                "education_centrality": classification.get("education_centrality")
                or "",
                "economic_relevance": classification.get("economic_relevance") or "",
                "confidence": classification.get("confidence") or "",
                "direct_evidence": " | ".join(classification.get("evidence") or []),
                "ai_reason": classification.get("reason") or "",
                "human_decision": paper["human_decision"],
                "human_note": paper["reviewer_notes"] or "",
            }
        )
    return "\ufeff" + output.getvalue()


def export_personal_task_markdown(
    database: Database,
    task_id: int,
    *,
    papers: Sequence[Mapping[str, object]] | None = None,
) -> str:
    selected = papers if papers is not None else list_personal_task_papers(database, task_id)
    lines = [
        "# Personal Research Export / 个人研究记录",
        "",
        "> AI 判断与人工备注均为个人研究辅助信息，不代表人工最终决定。",
        "",
    ]
    for paper in selected:
        processing = paper.get("processing")
        processing = processing if isinstance(processing, dict) else {}
        classification = processing.get("classification")
        classification = classification if isinstance(classification, dict) else {}
        lines.extend(
            [
                f"## {paper['title']}",
                "",
                f"- Chinese title: {paper.get('chinese_title') or ''}",
                f"- Authors: {'; '.join(paper.get('authors') or [])}",
                f"- Journal: {paper.get('journal') or ''}",
                f"- Publication date: {paper.get('publication_date') or ''}",
                f"- DOI: {paper.get('doi') or ''}",
                f"- Source URL: {paper.get('source_url') or ''}",
                f"- Abstract source: {paper.get('abstract_source') or ''}",
                f"- Abstract translation source: {paper.get('abstract_translation_source') or ''}",
                f"- Automatic decision: {classification.get('decision') or ''}",
                f"- Education centrality: {classification.get('education_centrality') or ''}",
                f"- Economic relevance: {classification.get('economic_relevance') or ''}",
                f"- Confidence: {classification.get('confidence') or ''}",
                f"- Direct evidence: {' | '.join(classification.get('evidence') or [])}",
                f"- AI reason: {classification.get('reason') or ''}",
                f"- Human decision: {paper.get('human_decision') or 'undecided'}",
                f"- Human note: {paper.get('reviewer_notes') or ''}",
                "",
                "### Original Abstract",
                "",
                str(paper.get("abstract") or "Original abstract unavailable."),
                "",
                "### Chinese Abstract Translation",
                "",
                str(paper.get("chinese_abstract") or "Chinese translation unavailable."),
                "",
            ]
        )
    return "\n".join(lines)


def export_personal_mentor_csv(
    papers: Sequence[Mapping[str, object]],
) -> str:
    fields = [
        "paper_id",
        "doi",
        "original_title",
        "chinese_title",
        "authors",
        "journal",
        "publication_date",
        "original_abstract",
        "abstract_source",
        "chinese_abstract",
        "abstract_translation_source",
        "source_url",
        "human_note",
    ]
    output = io.StringIO(newline="")
    writer = csv.DictWriter(output, fieldnames=fields)
    writer.writeheader()
    for paper in papers:
        if paper.get("human_decision") != "related":
            continue
        writer.writerow(
            {
                "paper_id": paper["id"],
                "doi": paper.get("doi") or "",
                "original_title": paper["title"],
                "chinese_title": paper.get("chinese_title") or "",
                "authors": "; ".join(paper.get("authors") or []),
                "journal": paper.get("journal") or "",
                "publication_date": paper.get("publication_date") or "",
                "original_abstract": paper.get("abstract") or "",
                "abstract_source": paper.get("abstract_source") or "",
                "chinese_abstract": paper.get("chinese_abstract") or "",
                "abstract_translation_source": paper.get(
                    "abstract_translation_source"
                )
                or "",
                "source_url": paper.get("source_url") or "",
                "human_note": paper.get("reviewer_notes") or "",
            }
        )
    return "\ufeff" + output.getvalue()


def export_personal_mentor_markdown(
    papers: Sequence[Mapping[str, object]],
) -> str:
    lines = ["# EduEcon Radar 导师版", ""]
    related = [paper for paper in papers if paper.get("human_decision") == "related"]
    if not related:
        return "# EduEcon Radar 导师版\n\n当前任务没有人工确认保留的论文。\n"
    for paper in related:
        lines.extend(
            [
                f"## {paper['title']}",
                "",
                f"**中文标题：** {paper.get('chinese_title') or 'Unavailable'}",
                f"**作者：** {', '.join(paper.get('authors') or []) or 'Unknown'}",
                f"**期刊：** {paper.get('journal') or ''}",
                f"**发表日期：** {paper.get('publication_date') or 'Unavailable'}",
                f"**DOI：** {paper.get('doi') or 'Unavailable'}",
                f"**Source Link：** {paper.get('source_url') or 'Unavailable'}",
                "",
                "### Original Abstract / 论文原版摘要",
                "",
                str(paper.get("abstract") or "Original abstract unavailable."),
                f"Source: {paper.get('abstract_source') or 'Unavailable'}",
                "",
                "### Chinese Abstract Translation / 中文摘要翻译",
                "",
                str(paper.get("chinese_abstract") or "Chinese translation unavailable."),
                f"Source: {paper.get('abstract_translation_source') or 'Unavailable'}",
                "",
                "### Human Note / 人工备注",
                "",
                str(paper.get("reviewer_notes") or ""),
                "",
            ]
        )
    return "\n".join(lines)


def load_personal_state(database: Database, task_id: int) -> dict[str, object]:
    with database.connect() as connection:
        row = connection.execute(
            "SELECT result_summary_json,source_config_json FROM workflow_batches WHERE id=?",
            (task_id,),
        ).fetchone()
    if not row:
        raise ValueError(f"Workflow task not found: {task_id}")
    source = json.loads(row["source_config_json"] or "{}")
    if source.get("runtime_mode") != "personal_local":
        raise ValueError(f"Task {task_id} is not a personal-local task")
    payload = json.loads(row["result_summary_json"] or "{}")
    state = payload.get("personal_local")
    if not isinstance(state, dict):
        raise ValueError(f"Task {task_id} has no personal-local state")
    return state


def _save_personal_state(
    database: Database, task_id: int, state: dict[str, object]
) -> None:
    with database.connect() as connection:
        row = connection.execute(
            "SELECT result_summary_json FROM workflow_batches WHERE id=?", (task_id,)
        ).fetchone()
        if not row:
            raise ValueError(f"Workflow task not found: {task_id}")
        payload = json.loads(row["result_summary_json"] or "{}")
        payload["personal_local"] = state
        connection.execute(
            "UPDATE workflow_batches SET result_summary_json=?,updated_at=? WHERE id=?",
            (json.dumps(payload, ensure_ascii=False, sort_keys=True), utcnow(), task_id),
        )


def _refresh_journal_processing_summary(state: dict[str, object]) -> None:
    results = state.get("journal_results")
    items = state.get("items")
    if not isinstance(results, list) or not isinstance(items, dict):
        return
    for result in results:
        if not isinstance(result, dict):
            continue
        paper_ids = result.get("matched_paper_ids") or result.get("paper_ids") or []
        selected = [
            dict(items.get(str(paper_id)) or {}) for paper_id in paper_ids
        ]
        title_statuses = [
            dict(item.get("title_translation") or {}).get("status")
            for item in selected
        ]
        abstract_statuses = [
            dict(item.get("abstract_retrieval") or {}).get("status")
            for item in selected
        ]
        result["title_translation_succeeded"] = sum(
            value in {"translated", "manual_title_preserved"}
            for value in title_statuses
        )
        result["manual_title_preserved"] = sum(
            value == "manual_title_preserved" for value in title_statuses
        )
        result["title_translation_retry"] = sum(
            value in {"title_translation_pending", "title_translation_retry"}
            for value in title_statuses
        )
        result["abstract_auto_succeeded"] = sum(
            value == "abstract_ready" for value in abstract_statuses
        )
        result["abstract_retry"] = sum(
            value == "needs_retry" for value in abstract_statuses
        )
        result["abstract_needs_manual"] = sum(
            value in {"needs_manual_abstract", "failed"}
            for value in abstract_statuses
        )


def _update_personal_item(
    database: Database,
    task_id: int,
    paper_id: int,
    **updates: object,
) -> None:
    if updates.get("status") not in PERSONAL_ITEM_STATUSES:
        raise ValueError(f"Invalid personal item status: {updates.get('status')}")
    state = load_personal_state(database, task_id)
    items = state["items"]
    current = dict(items.get(str(paper_id)) or {})
    if not current:
        raise ValueError(f"Paper {paper_id} is not in personal task {task_id}")
    current.update(updates)
    current["updated_at"] = utcnow()
    items[str(paper_id)] = current
    _save_personal_state(database, task_id, state)


def _load_personal_paper(
    database: Database, task_id: int, paper_id: int
) -> dict[str, object]:
    with database.connect() as connection:
        row = connection.execute(
            """SELECT p.*,j.name journal
            FROM workflow_batch_items wbi JOIN papers p ON p.id=wbi.paper_id
            JOIN journals j ON j.id=p.journal_id
            WHERE wbi.batch_id=? AND p.id=?""",
            (task_id, paper_id),
        ).fetchone()
    if not row:
        raise ValueError(f"Paper {paper_id} does not belong to task {task_id}")
    paper = dict(row)
    metadata = json.loads(paper.get("source_metadata_json") or "{}")
    paper["abstract_enrichment"] = metadata.get("abstract_enrichment")
    published = paper.get("issue_date") or paper.get("online_date")
    paper["year"] = str(published)[:4] if published else None
    return paper


def _has_direct_evidence(evidence: list[str], source: str) -> bool:
    normalized_source = _evidence_text(source)
    return any(
        len(_evidence_text(item)) >= 12 and _evidence_text(item) in normalized_source
        for item in evidence
    )


def _evidence_text(value: str) -> str:
    return " ".join(re.sub(r"\s+", " ", value).casefold().split())


def _merge_usage(target: dict[str, int], stats: object) -> None:
    target["ai_calls"] += int(getattr(stats, "ai_calls", 0) or 0)
    target["input_tokens"] += int(getattr(stats, "input_tokens", 0) or 0)
    target["output_tokens"] += int(getattr(stats, "output_tokens", 0) or 0)


def _human_decision(row: dict[str, object]) -> str:
    if row.get("final_decision") == "include":
        return "related"
    if row.get("final_decision") == "exclude":
        return "irrelevant"
    if row.get("review_status") == "needs_revision":
        return "uncertain"
    return "undecided"


def _normalize_human_decision(value: object) -> str:
    normalized = str(value or "").strip().casefold() or "undecided"
    if normalized not in HUMAN_DECISIONS:
        raise ValueError("Unsupported personal human decision")
    return normalized


def _normalize_note(value: object) -> str | None:
    normalized = str(value).strip() if value is not None else ""
    return normalized or None


