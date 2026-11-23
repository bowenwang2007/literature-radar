# --- not part of the product core: see docs/PRODUCT_CORE.md ---
# LEGACY (CLI/tests). The v0.5 orchestrator is not on the product path;
# ``save_manual_abstract`` now delegates to ``services/manual_abstract.py``.

from __future__ import annotations

import json
from contextlib import suppress
from dataclasses import asdict, dataclass
from datetime import date
from pathlib import Path
from types import SimpleNamespace
from typing import Any

from eduecon_radar.config import Settings
from eduecon_radar.db import Database, utcnow
from eduecon_radar.domain.semantic_judgment import DeepSeekSemanticAdapter
from eduecon_radar.domain.storage import DomainEvidenceRepository
from eduecon_radar.journals import JournalFile
from eduecon_radar.resources import runtime_path
from eduecon_radar.services.abstract_enrichment import (
    AbstractEnrichmentService,
    CrossrefEnrichmentAdapter,
    DOAJEnrichmentAdapter,
    OpenAlexEnrichmentAdapter,
    RePEcOAIAdapter,
    SemanticScholarEnrichmentAdapter,
    load_repec_series_registry,
)
from eduecon_radar.services.abstract_source_config import load_abstract_sources
from eduecon_radar.services.ai_budget import AIBudgetManager, BudgetBlocked
from eduecon_radar.services.crossref import CrossrefClient
from eduecon_radar.services.deepseek import DeepSeekClient
from eduecon_radar.services.domain_evidence import DomainEvidenceService
from eduecon_radar.services.ingestion import SyncService
from eduecon_radar.services.migrations import (
    require_data_integrity_schema,
    require_p0_hardening_schema,
)
from eduecon_radar.services.pilot_runtime import (
    PilotAuthorizationService,
    PilotReservationGuard,
)
from eduecon_radar.services.resilient_http import ResilientHttpClient
from eduecon_radar.services.runtime_bundle import (
    RuntimeBundleResolver,
    RuntimeModelIdentity,
)
from eduecon_radar.services.semantic_screening import (
    SemanticScreeningAnalysis,
    SemanticScreeningService,
)
from eduecon_radar.services.task_lock import (
    TaskLease,
    TaskLeaseLost,
    TaskLeaseManager,
)
from eduecon_radar.services.task_review import (
    TaskReviewInput,
    mark_task_recovery,
    save_task_reviews,
    task_review_counts,
)
from eduecon_radar.services.translation import (
    TranslationParameters,
    TranslationService,
)
from eduecon_radar.services.workflow import (
    add_batch_items,
    batch_paper_ids,
    create_workflow_batch,
    get_workflow_batch,
)
from eduecon_radar.services.workflow_resume import (
    WorkflowCheckpointStore,
    fingerprint,
)

PROTOTYPE_STEPS = (
    "search_scope",
    "automatic_processing",
    "missed_recovery",
    "final_review",
    "export",
)
V05_POLICY_VERSION = "v05-personal-high-recall-v1"


@dataclass(frozen=True)
class LiteratureSearchScope:
    journals: tuple[str, ...]
    date_from: date
    date_to: date
    per_journal_limit: int
    preset_name: str | None = None

    def __post_init__(self) -> None:
        normalized = tuple(dict.fromkeys(item.strip() for item in self.journals if item.strip()))
        object.__setattr__(self, "journals", normalized)
        if not normalized:
            raise ValueError("请至少选择一本期刊")
        if self.date_from > self.date_to:
            raise ValueError("起始日期不得晚于结束日期")
        if not 1 <= self.per_journal_limit <= 1000:
            raise ValueError("每刊最大数量必须在 1 到 1000 之间")


@dataclass(frozen=True)
class FinalReviewInput:
    keep: bool | None = None
    chinese_title: str | None = None
    chinese_abstract: str | None = None
    note: str | None = None
    decision: str = "undecided"

    def resolved_decision(self) -> str:
        if self.decision not in {"include", "exclude", "undecided"}:
            raise ValueError(f"Unsupported human final decision: {self.decision}")
        if self.decision != "undecided":
            return self.decision
        if self.keep is True:
            return "include"
        if self.keep is False:
            return "exclude"
        return "undecided"


def create_prototype_literature_task(
    database: Database,
    *,
    name: str,
    scope: LiteratureSearchScope,
    pilot_context: dict[str, object] | None = None,
) -> int:
    """Create the durable v0.5 batch; retained name keeps Phase 0-2 callers compatible."""
    source_config: dict[str, object] = {
        "journals": list(scope.journals),
        "date_from": scope.date_from.isoformat(),
        "date_to": scope.date_to.isoformat(),
        "since": scope.date_from.isoformat(),
        "limit": scope.per_journal_limit,
        "preset_name": scope.preset_name,
        "v05_literature": True,
    }
    if pilot_context:
        source_config.update(pilot_context)
    task_id = create_workflow_batch(
        database,
        name=name,
        source_type="sync",
        source_config=source_config,
    )
    _save_state(
        database,
        task_id,
        {
            "current_step": "automatic_processing",
            "mock_only": False,
            "decision_policy": V05_POLICY_VERSION,
            "result": _empty_result(),
            "decisions": {},
            "recovered_ids": [],
            "candidate_ids": [],
            "final_reviewed_ids": [],
            "pilot": bool(pilot_context),
            "release_status": "NOT_RELEASED" if pilot_context else "released",
        },
    )
    return task_id


def create_controlled_pilot_literature_task(
    database: Database,
    *,
    name: str,
    scope: LiteratureSearchScope,
    authorization_service: PilotAuthorizationService,
    authorization_id: str,
    models: RuntimeModelIdentity,
    task_key: str,
) -> int:
    authorization = authorization_service.require(authorization_id, models)
    requested_papers = len(scope.journals) * scope.per_journal_limit
    claim = authorization_service.claim_task(
        authorization_id=authorization_id,
        models=models,
        task_key=task_key,
        requested_papers=requested_papers,
    )
    if claim.get("task_id") is not None:
        return int(claim["task_id"])
    existing_task_id = _controlled_pilot_task_for_key(
        database,
        authorization_id=authorization.authorization_id,
        task_key=task_key,
    )
    if existing_task_id is not None:
        authorization_service.bind_task(task_key, existing_task_id)
        return existing_task_id
    snapshot = authorization_service.runtime_snapshot(models)
    pilot_context = {
        "runtime_mode": "controlled_pilot",
        "pilot_authorization_id": authorization.authorization_id,
        "pilot_task_key": task_key,
        "runtime_config_sha256": authorization.runtime_config_sha256,
        "prompt_bundle_sha256": authorization.prompt_bundle_sha256,
        "pilot_models_snapshot": authorization.models,
        "pilot_limits_snapshot": asdict(authorization.limits),
        "runtime_bundle_sha256": snapshot["runtime"].bundle_sha256,
    }
    try:
        task_id = create_prototype_literature_task(
            database, name=name, scope=scope, pilot_context=pilot_context
        )
    except Exception:
        if (
            _controlled_pilot_task_for_key(
                database,
                authorization_id=authorization.authorization_id,
                task_key=task_key,
            )
            is None
        ):
            authorization_service.release_unbound_task(task_key)
        raise
    authorization_service.bind_task(task_key, task_id)
    return task_id


def _controlled_pilot_task_for_key(
    database: Database,
    *,
    authorization_id: str,
    task_key: str,
) -> int | None:
    matches: list[int] = []
    with database.connect() as connection:
        rows = connection.execute(
            "SELECT id,source_config_json FROM workflow_batches ORDER BY id"
        ).fetchall()
    for row in rows:
        source_config = _safe_json_object(row["source_config_json"])
        if (
            source_config.get("runtime_mode") == "controlled_pilot"
            and source_config.get("pilot_authorization_id") == authorization_id
            and source_config.get("pilot_task_key") == task_key
        ):
            matches.append(int(row["id"]))
    if len(matches) > 1:
        raise RuntimeError("controlled_pilot_task_key_conflict")
    return matches[0] if matches else None


def set_prototype_task_step(
    database: Database,
    task_id: int,
    step: str,
    *,
    mock_result: dict[str, int] | None = None,
) -> None:
    if step not in PROTOTYPE_STEPS:
        raise ValueError(f"Unsupported literature task step: {step}")
    state = prototype_task_state(database, task_id)
    previous = str(state["current_step"])
    previous_index = PROTOTYPE_STEPS.index(previous)
    requested_index = PROTOTYPE_STEPS.index(step)
    if requested_index < previous_index:
        raise ValueError("Literature tasks cannot move backwards")
    if requested_index > previous_index + 1:
        raise ValueError("Literature tasks must advance one step at a time")
    payload = _state_payload(database, task_id)
    payload["current_step"] = step
    if mock_result is not None:
        payload["result"] = mock_result
    _save_state(database, task_id, payload)


def prototype_task_state(database: Database, task_id: int) -> dict[str, object]:
    batch = get_workflow_batch(database, task_id)
    source_config = dict(batch.get("source_config") or {})
    if not (source_config.get("v05_literature") or source_config.get("prototype_only")):
        raise ValueError("This is not a v0.5 literature task")
    payload = _state_payload_from_batch(batch)
    return {
        "task_id": task_id,
        "name": batch["name"],
        "status": batch["status"],
        "scope": source_config,
        **payload,
    }


class V05LiteratureOrchestrator:
    """Thin integration over existing sync, translation, evidence and semantic services."""

    def __init__(
        self,
        *,
        database: Database,
        enrichment_service: AbstractEnrichmentService | None = None,
        journal_file: JournalFile,
        sync_service: SyncService,
        translation_service: TranslationService,
        evidence_service: DomainEvidenceService,
        semantic_service: SemanticScreeningService,
        lease_manager: TaskLeaseManager | None = None,
        runtime_identity: dict[str, object] | None = None,
    ) -> None:
        self.database = database
        self.journal_file = journal_file
        self.enrichment_service = enrichment_service
        self.sync_service = sync_service
        self.translation_service = translation_service
        self.evidence_service = evidence_service
        self.semantic_service = semantic_service
        self.lease_manager = lease_manager
        self.runtime_identity = dict(runtime_identity or {"mode": "injected_test_runtime"})
        self._active_lease: TaskLease | None = None

    def run(self, task_id: int) -> dict[str, object]:
        if self.lease_manager is None:
            return self._run_with_lease(task_id)
        lease = self.lease_manager.acquire(
            task_id,
            owner_id=self.lease_manager.new_owner_id(),
            stage="automatic_processing",
        )
        self._active_lease = lease
        try:
            return self._run_with_lease(task_id)
        finally:
            with suppress(TaskLeaseLost):
                self.lease_manager.release(lease)
            self._active_lease = None

    def _run_with_lease(self, task_id: int) -> dict[str, object]:
        budget_manager = getattr(self.translation_service, "budget_manager", None)
        if budget_manager is not None:
            budget_manager.ensure_default_budget(task_id)
        state = prototype_task_state(self.database, task_id)
        scope = state["scope"]
        assert isinstance(scope, dict)
        result = dict(state.get("result") or _empty_result())
        decisions = dict(state.get("decisions") or {})
        failures: dict[str, str] = dict(state.get("failures") or {})
        batch = get_workflow_batch(self.database, task_id)
        checkpoints = WorkflowCheckpointStore(self.database)
        checkpoints.initialize_all(task_id)
        if (
            checkpoints.get(task_id, "composite").status == "completed"
            and state.get("current_step") != "automatic_processing"
        ):
            return state
        self._set_batch_status(task_id, "running")
        try:
            if not batch.get("sync_completed_at"):
                self._guard("sync")
                scope_fp = fingerprint(scope)
                sync_checkpoint = checkpoints.start(
                    task_id,
                    "sync",
                    total=len(tuple(scope["journals"])),
                    config_fp="crossref-sync-v1",
                    input_fp=scope_fp,
                )
                selected = _selected_journals(self.journal_file, tuple(scope["journals"]))
                sync_stats = self.sync_service.sync(
                    selected,
                    since=date.fromisoformat(str(scope["date_from"])),
                    until=date.fromisoformat(str(scope["date_to"])),
                    limit=int(scope["limit"]),
                )
                self._guard("sync")
                exact_ids = tuple(int(item) for item in sync_stats["affected_paper_ids"])
                add_batch_items(self.database, task_id, exact_ids)
                result.update(
                    {
                        "journals_checked": int(sync_stats["journals_checked"]),
                        "records_fetched": int(sync_stats["records_fetched"]),
                        "records_inserted": int(sync_stats["records_inserted"]),
                        "records_updated": int(sync_stats["records_updated"]),
                        "sync_failed": int(sync_stats["records_failed"]),
                    }
                )
                self._mark_batch_stage(task_id, "sync_completed_at")
                checkpoints.finish(
                    task_id,
                    "sync",
                    sync_checkpoint,
                    succeeded=int(sync_stats["journals_checked"]),
                    failed=int(sync_stats["records_failed"]),
                    failed_item_ids=(),
                )
            elif checkpoints.get(task_id, "sync").status != "completed":
                sync_checkpoint = checkpoints.start(
                    task_id,
                    "sync",
                    total=0,
                    config_fp="crossref-sync-v1",
                    input_fp=fingerprint(scope),
                )
                checkpoints.finish(task_id, "sync", sync_checkpoint, succeeded=0, skipped=1)

            paper_ids = tuple(batch_paper_ids(self.database, task_id))
            result["total"] = len(paper_ids)
            if self.enrichment_service:
                input_fp = fingerprint(paper_ids)
                config_fp = (
                    f"abstract-enrichment-{self.enrichment_service.policy_version}"
                )
                if checkpoints.should_run(task_id, "abstract_enrichment", config_fp, input_fp):
                    self._guard("abstract_enrichment")
                    checkpoint = checkpoints.start(
                        task_id,
                        "abstract_enrichment",
                        total=len(paper_ids),
                        config_fp=config_fp,
                        input_fp=input_fp,
                        item_ids=paper_ids,
                    )
                    enrich_stats = self.enrichment_service.enrich_papers(
                        paper_ids, workflow_batch_id=task_id
                    )
                    self._guard("abstract_enrichment")
                    result["enriched"] = enrich_stats.enriched
                    result["enrichment_failed"] = enrich_stats.failed
                    result["enrichment_by_source"] = dict(enrich_stats.by_source)
                    checkpoints.finish(
                        task_id,
                        "abstract_enrichment",
                        checkpoint,
                        succeeded=enrich_stats.enriched,
                        failed=enrich_stats.failed,
                        skipped=enrich_stats.skipped_have_abstract,
                        failed_item_ids=_abstract_failed_ids(self.database, paper_ids),
                    )
            input_fp = fingerprint(paper_ids)
            title_fp = "title-translation-6c.1"
            if checkpoints.should_run(task_id, "title_translation", title_fp, input_fp):
                self._guard("title_translation")
                checkpoint = checkpoints.start(
                    task_id,
                    "title_translation",
                    total=len(paper_ids),
                    config_fp=title_fp,
                    input_fp=input_fp,
                    item_ids=paper_ids,
                )
                title_stats = _translate_in_chunks(
                    self.translation_service,
                    paper_ids,
                    titles=True,
                    abstracts=False,
                    task_id=task_id,
                )
                self._guard("title_translation")
                result["title_translated"] = _count_translated_titles(self.database, paper_ids)
                result["title_translation_failed"] = title_stats["failed"]
                checkpoints.finish(
                    task_id,
                    "title_translation",
                    checkpoint,
                    succeeded=result["title_translated"],
                    failed=title_stats["failed"],
                    skipped=title_stats.get("skipped", 0),
                    failed_item_ids=_missing_translation_ids(
                        self.database, paper_ids, "chinese_title"
                    ),
                )

            evidence_by_id: dict[int, Any] = {}
            evidence_fp = fingerprint(
                (
                    paper_ids,
                    getattr(getattr(self.evidence_service, "bundle", None), "config_sha256", ""),
                )
            )
            evidence_checkpoint = checkpoints.start(
                task_id,
                "evidence",
                total=len(paper_ids),
                config_fp="domain-evidence-v1",
                input_fp=evidence_fp,
                item_ids=paper_ids,
            )
            evidence_failed_ids: list[int] = []
            for paper_id in paper_ids:
                try:
                    self._guard("evidence")
                    bundle = getattr(self.evidence_service, "bundle", None)
                    stored = _stored_evidence(
                        self.database,
                        paper_id,
                        task_id,
                        getattr(bundle, "config_sha256", None),
                    )
                    evidence_by_id[paper_id] = stored or self.evidence_service.analyze_paper(
                        paper_id, workflow_batch_id=task_id
                    )
                except Exception as exc:  # one paper must not stop its batch
                    failures[str(paper_id)] = f"evidence: {type(exc).__name__}: {str(exc)[:300]}"
                    evidence_failed_ids.append(paper_id)

            checkpoints.finish(
                task_id,
                "evidence",
                evidence_checkpoint,
                succeeded=len(evidence_by_id),
                failed=len(evidence_failed_ids),
                failed_item_ids=evidence_failed_ids,
                errors={str(item): failures[str(item)] for item in evidence_failed_ids},
            )

            prior_fast = checkpoints.get(task_id, "fast_semantic")
            retry_semantic_ids = tuple(prior_fast.failed_item_ids or prior_fast.pending_item_ids)
            semantic_ids = (
                retry_semantic_ids
                if prior_fast.status in {"partial_failed", "failed", "running"}
                and retry_semantic_ids
                else tuple(evidence_by_id)
            )
            semantic_fp = fingerprint(
                (
                    semantic_ids,
                    getattr(getattr(self.semantic_service, "config", None), "version", ""),
                )
            )
            fast_checkpoint = checkpoints.start(
                task_id,
                "fast_semantic",
                total=len(evidence_by_id),
                config_fp="fast-semantic-v1",
                input_fp=semantic_fp,
                item_ids=semantic_ids,
            )
            semantic = self.semantic_service.analyze_many(semantic_ids, workflow_batch_id=task_id)
            self._guard("composite")
            for paper_id, error in semantic.failures.items():
                failures[str(paper_id)] = f"semantic: {error}"
            for paper_id in semantic.completed:
                if str(failures.get(str(paper_id), "")).startswith("semantic:"):
                    failures.pop(str(paper_id), None)
            semantic_failed_ids = list(semantic.failures)
            checkpoints.finish(
                task_id,
                "fast_semantic",
                fast_checkpoint,
                succeeded=min(len(evidence_by_id), prior_fast.succeeded + len(semantic.completed)),
                failed=len(semantic_failed_ids),
                failed_item_ids=semantic_failed_ids,
                errors={str(item): str(semantic.failures[item]) for item in semantic_failed_ids},
            )
            prior_strong = checkpoints.get(task_id, "strong_semantic")
            strong_checkpoint = checkpoints.start(
                task_id,
                "strong_semantic",
                total=len(evidence_by_id),
                config_fp="strong-semantic-v1",
                input_fp=semantic_fp,
                item_ids=tuple(semantic.completed),
            )
            strong_failed_ids = [
                paper_id
                for paper_id, analysis in semantic.completed.items()
                if analysis.strong_failed
            ]
            checkpoints.finish(
                task_id,
                "strong_semantic",
                strong_checkpoint,
                succeeded=min(
                    len(evidence_by_id),
                    prior_strong.succeeded + len(semantic.completed) - len(strong_failed_ids),
                ),
                failed=len(strong_failed_ids),
                failed_item_ids=strong_failed_ids,
            )
            prior_composite = checkpoints.get(task_id, "composite")
            composite_checkpoint = checkpoints.start(
                task_id,
                "composite",
                total=len(evidence_by_id),
                config_fp=V05_POLICY_VERSION,
                input_fp=semantic_fp,
                item_ids=tuple(semantic.completed),
            )

            for paper_id in paper_ids:
                evidence = evidence_by_id.get(paper_id)
                analysis = semantic.completed.get(paper_id)
                if analysis is None and str(paper_id) in decisions:
                    continue
                decisions[str(paper_id)] = _decision_detail(
                    self.database, paper_id, evidence, analysis, failures.get(str(paper_id))
                )
            checkpoints.finish(
                task_id,
                "composite",
                composite_checkpoint,
                succeeded=min(
                    len(evidence_by_id),
                    prior_composite.succeeded + len(semantic.completed),
                ),
                failed=len(semantic_failed_ids),
                failed_item_ids=semantic_failed_ids,
            )

            new_counts = _result_counts(decisions, evidence_by_id, semantic.completed)
            new_counts["fast_semantic_completed"] = min(
                len(evidence_by_id),
                int(result.get("fast_semantic_completed", 0)) + len(semantic.completed),
            )
            new_counts["strong_review_completed"] = min(
                len(evidence_by_id),
                int(result.get("strong_review_completed", 0))
                + sum(item.strong is not None for item in semantic.completed.values()),
            )
            new_counts["composite_completed"] = min(
                len(evidence_by_id),
                int(result.get("composite_completed", 0)) + len(semantic.completed),
            )
            result.update(new_counts)
            payload = _state_payload(self.database, task_id)
            payload.update(
                {
                    "current_step": "missed_recovery",
                    "mock_only": False,
                    "decision_policy": V05_POLICY_VERSION,
                    "result": result,
                    "decisions": decisions,
                    "failures": failures,
                    "runtime_bundle": self.runtime_identity,
                }
            )
            self._save_task_state(task_id, payload, status="ready")
            recovery_ids = tuple(
                int(item)
                for item, detail in decisions.items()
                if detail.get("decision") == "exclude"
            )
            checkpoints.start(
                task_id,
                "missed_recovery",
                total=len(recovery_ids),
                config_fp=V05_POLICY_VERSION,
                input_fp=fingerprint(recovery_ids),
                item_ids=recovery_ids,
            )
            self._mark_batch_stage(task_id, "screening_completed_at")
            return prototype_task_state(self.database, task_id)
        except BudgetBlocked as exc:
            payload = _state_payload(self.database, task_id)
            payload["budget_status"] = "budget_blocked"
            payload["budget_blocked_reason"] = exc.reason
            self._save_task_state(task_id, payload, status="paused")
            raise
        except Exception as exc:
            self._set_batch_status(task_id, "failed", error=str(exc)[:1000])
            raise

    def confirm_recovery(self, task_id: int, recovered_ids: tuple[int, ...]) -> dict[str, object]:
        if self.lease_manager is None:
            return self._confirm_recovery_with_lease(task_id, recovered_ids)
        lease = self.lease_manager.acquire(
            task_id,
            owner_id=self.lease_manager.new_owner_id(),
            stage="missed_recovery",
        )
        self._active_lease = lease
        try:
            return self._confirm_recovery_with_lease(task_id, recovered_ids)
        finally:
            with suppress(TaskLeaseLost):
                self.lease_manager.release(lease)
            self._active_lease = None

    def _confirm_recovery_with_lease(
        self, task_id: int, recovered_ids: tuple[int, ...]
    ) -> dict[str, object]:
        state = prototype_task_state(self.database, task_id)
        decisions = dict(state.get("decisions") or {})
        excluded = {
            int(paper_id)
            for paper_id, detail in decisions.items()
            if detail.get("decision") == "exclude"
        }
        recovered = tuple(dict.fromkeys(int(item) for item in recovered_ids))
        if not set(recovered) <= excluded:
            raise ValueError("Only papers in this task's exclude set may be recovered")
        candidates = tuple(
            int(paper_id)
            for paper_id, detail in decisions.items()
            if detail.get("semantic_decision") in {"include", "borderline"}
            or detail.get("execution_state")
            in {"retryable_error", "permanent_error", "needs_abstract", "needs_manual_review"}
            or int(paper_id) in recovered
        )
        payload = _state_payload(self.database, task_id)
        payload["recovered_ids"] = list(recovered)
        payload["candidate_ids"] = list(candidates)
        for paper_id in excluded:
            mark_task_recovery(
                self.database, task_id, paper_id, recovered=paper_id in set(recovered)
            )
        self._save_task_state(task_id, payload)
        checkpoints = WorkflowCheckpointStore(self.database)
        recovery_checkpoint = checkpoints.get(task_id, "missed_recovery")
        checkpoints.finish(
            task_id,
            "missed_recovery",
            recovery_checkpoint,
            succeeded=len(excluded),
            skipped=max(0, len(excluded) - len(recovered)),
        )
        translation_checkpoint = checkpoints.start(
            task_id,
            "abstract_translation",
            total=len(candidates),
            config_fp="abstract-translation-v1",
            input_fp=fingerprint(candidates),
            item_ids=candidates,
        )
        self._guard("abstract_translation")
        abstract_stats = _translate_in_chunks(
            self.translation_service,
            candidates,
            titles=False,
            abstracts=True,
            task_id=task_id,
        )
        self._guard("abstract_translation")
        result = dict(payload.get("result") or {})
        result["abstract_translated"] = _count_translated_abstracts(self.database, candidates)
        result["abstract_unavailable"] = abstract_stats["unavailable"]
        result["abstract_translation_failed"] = abstract_stats["failed"]
        payload["result"] = result
        payload["current_step"] = "final_review"
        self._save_task_state(task_id, payload, status="ready_for_review")
        missing_abstract_translations = _missing_translation_ids(
            self.database, candidates, "chinese_abstract"
        )
        checkpoints.finish(
            task_id,
            "abstract_translation",
            translation_checkpoint,
            succeeded=result["abstract_translated"],
            failed=abstract_stats["failed"],
            skipped=abstract_stats["unavailable"],
            failed_item_ids=missing_abstract_translations,
        )
        checkpoints.start(
            task_id,
            "final_review",
            total=len(candidates),
            config_fp="human-review-v1",
            input_fp=fingerprint(candidates),
            item_ids=candidates,
        )
        self._mark_batch_stage(task_id, "translation_completed_at")
        return prototype_task_state(self.database, task_id)

    def retry_abstract_failures(
        self, task_id: int, *, temporary_only: bool = False, all_missing: bool = False
    ) -> dict[str, object]:
        if self.lease_manager is None:
            return self._retry_abstract_failures_with_lease(
                task_id, temporary_only=temporary_only, all_missing=all_missing
            )
        lease = self.lease_manager.acquire(
            task_id,
            owner_id=self.lease_manager.new_owner_id(),
            stage="abstract_enrichment",
        )
        self._active_lease = lease
        try:
            return self._retry_abstract_failures_with_lease(
                task_id, temporary_only=temporary_only, all_missing=all_missing
            )
        finally:
            with suppress(TaskLeaseLost):
                self.lease_manager.release(lease)
            self._active_lease = None

    def _retry_abstract_failures_with_lease(
        self, task_id: int, *, temporary_only: bool = False, all_missing: bool = False
    ) -> dict[str, object]:
        if not self.enrichment_service:
            return prototype_task_state(self.database, task_id)
        checkpoints = WorkflowCheckpointStore(self.database)
        previous = checkpoints.get(task_id, "abstract_enrichment")
        if temporary_only and not previous.retryable:
            return prototype_task_state(self.database, task_id)
        paper_ids = tuple(batch_paper_ids(self.database, task_id))
        target_ids = (
            _papers_missing_abstract(self.database, paper_ids)
            if all_missing
            else tuple(previous.failed_item_ids or previous.pending_item_ids)
        )
        if temporary_only:
            target_ids = tuple(
                item for item in target_ids if _abstract_retryable(self.database, item)
            )
        if not target_ids:
            return prototype_task_state(self.database, task_id)
        checkpoint = checkpoints.start(
            task_id,
            "abstract_enrichment",
            total=len(target_ids),
            config_fp=f"abstract-enrichment-{self.enrichment_service.policy_version}",
            input_fp=fingerprint(target_ids),
            item_ids=target_ids,
        )
        self._guard("abstract_enrichment")
        stats = self.enrichment_service.enrich_papers(
            target_ids, workflow_batch_id=task_id, retry_failed_only=True
        )
        self._guard("abstract_enrichment")
        failed_ids = _abstract_failed_ids(self.database, target_ids)
        checkpoints.finish(
            task_id,
            "abstract_enrichment",
            checkpoint,
            succeeded=stats.enriched,
            failed=len(failed_ids),
            skipped=stats.skipped_have_abstract,
            failed_item_ids=failed_ids,
            retryable=any(_abstract_retryable(self.database, item) for item in failed_ids),
        )
        payload = _state_payload(self.database, task_id)
        result = dict(payload.get("result") or {})
        result["enriched"] = int(result.get("enriched", 0)) + stats.enriched
        result["enrichment_failed"] = len(_papers_missing_abstract(self.database, paper_ids))
        by_source = dict(result.get("enrichment_by_source") or {})
        for source, count in stats.by_source.items():
            by_source[source] = int(by_source.get(source, 0)) + count
        result["enrichment_by_source"] = by_source
        payload["result"] = result
        self._save_task_state(task_id, payload)
        return prototype_task_state(self.database, task_id)

    def run_stage(
        self, batch_id: int, stage: str, item_ids: tuple[int, ...] | None = None
    ) -> dict[str, object]:
        if stage == "abstract_enrichment":
            return self.retry_abstract_failures(batch_id)
        return self.run(batch_id)

    def save_and_reprocess_manual_abstract(
        self,
        task_id: int,
        paper_id: int,
        abstract: str,
        source_url: str | None = None,
    ) -> dict[str, object]:
        """Save one manual abstract, translate it, and rerun only that paper's domain chain."""
        if self.lease_manager is None:
            return self._save_and_reprocess_manual_abstract(
                task_id, paper_id, abstract, source_url
            )
        lease = self.lease_manager.acquire(
            task_id,
            owner_id=self.lease_manager.new_owner_id(),
            stage="manual_abstract_reprocess",
        )
        self._active_lease = lease
        try:
            return self._save_and_reprocess_manual_abstract(
                task_id, paper_id, abstract, source_url
            )
        finally:
            with suppress(TaskLeaseLost):
                self.lease_manager.release(lease)
            self._active_lease = None

    def _save_and_reprocess_manual_abstract(
        self,
        task_id: int,
        paper_id: int,
        abstract: str,
        source_url: str | None,
    ) -> dict[str, object]:
        if paper_id not in set(batch_paper_ids(self.database, task_id)):
            raise ValueError(f"Paper {paper_id} does not belong to task {task_id}")
        payload = _state_payload(self.database, task_id)
        reprocessing = dict(payload.get("manual_reprocessing") or {})
        reprocessing[str(paper_id)] = {
            "status": "running",
            "stages": [
                "manual_abstract",
                "abstract_translation",
                "evidence",
                "fast_semantic",
                "strong_semantic_if_escalated",
                "composite",
            ],
            "updated_at": utcnow(),
        }
        payload["manual_reprocessing"] = reprocessing
        self._save_task_state(task_id, payload)
        try:
            self._guard("manual_abstract")
            translation = translate_manual_abstract(
                self.database,
                paper_id,
                abstract,
                source_url,
                self.translation_service,
                task_id=task_id,
            )
            if "error" in translation:
                raise ValueError(str(translation["error"]))
            if not translation.get("abstract_changed", True):
                reprocessing[str(paper_id)] = {
                    "status": "completed",
                    "execution_state": "unchanged_source_no_semantic_rerun",
                    "updated_at": utcnow(),
                }
                payload["manual_reprocessing"] = reprocessing
                self._save_task_state(task_id, payload)
                return {
                    "paper_id": paper_id,
                    "translation": translation,
                    "decision": dict(
                        (payload.get("decisions") or {}).get(str(paper_id)) or {}
                    ),
                    "status": "completed",
                    "skipped": "英文摘要内容未变化，未重复运行下游语义处理",
                }
            self._guard("evidence")
            evidence = self.evidence_service.analyze_paper(
                paper_id, workflow_batch_id=task_id
            )
            self._guard("fast_semantic")
            semantic = self.semantic_service.analyze_many(
                (paper_id,), workflow_batch_id=task_id
            )
            analysis = semantic.completed.get(paper_id)
            failure = semantic.failures.get(paper_id)
            detail = _decision_detail(
                self.database, paper_id, evidence, analysis, failure
            )
            decisions = dict(payload.get("decisions") or {})
            decisions[str(paper_id)] = detail
            payload["decisions"] = decisions
            candidates = {int(item) for item in payload.get("candidate_ids") or []}
            candidates.add(paper_id)
            payload["candidate_ids"] = sorted(candidates)
            reprocessing[str(paper_id)] = {
                "status": (
                    "completed"
                    if detail.get("execution_state")
                    in {"completed", "needs_manual_review"}
                    else "retryable_error"
                ),
                "execution_state": detail.get("execution_state"),
                "updated_at": utcnow(),
            }
            payload["manual_reprocessing"] = reprocessing
            self._save_task_state(task_id, payload)
            return {
                "paper_id": paper_id,
                "translation": translation,
                "decision": detail,
                "status": reprocessing[str(paper_id)]["status"],
            }
        except Exception as exc:
            reprocessing[str(paper_id)] = {
                "status": "retryable_error",
                "error": f"{type(exc).__name__}: {str(exc)[:300]}",
                "updated_at": utcnow(),
            }
            payload["manual_reprocessing"] = reprocessing
            decisions = dict(payload.get("decisions") or {})
            decisions[str(paper_id)] = {
                "decision": None,
                "semantic_decision": None,
                "execution_state": "retryable_error",
                "human_final_decision": "undecided",
                "decision_basis": "manual_reprocessing_failed",
                "reason": reprocessing[str(paper_id)]["error"],
                "semantic_failed": True,
            }
            payload["decisions"] = decisions
            self._save_task_state(task_id, payload)
            raise

    def _mark_batch_stage(self, task_id: int, field: str) -> None:
        if field not in {
            "sync_completed_at",
            "screening_completed_at",
            "translation_completed_at",
        }:
            raise ValueError("Unsupported batch stage")
        with self.database.connect() as connection:
            self._assert_lease(connection)
            connection.execute(
                f"UPDATE workflow_batches SET {field}=COALESCE({field},?),updated_at=? WHERE id=?",  # noqa: S608 -- field is allow-listed above
                (utcnow(), utcnow(), task_id),
            )

    def _set_batch_status(self, task_id: int, status: str, *, error: str | None = None) -> None:
        with self.database.connect() as connection:
            self._assert_lease(connection)
            connection.execute(
                "UPDATE workflow_batches SET status=?,last_error=?,updated_at=? WHERE id=?",
                (status, error, utcnow(), task_id),
            )

    def _save_task_state(
        self,
        task_id: int,
        payload: dict[str, Any],
        *,
        status: str | None = None,
        completed: bool = False,
    ) -> None:
        _save_state(
            self.database,
            task_id,
            payload,
            status=status,
            completed=completed,
            lease_manager=self.lease_manager,
            lease=self._active_lease,
        )

    def _guard(self, stage: str) -> None:
        if self.lease_manager is None or self._active_lease is None:
            return
        self._active_lease = self.lease_manager.heartbeat(
            self._active_lease, stage=stage
        )

    def _assert_lease(self, connection) -> None:
        if self.lease_manager is not None and self._active_lease is not None:
            self.lease_manager.assert_current(self._active_lease, connection=connection)


def build_live_v05_orchestrator(
    database: Database,
    settings: Settings,
    journal_file: JournalFile,
    *,
    runtime_mode: str = "production",
    allow_unreleased_bundle: bool = False,
    pilot_authorization_id: str | None = None,
    pilot_authorization_path: Path | None = None,
    pilot_task_id: int | None = None,
) -> V05LiteratureOrchestrator:
    require_p0_hardening_schema(database)
    if not settings.deepseek_api_key:
        raise RuntimeError("DEEPSEEK_API_KEY is not configured")
    resolver = RuntimeBundleResolver(settings.project_root)
    models = RuntimeModelIdentity(
        provider="deepseek-compatible",
        fast=settings.deepseek_model_fast,
        strong=settings.deepseek_model_strong,
        translation=settings.deepseek_model_fast,
    )
    pilot_service = None
    if runtime_mode == "controlled_pilot":
        if allow_unreleased_bundle or not pilot_authorization_id:
            raise RuntimeError("controlled_pilot requires one explicit authorization")
        pilot_service = PilotAuthorizationService(
            settings.project_root, authorization_path=pilot_authorization_path
        )
        authorization = pilot_service.require(
            pilot_authorization_id, models, task_id=pilot_task_id
        )
        if authorization.scope == "resume_existing_task":
            if pilot_task_id is None:
                raise RuntimeError(
                    "controlled_pilot resume authorization requires its exact task"
                )
            authorization = pilot_service.require_resume_task(
                database, pilot_authorization_id, models, pilot_task_id
            )
            pilot_service.bind_resume_use(pilot_authorization_id, pilot_task_id)
        runtime = resolver.resolve_controlled_pilot(authorization, models=models)
    else:
        runtime = resolver.resolve(
            mode=runtime_mode,  # type: ignore[arg-type]
            allow_unreleased_bundle=allow_unreleased_bundle,
            development_models=models if allow_unreleased_bundle else None,
        )
    require_data_integrity_schema(database)
    deepseek = DeepSeekClient(
        api_key=settings.deepseek_api_key,
        base_url=settings.deepseek_base_url,
        timeout_seconds=settings.deepseek_timeout_seconds,
        max_attempts=settings.deepseek_max_retries,
    )
    lease_manager = TaskLeaseManager(database)
    budget_manager = AIBudgetManager(
        database,
        reservation_hook=(
            PilotReservationGuard(pilot_service, pilot_authorization_id, models)
            if pilot_service is not None and pilot_authorization_id is not None
            else None
        ),
    )
    lease_ref: dict[str, V05LiteratureOrchestrator] = {}

    def guard() -> None:
        orchestrator = lease_ref.get("orchestrator")
        if orchestrator is not None:
            orchestrator._guard("paid_processing")

    semantic_config = runtime.semantic_config
    fixture_mode = runtime.unreleased_override and runtime_mode != "controlled_pilot"
    controlled_pilot_mode = runtime_mode == "controlled_pilot"
    if fixture_mode or controlled_pilot_mode:
        semantic_config = semantic_config.model_copy(
            update={
                "fast": semantic_config.fast.model_copy(
                    update={
                        "model": runtime.models.fast,
                        "enabled": (
                            True if controlled_pilot_mode else semantic_config.fast.enabled
                        ),
                    }
                ),
                "strong_review": semantic_config.strong_review.model_copy(
                    update={
                        "model": runtime.models.strong,
                        "enabled": (
                            True
                            if controlled_pilot_mode
                            else semantic_config.strong_review.enabled
                        ),
                    }
                ),
            }
        )
    abstract_config = load_abstract_sources(
        runtime_path(settings.project_root, "config/abstract_sources.yaml")
    )
    abstract_sources = {item.name: item for item in abstract_config.sources}

    def abstract_http(source: str) -> ResilientHttpClient:
        source_config = abstract_sources[source]
        return ResilientHttpClient(
            timeout=source_config.timeout_seconds,
            max_retries=source_config.max_retries,
            rate_limit_per_second=source_config.rate_limit_per_second,
        )

    orchestrator = V05LiteratureOrchestrator(
        database=database,
        enrichment_service=AbstractEnrichmentService(
            database,
            adapters=[
                CrossrefEnrichmentAdapter(client=abstract_http("crossref")),
                RePEcOAIAdapter(
                    client=abstract_http("repec"),
                    crossref_client=abstract_http("crossref"),
                    series_by_issn=load_repec_series_registry(
                        runtime_path(settings.project_root, "config/repec_series.json")
                    ),
                ),
                DOAJEnrichmentAdapter(client=abstract_http("doaj")),
                SemanticScholarEnrichmentAdapter(
                    client=abstract_http("semantic_scholar")
                ),
                OpenAlexEnrichmentAdapter(
                    settings.openalex_api_key, client=abstract_http("openalex")
                ),
            ],
            source_configs=abstract_sources,
            policy_version=abstract_config.version,
        ),
        journal_file=journal_file,
        sync_service=SyncService(database, CrossrefClient()),
        translation_service=TranslationService(
            database,
            settings.prompts_dir,
            deepseek,
            runtime.models.translation,
            execution_guard=guard,
            budget_manager=budget_manager,
        ),
        evidence_service=DomainEvidenceService(database, runtime.domain_bundle),
        semantic_service=SemanticScreeningService(
            database=database,
            config=semantic_config,
            prompts=runtime.prompt_builder,
            client=DeepSeekSemanticAdapter(deepseek),
            allow_live_ai=not fixture_mode,
            fixture_mode=fixture_mode,
            runtime_mode=runtime.mode,
            execution_guard=guard,
            budget_manager=budget_manager,
        ),
        lease_manager=lease_manager,
        runtime_identity=runtime.audit_summary(),
    )
    lease_ref["orchestrator"] = orchestrator
    return orchestrator


def list_task_papers(
    database: Database,
    task_id: int,
    *,
    decisions: tuple[str, ...] | None = None,
    state: dict[str, object] | None = None,
) -> list[dict[str, object]]:
    task_state = state if state is not None else prototype_task_state(database, task_id)
    details = dict(task_state.get("decisions") or {})
    candidate_ids = {int(item) for item in task_state.get("candidate_ids") or []}
    recovered_ids = {int(item) for item in task_state.get("recovered_ids") or []}
    with database.connect() as connection:
        rows = connection.execute(
            """SELECT p.id,p.title,p.abstract,p.authors_json,p.doi,p.source_url,
            COALESCE(p.issue_date,p.online_date) publication_date,j.name journal,
            COALESCE(tr.title_override,pt.chinese_title) chinese_title,
            COALESCE(tr.abstract_override,pt.chinese_abstract) chinese_abstract,
            pt.abstract_translation_status,pt.abstract_translation_source,
            pt.model_name translation_model_name,pt.updated_at translation_updated_at,
            p.source_metadata_json,
            COALESCE(tr.review_status,'pending') review_status,
            tr.final_decision human_final_decision,tr.reviewer_notes reviewer_note
            FROM workflow_batch_items wbi JOIN papers p ON p.id=wbi.paper_id
            JOIN journals j ON j.id=p.journal_id
            LEFT JOIN paper_translations pt ON pt.paper_id=p.id
            LEFT JOIN workflow_task_reviews tr
              ON tr.task_id=wbi.batch_id AND tr.paper_id=wbi.paper_id
            WHERE wbi.batch_id=? ORDER BY wbi.position""",
            (task_id,),
        ).fetchall()
    output: list[dict[str, object]] = []
    for row in rows:
        item = dict(row)
        detail = dict(details.get(str(item["id"])) or {})
        semantic_decision = detail.get("semantic_decision", detail.get("decision"))
        if decisions and semantic_decision not in decisions:
            continue
        item["authors"] = json.loads(item.pop("authors_json") or "[]")
        metadata = _safe_json_object(item.pop("source_metadata_json") or "{}")
        item["abstract_provenance"] = metadata.get("abstract_provenance") or {
            "source": metadata.get("abstract_source")
        }
        item.update(detail)
        item["semantic_decision"] = semantic_decision
        item["decision"] = semantic_decision
        item["human_final_decision"] = item.get("human_final_decision") or "undecided"
        item["recovered"] = int(item["id"]) in recovered_ids
        item["candidate"] = int(item["id"]) in candidate_ids
        output.append(item)
    return output


def task_processing_counts(database: Database, task_id: int) -> dict[str, int]:
    """Return mutually clear task counts from current paper and translation state."""
    with database.connect() as connection:
        rows = connection.execute(
            """SELECT p.abstract,p.source_metadata_json,pt.chinese_title,pt.chinese_abstract,
            pt.abstract_translation_status
            FROM workflow_batch_items wbi
            JOIN papers p ON p.id=wbi.paper_id
            LEFT JOIN paper_translations pt ON pt.paper_id=p.id
            WHERE wbi.batch_id=?""",
            (task_id,),
        ).fetchall()
    counts = {
        "total_papers": len(rows),
        "english_abstracts": 0,
        "chinese_abstracts": 0,
        "missing_abstracts": 0,
        "retryable_failures": 0,
        "permanent_failures": 0,
        "translated_titles": 0,
        "original_metadata_abstracts": 0,
        "automatic_abstracts": 0,
        "manual_abstracts": 0,
        "chinese_translation_pending": 0,
        "strategy_exhausted": 0,
        "needs_manual_abstract": 0,
    }
    for row in rows:
        has_abstract = bool((row["abstract"] or "").strip())
        counts["english_abstracts"] += int(has_abstract)
        counts["missing_abstracts"] += int(not has_abstract)
        counts["chinese_abstracts"] += int(bool((row["chinese_abstract"] or "").strip()))
        counts["translated_titles"] += int(bool((row["chinese_title"] or "").strip()))
        metadata = _safe_json_object(row["source_metadata_json"])
        provenance = metadata.get("abstract_provenance")
        source = str(
            metadata.get("abstract_source")
            or (
                provenance.get("source")
                if isinstance(provenance, dict)
                else ""
            )
            or ""
        )
        if has_abstract:
            if source == "manual":
                counts["manual_abstracts"] += 1
            elif metadata.get("abstract"):
                counts["original_metadata_abstracts"] += 1
            elif source:
                counts["automatic_abstracts"] += 1
            if (
                not (row["chinese_abstract"] or "").strip()
                or row["abstract_translation_status"] in {"pending", "failed", "stale"}
            ):
                counts["chinese_translation_pending"] += 1
        audit = metadata.get("abstract_enrichment")
        if not isinstance(audit, dict) or has_abstract:
            continue
        state = audit.get("execution_state")
        if state == "retryable_error" or audit.get("retryable"):
            counts["retryable_failures"] += 1
        elif state == "permanent_error":
            counts["permanent_failures"] += 1
        if audit.get("strategy_exhausted"):
            counts["strategy_exhausted"] += 1
        if state == "needs_manual_review" or audit.get("strategy_exhausted"):
            counts["needs_manual_abstract"] += 1
    return counts


def save_final_reviews(
    database: Database,
    task_id: int,
    reviews: dict[int, FinalReviewInput],
) -> dict[int, str]:
    require_p0_hardening_schema(database)
    state = prototype_task_state(database, task_id)
    candidates = {int(item) for item in state.get("candidate_ids") or []}
    unknown = set(reviews) - candidates
    if unknown:
        raise ValueError(f"Papers are not candidates in this task: {sorted(unknown)}")
    changed_reviews: dict[
        int, tuple[FinalReviewInput, dict[str, object], tuple[str, str, str, str]]
    ] = {}
    persisted_states = _persisted_review_states(
        database, task_id, tuple(reviews)
    )
    for paper_id, review in reviews.items():
        persisted = persisted_states[paper_id]
        current = (
            persisted["decision"],
            persisted["effective_title"],
            persisted["effective_abstract"],
            persisted["note"],
        )
        desired = (
            review.resolved_decision(),
            (
                _normalized_review_text(review.chinese_title)
                if review.chinese_title is not None
                else str(current[1])
            ),
            (
                _normalized_review_text(review.chinese_abstract)
                if review.chinese_abstract is not None
                else str(current[2])
            ),
            (
                _normalized_review_text(review.note)
                if review.note is not None
                else str(current[3])
            ),
        )
        if desired != current:
            changed_reviews[paper_id] = (review, persisted, desired)
    if not changed_reviews and candidates:
        return {}
    errors: dict[int, str] = {}
    completed = {int(item) for item in state.get("final_reviewed_ids") or []}
    values: dict[int, TaskReviewInput] = {}
    for paper_id, (_review, persisted, desired) in changed_reviews.items():
        final_decision, title, abstract, note = desired
        existing = persisted["review"]
        values[paper_id] = TaskReviewInput(
            review_status=(
                "approved"
                if final_decision == "include"
                else "rejected"
                if final_decision == "exclude"
                else "pending"
            ),
            final_decision=(
                final_decision
                if final_decision in {"include", "exclude"}
                else None
            ),
            reincluded=bool(existing.get("reincluded")),
            recovery_status=str(
                existing.get("recovery_status") or "not_applicable"
            ),
            reviewer_notes=note,
            title_override=None if title == persisted["base_title"] else title,
            abstract_override=(
                None if abstract == persisted["base_abstract"] else abstract
            ),
            needs_edit=bool(
                title != persisted["base_title"]
                or abstract != persisted["base_abstract"]
            ),
            source="v05_human",
        )
    try:
        save_task_reviews(database, task_id, values)
    except Exception as exc:
        message = f"{type(exc).__name__}: {str(exc)[:300]}"
        errors = {paper_id: message for paper_id in values}
    if not errors:
        for paper_id, (_, _, desired) in changed_reviews.items():
            if desired[0] in {"include", "exclude"}:
                completed.add(paper_id)
            else:
                completed.discard(paper_id)
    payload = _state_payload(database, task_id)
    payload["final_reviewed_ids"] = sorted(completed)
    payload["final_review_errors"] = {str(key): value for key, value in errors.items()}
    payload["task_review_counts"] = task_review_counts(database, task_id)
    ready_for_export = not errors and candidates <= completed
    if ready_for_export:
        payload["current_step"] = "export"
        _save_state(database, task_id, payload, status="completed", completed=True)
    else:
        _save_state(database, task_id, payload, status="ready_for_review")
    checkpoints = WorkflowCheckpointStore(database)
    checkpoint = checkpoints.get(task_id, "final_review")
    if checkpoint.status == "pending":
        checkpoint = checkpoints.start(
            task_id,
            "final_review",
            total=len(candidates),
            config_fp="human-review-v1",
            input_fp=fingerprint(tuple(sorted(candidates))),
            item_ids=tuple(sorted(candidates)),
        )
    failed_ids = sorted(set(candidates) - completed | set(errors))
    checkpoints.finish(
        task_id,
        "final_review",
        checkpoint,
        succeeded=len(completed & candidates),
        failed=len(failed_ids),
        failed_item_ids=failed_ids,
        errors={str(key): value for key, value in errors.items()},
    )
    if ready_for_export:
        mark_export_completed(database, task_id)
    return errors


def _persisted_review_states(
    database: Database, task_id: int, paper_ids: tuple[int, ...]
) -> dict[int, dict[str, object]]:
    if not paper_ids:
        return {}
    placeholders = ",".join("?" for _ in paper_ids)
    with database.connect() as connection:
        rows = connection.execute(
            f"""SELECT wbi.paper_id,pt.chinese_title base_title,
            pt.chinese_abstract base_abstract,tr.final_decision,tr.reviewer_notes,
            tr.title_override,tr.abstract_override,tr.reincluded,tr.recovery_status,
            tr.reviewer,tr.needs_edit,tr.source
            FROM workflow_batch_items wbi
            LEFT JOIN paper_translations pt ON pt.paper_id=wbi.paper_id
            LEFT JOIN workflow_task_reviews tr
              ON tr.task_id=wbi.batch_id AND tr.paper_id=wbi.paper_id
            WHERE wbi.batch_id=? AND wbi.paper_id IN ({placeholders})""",  # noqa: S608
            (task_id, *paper_ids),
        ).fetchall()
    if len(rows) != len(set(paper_ids)):
        found = {int(row["paper_id"]) for row in rows}
        raise ValueError(
            f"Papers do not belong to task {task_id}: {sorted(set(paper_ids) - found)}"
        )
    output: dict[int, dict[str, object]] = {}
    for row in rows:
        base_title = _normalized_review_text(row["base_title"])
        base_abstract = _normalized_review_text(row["base_abstract"])
        title_override = row["title_override"]
        abstract_override = row["abstract_override"]
        existing = {
            "reincluded": row["reincluded"],
            "recovery_status": row["recovery_status"],
            "reviewer": row["reviewer"],
            "needs_edit": row["needs_edit"],
            "source": row["source"],
        }
        output[int(row["paper_id"])] = {
            "decision": (
                str(row["final_decision"])
                if row["final_decision"] in {"include", "exclude"}
                else "undecided"
            ),
            "base_title": base_title,
            "base_abstract": base_abstract,
            "effective_title": (
                _normalized_review_text(title_override)
                if title_override is not None
                else base_title
            ),
            "effective_abstract": (
                _normalized_review_text(abstract_override)
                if abstract_override is not None
                else base_abstract
            ),
            "note": _normalized_review_text(row["reviewer_notes"]),
            "review": existing,
        }
    return output


def _normalized_review_text(value: object) -> str:
    return str(value or "").strip()


def mark_export_completed(database: Database, task_id: int) -> None:
    """Record idempotent export readiness without changing reviews or translations."""
    checkpoints = WorkflowCheckpointStore(database)
    checkpoint = checkpoints.get(task_id, "export")
    if checkpoint.status == "completed":
        return
    with database.connect() as connection:
        total = int(
            connection.execute(
                "SELECT COUNT(*) FROM workflow_batch_items WHERE batch_id=?", (task_id,)
            ).fetchone()[0]
        )
    checkpoint = checkpoints.start(
        task_id,
        "export",
        total=total,
        config_fp="export-policy-v1",
        input_fp=fingerprint((task_id, total)),
    )
    checkpoints.finish(task_id, "export", checkpoint, succeeded=total)
    _save_export_manifest(database, task_id)


def _save_export_manifest(database: Database, task_id: int) -> None:
    payload = _state_payload(database, task_id)
    if not isinstance(payload.get("export_manifest"), dict):
        payload["export_manifest"] = {
            "task_id": task_id,
            "review_scope": "workflow_task_reviews",
            "generated_at": utcnow(),
        }
    _save_state(database, task_id, payload)


def save_manual_abstract(
    database: Database,
    paper_id: int,
    abstract: str,
    source_url: str | None = None,
) -> dict:
    """Historical entry point; the implementation now lives in the product core.

    Kept as a delegation so existing callers and tests keep working without
    importing the research/governance stack through this module.
    """
    from eduecon_radar.services.manual_abstract import (
        save_manual_abstract as _save_manual_abstract,
    )

    return _save_manual_abstract(database, paper_id, abstract, source_url)


def translate_manual_abstract(
    database: Database,
    paper_id: int,
    abstract: str,
    source_url: str | None,
    translation_service: TranslationService,
    task_id: int | None = None,
) -> dict:
    save_result = save_manual_abstract(database, paper_id, abstract, source_url)
    if "error" in save_result:
        return save_result
    with database.connect() as connection:
        row = connection.execute(
            "SELECT chinese_abstract,abstract_translation_source,"
            "abstract_translation_status FROM paper_translations"
            " WHERE paper_id=?",
            (paper_id,),
        ).fetchone()
    if row and row["chinese_abstract"]:
        if (
            row["abstract_translation_source"] == "human_edited"
            and row["abstract_translation_status"] != "stale"
        ):
            return {
                "success": True,
                "paper_id": paper_id,
                "chinese": row["chinese_abstract"],
                "skipped": "已有中文摘要且来源为人工编辑，未覆盖",
                "abstract_changed": bool(save_result.get("abstract_changed")),
            }
        if row["abstract_translation_status"] != "stale":
            return {
                "success": True,
                "paper_id": paper_id,
                "chinese": row["chinese_abstract"],
                "skipped": "已有正式中文摘要，未覆盖",
                "abstract_changed": bool(save_result.get("abstract_changed")),
            }
    stats = translation_service.run(
        TranslationParameters(
            review_status=None,
            limit=1,
            missing_only=True,
            overwrite_human=False,
            paper_ids=(paper_id,),
            translate_titles=False,
            translate_abstracts=True,
            task_id=task_id,
        )
    )
    if stats.failed:
        return {
            "success": False,
            "paper_id": paper_id,
            "error": "翻译失败（保留英文原摘要）",
            "abstract_changed": bool(save_result.get("abstract_changed")),
        }
    return {
        "success": True,
        "paper_id": paper_id,
        "translated": True,
        "abstract_changed": bool(save_result.get("abstract_changed")),
    }


def _selected_journals(journal_file: JournalFile, names: tuple[str, ...]) -> JournalFile:
    requested = set(names)
    selected = [
        item for item in journal_file.journals if item.enabled and item.short_name in requested
    ]
    found = {item.short_name for item in selected}
    if found != requested:
        raise ValueError(f"Unknown or disabled journals: {sorted(requested - found)}")
    return JournalFile(ranking=journal_file.ranking, journals=selected)


def _translate_in_chunks(
    service: TranslationService,
    paper_ids: tuple[int, ...],
    *,
    titles: bool,
    abstracts: bool,
    task_id: int,
) -> dict[str, int]:
    totals = {"failed": 0, "unavailable": 0}
    for offset in range(0, len(paper_ids), 100):
        chunk = paper_ids[offset : offset + 100]
        if not chunk:
            continue
        stats = service.run(
            TranslationParameters(
                review_status=None,
                limit=len(chunk),
                missing_only=True,
                paper_ids=chunk,
                translate_titles=titles,
                translate_abstracts=abstracts,
                task_id=task_id,
            )
        )
        totals["failed"] += stats.failed
        totals["unavailable"] += stats.unavailable
    return totals


def _stored_evidence(
    database: Database, paper_id: int, task_id: int, config_sha256: str | None
) -> Any | None:
    if config_sha256 is None:
        return None
    run = DomainEvidenceRepository(database).latest_for_paper(paper_id, workflow_batch_id=task_id)
    if not run or run.get("workflow_batch_id") != task_id:
        return None
    if run.get("config_bundle_sha256") != config_sha256:
        return None
    payload = run["evidence_payload"]
    scoring = payload["scoring"]
    priority = payload["priority"]
    return SimpleNamespace(
        scoring=SimpleNamespace(rule_evidence_tier=scoring["rule_evidence_tier"]),
        priority=SimpleNamespace(
            priority_tier=priority["priority_tier"],
            priority_score=priority.get("priority_score"),
        ),
    )


def _decision_detail(
    database: Database,
    paper_id: int,
    evidence: Any,
    analysis: SemanticScreeningAnalysis | None,
    failure: str | None,
) -> dict[str, object]:
    rule_tier = evidence.scoring.rule_evidence_tier if evidence else "little_evidence"
    priority_tier = evidence.priority.priority_tier if evidence else "urgent"
    priority_score = evidence.priority.priority_score if evidence else 100.0
    with database.connect() as connection:
        row = connection.execute("SELECT abstract FROM papers WHERE id=?", (paper_id,)).fetchone()
    has_abstract = bool(row and row["abstract"])
    if analysis is None:
        execution_state = "needs_abstract" if not has_abstract else "retryable_error"
        return {
            "decision": None,
            "semantic_decision": None,
            "execution_state": execution_state,
            "human_final_decision": "undecided",
            "decision_basis": "semantic_result_unavailable",
            "confidence": None,
            "reason": failure or "Semantic result unavailable",
            "rule_evidence_tier": rule_tier,
            "priority_tier": priority_tier,
            "priority_score": priority_score,
            "strong_recommendation": priority_tier in {"urgent", "high"},
            "semantic_failed": True,
            "priority_label": "缺少摘要" if not has_abstract else "处理失败，可重试",
            "semantic_score": None,
            "score_is_probability": False,
            "uncertainty_label": "high",
            "missed_priority_label": priority_tier,
        }
    fast = analysis.fast
    strong = analysis.strong
    strong_decision = strong.recommended_final_decision if strong else None
    conflict = bool(analysis.escalation.conflict_types) or (
        strong_decision is not None and fast.decision != strong_decision
    )
    rule_positive = rule_tier in {"strong_positive", "moderate_positive"}
    insufficient = (
        not has_abstract
        or fast.information_sufficiency == "insufficient"
        or (strong is not None and strong.decision_sufficiency == "insufficient")
    )
    if not has_abstract:
        decision = None
        execution_state = "needs_abstract"
        basis = "abstract_required_for_semantic_decision"
    elif analysis.strong_failed:
        decision = None
        execution_state = "retryable_error"
        basis = "strong_semantic_execution_failed"
    elif conflict or fast.decision == "borderline" or strong_decision == "borderline":
        decision = "borderline"
        execution_state = "needs_manual_review"
        basis = "semantic_or_rule_conflict"
    elif insufficient:
        decision = None
        execution_state = "needs_manual_review"
        basis = "information_insufficient"
    elif fast.decision == "include" or strong_decision == "include":
        decision = "include"
        execution_state = "completed"
        basis = "positive_semantic_judgment"
    elif (
        fast.decision == "exclude"
        and (not analysis.escalation.escalate or strong_decision == "exclude")
        and not rule_positive
    ):
        decision = "exclude"
        execution_state = "completed"
        basis = "consistent_negative_semantic_judgment"
    else:
        decision = None
        execution_state = "needs_manual_review"
        basis = "high_recall_safety_fallback"
    return {
        "decision": decision,
        "semantic_decision": decision,
        "execution_state": execution_state,
        "human_final_decision": "undecided",
        "decision_basis": basis,
        "confidence": fast.confidence,
        "reason": fast.reason,
        "rule_evidence_tier": rule_tier,
        "priority_tier": analysis.missed_priority.missed_review_priority_tier,
        "priority_score": analysis.missed_priority.provisional_score,
        "priority_reasons": list(analysis.missed_priority.missed_review_priority_reasons),
        "strong_recommendation": analysis.missed_priority.missed_review_priority_tier
        in {"urgent", "high"},
        "fast_decision": fast.decision,
        "strong_decision": strong_decision,
        "strong_ran": strong is not None,
        "semantic_failed": analysis.strong_failed,
        "conflicts": list(analysis.composite.conflict_types),
        "priority_label": _compute_priority_label(
            decision, float(fast.confidence), rule_tier, conflict
        ),
        "semantic_score": fast.semantic_score,
        "score_is_probability": False,
        "uncertainty_label": analysis.composite.uncertainty_level,
        "missed_priority_label": analysis.missed_priority.missed_review_priority_tier,
    }


def _result_counts(
    decisions: dict[str, Any], evidence: dict[int, Any], semantic: dict[int, Any]
) -> dict[str, int]:
    values = list(decisions.values())
    evidence_values = list(evidence.values())
    return {
        "evidence_extracted": len(evidence),
        "strong_positive_evidence": sum(
            item.scoring.rule_evidence_tier == "strong_positive" for item in evidence_values
        ),
        "mixed_evidence": sum(
            item.scoring.rule_evidence_tier == "weak_or_mixed" for item in evidence_values
        ),
        "information_limited": sum(
            item.get("decision_basis") == "information_insufficient" for item in values
        ),
        "fast_semantic_completed": len(semantic),
        "strong_review_completed": sum(item.strong is not None for item in semantic.values()),
        "semantic_conflicts": sum(
            bool(item.composite.conflict_types) for item in semantic.values()
        ),
        "composite_completed": len(semantic),
        "provisional_include": sum(item.get("decision") == "include" for item in values),
        "provisional_borderline": sum(item.get("decision") == "borderline" for item in values),
        "provisional_exclude": sum(item.get("decision") == "exclude" for item in values),
        "semantic_failed": sum(
            item.get("execution_state") in {"retryable_error", "permanent_error"}
            for item in values
        ),
        "semantic_needs_abstract": sum(
            item.get("execution_state") == "needs_abstract" for item in values
        ),
        "semantic_needs_manual_review": sum(
            item.get("execution_state") == "needs_manual_review" for item in values
        ),
    }


def _count_translated_titles(database: Database, paper_ids: tuple[int, ...]) -> int:
    return _translation_count(database, paper_ids, "chinese_title")


def _count_translated_abstracts(database: Database, paper_ids: tuple[int, ...]) -> int:
    return _translation_count(database, paper_ids, "chinese_abstract")


def _translation_count(database: Database, paper_ids: tuple[int, ...], field: str) -> int:
    if not paper_ids:
        return 0
    if field not in {"chinese_title", "chinese_abstract"}:
        raise ValueError("Unsupported translation field")
    placeholders = ",".join("?" for _ in paper_ids)
    with database.connect() as connection:
        return int(
            connection.execute(
                f"SELECT COUNT(*) FROM paper_translations WHERE {field} IS NOT NULL "
                f"AND paper_id IN ({placeholders})",  # noqa: S608 -- field allow-listed
                paper_ids,
            ).fetchone()[0]
        )


def _state_payload(database: Database, task_id: int) -> dict[str, Any]:
    return _state_payload_from_batch(get_workflow_batch(database, task_id))


def _state_payload_from_batch(batch: dict[str, object]) -> dict[str, Any]:
    summary = dict(batch.get("result_summary") or {})
    payload = summary.get("v05_literature")
    if isinstance(payload, dict):
        return dict(payload)
    legacy = summary.get("v05_prototype")
    if isinstance(legacy, dict):
        return dict(legacy)
    return {"current_step": "search_scope", "result": _empty_result()}


def _save_state(
    database: Database,
    task_id: int,
    payload: dict[str, Any],
    *,
    status: str | None = None,
    completed: bool = False,
    lease_manager: TaskLeaseManager | None = None,
    lease: TaskLease | None = None,
) -> None:
    now = utcnow()
    with database.connect() as connection:
        connection.execute("BEGIN IMMEDIATE")
        if lease_manager is not None:
            if lease is None:
                raise TaskLeaseLost("A durable task lease is required for this state update")
            lease_manager.assert_current(lease, connection=connection)
        row = connection.execute(
            "SELECT result_summary_json FROM workflow_batches WHERE id=?", (task_id,)
        ).fetchone()
        summary = _safe_json_object(row[0] if row else None)
        summary["v05_literature"] = payload
        connection.execute(
            """UPDATE workflow_batches SET status=COALESCE(?,status),
            result_summary_json=?,completed_at=CASE WHEN ? THEN ? ELSE completed_at END,
            updated_at=? WHERE id=?""",
            (
                status,
                json.dumps(summary, ensure_ascii=False, sort_keys=True),
                int(completed),
                now,
                now,
                task_id,
            ),
        )


def _safe_json_object(value: object) -> dict[str, Any]:
    try:
        parsed = json.loads(value or "{}")
    except (TypeError, ValueError, json.JSONDecodeError):
        return {}
    return parsed if isinstance(parsed, dict) else {}


def _papers_missing_abstract(database: Database, paper_ids: tuple[int, ...]) -> tuple[int, ...]:
    if not paper_ids:
        return ()
    placeholders = ",".join("?" for _ in paper_ids)
    with database.connect() as connection:
        rows = connection.execute(
            f"SELECT id FROM papers WHERE id IN ({placeholders}) "  # noqa: S608
            "AND (abstract IS NULL OR abstract='') ORDER BY id",
            paper_ids,
        ).fetchall()
    return tuple(int(row[0]) for row in rows)


def _abstract_failed_ids(database: Database, paper_ids: tuple[int, ...]) -> tuple[int, ...]:
    return _papers_missing_abstract(database, paper_ids)


def _abstract_retryable(database: Database, paper_id: int) -> bool:
    with database.connect() as connection:
        row = connection.execute(
            "SELECT source_metadata_json FROM papers WHERE id=?", (paper_id,)
        ).fetchone()
    metadata = _safe_json_object(row[0] if row else None)
    audit = metadata.get("abstract_enrichment")
    return bool(isinstance(audit, dict) and audit.get("retryable"))


def _missing_translation_ids(
    database: Database, paper_ids: tuple[int, ...], field: str
) -> tuple[int, ...]:
    if not paper_ids:
        return ()
    if field not in {"chinese_title", "chinese_abstract"}:
        raise ValueError("unsupported translation field")
    placeholders = ",".join("?" for _ in paper_ids)
    with database.connect() as connection:
        rows = connection.execute(
            f"SELECT p.id FROM papers p LEFT JOIN paper_translations pt ON pt.paper_id=p.id "
            f"WHERE p.id IN ({placeholders}) AND pt.{field} IS NULL",  # noqa: S608
            paper_ids,
        ).fetchall()
    return tuple(int(row[0]) for row in rows)


def _compute_priority_label(
    decision: str,
    confidence: float,
    rule_tier: str,
    has_conflict: bool,
) -> str:
    if (
        decision == "include"
        and confidence >= 0.7
        and rule_tier in {"strong_positive", "moderate_positive"}
    ):
        return "核心教育经济学"
    if decision == "include" and confidence >= 0.7:
        return "高度相关"
    if decision == "include":
        return "建议人工检查"
    if has_conflict:
        return "建议人工检查"
    if decision == "borderline":
        return "建议人工检查"
    return "低相关"


def _empty_result() -> dict[str, int]:
    return {
        "total": 0,
        "enriched": 0,
        "enrichment_failed": 0,
        "title_translated": 0,
        "provisional_include": 0,
        "provisional_borderline": 0,
        "provisional_exclude": 0,
        "semantic_failed": 0,
        "semantic_needs_abstract": 0,
        "semantic_needs_manual_review": 0,
        "evidence_extracted": 0,
        "fast_semantic_completed": 0,
        "strong_review_completed": 0,
        "composite_completed": 0,
    }
