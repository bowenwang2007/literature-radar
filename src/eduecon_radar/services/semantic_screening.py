# --- not part of the product core: see docs/PRODUCT_CORE.md ---
# RESEARCH (v0.5 semantic engine). Not on the product path.

from __future__ import annotations

import json
from collections.abc import Callable
from dataclasses import dataclass
from typing import Any, Literal

from eduecon_radar.db import Database
from eduecon_radar.domain.composite_decision import CompositeDecisionPolicy
from eduecon_radar.domain.escalation import EscalationPolicy
from eduecon_radar.domain.semantic_judgment import (
    FastSemanticJudge,
    SemanticAIClient,
    StrongSemanticReviewer,
)
from eduecon_radar.domain.semantic_models import (
    CompositeDecisionResult,
    EscalationDecision,
    FastSemanticResult,
    SemanticEngineConfig,
    SemanticMissedPriority,
    SemanticPaperInput,
    StrongSemanticReviewResult,
)
from eduecon_radar.domain.semantic_prompting import (
    SemanticPromptBuilder,
    canonical_sha256,
    semantic_input_fingerprint,
)
from eduecon_radar.domain.semantic_storage import (
    DomainSemanticRepository,
    StoredCompositeDecision,
    StoredSemanticRun,
)
from eduecon_radar.domain.storage import DomainEvidenceRepository
from eduecon_radar.services.ai_budget import AIBudgetManager, BudgetBlocked, estimate_tokens
from eduecon_radar.services.task_lock import TaskLeaseError


@dataclass(frozen=True)
class SemanticScreeningAnalysis:
    fast: FastSemanticResult
    escalation: EscalationDecision
    strong: StrongSemanticReviewResult | None
    composite: CompositeDecisionResult
    missed_priority: SemanticMissedPriority
    fast_storage: StoredSemanticRun
    strong_storage: StoredSemanticRun | None
    composite_storage: StoredCompositeDecision
    strong_failed: bool = False


@dataclass(frozen=True)
class SemanticBatchAnalysis:
    completed: dict[int, SemanticScreeningAnalysis]
    failures: dict[int, str]


class SemanticScreeningService:
    """Explicit Phase 4 shadow runner. It never writes legacy screening or review tables."""

    def __init__(
        self,
        *,
        database: Database,
        config: SemanticEngineConfig,
        prompts: SemanticPromptBuilder,
        client: SemanticAIClient,
        fixture_mode: bool = False,
        allow_live_ai: bool = False,
        runtime_mode: Literal["production", "controlled_pilot"] = "production",
        execution_guard: Callable[[], None] | None = None,
        budget_manager: AIBudgetManager | None = None,
    ) -> None:
        live_ready = (
            allow_live_ai
            and config.live_ai_allowed
            and config.status == "released"
            and config.fast.enabled
            and config.fast.status == "released"
            and config.strong_review.enabled
            and config.strong_review.status == "released"
        )
        pilot_ready = (
            runtime_mode == "controlled_pilot"
            and allow_live_ai
            and config.fast.enabled
            and config.strong_review.enabled
        )
        if not fixture_mode and not (live_ready or pilot_ready):
            raise RuntimeError(
                "Phase 4 semantic execution is disabled unless fixture mode or an explicitly "
                "enabled released live configuration is supplied"
            )
        self.database = database
        self.config = config
        self.prompts = prompts
        self.client = client
        self.runtime_mode = runtime_mode
        self.repository = DomainSemanticRepository(database)
        self.execution_guard = execution_guard
        self.budget_manager = budget_manager

    def analyze_paper(
        self,
        paper_id: int,
        *,
        run_strong_if_escalated: bool = True,
        workflow_batch_id: int | None = None,
    ) -> SemanticScreeningAnalysis:
        evidence_run = DomainEvidenceRepository(self.database).latest_for_paper(
            paper_id, workflow_batch_id=workflow_batch_id
        )
        if evidence_run is None:
            raise RuntimeError("A Phase 3 domain evidence run is required")
        paper = load_semantic_paper_input(self.database, paper_id, evidence_run)

        fast_prompt = self.prompts.build_fast(paper)
        fast_fingerprint = semantic_input_fingerprint(
            paper=paper,
            prompt=fast_prompt,
            model_role="fast",
            model_name=self.config.fast.model,
            engine_version=self.config.version,
            escalation_policy_version=self.config.escalation.version,
        )
        existing_fast = self.repository.successful_by_fingerprint(fast_fingerprint, "fast")
        if existing_fast:
            fast = FastSemanticResult.model_validate(existing_fast["result"])
            fast_storage = StoredSemanticRun(int(existing_fast["id"]), True, True)
        else:
            self._guard()
            judge = FastSemanticJudge(config=self.config, prompts=self.prompts, client=self.client)
            reservation = self._reserve(
                workflow_batch_id,
                "fast_semantic",
                self.config.fast.provider,
                self.config.fast.model,
                fast_fingerprint,
                fast_prompt.system_prompt + fast_prompt.user_prompt,
                self.config.fast.max_tokens,
            )
            try:
                judgment = judge.judge(paper)
                self._guard()
                self._settle(
                    reservation, judgment.input_tokens, judgment.output_tokens
                )
            except TaskLeaseError:
                raise
            except BudgetBlocked:
                raise
            except Exception as exc:
                self._settle(reservation, None, None, failed=True, error=exc)
                self.repository.save_failure(
                    paper=paper,
                    role="fast",
                    fingerprint=fast_fingerprint,
                    prompt=fast_prompt,
                    model_provider=self.config.fast.provider,
                    model_name=self.config.fast.model,
                    engine_version=self.config.version,
                    escalation_policy_version=self.config.escalation.version,
                    error=exc,
                )
                raise
            fast = FastSemanticResult.model_validate(judgment.result)
            fast_storage = self.repository.save_success(
                paper=paper,
                role="fast",
                fingerprint=fast_fingerprint,
                prompt=judgment.prompt,
                model_provider=self.config.fast.provider,
                model_name=fast.model_name,
                engine_version=self.config.version,
                escalation_policy_version=self.config.escalation.version,
                result=fast,
                input_tokens=judgment.input_tokens,
                output_tokens=judgment.output_tokens,
                estimated_cost=judgment.estimated_cost,
            )

        escalation = EscalationPolicy(self.config.escalation).decide(paper, fast)
        strong: StrongSemanticReviewResult | None = None
        strong_storage: StoredSemanticRun | None = None
        strong_failed = False
        if escalation.escalate and run_strong_if_escalated:
            strong_prompt = self.prompts.build_strong(paper, fast, escalation)
            strong_fingerprint = semantic_input_fingerprint(
                paper=paper,
                prompt=strong_prompt,
                model_role="strong_review",
                model_name=self.config.strong_review.model,
                engine_version=self.config.version,
                escalation_policy_version=self.config.escalation.version,
                fast_result=fast,
            )
            existing_strong = self.repository.successful_by_fingerprint(
                strong_fingerprint, "strong_review"
            )
            if existing_strong:
                strong = StrongSemanticReviewResult.model_validate(existing_strong["result"])
                strong_storage = StoredSemanticRun(int(existing_strong["id"]), True, True)
            else:
                self._guard()
                reviewer = StrongSemanticReviewer(
                    config=self.config, prompts=self.prompts, client=self.client
                )
                reservation = self._reserve(
                    workflow_batch_id,
                    "strong_semantic",
                    self.config.strong_review.provider,
                    self.config.strong_review.model,
                    strong_fingerprint,
                    strong_prompt.system_prompt + strong_prompt.user_prompt,
                    self.config.strong_review.max_tokens,
                )
                try:
                    judgment = reviewer.review(paper, fast, escalation)
                    self._guard()
                    self._settle(
                        reservation, judgment.input_tokens, judgment.output_tokens
                    )
                    strong = StrongSemanticReviewResult.model_validate(judgment.result)
                    strong_storage = self.repository.save_success(
                        paper=paper,
                        role="strong_review",
                        fingerprint=strong_fingerprint,
                        prompt=judgment.prompt,
                        model_provider=self.config.strong_review.provider,
                        model_name=strong.model_name,
                        engine_version=self.config.version,
                        escalation_policy_version=self.config.escalation.version,
                        result=strong,
                        input_tokens=judgment.input_tokens,
                        output_tokens=judgment.output_tokens,
                        estimated_cost=judgment.estimated_cost,
                        parent_fast_run_id=fast_storage.id,
                    )
                except TaskLeaseError:
                    raise
                except BudgetBlocked:
                    raise
                except Exception as exc:
                    self._settle(reservation, None, None, failed=True, error=exc)
                    strong_storage = self.repository.save_failure(
                        paper=paper,
                        role="strong_review",
                        fingerprint=strong_fingerprint,
                        prompt=strong_prompt,
                        model_provider=self.config.strong_review.provider,
                        model_name=self.config.strong_review.model,
                        engine_version=self.config.version,
                        escalation_policy_version=self.config.escalation.version,
                        error=exc,
                        parent_fast_run_id=fast_storage.id,
                    )
                    strong_failed = True

        policy = CompositeDecisionPolicy(self.config.composite)
        composite = policy.combine(paper, fast, escalation, strong, strong_failed=strong_failed)
        priority = policy.missed_priority(paper, composite, escalation, fast, strong)
        composite_fingerprint = canonical_sha256(
            {
                "paper_input": paper.model_dump(mode="json"),
                "fast_run_id": fast_storage.id,
                "fast": fast.model_dump(mode="json"),
                "strong_run_id": strong_storage.id
                if strong_storage and strong_storage.succeeded
                else None,
                "strong": strong.model_dump(mode="json") if strong else None,
                "escalation": escalation.model_dump(mode="json"),
                "composite_policy": self.config.composite.model_dump(mode="json"),
                "strong_failed": strong_failed,
            }
        )
        self._guard()
        composite_storage = self.repository.save_composite(
            paper=paper,
            fast_run_id=fast_storage.id,
            strong_run_id=(
                strong_storage.id if strong_storage and strong_storage.succeeded else None
            ),
            fingerprint=composite_fingerprint,
            escalation=escalation,
            composite=composite,
            priority=priority,
        )
        return SemanticScreeningAnalysis(
            fast=fast,
            escalation=escalation,
            strong=strong,
            composite=composite,
            missed_priority=priority,
            fast_storage=fast_storage,
            strong_storage=strong_storage,
            composite_storage=composite_storage,
            strong_failed=strong_failed,
        )

    def analyze_many(
        self,
        paper_ids: tuple[int, ...],
        *,
        run_strong_if_escalated: bool = True,
        workflow_batch_id: int | None = None,
    ) -> SemanticBatchAnalysis:
        """Continue after per-paper failures without inventing substitute decisions."""
        completed: dict[int, SemanticScreeningAnalysis] = {}
        failures: dict[int, str] = {}
        for paper_id in dict.fromkeys(paper_ids):
            try:
                self._guard()
                completed[paper_id] = self.analyze_paper(
                    paper_id,
                    run_strong_if_escalated=run_strong_if_escalated,
                    workflow_batch_id=workflow_batch_id,
                )
            except TaskLeaseError:
                raise
            except BudgetBlocked:
                raise
            except Exception as exc:
                failures[paper_id] = f"{type(exc).__name__}: {str(exc)[:500]}"
        return SemanticBatchAnalysis(completed=completed, failures=failures)

    def _guard(self) -> None:
        if self.execution_guard is not None:
            self.execution_guard()

    def _reserve(
        self,
        task_id: int | None,
        stage: str,
        provider: str,
        model: str,
        identity: str,
        prompt: str,
        output_limit: int,
    ):
        if self.budget_manager is None:
            return None
        if task_id is None:
            raise BudgetBlocked("task_id_required")
        input_tokens, output_tokens = estimate_tokens(prompt, output_limit)
        return self.budget_manager.reserve(
            task_id=task_id,
            stage=stage,
            provider=provider,
            model_name=model,
            idempotency_key=identity,
            estimated_input_tokens=input_tokens,
            estimated_output_tokens=output_tokens,
        )

    def _settle(
        self,
        reservation,
        input_tokens: int | None,
        output_tokens: int | None,
        *,
        failed: bool = False,
        error: Exception | None = None,
    ) -> None:
        if self.budget_manager is not None and reservation is not None:
            self.budget_manager.settle(
                reservation,
                input_tokens=input_tokens,
                output_tokens=output_tokens,
                failed=failed,
                error_type=type(error).__name__ if error else None,
            )


def load_semantic_paper_input(
    database: Database, paper_id: int, evidence_run: dict[str, Any]
) -> SemanticPaperInput:
    with database.connect() as connection:
        translation_table = connection.execute(
            "SELECT 1 FROM sqlite_master WHERE type='table' AND name='paper_translations'"
        ).fetchone()
        chinese_join = (
            "LEFT JOIN paper_translations pt ON pt.paper_id=p.id" if translation_table else ""
        )
        chinese_column = "pt.chinese_title" if translation_table else "NULL"
        row = connection.execute(
            f"""SELECT p.id,p.doi,p.title,p.abstract,p.authors_json,p.keywords_json,
            p.jel_codes_json,j.name journal,COALESCE(p.issue_date,p.online_date) publication_date,
            {chinese_column} chinese_title
            FROM papers p JOIN journals j ON j.id=p.journal_id {chinese_join}
            WHERE p.id=?""",  # noqa: S608 -- only static fragments selected above
            (paper_id,),
        ).fetchone()
    if not row:
        raise ValueError(f"Paper not found: {paper_id}")
    payload = evidence_run["evidence_payload"]
    evidence = payload["evidence"]
    scoring = payload["scoring"]
    matched = tuple(str(item["concept_id"]) for item in evidence.get("matched_concepts", []))
    exclusions = tuple(
        str(item["context_id"]) for item in evidence.get("matched_exclusion_contexts", [])
    )
    codes = tuple(
        str(item["code"])
        for item in evidence.get("subject_code_signals", [])
        if item.get("matched")
    )
    metadata = evidence["metadata_evidence"]
    return SemanticPaperInput(
        paper_id=int(row["id"]),
        doi=row["doi"],
        original_title=row["title"],
        chinese_title=row["chinese_title"],
        original_abstract=row["abstract"],
        journal=row["journal"],
        publication_date=row["publication_date"],
        authors=_string_tuple(row["authors_json"]),
        keywords=_string_tuple(row["keywords_json"]),
        subject_codes=codes or _string_tuple(row["jel_codes_json"]),
        matched_ontology_concepts=matched,
        exclusion_contexts=exclusions,
        journal_prior_tier=evidence["journal_prior_signal"].get("prior_tier"),
        metadata_confidence=float(metadata["overall_metadata_confidence"]),
        title_opacity=evidence["opaque_title_features"]["opacity_level"],
        evidence_limitations=tuple(metadata.get("limitations", []))
        + tuple(evidence.get("extraction_warnings", [])),
        rule_evidence_tier=scoring["rule_evidence_tier"],
        concept_family_coverage=tuple(evidence.get("concept_family_coverage", [])),
        domain_evidence_run_id=int(evidence_run["id"]),
        workflow_batch_id=evidence_run.get("workflow_batch_id"),
        domain_evidence_hash=str(evidence_run["config_bundle_sha256"]),
        profile_id=str(evidence_run["profile_id"]),
        ontology_version=str(evidence_run["ontology_version"]),
        scoring_version=str(evidence_run["scoring_version"]),
    )


def _string_tuple(raw: object) -> tuple[str, ...]:
    try:
        value = json.loads(raw) if isinstance(raw, str) else raw
    except json.JSONDecodeError:
        return ()
    if not isinstance(value, list):
        return ()
    return tuple(item for item in value if isinstance(item, str))
