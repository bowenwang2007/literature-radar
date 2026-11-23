from __future__ import annotations

import json
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Protocol

from eduecon_radar.db import Database, utcnow
from eduecon_radar.domain.composite_decision import CompositeDecisionPolicy
from eduecon_radar.domain.escalation import EscalationPolicy
from eduecon_radar.domain.evidence import EvidenceExtractor, PaperEvidenceInput
from eduecon_radar.domain.goldset_archive_gate import validate_archive_manifest
from eduecon_radar.domain.goldset_calibration import load_human_gold_snapshot
from eduecon_radar.domain.goldset_storage import gold_set_schema_ready
from eduecon_radar.domain.models import DomainConfigBundle
from eduecon_radar.domain.scoring import RuleEvidenceScorer
from eduecon_radar.domain.semantic_judgment import (
    FastSemanticJudge,
    SemanticAIClient,
    StrongSemanticReviewer,
)
from eduecon_radar.domain.semantic_models import SemanticEngineConfig, SemanticPaperInput
from eduecon_radar.domain.semantic_prompting import (
    SemanticPromptBuilder,
    canonical_sha256,
)

CONFIRMATION = "RUN CONTROLLED GOLD PILOT"


@dataclass(frozen=True)
class PilotBudget:
    max_papers: int
    max_fast_calls: int
    max_strong_calls: int
    max_input_tokens: int
    max_output_tokens: int
    budget_limit: float
    max_retries: int = 1
    failure_rate_threshold: float = 0.25


@dataclass(frozen=True)
class PilotRequest:
    gold_set_snapshot: Path
    gold_set_sha256: str
    output_database: Path
    archive_manifest: Path
    production_database: Path
    model_bundle_version: str
    expected_config_sha256: str
    run_uid: str
    allow_live_ai: bool
    confirmation: str
    budget: PilotBudget
    allow_resume: bool = False


@dataclass(frozen=True)
class PilotPreflight:
    snapshot: dict
    snapshot_database_id: int
    prompt_bundle_sha256: str


@dataclass(frozen=True)
class PilotStats:
    run_uid: str
    status: str
    considered: int
    succeeded: int
    failed: int
    skipped_existing: int
    fast_calls: int
    strong_calls: int
    input_tokens: int
    output_tokens: int
    estimated_cost: float | None
    stop_reason: str | None


class GoldPilotProcessor(Protocol):
    def __call__(self, item: dict, *, allow_strong: bool) -> dict: ...


def validate_pilot_preflight(
    request: PilotRequest,
    *,
    bundle: DomainConfigBundle,
    engine: SemanticEngineConfig,
    prompts: SemanticPromptBuilder,
    api_key: str | None,
) -> PilotPreflight:
    if not request.allow_live_ai:
        raise ValueError("Live pilot requires --allow-live-ai")
    if request.confirmation != CONFIRMATION:
        raise ValueError(f'Live pilot requires --confirm "{CONFIRMATION}"')
    _validate_budget(request.budget)
    production = request.production_database.resolve()
    output = request.output_database.resolve()
    snapshot_path = request.gold_set_snapshot.resolve()
    if output == production or snapshot_path == production:
        raise ValueError("Production database paths are forbidden for Gold pilot input/output")
    if not output.is_file() or not gold_set_schema_ready(Database(output)):
        raise ValueError("Pilot output must be an existing independent Gold Set database")
    archive = validate_archive_manifest(
        request.archive_manifest,
        production_database=production,
        active_database=production,
    )
    if not archive.valid:
        raise ValueError("Production archive gate failed: " + ", ".join(archive.reasons))
    snapshot = load_human_gold_snapshot(snapshot_path, request.gold_set_sha256)
    if snapshot.get("source_db_sha256") != archive.source_sha256:
        raise ValueError("Gold snapshot source hash does not match the archived production source")
    if request.expected_config_sha256.upper() != bundle.config_sha256:
        raise ValueError("Domain configuration SHA-256 does not match the requested bundle")
    if not engine.live_ai_allowed or engine.status != "released":
        raise ValueError("Semantic engine is not explicitly released for controlled live AI")
    if not engine.fast.enabled or engine.fast.model == "not-enabled":
        raise ValueError("Fast semantic model is not enabled")
    if not engine.strong_review.enabled or engine.strong_review.model == "not-enabled":
        raise ValueError("Strong semantic model is not enabled")
    if not (api_key or "").strip():
        raise ValueError("A DeepSeek API key is required for an explicitly confirmed live pilot")
    prompts.validate()
    prompt_hash = semantic_prompt_bundle_sha256(prompts)
    with Database(output).connect() as connection:
        row = connection.execute(
            "SELECT id FROM gold_set_snapshots WHERE snapshot_sha256=?",
            (request.gold_set_sha256.upper(),),
        ).fetchone()
    if row is None:
        raise ValueError("Gold snapshot is not registered in the selected Gold Set database")
    return PilotPreflight(snapshot, int(row["id"]), prompt_hash)


def semantic_prompt_bundle_sha256(prompts: SemanticPromptBuilder) -> str:
    return canonical_sha256(
        [
            {
                "prompt_id": entry.prompt_id,
                "version": entry.version,
                "system_sha256": entry.system_sha256,
                "user_sha256": entry.user_sha256,
            }
            for entry in prompts.manifest.prompts
        ]
    )


class ControlledGoldPilot:
    def __init__(self, database: Database) -> None:
        if not gold_set_schema_ready(database):
            raise RuntimeError("Gold Set schema is not installed")
        self.database = database

    def run(
        self,
        request: PilotRequest,
        preflight: PilotPreflight,
        processor: GoldPilotProcessor,
    ) -> PilotStats:
        run_id, previous = self._start_or_resume(request, preflight)
        items = preflight.snapshot["items"][: request.budget.max_papers]
        stats = {
            "considered": 0,
            "succeeded": 0,
            "failed": 0,
            "skipped_existing": 0,
            "fast_calls": 0,
            "strong_calls": 0,
            "input_tokens": 0,
            "output_tokens": 0,
            "estimated_cost": None,
        }
        existing_costs: list[float] = []
        if previous:
            stats.update(previous)
            if previous.get("estimated_cost") is not None:
                existing_costs.append(float(previous["estimated_cost"]))
        stop_reason: str | None = None
        for raw_item in items:
            safe_item = _pilot_safe_item(raw_item)
            fingerprint = canonical_sha256(
                {
                    "item": safe_item,
                    "snapshot_sha256": request.gold_set_sha256.upper(),
                    "model_bundle_version": request.model_bundle_version,
                    "config_bundle_sha256": request.expected_config_sha256.upper(),
                    "prompt_bundle_sha256": preflight.prompt_bundle_sha256,
                }
            )
            if self._successful_fingerprint_exists(run_id, fingerprint):
                stats["skipped_existing"] += 1
                continue
            stop_reason = _budget_stop_reason(stats, request.budget)
            if stop_reason:
                break
            stats["considered"] += 1
            error: Exception | None = None
            result: dict | None = None
            attempts = 0
            remaining_fast_calls = request.budget.max_fast_calls - stats["fast_calls"]
            attempt_limit = min(request.budget.max_retries + 1, remaining_fast_calls)
            remaining_strong_calls = request.budget.max_strong_calls - stats["strong_calls"]
            allow_strong = remaining_strong_calls >= attempt_limit
            for _ in range(attempt_limit):
                attempts += 1
                try:
                    result = processor(safe_item, allow_strong=allow_strong)
                    error = None
                    break
                except Exception as exc:  # noqa: BLE001 - persisted as classified pilot failure
                    error = exc
            if error is not None or result is None:
                stats["failed"] += 1
                stats["fast_calls"] += attempts
                self._save_result(
                    run_id,
                    safe_item["item_id"],
                    fingerprint,
                    result=None,
                    failure_type=type(error).__name__ if error else "unknown",
                )
            else:
                fast_calls = max(int(result.get("fast_calls", 1)), attempts)
                strong_calls = int(result.get("strong_calls", 0)) * attempts
                input_tokens = int(result.get("input_tokens", 0))
                output_tokens = int(result.get("output_tokens", 0))
                estimated_cost = result.get("estimated_cost")
                stats["succeeded"] += 1
                stats["fast_calls"] += fast_calls
                stats["strong_calls"] += strong_calls
                stats["input_tokens"] += input_tokens
                stats["output_tokens"] += output_tokens
                if estimated_cost is not None:
                    existing_costs.append(float(estimated_cost))
                stats["estimated_cost"] = sum(existing_costs) if existing_costs else None
                self._save_result(
                    run_id,
                    safe_item["item_id"],
                    fingerprint,
                    result=result,
                    failure_type=None,
                )
            if stats["considered"] >= 3 and (
                stats["failed"] / stats["considered"] > request.budget.failure_rate_threshold
            ):
                stop_reason = "failure_rate_threshold"
                break
        all_done = stats["succeeded"] + stats["failed"] >= len(items)
        stop_reason = (
            None if all_done else stop_reason or _budget_stop_reason(stats, request.budget)
        )
        status = "completed" if all_done and not stop_reason else "partial"
        self._finish(run_id, status, stats, stop_reason)
        return PilotStats(
            run_uid=request.run_uid,
            status=status,
            stop_reason=stop_reason,
            **stats,
        )

    def _start_or_resume(
        self, request: PilotRequest, preflight: PilotPreflight
    ) -> tuple[int, dict | None]:
        with self.database.connect() as connection:
            existing = connection.execute(
                "SELECT * FROM pilot_runs WHERE run_uid=?", (request.run_uid,)
            ).fetchone()
            if existing:
                if not request.allow_resume:
                    raise FileExistsError("Pilot run already exists; use --resume-pilot-run")
                if existing["status"] not in {"partial", "failed"}:
                    raise ValueError("Only partial or failed pilot runs may be resumed")
                return int(existing["id"]), json.loads(existing["stats_json"])
            now = utcnow()
            cursor = connection.execute(
                """INSERT INTO pilot_runs
                (run_uid,gold_set_snapshot_id,model_bundle_version,config_bundle_sha256,
                 prompt_bundle_sha256,budget_json,stats_json,status,created_at,updated_at)
                VALUES (?,?,?,?,?,?,?,'running',?,?)""",
                (
                    request.run_uid,
                    preflight.snapshot_database_id,
                    request.model_bundle_version,
                    request.expected_config_sha256.upper(),
                    preflight.prompt_bundle_sha256,
                    json.dumps(asdict(request.budget), sort_keys=True),
                    "{}",
                    now,
                    now,
                ),
            )
        return int(cursor.lastrowid), None

    def _successful_fingerprint_exists(self, run_id: int, fingerprint: str) -> bool:
        with self.database.connect() as connection:
            return (
                connection.execute(
                    """SELECT 1 FROM pilot_results WHERE pilot_run_id=?
                    AND request_fingerprint=? AND succeeded=1""",
                    (run_id, fingerprint),
                ).fetchone()
                is not None
            )

    def _save_result(
        self,
        run_id: int,
        item_id: str,
        fingerprint: str,
        *,
        result: dict | None,
        failure_type: str | None,
    ) -> None:
        with self.database.connect() as connection:
            connection.execute(
                """INSERT INTO pilot_results
                (pilot_run_id,snapshot_item_id,request_fingerprint,result_json,succeeded,
                 failure_type,input_tokens,output_tokens,estimated_cost,created_at)
                VALUES (?,?,?,?,?,?,?,?,?,?)
                ON CONFLICT(pilot_run_id,request_fingerprint) DO UPDATE SET
                 result_json=excluded.result_json,succeeded=excluded.succeeded,
                 failure_type=excluded.failure_type,input_tokens=excluded.input_tokens,
                 output_tokens=excluded.output_tokens,estimated_cost=excluded.estimated_cost,
                 created_at=excluded.created_at""",
                (
                    run_id,
                    item_id,
                    fingerprint,
                    json.dumps(result, ensure_ascii=False, sort_keys=True) if result else None,
                    int(result is not None),
                    failure_type,
                    int(result.get("input_tokens", 0)) if result else None,
                    int(result.get("output_tokens", 0)) if result else None,
                    result.get("estimated_cost") if result else None,
                    utcnow(),
                ),
            )

    def _finish(self, run_id: int, status: str, stats: dict, stop_reason: str | None) -> None:
        with self.database.connect() as connection:
            connection.execute(
                """UPDATE pilot_runs SET stats_json=?,status=?,failure_summary=?,updated_at=?
                WHERE id=?""",
                (json.dumps(stats, sort_keys=True), status, stop_reason, utcnow(), run_id),
            )


class SemanticGoldPilotProcessor:
    """Phase 4 engine adapter for an explicitly confirmed Gold pilot only."""

    def __init__(
        self,
        *,
        bundle: DomainConfigBundle,
        engine: SemanticEngineConfig,
        prompts: SemanticPromptBuilder,
        client: SemanticAIClient,
    ) -> None:
        self.bundle = bundle
        self.engine = engine
        self.fast = FastSemanticJudge(config=engine, prompts=prompts, client=client)
        self.strong = StrongSemanticReviewer(config=engine, prompts=prompts, client=client)
        self.escalation = EscalationPolicy(engine.escalation)
        self.composite = CompositeDecisionPolicy(engine.composite)

    def __call__(self, item: dict, *, allow_strong: bool) -> dict:
        evidence = EvidenceExtractor(self.bundle).extract(
            PaperEvidenceInput(
                paper_id=int(item["pilot_paper_id"]),
                doi=item.get("doi"),
                title=item["original_title"],
                abstract=item.get("original_abstract"),
                authors=tuple(item.get("authors", [])),
                journal=item["journal"],
                publication_date=item.get("publication_date"),
                keywords=item.get("keywords", []),
                jel_codes=item.get("subject_codes", []),
                source_metadata={},
            )
        )
        scoring = RuleEvidenceScorer(self.bundle.scoring).score(evidence)
        paper = SemanticPaperInput(
            paper_id=int(item["pilot_paper_id"]),
            doi=item.get("doi"),
            original_title=item["original_title"],
            original_abstract=item.get("original_abstract"),
            journal=item["journal"],
            publication_date=item.get("publication_date"),
            authors=tuple(item.get("authors", [])),
            keywords=tuple(item.get("keywords", [])),
            subject_codes=tuple(item.get("subject_codes", [])),
            matched_ontology_concepts=tuple(
                sorted({match.concept_id for match in evidence.matched_concepts})
            ),
            exclusion_contexts=tuple(
                sorted({match.context_id for match in evidence.matched_exclusion_contexts})
            ),
            journal_prior_tier=evidence.journal_prior_signal.prior_tier,
            metadata_confidence=evidence.metadata_evidence.overall_metadata_confidence,
            title_opacity=evidence.opaque_title_features.opacity_level,
            evidence_limitations=evidence.metadata_evidence.limitations,
            rule_evidence_tier=scoring.rule_evidence_tier,
            concept_family_coverage=evidence.concept_family_coverage,
            domain_evidence_run_id=0,
            domain_evidence_hash=canonical_sha256(evidence.model_dump(mode="json")),
            ontology_version=self.bundle.ontology.version,
            scoring_version=self.bundle.scoring.scoring_version,
        )
        fast = self.fast.judge(paper)
        escalation = self.escalation.decide(paper, fast.result)
        strong = None
        if allow_strong and escalation.escalate:
            strong = self.strong.review(paper, fast.result, escalation)
        composite = self.composite.combine(
            paper,
            fast.result,
            escalation,
            strong.result if strong else None,
        )
        missed = self.composite.missed_priority(
            paper,
            composite,
            escalation,
            fast.result,
            strong.result if strong else None,
        )
        cost_values = (fast.estimated_cost, strong.estimated_cost if strong else None)
        costs = [value for value in cost_values if value is not None]
        return {
            "item_id": item["item_id"],
            "rule_decision": _rule_decision(scoring.rule_evidence_tier),
            "rule_evidence_tier": scoring.rule_evidence_tier,
            "fast_decision": fast.result.decision,
            "strong_decision": strong.result.recommended_final_decision if strong else None,
            "strong_triggered": bool(strong),
            "composite_decision": composite.provisional_decision,
            "semantic_score": fast.result.semantic_score,
            "confidence": fast.result.confidence,
            "abstain": fast.result.abstain,
            "missed_priority_tier": missed.missed_review_priority_tier,
            "missed_priority_score": None,
            "fast_calls": 1,
            "strong_calls": int(strong is not None),
            "input_tokens": fast.input_tokens + (strong.input_tokens if strong else 0),
            "output_tokens": fast.output_tokens + (strong.output_tokens if strong else 0),
            "estimated_cost": sum(costs) if costs else None,
        }


def _pilot_safe_item(item: dict) -> dict:
    """Strict allow-list: the human final label can never cross the AI boundary."""
    allowed = {
        "item_id",
        "doi",
        "original_title",
        "original_abstract",
        "authors",
        "journal",
        "journal_group",
        "publication_date",
        "keywords",
        "subject_codes",
        "difficulty_tags",
    }
    safe = {name: item.get(name) for name in allowed}
    safe["item_id"] = str(item["item_id"])
    safe["pilot_paper_id"] = int(canonical_sha256(safe["item_id"])[0:8], 16)
    serialized = json.dumps(safe, ensure_ascii=False, sort_keys=True)
    if "final_label" in serialized or "gold_label" in serialized:
        raise RuntimeError("Gold label isolation failed")
    return safe


def _validate_budget(budget: PilotBudget) -> None:
    positive = (
        budget.max_papers,
        budget.max_fast_calls,
        budget.max_input_tokens,
        budget.max_output_tokens,
        budget.budget_limit,
    )
    if any(value <= 0 for value in positive):
        raise ValueError("Pilot paper, call, token and cost budgets must all be positive")
    if budget.max_strong_calls < 0:
        raise ValueError("Pilot max_strong_calls cannot be negative")
    if budget.max_retries < 0:
        raise ValueError("Pilot max_retries cannot be negative")
    if not 0 < budget.failure_rate_threshold <= 1:
        raise ValueError("Pilot failure_rate_threshold must be in (0,1]")


def _budget_stop_reason(stats: dict, budget: PilotBudget) -> str | None:
    if stats["considered"] >= budget.max_papers:
        return "max_papers"
    if stats["fast_calls"] >= budget.max_fast_calls:
        return "max_fast_calls"
    if stats["input_tokens"] >= budget.max_input_tokens:
        return "max_input_tokens"
    if stats["output_tokens"] >= budget.max_output_tokens:
        return "max_output_tokens"
    cost = stats.get("estimated_cost")
    if cost is not None and cost >= budget.budget_limit:
        return "budget_limit"
    return None


def _rule_decision(tier: str) -> str:
    if tier in {"strong_positive", "moderate_positive"}:
        return "include"
    if tier == "weak_or_mixed":
        return "borderline"
    return "exclude"
