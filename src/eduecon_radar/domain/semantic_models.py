from __future__ import annotations

from typing import Literal

from pydantic import BaseModel, ConfigDict, Field, model_validator

Decision = Literal["include", "borderline", "exclude"]
Sufficiency = Literal["high", "medium", "low", "insufficient"]


class SemanticModel(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True)


class FastSemanticResult(SemanticModel):
    paper_id: int
    semantic_score: float = Field(ge=0, le=100)
    decision: Decision
    confidence: float = Field(ge=0, le=1)
    education_is_core: bool
    economics_mechanism_present: bool
    education_dimension: str
    economics_dimension: str
    reason: str
    positive_signals: tuple[str, ...] = ()
    negative_signals: tuple[str, ...] = ()
    evidence_basis: tuple[str, ...]
    limitations: tuple[str, ...] = ()
    title_evidence_sufficiency: Sufficiency
    abstract_evidence_sufficiency: Sufficiency
    information_sufficiency: Sufficiency
    suspected_false_friend: bool = False
    adjacent_domain: str | None = None
    needs_strong_review: bool = False
    title_only_decision: Decision
    title_only_confidence: float = Field(ge=0, le=1)
    abstract_aware_decision: Decision
    abstract_aware_confidence: float = Field(ge=0, le=1)
    title_abstract_conflict: bool = False
    abstain: bool = False
    model_name: str
    prompt_version: str


class StrongSemanticReviewResult(SemanticModel):
    paper_id: int
    independent_decision: Decision
    semantic_score: float = Field(ge=0, le=100)
    confidence: float = Field(ge=0, le=1)
    agrees_with_fast: bool
    agrees_with_rule: bool | None = None
    conflict_resolution: str
    decisive_evidence: tuple[str, ...]
    remaining_uncertainty: tuple[str, ...] = ()
    evidence_basis: tuple[str, ...]
    limitations: tuple[str, ...] = ()
    recommended_final_decision: Decision
    decision_sufficiency: Sufficiency
    abstain: bool = False
    model_name: str
    prompt_version: str


class EscalationDecision(SemanticModel):
    escalate: bool
    escalation_tier: Literal["mandatory", "recommended", "not_needed"]
    reasons: tuple[str, ...]
    rule_semantic_conflict_type: str
    conflict_types: tuple[str, ...] = ()
    evidence_snapshot: dict[str, object]
    policy_version: str


class CompositeDecisionResult(SemanticModel):
    provisional_decision: Decision
    decision_basis: tuple[str, ...]
    rule_component: str
    fast_semantic_component: Decision
    strong_semantic_component: Decision | None = None
    conflict_types: tuple[str, ...]
    uncertainty_level: Literal["low", "medium", "high"]
    needs_human_missed_review: bool
    needs_final_human_review: bool = True
    publishable: bool = False
    calibration_status: Literal["uncalibrated"] = "uncalibrated"
    policy_version: str
    warnings: tuple[str, ...] = ()
    abstained: bool = False

    @model_validator(mode="after")
    def experimental_only(self) -> CompositeDecisionResult:
        if self.publishable:
            raise ValueError("Phase 4 composite decisions cannot be publishable")
        return self


class SemanticMissedPriority(SemanticModel):
    missed_review_priority_tier: Literal["urgent", "high", "normal", "low"]
    missed_review_priority_reasons: tuple[str, ...]
    provisional_score: float | None = None
    calibrated: bool = False
    policy_version: str


class SemanticPaperInput(SemanticModel):
    paper_id: int
    doi: str | None = None
    original_title: str
    chinese_title: str | None = None
    original_abstract: str | None = None
    journal: str
    publication_date: str | None = None
    authors: tuple[str, ...] = ()
    keywords: tuple[str, ...] = ()
    subject_codes: tuple[str, ...] = ()
    matched_ontology_concepts: tuple[str, ...] = ()
    exclusion_contexts: tuple[str, ...] = ()
    journal_prior_tier: str | None = None
    metadata_confidence: float = Field(ge=0, le=1)
    title_opacity: Literal["low", "medium", "high"]
    evidence_limitations: tuple[str, ...] = ()
    rule_evidence_tier: str
    concept_family_coverage: tuple[str, ...] = ()
    domain_evidence_run_id: int
    workflow_batch_id: int | None = None
    domain_evidence_hash: str
    profile_id: str = "education_economics"
    ontology_version: str
    scoring_version: str


class SemanticRoleConfig(SemanticModel):
    provider: str
    model: str
    prompt_id: str
    prompt_version: str
    temperature: float = 0
    max_tokens: int
    timeout_seconds: float
    enabled: bool = False
    status: Literal["draft", "test_fixture", "released"] = "draft"


class EscalationPolicyConfig(SemanticModel):
    version: str
    low_confidence_max: float | None = Field(default=None, ge=0, le=1)
    no_abstract_confidence_cap: float | None = Field(default=None, ge=0, le=1)
    reason_conflict_detection_enabled: bool = True


class CompositePolicyConfig(SemanticModel):
    version: str
    calibration_status: Literal["uncalibrated"] = "uncalibrated"
    publishable: bool = False


class SemanticEngineConfig(SemanticModel):
    profile_id: str
    version: str
    status: Literal["draft", "test_fixture", "released"]
    live_ai_allowed: bool = False
    fast: SemanticRoleConfig
    strong_review: SemanticRoleConfig
    escalation: EscalationPolicyConfig
    composite: CompositePolicyConfig

    @model_validator(mode="after")
    def production_safety(self) -> SemanticEngineConfig:
        if self.status != "released" and self.live_ai_allowed:
            raise ValueError("Draft/test semantic configuration cannot allow live AI")
        if self.composite.publishable:
            raise ValueError("Phase 4 composite policy cannot be publishable")
        return self
