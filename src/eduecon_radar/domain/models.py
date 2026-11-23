from __future__ import annotations

from typing import Any, Literal

from pydantic import BaseModel, ConfigDict, Field, model_validator

ConfigStatus = Literal["candidate", "draft", "released"]


class FrozenModel(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True)


class EvidenceSource(FrozenModel):
    source_type: str
    title: str
    identifier: str | None = None
    url: str | None = None
    year: int | None = None


class AdjacentDomain(FrozenModel):
    domain_id: str
    display_name: str
    weight_status: Literal["reserved", "draft"] = "reserved"


class ResearchProfile(FrozenModel):
    profile_id: str
    display_name: str
    version: str
    status: ConfigStatus
    primary_domain: str
    adjacent_domains: tuple[AdjacentDomain, ...] = ()
    ontology_version: str
    exclusion_context_version: str
    journal_prior_version: str
    subject_code_version: str
    scoring_version: str
    prompt_version: str


class ProfileFile(FrozenModel):
    profile: ResearchProfile


class OntologyConcept(FrozenModel):
    concept_id: str
    preferred_term: str
    display_name_zh: str
    description: str
    synonyms: tuple[str, ...] = ()
    phrases: tuple[str, ...] = ()
    parent_concept: str | None = None
    child_concepts: tuple[str, ...] = ()
    signal_strength: Literal["neutral", "weak", "moderate", "strong"] = "neutral"
    negative_contexts: tuple[str, ...] = ()
    evidence_sources: tuple[EvidenceSource, ...] = ()
    notes: str | None = None
    classification: Literal[
        "high_precision_anchor", "contextual_anchor", "ambiguous_signal", "disabled_candidate"
    ] = "disabled_candidate"
    matching_mode: Literal["exact_phrase", "normalized_phrase", "contextual_phrase"] = (
        "exact_phrase"
    )
    education_exceptions: tuple[str, ...] = ()
    enabled: bool = False
    status: ConfigStatus = "draft"
    version_added: str

    @model_validator(mode="after")
    def released_requires_evidence(self) -> OntologyConcept:
        if self.status == "released" and not self.evidence_sources:
            raise ValueError("Released ontology concepts require evidence_sources")
        return self


class OntologyDefinition(FrozenModel):
    profile_id: str
    version: str
    released_at: str | None = None
    status: ConfigStatus
    evidence_policy: str
    concepts: tuple[OntologyConcept, ...]


class OntologyFile(FrozenModel):
    ontology: OntologyDefinition


class ExclusionContext(FrozenModel):
    context_id: str
    preferred_term: str
    phrases: tuple[str, ...]
    description: str
    signal_strength: Literal["weak", "moderate", "strong"]
    exceptions: tuple[str, ...] = ()
    evidence_sources: tuple[EvidenceSource, ...] = ()
    enabled: bool = False
    hard_exclude: bool = False
    status: ConfigStatus = "draft"
    version_added: str

    @model_validator(mode="after")
    def cannot_be_hard_exclude(self) -> ExclusionContext:
        if self.hard_exclude:
            raise ValueError("Exclusion contexts may not be configured as hard excludes")
        if self.status == "released" and not self.evidence_sources:
            raise ValueError("Released exclusion contexts require evidence_sources")
        return self


class ExclusionDefinition(FrozenModel):
    profile_id: str
    version: str
    released_at: str | None = None
    status: ConfigStatus
    contexts: tuple[ExclusionContext, ...]


class ExclusionFile(FrozenModel):
    exclusion_contexts: ExclusionDefinition


class SubjectCode(FrozenModel):
    code: str
    label: str
    source: str
    signal_type: Literal["positive", "adjacent", "context", "unknown"]
    strength: Literal["neutral", "weak", "moderate", "strong"] = "neutral"
    applicable_contexts: tuple[str, ...] = ()
    evidence_sources: tuple[EvidenceSource, ...] = ()
    enabled: bool = False
    status: ConfigStatus = "candidate"
    version_added: str

    @model_validator(mode="after")
    def released_requires_evidence(self) -> SubjectCode:
        if self.status == "released" and not self.evidence_sources:
            raise ValueError("Released subject codes require evidence_sources")
        return self


class SubjectCodeDefinition(FrozenModel):
    profile_id: str
    version: str
    released_at: str | None = None
    status: ConfigStatus
    missing_signal_behavior: Literal["neutral"] = "neutral"
    codes: tuple[SubjectCode, ...]


class SubjectCodeFile(FrozenModel):
    subject_codes: SubjectCodeDefinition


class JournalPrior(FrozenModel):
    journal_id: str
    canonical_name: str
    aliases: tuple[str, ...] = ()
    group: Literal["monitoring", "reference", "both"]
    monitoring_enabled: bool = False
    reference_enabled: bool = False
    prior_tier: Literal["strong_positive", "moderate_positive", "neutral"] | None = None
    prior_value: float | None = None
    verified: bool = False
    print_issn: str | None = None
    online_issn: str | None = None
    verification_sources: tuple[EvidenceSource, ...] = ()
    notes: str | None = None
    status: ConfigStatus = "candidate"
    version_added: str

    @model_validator(mode="after")
    def released_must_be_verified(self) -> JournalPrior:
        if self.status == "released" and not self.verified:
            raise ValueError("Unverified journal priors cannot be released")
        if self.status != "released" and self.prior_value is not None:
            raise ValueError("Candidate or draft journal priors cannot publish numeric values")
        return self


class JournalPriorDefinition(FrozenModel):
    profile_id: str
    version: str
    released_at: str | None = None
    status: ConfigStatus
    journals: tuple[JournalPrior, ...]


class JournalPriorFile(FrozenModel):
    journal_priors: JournalPriorDefinition


class ScoringComponent(FrozenModel):
    component_id: Literal[
        "ontology_score",
        "subject_code_score",
        "journal_prior_score",
        "semantic_score",
        "exclusion_penalty",
        "metadata_confidence",
        "combined_score",
        "missed_risk_score",
    ]
    description: str
    enabled: bool = False
    weight: float | None = None
    status: Literal["placeholder", "draft", "calibrated"] = "placeholder"


class ScoringThresholds(FrozenModel):
    include_min: float | None = None
    borderline_min: float | None = None
    exclude_max: float | None = None


class EvidenceScoringParameters(FrozenModel):
    strength_values: dict[Literal["neutral", "weak", "moderate", "strong"], float]
    source_field_multipliers: dict[Literal["title", "abstract", "keywords"], float]
    concept_per_source_cap: float
    concept_family_cap: float
    subject_code_values: dict[Literal["positive", "adjacent", "context", "unknown"], float]
    journal_prior_values: dict[Literal["strong_positive", "moderate_positive", "neutral"], float]
    exclusion_strength_values: dict[Literal["weak", "moderate", "strong"], float]
    component_weights: dict[
        Literal[
            "ontology",
            "subject_code",
            "journal_prior",
            "exclusion",
            "concept_diversity",
        ],
        float,
    ]
    priority_reason_values: dict[str, float]
    priority_thresholds: dict[Literal["urgent", "high", "normal"], float]


class ScoringDefinition(FrozenModel):
    profile_id: str
    scoring_version: str
    status: ConfigStatus
    calibration_status: Literal["uncalibrated", "draft", "test_fixture", "calibrated"]
    gold_set_version: str | None = None
    components: tuple[ScoringComponent, ...]
    thresholds: ScoringThresholds
    evidence_parameters: EvidenceScoringParameters | None = None
    publishable: bool = False
    notes: str

    @model_validator(mode="after")
    def uncalibrated_has_no_published_values(self) -> ScoringDefinition:
        threshold_values = self.thresholds.model_dump().values()
        if self.calibration_status == "uncalibrated":
            if any(value is not None for value in threshold_values):
                raise ValueError("Uncalibrated scoring cannot publish decision thresholds")
            if any(component.weight is not None for component in self.components):
                raise ValueError("Uncalibrated scoring cannot publish component weights")
            if self.evidence_parameters is not None:
                raise ValueError("Uncalibrated scoring cannot publish evidence parameters")
        if self.calibration_status == "calibrated" and not self.gold_set_version:
            raise ValueError("Calibrated scoring requires a gold_set_version")
        if self.calibration_status != "calibrated" and self.publishable:
            raise ValueError("Only calibrated scoring may be publishable")
        return self


class ScoringFile(FrozenModel):
    scoring: ScoringDefinition


class OntologyCandidate(FrozenModel):
    candidate_id: str
    candidate_type: Literal[
        "concept", "synonym", "exclusion_context", "journal_prior", "subject_code", "weight"
    ]
    proposed_value: dict[str, Any]
    source_type: str
    source_reference: str | None = None
    supporting_paper_ids: tuple[int, ...] = ()
    proposed_at: str
    status: Literal["pending", "accepted", "rejected"] = "pending"
    reviewed_at: str | None = None
    reviewer_note: str | None = None


class CandidateDefinition(FrozenModel):
    profile_id: str
    version: str
    candidates: tuple[OntologyCandidate, ...] = ()


class CandidateFile(FrozenModel):
    ontology_candidates: CandidateDefinition


class DomainMetadata(FrozenModel):
    profile_id: str
    bundle_version: str
    status: ConfigStatus
    released_at: str | None = None
    hash_algorithm: Literal["SHA-256"] = "SHA-256"
    model_version: str
    prompt_version: str
    notes: str


class MetadataFile(FrozenModel):
    metadata: DomainMetadata


class DomainConfigBundle(FrozenModel):
    profile: ResearchProfile
    ontology: OntologyDefinition
    exclusions: ExclusionDefinition
    subject_codes: SubjectCodeDefinition
    journal_priors: JournalPriorDefinition
    scoring: ScoringDefinition
    candidates: CandidateDefinition
    metadata: DomainMetadata
    config_sha256: str = Field(pattern=r"^[A-F0-9]{64}$")
