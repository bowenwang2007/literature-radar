from __future__ import annotations

from collections import defaultdict
from typing import Literal

from pydantic import BaseModel, ConfigDict

from eduecon_radar.domain.evidence import DomainEvidenceResult
from eduecon_radar.domain.models import EvidenceScoringParameters, ScoringDefinition


class RuleScoringResult(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True)

    ontology_evidence_score: float | None
    subject_code_evidence_score: float | None
    journal_prior_evidence_score: float | None
    exclusion_evidence_penalty: float | None
    concept_diversity_score: float | None
    metadata_confidence: float
    provisional_rule_score: float | None
    rule_evidence_tier: Literal[
        "strong_positive", "moderate_positive", "weak_or_mixed", "little_evidence"
    ]
    evidence_summary: tuple[str, ...]
    warnings: tuple[str, ...]
    publishable: bool
    calibration_status: str
    scoring_version: str
    positive_negative_conflict: bool


class RuleEvidenceScorer:
    def __init__(self, scoring: ScoringDefinition) -> None:
        self.scoring = scoring

    def score(self, evidence: DomainEvidenceResult) -> RuleScoringResult:
        parameters = self.scoring.evidence_parameters
        qualitative_tier = self._qualitative_tier(evidence)
        summary = self._summary(evidence)
        warnings = list(evidence.extraction_warnings)
        if parameters is None:
            warnings.append("uncalibrated_no_numeric_evidence_parameters")
            return RuleScoringResult(
                ontology_evidence_score=None,
                subject_code_evidence_score=None,
                journal_prior_evidence_score=None,
                exclusion_evidence_penalty=None,
                concept_diversity_score=None,
                metadata_confidence=evidence.metadata_evidence.overall_metadata_confidence,
                provisional_rule_score=None,
                rule_evidence_tier=qualitative_tier,
                evidence_summary=summary,
                warnings=tuple(dict.fromkeys(warnings)),
                publishable=False,
                calibration_status=self.scoring.calibration_status,
                scoring_version=self.scoring.scoring_version,
                positive_negative_conflict=self._has_conflict(evidence),
            )
        components = self._numeric_components(evidence, parameters)
        provisional = sum(
            components[name] * parameters.component_weights[name]
            for name in (
                "ontology",
                "subject_code",
                "journal_prior",
                "exclusion",
                "concept_diversity",
            )
        )
        return RuleScoringResult(
            ontology_evidence_score=components["ontology"],
            subject_code_evidence_score=components["subject_code"],
            journal_prior_evidence_score=components["journal_prior"],
            exclusion_evidence_penalty=components["exclusion"],
            concept_diversity_score=components["concept_diversity"],
            metadata_confidence=evidence.metadata_evidence.overall_metadata_confidence,
            provisional_rule_score=provisional,
            rule_evidence_tier=qualitative_tier,
            evidence_summary=summary,
            warnings=tuple(dict.fromkeys(warnings)),
            publishable=bool(
                self.scoring.publishable and self.scoring.calibration_status == "calibrated"
            ),
            calibration_status=self.scoring.calibration_status,
            scoring_version=self.scoring.scoring_version,
            positive_negative_conflict=self._has_conflict(evidence),
        )

    @staticmethod
    def _numeric_components(
        evidence: DomainEvidenceResult, parameters: EvidenceScoringParameters
    ) -> dict[str, float]:
        family_scores: dict[str, float] = defaultdict(float)
        concept_source_scores: dict[tuple[str, str], float] = {}
        for item in evidence.matched_concepts:
            value = (
                parameters.strength_values[item.signal_strength]
                * parameters.source_field_multipliers[item.source_field]
            )
            key = (item.concept_id, item.source_field)
            concept_source_scores[key] = min(
                max(concept_source_scores.get(key, 0.0), value),
                parameters.concept_per_source_cap,
            )
        for (concept_id, _), value in concept_source_scores.items():
            family = next(
                item.concept_family
                for item in evidence.matched_concepts
                if item.concept_id == concept_id
            )
            family_scores[family] = min(
                family_scores[family] + value, parameters.concept_family_cap
            )
        ontology = sum(family_scores.values())
        subject = sum(
            parameters.subject_code_values[item.signal_type]
            for item in evidence.subject_code_signals
            if item.matched
        )
        journal = 0.0
        prior = evidence.journal_prior_signal
        if prior.numeric_eligible and prior.prior_tier:
            journal = parameters.journal_prior_values[prior.prior_tier]
        exclusion = -sum(
            parameters.exclusion_strength_values[item.negative_strength]
            for item in evidence.matched_exclusion_contexts
            if not item.exception_triggered
        )
        diversity = float(len(evidence.concept_family_coverage))
        return {
            "ontology": ontology,
            "subject_code": subject,
            "journal_prior": journal,
            "exclusion": exclusion,
            "concept_diversity": diversity,
        }

    @staticmethod
    def _qualitative_tier(evidence: DomainEvidenceResult) -> str:
        strengths = {item.signal_strength for item in evidence.matched_concepts}
        positive = evidence.positive_evidence_count
        negative = evidence.negative_evidence_count
        if "strong" in strengths and not negative:
            return "strong_positive"
        if "strong" in strengths or "moderate" in strengths:
            return "weak_or_mixed" if negative else "moderate_positive"
        if positive:
            return "weak_or_mixed"
        return "little_evidence"

    @staticmethod
    def _has_conflict(evidence: DomainEvidenceResult) -> bool:
        return evidence.positive_evidence_count > 0 and evidence.negative_evidence_count > 0

    @staticmethod
    def _summary(evidence: DomainEvidenceResult) -> tuple[str, ...]:
        values: list[str] = []
        if evidence.matched_concepts:
            values.append(
                "ontology:"
                + ",".join(sorted({item.concept_id for item in evidence.matched_concepts}))
            )
        if evidence.subject_code_signals:
            values.append(
                "subject_codes:" + ",".join(item.code for item in evidence.subject_code_signals)
            )
        if evidence.journal_prior_signal.signal_type != "neutral":
            values.append("journal_prior:" + evidence.journal_prior_signal.signal_type)
        if evidence.matched_exclusion_contexts:
            values.append(
                "exclusions:"
                + ",".join(item.context_id for item in evidence.matched_exclusion_contexts)
            )
        if evidence.metadata_evidence.limitations:
            values.append("limitations:" + ",".join(evidence.metadata_evidence.limitations))
        return tuple(values)
