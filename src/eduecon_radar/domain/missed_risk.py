from __future__ import annotations

from typing import Literal

from pydantic import BaseModel, ConfigDict

from eduecon_radar.domain.evidence import DomainEvidenceResult
from eduecon_radar.domain.models import EvidenceScoringParameters
from eduecon_radar.domain.scoring import RuleScoringResult


class EvidenceReviewPriority(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True)

    priority_tier: Literal["urgent", "high", "normal", "low"]
    priority_score: float | None
    priority_reasons: tuple[str, ...]
    publishable: bool = False
    phase: Literal["pre_semantic"] = "pre_semantic"
    scoring_version: str


class MissedRiskPrioritizer:
    def __init__(self, scoring_version: str, parameters: EvidenceScoringParameters | None) -> None:
        self.scoring_version = scoring_version
        self.parameters = parameters

    def prioritize(
        self, evidence: DomainEvidenceResult, scoring: RuleScoringResult
    ) -> EvidenceReviewPriority:
        reasons: list[str] = []
        if scoring.positive_negative_conflict:
            reasons.append("positive_negative_evidence_conflict")
        if not evidence.abstract_available and evidence.positive_evidence_count:
            reasons.append("positive_evidence_without_abstract")
        if (
            evidence.journal_prior_signal.signal_type == "verified_prior_signal"
            and evidence.positive_evidence_count <= 1
        ):
            reasons.append("strong_journal_prior_with_weak_concept_evidence")
        if evidence.opaque_title_features.opacity_level == "high":
            reasons.append("high_title_opacity")
        elif evidence.opaque_title_features.opaque_title:
            reasons.append("title_opacity")
        if len(evidence.concept_family_coverage) >= 2:
            reasons.append("multiple_concept_families")
        if evidence.extraction_warnings:
            reasons.append("malformed_metadata")
        if evidence.metadata_evidence.overall_metadata_confidence < 0.6:
            reasons.append("metadata_sparse")
        if not reasons and scoring.rule_evidence_tier == "little_evidence":
            reasons.append("complete_or_low_evidence_review_later")

        score = None
        if self.parameters is not None:
            score = sum(
                self.parameters.priority_reason_values.get(reason, 0.0) for reason in reasons
            )
            thresholds = self.parameters.priority_thresholds
            if score >= thresholds["urgent"]:
                tier = "urgent"
            elif score >= thresholds["high"]:
                tier = "high"
            elif score >= thresholds["normal"]:
                tier = "normal"
            else:
                tier = "low"
        elif any(
            reason in reasons
            for reason in (
                "positive_negative_evidence_conflict",
                "positive_evidence_without_abstract",
                "high_title_opacity",
            )
        ):
            tier = "high"
        elif reasons == ["complete_or_low_evidence_review_later"]:
            tier = "low"
        else:
            tier = "normal"
        return EvidenceReviewPriority(
            priority_tier=tier,
            priority_score=score,
            priority_reasons=tuple(reasons),
            scoring_version=self.scoring_version,
        )
