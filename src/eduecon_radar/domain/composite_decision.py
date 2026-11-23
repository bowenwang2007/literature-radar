from __future__ import annotations

from eduecon_radar.domain.semantic_models import (
    CompositeDecisionResult,
    CompositePolicyConfig,
    EscalationDecision,
    FastSemanticResult,
    SemanticMissedPriority,
    SemanticPaperInput,
    StrongSemanticReviewResult,
)


class CompositeDecisionPolicy:
    """Conservative, uncalibrated Phase 4 shadow policy; never a final approval."""

    def __init__(self, config: CompositePolicyConfig) -> None:
        self.config = config

    def combine(
        self,
        paper: SemanticPaperInput,
        fast: FastSemanticResult,
        escalation: EscalationDecision,
        strong: StrongSemanticReviewResult | None = None,
        *,
        strong_failed: bool = False,
    ) -> CompositeDecisionResult:
        warnings = ["UNCALIBRATED - EXPERIMENTAL SHADOW DECISION"]
        conflicts = list(escalation.conflict_types)
        if strong and strong.independent_decision != fast.decision:
            conflicts.append("fast_strong_conflict")
        if fast.title_abstract_conflict:
            conflicts.append("title_abstract_conflict")
        if strong_failed:
            warnings.append("strong_review_failed")

        selected = strong.recommended_final_decision if strong else fast.decision
        high_precision = {
            "returns_to_education",
            "teacher_quality",
            "school_choice",
            "college_admissions",
            "school_finance",
            "educational_attainment",
            "student_achievement",
            "early_childhood_education",
            "higher_education",
            "student_loans",
            "education_production_function",
            "vocational_education",
        }
        high_precision_anchor = bool(high_precision.intersection(paper.matched_ontology_concepts))
        content_anchor = high_precision_anchor or fast.education_is_core
        no_abstract = not bool((paper.original_abstract or "").strip())
        abstained = fast.abstain or bool(strong and strong.abstain)
        basis = [f"rule:{paper.rule_evidence_tier}", f"fast:{fast.decision}"]
        if strong:
            basis.append(f"strong:{strong.recommended_final_decision}")

        if strong and strong.recommended_final_decision == "exclude" and not high_precision_anchor:
            decision = "borderline" if conflicts else "exclude"
            basis.append("strong_negative_not_overridden_by_weak_rule_or_journal_prior")
        elif selected == "exclude" and not content_anchor and paper.journal_prior_tier is not None:
            decision = "exclude"
            basis.append("journal_prior_cannot_override_negative_content_semantics")
        elif no_abstract and high_precision_anchor and selected == "exclude":
            decision = "borderline"
            basis.append("clear_education_title_without_abstract_requires_human_review")
        elif no_abstract and not content_anchor and selected == "include":
            decision = "borderline"
            basis.append("vague_title_without_abstract_cannot_be_high_confidence_include")
        elif paper.journal_prior_tier is not None and not content_anchor and selected == "include":
            decision = "borderline"
            basis.append("journal_prior_is_priority_only_without_education_anchor")
        elif abstained or fast.information_sufficiency == "insufficient" or strong_failed:
            decision = "borderline"
            basis.append("insufficient_or_failed_review_conservative_borderline")
        elif escalation.rule_semantic_conflict_type == "rule_positive_semantic_exclude":
            decision = "borderline"
            basis.append("positive_rule_semantic_exclude_requires_human_resolution")
        elif fast.suspected_false_friend and not fast.education_is_core:
            decision = "exclude"
            basis.append("false_friend_without_core_education_dimension")
        elif fast.adjacent_domain and fast.education_is_core and fast.economics_mechanism_present:
            decision = "include" if selected == "include" else "borderline"
            basis.append("adjacent_domain_with_substantive_education_economics_link")
        elif conflicts:
            decision = "borderline"
            basis.append("unresolved_conflict_conservative_borderline")
        else:
            decision = selected

        uncertainty = (
            "high"
            if abstained or conflicts or strong_failed
            else ("medium" if fast.information_sufficiency in {"low", "insufficient"} else "low")
        )
        return CompositeDecisionResult(
            provisional_decision=decision,
            decision_basis=tuple(basis),
            rule_component=paper.rule_evidence_tier,
            fast_semantic_component=fast.decision,
            strong_semantic_component=strong.independent_decision if strong else None,
            conflict_types=tuple(dict.fromkeys(conflicts)),
            uncertainty_level=uncertainty,
            needs_human_missed_review=decision in {"exclude", "borderline"} or bool(conflicts),
            publishable=False,
            calibration_status="uncalibrated",
            policy_version=self.config.version,
            warnings=tuple(warnings),
            abstained=abstained,
        )

    def missed_priority(
        self,
        paper: SemanticPaperInput,
        composite: CompositeDecisionResult,
        escalation: EscalationDecision,
        fast: FastSemanticResult | None = None,
        strong: StrongSemanticReviewResult | None = None,
    ) -> SemanticMissedPriority:
        reasons: list[str] = []
        if composite.provisional_decision == "exclude":
            reasons.append("provisional_exclusion_requires_missed-review_visibility")
        if composite.conflict_types:
            reasons.append("rule_or_semantic_conflict")
        if paper.title_opacity == "high":
            reasons.append("opaque_title")
        if not paper.original_abstract:
            reasons.append("abstract_missing")
        if escalation.escalation_tier == "mandatory":
            reasons.append("mandatory_escalation")
        if fast and fast.decision == "exclude":
            reasons.append("fast_semantic_exclude")
            if "low_confidence" in escalation.reasons:
                reasons.append("low_confidence_exclude")
            if fast.adjacent_domain:
                reasons.append("adjacent_domain_exclusion")
        if strong and strong.remaining_uncertainty:
            reasons.append("strong_review_remaining_uncertainty")
        if "prior_semantic_conflict" in escalation.conflict_types:
            reasons.append("journal_prior_semantic_conflict")
        if "positive_negative_evidence_conflict" in escalation.conflict_types:
            reasons.append("positive_negative_evidence_conflict")
        if "fast_strong_conflict" in composite.conflict_types:
            reasons.append("fast_strong_conflict")
        if paper.metadata_confidence < 0.5:
            reasons.append("sparse_metadata")
        if "mandatory_escalation" in reasons:
            tier = "urgent"
        elif len(reasons) >= 2:
            tier = "high"
        elif reasons:
            tier = "normal"
        else:
            tier = "low"
        return SemanticMissedPriority(
            missed_review_priority_tier=tier,
            missed_review_priority_reasons=tuple(reasons or ["no_special_semantic_risk"]),
            provisional_score=None,
            calibrated=False,
            policy_version=self.config.version,
        )
