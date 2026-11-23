from __future__ import annotations

from eduecon_radar.domain.semantic_models import (
    EscalationDecision,
    EscalationPolicyConfig,
    FastSemanticResult,
    SemanticPaperInput,
)


class EscalationPolicy:
    def __init__(self, config: EscalationPolicyConfig) -> None:
        self.config = config

    def decide(self, paper: SemanticPaperInput, fast: FastSemanticResult) -> EscalationDecision:
        reasons: list[str] = []
        conflicts: list[str] = []
        mandatory = False
        conflict = "no_conflict"
        positive_rule = paper.rule_evidence_tier in {"strong_positive", "moderate_positive"}
        weak_rule = paper.rule_evidence_tier == "little_evidence"

        if positive_rule and fast.decision == "exclude":
            conflict = "rule_positive_semantic_exclude"
            conflicts.append(conflict)
            reasons.append(conflict)
            mandatory = True
        elif weak_rule and fast.decision == "include":
            conflict = "rule_weak_semantic_include"
            conflicts.append(conflict)
            reasons.append(conflict)
        if fast.title_abstract_conflict:
            conflict = "title_abstract_conflict"
            conflicts.append(conflict)
            reasons.append(conflict)
            mandatory = True
        if fast.positive_signals and fast.negative_signals:
            reasons.append("positive_negative_signal_conflict")
            conflicts.append("positive_negative_evidence_conflict")
        if fast.decision == "borderline" or fast.abstain:
            reasons.append("borderline_or_abstention")
        if fast.needs_strong_review:
            reasons.append("fast_requested_strong_review")
        if (
            self.config.low_confidence_max is not None
            and fast.confidence <= self.config.low_confidence_max
        ):
            reasons.append("low_confidence")
        if not paper.original_abstract:
            reasons.append("abstract_missing")
            conflicts.append("insufficient_information")
        if paper.title_opacity == "high":
            reasons.append("opaque_title")
        if (
            paper.journal_prior_tier in {"verified_domain", "strong_positive"}
            and fast.decision == "exclude"
        ):
            reasons.append("journal_prior_semantic_exclude")
            conflicts.append("prior_semantic_conflict")
        if fast.suspected_false_friend and fast.economics_mechanism_present:
            reasons.append("false_friend_with_economics_signal")
        if paper.evidence_limitations:
            reasons.append("evidence_limitations")
            conflicts.append("insufficient_information")
        if self.config.reason_conflict_detection_enabled:
            if (
                fast.decision == "exclude"
                and fast.education_is_core
                and fast.economics_mechanism_present
            ):
                reasons.append("decision_reason_contradiction")
                mandatory = True
            if (
                fast.decision == "include"
                and fast.suspected_false_friend
                and not fast.education_is_core
            ):
                reasons.append("decision_reason_contradiction")
                mandatory = True

        unique = tuple(dict.fromkeys(reasons))
        tier = "mandatory" if mandatory else "recommended" if unique else "not_needed"
        return EscalationDecision(
            escalate=bool(unique),
            escalation_tier=tier,
            reasons=unique,
            rule_semantic_conflict_type=conflict,
            conflict_types=tuple(dict.fromkeys(conflicts)),
            evidence_snapshot={
                "rule_evidence_tier": paper.rule_evidence_tier,
                "fast_decision": fast.decision,
                "fast_confidence": fast.confidence,
                "abstract_available": bool(paper.original_abstract),
                "title_opacity": paper.title_opacity,
            },
            policy_version=self.config.version,
        )
