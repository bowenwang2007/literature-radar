from eduecon_radar.domain.composite_decision import CompositeDecisionPolicy
from eduecon_radar.domain.semantic_models import (
    CompositePolicyConfig,
    EscalationDecision,
    FastSemanticResult,
    SemanticPaperInput,
    StrongSemanticReviewResult,
)


def _paper(*, concepts=(), abstract="Evidence", prior=None, tier="little_evidence"):
    return SemanticPaperInput(
        paper_id=1,
        original_title="Fixture",
        original_abstract=abstract,
        journal="Journal",
        matched_ontology_concepts=concepts,
        journal_prior_tier=prior,
        metadata_confidence=0.8 if abstract else 0.4,
        title_opacity="low" if concepts else "high",
        evidence_limitations=() if abstract else ("missing_abstract",),
        rule_evidence_tier=tier,
        domain_evidence_run_id=1,
        domain_evidence_hash="fixture",
        ontology_version="v",
        scoring_version="v",
    )


def _fast(decision="include", *, core=True, sufficient="high"):
    return FastSemanticResult(
        paper_id=1,
        semantic_score=80 if decision == "include" else 15,
        decision=decision,
        confidence=0.9,
        education_is_core=core,
        economics_mechanism_present=core,
        education_dimension="education" if core else "none",
        economics_dimension="allocation",
        reason="fixture",
        evidence_basis=("title",),
        title_evidence_sufficiency="high",
        abstract_evidence_sufficiency=sufficient,
        information_sufficiency=sufficient,
        title_only_decision=decision,
        title_only_confidence=0.8,
        abstract_aware_decision=decision,
        abstract_aware_confidence=0.9,
        model_name="fixture",
        prompt_version="phase6d",
    )


def _strong(decision="exclude"):
    return StrongSemanticReviewResult(
        paper_id=1,
        independent_decision=decision,
        semantic_score=10,
        confidence=0.95,
        agrees_with_fast=False,
        conflict_resolution="fixture",
        decisive_evidence=("abstract",),
        evidence_basis=("abstract",),
        recommended_final_decision=decision,
        decision_sufficiency="high",
        model_name="fixture",
        prompt_version="phase6d",
    )


def _escalation(*, conflict=()):
    return EscalationDecision(
        escalate=bool(conflict),
        escalation_tier="mandatory" if conflict else "not_needed",
        reasons=conflict,
        rule_semantic_conflict_type="none",
        conflict_types=conflict,
        evidence_snapshot={},
        policy_version="phase6d",
    )


def _combine(paper, fast, strong=None, escalation=None):
    policy = CompositeDecisionPolicy(CompositePolicyConfig(version="phase6d-composite-v1"))
    return policy.combine(paper, fast, escalation or _escalation(), strong)


def test_journal_prior_never_overrides_explicit_negative_semantics():
    result = _combine(_paper(concepts=(), prior="strong_positive"), _fast("exclude", core=False))
    assert result.provisional_decision == "exclude"
    assert "journal_prior_cannot_override_negative_content_semantics" in result.decision_basis


def test_journal_prior_without_anchor_cannot_create_include():
    result = _combine(_paper(concepts=(), prior="strong_positive"), _fast("include", core=False))
    assert result.provisional_decision == "borderline"


def test_weak_ontology_cannot_override_strong_exclude():
    result = _combine(
        _paper(concepts=("human_capital",), tier="weak_or_mixed"),
        _fast("include", core=False),
        _strong("exclude"),
    )
    assert result.provisional_decision != "include"
    assert "strong_negative_not_overridden_by_weak_rule_or_journal_prior" in result.decision_basis


def test_no_abstract_clear_education_title_reaches_human_review():
    result = _combine(
        _paper(concepts=("school_choice",), abstract=None, tier="strong_positive"),
        _fast("exclude", core=False, sufficient="low"),
    )
    assert result.provisional_decision == "borderline"
    assert result.needs_human_missed_review


def test_no_abstract_vague_title_cannot_be_include():
    result = _combine(
        _paper(concepts=(), abstract=None, prior="strong_positive"),
        _fast("include", core=False, sufficient="insufficient"),
    )
    assert result.provisional_decision == "borderline"


def test_fast_strong_conflict_is_visible_borderline():
    result = _combine(
        _paper(concepts=("returns_to_education",), tier="strong_positive"),
        _fast("include"),
        _strong("exclude"),
        _escalation(conflict=("evidence_semantic",)),
    )
    assert result.provisional_decision == "borderline"
    assert {"evidence_semantic", "fast_strong_conflict"} <= set(result.conflict_types)
    assert result.uncertainty_level == "high"


def test_education_exception_protects_theory_method_recall():
    result = _combine(_paper(concepts=("school_choice",), tier="strong_positive"), _fast("include"))
    assert result.provisional_decision == "include"
