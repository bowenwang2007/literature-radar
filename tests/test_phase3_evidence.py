from __future__ import annotations

from pathlib import Path

import pytest

from eduecon_radar.domain.evidence import (
    EvidenceExtractor,
    PaperEvidenceInput,
    normalize_evidence_text,
    phrase_occurrences,
)
from eduecon_radar.domain.loader import DomainConfigLoader
from eduecon_radar.domain.missed_risk import MissedRiskPrioritizer
from eduecon_radar.domain.models import (
    EvidenceScoringParameters,
    EvidenceSource,
    ExclusionContext,
    ExclusionDefinition,
    JournalPrior,
    JournalPriorDefinition,
    OntologyConcept,
    OntologyDefinition,
    ScoringDefinition,
    SubjectCode,
    SubjectCodeDefinition,
)
from eduecon_radar.domain.scoring import RuleEvidenceScorer

ROOT = Path(__file__).resolve().parents[1]


def phase3_bundle(*, numeric: bool = False):
    base = DomainConfigLoader(
        ROOT / "config" / "domains" / "education_economics"
    ).load()
    evidence_source = EvidenceSource(source_type="fixture", title="Synthetic test source")
    concepts = (
        OntologyConcept(
            concept_id="education",
            preferred_term="education",
            display_name_zh="教育",
            description="fixture root",
            child_concepts=("school", "student_outcomes", "human_capital"),
            signal_strength="weak",
            enabled=True,
            status="draft",
            version_added="fixture",
        ),
        OntologyConcept(
            concept_id="school",
            preferred_term="school",
            display_name_zh="学校",
            description="fixture family",
            synonyms=("schooling",),
            child_concepts=("school_choice",),
            parent_concept="education",
            signal_strength="weak",
            enabled=True,
            status="draft",
            version_added="fixture",
        ),
        OntologyConcept(
            concept_id="school_choice",
            preferred_term="school choice",
            display_name_zh="学校选择",
            description="fixture specific concept",
            synonyms=("education choice",),
            phrases=("school-choice",),
            parent_concept="school",
            signal_strength="strong",
            enabled=True,
            status="draft",
            version_added="fixture",
        ),
        OntologyConcept(
            concept_id="student_outcomes",
            preferred_term="student achievement",
            display_name_zh="学生结果",
            description="fixture outcome",
            synonyms=("learning outcomes",),
            parent_concept="education",
            signal_strength="moderate",
            enabled=True,
            status="draft",
            version_added="fixture",
        ),
        OntologyConcept(
            concept_id="human_capital",
            preferred_term="human capital",
            display_name_zh="人力资本",
            description="fixture labor concept",
            synonyms=("labor market returns",),
            parent_concept="education",
            signal_strength="strong",
            enabled=True,
            status="draft",
            version_added="fixture",
        ),
    )
    ontology = OntologyDefinition(
        profile_id=base.profile.profile_id,
        version="fixture-ontology-v1",
        status="draft",
        evidence_policy="synthetic fixture only",
        concepts=concepts,
    )
    exclusions = ExclusionDefinition(
        profile_id=base.profile.profile_id,
        version="fixture-exclusion-v1",
        status="draft",
        contexts=(
            ExclusionContext(
                context_id="medical_education",
                preferred_term="medical education",
                phrases=("medical education",),
                description="fixture negative context",
                signal_strength="moderate",
                exceptions=("education economics exception",),
                evidence_sources=(evidence_source,),
                enabled=True,
                status="draft",
                version_added="fixture",
            ),
            ExclusionContext(
                context_id="model_training",
                preferred_term="model training",
                phrases=("training data",),
                description="fixture computing context",
                signal_strength="strong",
                enabled=True,
                status="draft",
                version_added="fixture",
            ),
        ),
    )
    subject_codes = SubjectCodeDefinition(
        profile_id=base.profile.profile_id,
        version="fixture-subject-v1",
        status="draft",
        missing_signal_behavior="neutral",
        codes=(
            SubjectCode(
                code="I21",
                label="fixture education code",
                source="JEL",
                signal_type="positive",
                strength="strong",
                enabled=True,
                status="draft",
                version_added="fixture",
            ),
            SubjectCode(
                code="J24",
                label="fixture adjacent code",
                source="JEL",
                signal_type="adjacent",
                strength="moderate",
                enabled=True,
                status="draft",
                version_added="fixture",
            ),
        ),
    )
    journals = JournalPriorDefinition(
        profile_id=base.profile.profile_id,
        version="fixture-journal-v1",
        status="draft",
        journals=(
            JournalPrior(
                journal_id="fixture_domain",
                canonical_name="Fixture Domain Journal",
                aliases=("FDJ", "Fixture-Domain Journal"),
                group="both",
                monitoring_enabled=True,
                reference_enabled=True,
                prior_tier="strong_positive",
                prior_value=1.0,
                verified=True,
                status="released",
                verification_sources=(evidence_source,),
                version_added="fixture",
            ),
            JournalPrior(
                journal_id="candidate",
                canonical_name="Candidate Education Journal",
                group="reference",
                verified=False,
                status="candidate",
                version_added="fixture",
            ),
        ),
    )
    parameters = fixture_parameters() if numeric else None
    scoring = ScoringDefinition(
        profile_id=base.profile.profile_id,
        scoring_version="fixture-scoring-v1",
        status="draft",
        calibration_status="test_fixture" if numeric else "uncalibrated",
        gold_set_version=None,
        components=base.scoring.components,
        thresholds=base.scoring.thresholds,
        evidence_parameters=parameters,
        publishable=False,
        notes="synthetic fixture",
    )
    return base.model_copy(
        update={
            "ontology": ontology,
            "exclusions": exclusions,
            "subject_codes": subject_codes,
            "journal_priors": journals,
            "scoring": scoring,
            "config_sha256": "A" * 64,
        }
    )


def fixture_parameters() -> EvidenceScoringParameters:
    return EvidenceScoringParameters(
        strength_values={"neutral": 0, "weak": 1, "moderate": 2, "strong": 3},
        source_field_multipliers={"title": 2, "abstract": 1, "keywords": 1},
        concept_per_source_cap=4,
        concept_family_cap=5,
        subject_code_values={"positive": 3, "adjacent": 1, "context": 0, "unknown": 0},
        journal_prior_values={"strong_positive": 3, "moderate_positive": 2, "neutral": 0},
        exclusion_strength_values={"weak": 1, "moderate": 2, "strong": 3},
        component_weights={
            "ontology": 1,
            "subject_code": 1,
            "journal_prior": 1,
            "exclusion": 1,
            "concept_diversity": 1,
        },
        priority_reason_values={
            "positive_negative_evidence_conflict": 5,
            "positive_evidence_without_abstract": 5,
            "strong_journal_prior_with_weak_concept_evidence": 4,
            "high_title_opacity": 4,
            "title_opacity": 2,
            "multiple_concept_families": 2,
            "malformed_metadata": 2,
            "metadata_sparse": 2,
            "complete_or_low_evidence_review_later": 0,
        },
        priority_thresholds={"urgent": 8, "high": 5, "normal": 2},
    )


def paper(**overrides) -> PaperEvidenceInput:
    values = {
        "paper_id": 1,
        "title": "School Choice and Student Achievement",
        "abstract": "School choice may improve learning outcomes and human capital.",
        "journal": "Fixture Economics Journal",
        "keywords": [],
        "jel_codes": [],
        "source_metadata": {},
    }
    values.update(overrides)
    return PaperEvidenceInput(**values)


@pytest.mark.parametrize(
    ("text", "phrase"),
    [
        ("SCHOOL CHOICE reforms", "school choice"),
        ("school-choice reforms", "school choice"),
        ("school   choice reforms", "school-choice"),
        ("Human Capital", "human capital"),
    ],
)
def test_phrase_matching_normalizes_case_hyphens_and_spacing(text, phrase):
    assert phrase_occurrences(text, phrase) == 1


def test_word_boundaries_prevent_arbitrary_substrings():
    assert phrase_occurrences("preschoolchoiceartifact", "school") == 0
    assert phrase_occurrences("schooling", "school") == 0
    assert phrase_occurrences("school", "school") == 1


def test_normalization_is_unicode_and_ascii_case_stable():
    assert normalize_evidence_text("ＳＣＨＯＯＬ–Choice") == "school choice"


def test_synonym_and_specific_phrase_match_with_complete_provenance():
    result = EvidenceExtractor(phase3_bundle()).extract(
        paper(title="Education Choice and Learning Outcomes")
    )
    ids = {item.concept_id for item in result.matched_concepts}
    assert {"school_choice", "student_outcomes"} <= ids
    match = next(item for item in result.matched_concepts if item.concept_id == "school_choice")
    assert match.match_type == "synonym_phrase"
    assert match.provenance.source_field == "title"
    assert match.evidence_span == "education choice"


def test_parent_and_generic_matches_are_suppressed_by_specific_child():
    result = EvidenceExtractor(phase3_bundle()).extract(
        paper(title="Education and School Choice")
    )
    ids = {item.concept_id for item in result.matched_concepts}
    assert "school_choice" in ids
    assert "school" not in ids
    assert "education" not in ids


def test_repetition_records_raw_count_but_does_not_create_repeated_evidence_rows():
    repeated = "school choice " * 10
    result = EvidenceExtractor(phase3_bundle()).extract(
        paper(title=repeated, abstract=repeated)
    )
    matches = [item for item in result.matched_concepts if item.concept_id == "school_choice"]
    assert len(matches) == 2
    assert {item.source_field for item in matches} == {"title", "abstract"}
    assert all(item.raw_match_count == 10 for item in matches)


def test_multiple_concept_families_are_reported_once():
    result = EvidenceExtractor(phase3_bundle()).extract(paper())
    assert set(result.concept_family_coverage) == {
        "school",
        "student_outcomes",
        "human_capital",
    }


def test_malformed_metadata_is_warning_and_not_negative_evidence():
    result = EvidenceExtractor(phase3_bundle()).extract(
        paper(keywords="not-json", jel_codes={"bad": "shape"}, source_metadata="{")
    )
    assert "malformed_keywords" in result.extraction_warnings
    assert "malformed_jel_codes" in result.extraction_warnings
    assert "malformed_source_metadata" in result.extraction_warnings
    assert result.negative_evidence_count == 0


def test_exclusion_context_is_negative_evidence_not_a_decision():
    result = EvidenceExtractor(phase3_bundle()).extract(
        paper(title="Medical Education and School Choice")
    )
    assert result.matched_exclusion_contexts[0].context_id == "medical_education"
    assert not result.matched_exclusion_contexts[0].exception_triggered
    score = RuleEvidenceScorer(phase3_bundle().scoring).score(result)
    assert score.positive_negative_conflict
    assert not hasattr(score, "decision")


def test_exclusion_exception_is_recorded_and_prevents_negative_count():
    result = EvidenceExtractor(phase3_bundle()).extract(
        paper(title="Medical Education", abstract="Education economics exception")
    )
    assert result.matched_exclusion_contexts[0].exception_triggered
    assert result.negative_evidence_count == 0


def test_model_training_phrase_does_not_match_teacher_training():
    result = EvidenceExtractor(phase3_bundle()).extract(
        paper(title="Teacher Training and Student Achievement")
    )
    assert not result.matched_exclusion_contexts


@pytest.mark.parametrize(
    ("codes", "expected"),
    [
        (["I21"], [("I21", "positive", True)]),
        (["J24"], [("J24", "adjacent", True)]),
        (["I21", "J24"], [("I21", "positive", True), ("J24", "adjacent", True)]),
        (["Z99"], [("Z99", "unknown", False)]),
        ([], []),
    ],
)
def test_subject_codes_are_independent_neutral_signals(codes, expected):
    result = EvidenceExtractor(phase3_bundle()).extract(paper(jel_codes=codes))
    actual = [
        (item.code, item.signal_type, item.matched)
        for item in result.subject_code_signals
    ]
    assert actual == expected


def test_candidate_journal_is_non_numeric_and_verified_alias_is_numeric_eligible():
    extractor = EvidenceExtractor(phase3_bundle())
    candidate = extractor.extract(paper(journal="Candidate Education Journal"))
    verified = extractor.extract(paper(journal="fixture-domain journal"))
    general = extractor.extract(paper(journal="General Economics Journal"))
    assert candidate.journal_prior_signal.signal_type == "candidate_prior_signal"
    assert not candidate.journal_prior_signal.numeric_eligible
    assert verified.journal_prior_signal.signal_type == "verified_prior_signal"
    assert verified.journal_prior_signal.numeric_eligible
    assert general.journal_prior_signal.signal_type == "neutral"
    assert not hasattr(general.journal_prior_signal, "issn")


def test_missing_metadata_lowers_confidence_not_relevance_evidence():
    full = EvidenceExtractor(phase3_bundle()).extract(
        paper(keywords=["school choice"], jel_codes=["I21"])
    )
    sparse = EvidenceExtractor(phase3_bundle()).extract(
        paper(abstract=None, keywords=[], jel_codes=[])
    )
    assert (
        sparse.metadata_evidence.overall_metadata_confidence
        < full.metadata_evidence.overall_metadata_confidence
    )
    assert "missing_abstract" in sparse.metadata_evidence.limitations
    assert sparse.positive_evidence_count > 0


def test_missing_title_is_an_explicit_metadata_limitation():
    result = EvidenceExtractor(phase3_bundle()).extract(
        paper(title="", abstract="Student achievement and human capital")
    )
    assert not result.title_available
    assert result.metadata_evidence.title_confidence == 0
    assert "missing_title" in result.metadata_evidence.limitations
    assert result.positive_evidence_count > 0


def test_short_title_is_not_opaque_when_it_contains_specific_domain_evidence():
    result = EvidenceExtractor(phase3_bundle()).extract(paper(title="School Choice"))
    assert result.opaque_title_features.opacity_level == "low"


def test_short_generic_title_without_abstract_is_highly_opaque():
    result = EvidenceExtractor(phase3_bundle()).extract(
        paper(title="The Long Shadow?", abstract=None)
    )
    assert result.opaque_title_features.opaque_title
    assert result.opaque_title_features.opacity_level == "high"
    assert "no_abstract_and_no_title_evidence" in result.opaque_title_features.opacity_reasons


def test_uncalibrated_formal_scoring_has_no_numeric_or_publishable_score():
    bundle = phase3_bundle()
    evidence = EvidenceExtractor(bundle).extract(paper())
    result = RuleEvidenceScorer(bundle.scoring).score(evidence)
    assert result.provisional_rule_score is None
    assert result.ontology_evidence_score is None
    assert not result.publishable
    assert result.rule_evidence_tier != "exclude"


def test_fixture_scoring_is_deterministic_capped_and_config_driven():
    bundle = phase3_bundle(numeric=True)
    evidence = EvidenceExtractor(bundle).extract(
        paper(title="School Choice " * 10, abstract="School Choice " * 10)
    )
    first = RuleEvidenceScorer(bundle.scoring).score(evidence)
    second = RuleEvidenceScorer(bundle.scoring).score(evidence)
    assert first == second
    assert first.ontology_evidence_score == fixture_parameters().concept_family_cap
    assert first.provisional_rule_score is not None
    assert not first.publishable


def test_fixture_parameter_change_changes_score_without_changing_evidence():
    bundle = phase3_bundle(numeric=True)
    evidence = EvidenceExtractor(bundle).extract(paper())
    first = RuleEvidenceScorer(bundle.scoring).score(evidence)
    changed_parameters = bundle.scoring.evidence_parameters.model_copy(
        update={
            "component_weights": {
                **bundle.scoring.evidence_parameters.component_weights,
                "ontology": 2,
            }
        }
    )
    changed_scoring = bundle.scoring.model_copy(
        update={"evidence_parameters": changed_parameters}
    )
    second = RuleEvidenceScorer(changed_scoring).score(evidence)
    assert second.provisional_rule_score > first.provisional_rule_score


def test_priority_is_pre_semantic_explainable_and_not_probability():
    bundle = phase3_bundle(numeric=True)
    evidence = EvidenceExtractor(bundle).extract(
        paper(title="Medical Education and School Choice", abstract=None)
    )
    scoring = RuleEvidenceScorer(bundle.scoring).score(evidence)
    priority = MissedRiskPrioritizer(
        bundle.scoring.scoring_version, bundle.scoring.evidence_parameters
    ).prioritize(evidence, scoring)
    assert priority.priority_tier == "urgent"
    assert priority.phase == "pre_semantic"
    assert "positive_negative_evidence_conflict" in priority.priority_reasons
    assert "positive_evidence_without_abstract" in priority.priority_reasons
    assert not priority.publishable
    assert not hasattr(priority, "false_negative_probability")


def test_complete_little_evidence_is_reviewed_later_not_excluded():
    bundle = phase3_bundle(numeric=True)
    evidence = EvidenceExtractor(bundle).extract(
        paper(
            title="Monetary Policy Transmission through Bank Balance Sheets",
            abstract="Complete macroeconomic metadata",
        )
    )
    scoring = RuleEvidenceScorer(bundle.scoring).score(evidence)
    priority = MissedRiskPrioritizer(
        bundle.scoring.scoring_version, bundle.scoring.evidence_parameters
    ).prioritize(evidence, scoring)
    assert scoring.rule_evidence_tier == "little_evidence"
    assert priority.priority_tier == "low"
    assert not hasattr(scoring, "ai_decision")
