from pathlib import Path

from eduecon_radar.domain.evidence import EvidenceExtractor, PaperEvidenceInput
from eduecon_radar.domain.loader import DomainConfigLoader
from eduecon_radar.domain.offline_screening import load_screening_fixtures

ROOT = Path(__file__).resolve().parents[1]
CONFIG = ROOT / "config" / "domains" / "education_economics"
FIXTURES = ROOT / "tests" / "fixtures" / "education_economics_screening" / "papers.yaml"


def test_only_twelve_high_precision_concepts_are_enabled():
    bundle = DomainConfigLoader(CONFIG).load()
    enabled = [item for item in bundle.ontology.concepts if item.enabled]
    assert len(enabled) == 12
    assert all(item.classification == "high_precision_anchor" for item in enabled)
    forbidden = {"learning", "inequality", "investment", "matching", "human_capital"}
    assert not forbidden.intersection(item.concept_id for item in enabled)


def test_every_enabled_concept_has_positive_negative_and_boundary_fixture_protection():
    bundle = DomainConfigLoader(CONFIG).load()
    fixtures = load_screening_fixtures(FIXTURES)
    for concept in (item for item in bundle.ontology.concepts if item.enabled):
        positive = sum(
            concept.concept_id in item.get("concept_expectations", {}).get("positive", [])
            for item in fixtures
        )
        negative = sum(
            concept.concept_id in item.get("concept_expectations", {}).get("negative", [])
            for item in fixtures
        )
        boundary = sum(
            concept.concept_id in item.get("concept_expectations", {}).get("boundary", [])
            for item in fixtures
        )
        assert positive >= 2, concept.concept_id
        assert negative >= 2, concept.concept_id
        assert boundary >= 1, concept.concept_id
        assert concept.education_exceptions and concept.negative_contexts
        assert concept.matching_mode in {"normalized_phrase", "contextual_phrase"}


def test_high_precision_phrase_matches_but_ambiguous_single_words_do_not():
    extractor = EvidenceExtractor(DomainConfigLoader(CONFIG).load())
    positive = extractor.extract(
        PaperEvidenceInput(
            paper_id=1,
            title="School choice and student assignment",
            abstract=None,
            keywords=[],
            jel_codes=[],
            journal="Test",
            source_metadata={},
        )
    )
    negative = extractor.extract(
        PaperEvidenceInput(
            paper_id=2,
            title="Learning and inequality in matching markets",
            abstract=None,
            keywords=[],
            jel_codes=[],
            journal="Test",
            source_metadata={},
        )
    )
    assert "school_choice" in {item.concept_id for item in positive.matched_concepts}
    assert not negative.matched_concepts


def test_theory_exclusion_is_suppressed_by_education_exception():
    extractor = EvidenceExtractor(DomainConfigLoader(CONFIG).load())
    result = extractor.extract(
        PaperEvidenceInput(
            paper_id=1,
            title="Mechanism design for school choice",
            abstract="A mechanism design model studies student assignment and schooling.",
            keywords=[],
            jel_codes=[],
            journal="Test",
            source_metadata={},
        )
    )
    matches = [
        item
        for item in result.matched_exclusion_contexts
        if item.context_id == "non_education_mechanism_design"
    ]
    assert matches and matches[0].exception_triggered


def test_all_enabled_exclusions_have_complete_education_exceptions():
    bundle = DomainConfigLoader(CONFIG).load()
    required = {
        "school choice",
        "college admissions",
        "student assignment",
        "teacher labor market",
        "education signaling",
        "schooling",
        "education production function",
    }
    enabled = [item for item in bundle.exclusions.contexts if item.enabled]
    assert {item.context_id for item in enabled} == {
        "abstract_economic_theory",
        "econometric_theory",
        "machine_learning_theory",
        "non_education_matching",
        "non_education_mechanism_design",
        "mathematical_economics",
        "theoretical_industrial_organization",
    }
    assert all(required <= set(item.exceptions) for item in enabled)
