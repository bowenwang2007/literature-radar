from pathlib import Path

from eduecon_radar.domain.offline_screening import (
    load_screening_fixtures,
    stabilized_fixture_decision,
)

FIXTURES = Path(__file__).parent / "fixtures" / "education_economics_screening" / "papers.yaml"


def test_fixed_screening_fixture_has_sixty_complete_unique_cases():
    papers = load_screening_fixtures(FIXTURES)
    assert len(papers) == 64
    assert len({item["id"] for item in papers}) == 64
    tags = {tag for item in papers for tag in item["tags"]}
    assert {
        "pure_theory",
        "education_theory",
        "education_empirical",
        "boundary",
        "no_abstract",
        "conflict",
    } <= tags


def test_pure_theory_never_high_confidence_includes():
    papers = load_screening_fixtures(FIXTURES)
    theory = [item for item in papers if "pure_theory" in item["tags"]]
    assert len(theory) >= 16
    assert all(stabilized_fixture_decision(item) != "include" for item in theory)


def test_education_theory_exceptions_are_not_excluded():
    papers = load_screening_fixtures(FIXTURES)
    theory = [item for item in papers if "education_theory" in item["tags"]]
    assert len(theory) >= 12
    assert all(stabilized_fixture_decision(item) != "exclude" for item in theory)


def test_no_abstract_clear_education_reaches_review_but_vague_title_never_includes():
    papers = load_screening_fixtures(FIXTURES)
    no_abstract = [item for item in papers if "no_abstract" in item["tags"]]
    clear = [item for item in no_abstract if item["evidence_signals"].get("ontology")]
    vague = [item for item in no_abstract if not item["evidence_signals"].get("ontology")]
    assert clear and all(stabilized_fixture_decision(item) != "exclude" for item in clear)
    assert vague and all(stabilized_fixture_decision(item) != "include" for item in vague)


def test_journal_prior_does_not_override_negative_semantics():
    papers = load_screening_fixtures(FIXTURES)
    cases = [item for item in papers if "journal_prior_negative" in item["tags"]]
    assert cases and all(stabilized_fixture_decision(item) == "exclude" for item in cases)
