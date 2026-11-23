from eduecon_radar.schemas import PaperMetadata
from eduecon_radar.services.rule_screening import screen_by_rules


def test_positive_terms_enter_ai_screening():
    result = screen_by_rules(
        PaperMetadata(
            title="Teacher quality and student achievement",
            abstract="We study schools and test scores.",
        )
    )
    assert result.should_screen is True
    assert "teacher" in result.positive_hits


def test_student_sample_warning_does_not_auto_include():
    result = screen_by_rules(
        PaperMetadata(
            title="Risk preferences",
            abstract="The sample consisted of university students.",
        )
    )
    assert result.should_screen is True
    assert "student_sample_only" in result.risk_hits


def test_irrelevant_complete_metadata_is_filtered():
    result = screen_by_rules(
        PaperMetadata(title="Exchange rates and trade", abstract="A study of currency markets.")
    )
    assert result.should_screen is False


def test_missing_abstract_is_retained():
    result = screen_by_rules(PaperMetadata(title="Ambiguous new article"))
    assert result.should_screen is True
