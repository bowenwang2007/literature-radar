from pathlib import Path

from eduecon_radar.domain.offline_screening import (
    evaluate_screening_fixtures,
    load_screening_fixtures,
    regression_gates,
)

FIXTURES = Path(__file__).parent / "fixtures" / "education_economics_screening" / "papers.yaml"


def test_offline_evaluation_outputs_all_required_metrics_and_passes_gates():
    report = evaluate_screening_fixtures(load_screening_fixtures(FIXTURES))
    assert report.fixture_total == 64
    assert report.exact_decision_accuracy >= 0.95
    assert report.allowed_decision_accuracy == 1
    assert report.positive_recall == 1
    assert report.negative_specificity == 1
    assert report.pure_theory_false_positive_rate == 0
    assert report.education_theory_false_negative_rate == 0
    assert report.no_abstract_recall == 1
    assert 0 < report.borderline_rate < 1
    assert report.conflict_rate > 0
    assert all(regression_gates(report).values())


def test_candidate_does_not_regress_phase6c_critical_baseline():
    report = evaluate_screening_fixtures(load_screening_fixtures(FIXTURES))
    phase6c_baseline = {
        "allowed_decision_accuracy": 0.90,
        "positive_recall": 0.90,
        "negative_specificity": 0.90,
        "no_abstract_recall": 0.50,
    }
    assert report.allowed_decision_accuracy >= phase6c_baseline["allowed_decision_accuracy"]
    assert report.positive_recall >= phase6c_baseline["positive_recall"]
    assert report.negative_specificity >= phase6c_baseline["negative_specificity"]
    assert report.no_abstract_recall >= phase6c_baseline["no_abstract_recall"]


def test_report_is_explicitly_fixture_only_not_production_accuracy():
    module_doc = __import__("eduecon_radar.domain.offline_screening", fromlist=["__doc__"]).__doc__
    assert "not a production accuracy estimate" in module_doc
