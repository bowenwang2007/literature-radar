from __future__ import annotations

from eduecon_radar.services.error_analysis import analyze_screening_errors
from eduecon_radar.services.ontology_suggestion import suggest_ontology_candidates
from eduecon_radar.services.validation_report import validation_report_markdown


def test_error_analysis_classifies_false_positive_and_false_negative():
    rows = [
        {
            "id": 1,
            "source_paper_id": 10,
            "original_title": "An adjacent application",
            "original_abstract": "Available",
            "ai_decision": "include",
            "human_label": "exclude",
            "confidence": 0.95,
            "semantic_score": 92,
            "semantic": {
                "adjacent_domain": "health",
                "education_is_core": False,
            },
            "evidence": {},
        },
        {
            "id": 2,
            "source_paper_id": 11,
            "original_title": "Teacher labor market dynamics",
            "original_abstract": None,
            "ai_decision": "exclude",
            "human_label": "include",
            "semantic": {"title_evidence_sufficiency": "low"},
            "evidence": {
                "evidence": {
                    "matched_concepts": [],
                    "opaque_title_features": {"opacity_level": "high"},
                },
                "scoring": {"rule_evidence_tier": "little_evidence"},
            },
        },
    ]

    report = analyze_screening_errors(rows)

    assert report["false_positive"]["count"] == 1
    assert "adjacent economics" in report["false_positive"]["items"][0]["categories"]
    assert "semantic overconfidence" in report["false_positive"]["items"][0]["categories"]
    assert report["false_negative"]["count"] == 1
    categories = report["false_negative"]["items"][0]["categories"]
    assert {"missing abstract", "title ambiguity", "missing ontology concept"} <= set(
        categories
    )


def test_ontology_suggestions_are_candidates_only():
    errors = [
        {"title": "Teacher Labor Market Dynamics"},
        {"title": "Teacher Labor Market Regulation"},
        {"title": "Unrelated Unique Phrase"},
    ]

    suggestions = suggest_ontology_candidates(errors, minimum_frequency=2)

    candidate = next(item for item in suggestions if item["phrase"] == "teacher labor market")
    assert candidate["frequency"] == 2
    assert candidate["status"] == "candidate_requires_human_verification"


def test_validation_markdown_reports_false_negative_as_priority_metric():
    report = {
        "batch": {
            "batch_id": 3,
            "source_journals": ("JHR",),
            "date_range": ("2025-01-01", "2025-12-31"),
            "total_papers": 100,
        },
        "annotation_progress": {"annotated": 100, "total": 100},
        "performance": {
            "accuracy": 0.8,
            "precision": 0.81,
            "recall": 0.9,
            "f1": 0.85,
            "false_positive_rate": 0.2,
            "false_negative_rate": 0.1,
        },
        "errors": {
            "false_positive": {"count": 2, "category_counts": {}},
            "false_negative": {
                "count": 1,
                "category_counts": {"missing abstract": 1},
            },
        },
        "recommendations": ["Prioritize abstract enrichment."],
    }

    markdown = validation_report_markdown(report)

    assert "False Negative Rate: 10.00%" in markdown
    assert "Primary safety metric: False Negative Rate" in markdown
    assert "FN — missing abstract: 1" in markdown
