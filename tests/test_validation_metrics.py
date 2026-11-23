from __future__ import annotations

import pytest

from eduecon_radar.domain.validation_metrics import evaluate_screening


def test_validation_metrics_precision_and_recall():
    metrics = evaluate_screening(
        [
            {"ai_decision": "include", "human_label": "include"},
            {"ai_decision": "borderline", "human_label": "include"},
            {"ai_decision": "include", "human_label": "exclude"},
            {"ai_decision": "exclude", "human_label": "borderline"},
            {"ai_decision": "exclude", "human_label": "exclude"},
        ]
    )

    assert metrics.precision == pytest.approx(2 / 3)
    assert metrics.recall == pytest.approx(2 / 3)
    assert metrics.f1 == pytest.approx(2 / 3)
    assert metrics.accuracy == pytest.approx(2 / 5)


def test_validation_metrics_counts_false_negative_and_false_positive():
    metrics = evaluate_screening(
        [
            {"ai_decision": "exclude", "human_label": "include"},
            {"ai_decision": "include", "human_label": "exclude"},
            {"ai_decision": "exclude", "human_label": "exclude"},
        ]
    )

    assert metrics.false_negative == 1
    assert metrics.false_positive == 1
    assert metrics.false_negative_rate == 1.0
    assert metrics.false_positive_rate == pytest.approx(0.5)
