"""Deterministic Phase 6D fixture evaluation; not a production accuracy estimate."""

from __future__ import annotations

from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

import yaml


@dataclass(frozen=True)
class OfflineScreeningMetrics:
    fixture_total: int
    exact_decision_accuracy: float
    allowed_decision_accuracy: float
    positive_recall: float
    negative_specificity: float
    pure_theory_false_positive_rate: float
    education_theory_false_negative_rate: float
    no_abstract_recall: float
    borderline_rate: float
    conflict_rate: float

    def to_dict(self) -> dict[str, int | float]:
        return asdict(self)


def load_screening_fixtures(path: Path) -> list[dict[str, Any]]:
    payload = yaml.safe_load(path.read_text(encoding="utf-8"))
    if not isinstance(payload, list):
        raise ValueError("screening fixture file must contain a list")
    required = {
        "id",
        "title",
        "abstract",
        "journal",
        "has_abstract",
        "evidence_signals",
        "simulated_fast",
        "simulated_strong",
        "journal_prior",
        "expected_decision",
        "allowed_decisions",
        "expected_priority",
        "rationale",
        "tags",
    }
    seen: set[str] = set()
    for fixture in payload:
        if not isinstance(fixture, dict) or not required <= fixture.keys():
            raise ValueError("screening fixture is missing required fields")
        fixture_id = str(fixture["id"])
        if fixture_id in seen:
            raise ValueError(f"duplicate screening fixture id: {fixture_id}")
        seen.add(fixture_id)
        if bool(fixture["abstract"]) != bool(fixture["has_abstract"]):
            raise ValueError(f"has_abstract mismatch: {fixture_id}")
    return payload


def stabilized_fixture_decision(fixture: dict[str, Any]) -> str:
    fast = fixture["simulated_fast"]
    strong = fixture.get("simulated_strong")
    fast_decision = str(fast["decision"])
    strong_decision = str(strong["decision"]) if strong else None
    evidence = fixture["evidence_signals"]
    anchors = tuple(evidence.get("ontology") or ())
    exception = bool(evidence.get("education_exception"))

    if strong_decision and strong_decision != fast_decision:
        return "borderline"
    if strong_decision == "exclude":
        return "exclude"
    if not fixture["has_abstract"]:
        if anchors:
            return "include" if fast_decision == "include" else "borderline"
        return "exclude" if fast_decision == "exclude" else "borderline"
    if evidence.get("exclusions") and exception and anchors:
        return fast_decision if fast_decision != "exclude" else "borderline"
    # Journal prior is intentionally absent from the decision rule.
    return fast_decision


def evaluate_screening_fixtures(fixtures: list[dict[str, Any]]) -> OfflineScreeningMetrics:
    predictions = [stabilized_fixture_decision(item) for item in fixtures]
    total = len(fixtures)
    exact = sum(
        prediction == item["expected_decision"]
        for prediction, item in zip(predictions, fixtures, strict=True)
    )
    allowed = sum(
        prediction in item["allowed_decisions"]
        for prediction, item in zip(predictions, fixtures, strict=True)
    )
    positives = [
        index
        for index, item in enumerate(fixtures)
        if set(item["tags"]) & {"education_theory", "education_empirical"}
    ]
    negatives = [index for index, item in enumerate(fixtures) if "pure_theory" in item["tags"]]
    education_theory = [
        index for index, item in enumerate(fixtures) if "education_theory" in item["tags"]
    ]
    no_abstract_clear = [
        index
        for index, item in enumerate(fixtures)
        if "no_abstract" in item["tags"] and item["evidence_signals"].get("ontology")
    ]
    conflicts = [index for index, item in enumerate(fixtures) if "conflict" in item["tags"]]
    return OfflineScreeningMetrics(
        fixture_total=total,
        exact_decision_accuracy=_ratio(exact, total),
        allowed_decision_accuracy=_ratio(allowed, total),
        positive_recall=_ratio(
            sum(predictions[index] != "exclude" for index in positives), len(positives)
        ),
        negative_specificity=_ratio(
            sum(predictions[index] == "exclude" for index in negatives), len(negatives)
        ),
        pure_theory_false_positive_rate=_ratio(
            sum(predictions[index] == "include" for index in negatives), len(negatives)
        ),
        education_theory_false_negative_rate=_ratio(
            sum(predictions[index] == "exclude" for index in education_theory),
            len(education_theory),
        ),
        no_abstract_recall=_ratio(
            sum(predictions[index] != "exclude" for index in no_abstract_clear),
            len(no_abstract_clear),
        ),
        borderline_rate=_ratio(predictions.count("borderline"), total),
        conflict_rate=_ratio(len(conflicts), total),
    )


def regression_gates(metrics: OfflineScreeningMetrics) -> dict[str, bool]:
    return {
        "fixture_count": metrics.fixture_total >= 60,
        "pure_theory_false_positive": metrics.pure_theory_false_positive_rate == 0,
        "education_theory_recall": metrics.education_theory_false_negative_rate == 0,
        "no_abstract_recall": metrics.no_abstract_recall > 0,
        "allowed_decisions": metrics.allowed_decision_accuracy >= 0.95,
    }


def _ratio(numerator: int, denominator: int) -> float:
    return numerator / denominator if denominator else 0.0
