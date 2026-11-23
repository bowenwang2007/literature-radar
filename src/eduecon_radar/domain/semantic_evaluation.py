from __future__ import annotations

import json
from collections import defaultdict
from pathlib import Path
from typing import Literal

from pydantic import BaseModel, ConfigDict, Field, model_validator

from eduecon_radar.domain.semantic_models import Decision

POSITIVE = {"include", "borderline"}


class _Strict(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True)


class SemanticFixture(_Strict):
    case_id: str
    title: str
    abstract: str | None
    gold_label: Decision
    rule_decision: Decision
    fast_decision: Decision
    strong_decision: Decision | None = None
    composite_decision: Decision
    confidence: float = Field(ge=0, le=1)
    strong_triggered: bool = False
    difficulty_tags: tuple[str, ...] = ()
    semantic_missed_priority: str
    source_group: Literal["domain_journal", "general_journal"] = "general_journal"
    abstain: bool = False

    @model_validator(mode="after")
    def strong_fixture_is_complete(self) -> SemanticFixture:
        if self.strong_triggered and self.strong_decision is None:
            raise ValueError("A triggered strong-review fixture requires a strong_decision")
        return self


class DecisionMetrics(_Strict):
    accuracy: float
    positive_recall: float | None
    positive_precision: float | None
    false_negative_count: int
    false_negative_rate: float | None
    false_positive_count: int
    false_positive_rate: float | None
    borderline_proportion: float
    abstention_proportion: float
    confusion_matrix: dict[str, dict[str, int]]


class StrongIncrementalValue(_Strict):
    calls: int
    trigger_ratio: float
    corrected: int
    harmed: int
    maintained: int
    net_corrections: int
    calls_per_correction: float | None
    trigger_per_100_papers: float
    net_recall_change: float | None
    net_precision_change: float | None


class SemanticEvaluationReport(_Strict):
    report_status: str = "UNCALIBRATED - SIMULATION ONLY"
    sample_count: int
    rule_baseline: DecisionMetrics
    fast_semantic: DecisionMetrics
    strong_review_subset: DecisionMetrics | None
    provisional_composite: DecisionMetrics
    subgroup_reports: dict[str, dict[str, float | int]]
    confidence_buckets: dict[str, dict[str, float | int]]
    strong_review_incremental_value: StrongIncrementalValue
    semantic_missed_priority_counts: dict[str, int]
    top_k_missed_priority: dict[str, dict[str, int | float | None]]
    warnings: tuple[str, ...]


class SemanticEvaluationService:
    def load_fixtures(self, path: Path) -> tuple[SemanticFixture, ...]:
        payload = json.loads(path.read_text(encoding="utf-8"))
        return tuple(SemanticFixture.model_validate(item) for item in payload["fixtures"])

    def evaluate(self, fixtures: tuple[SemanticFixture, ...]) -> SemanticEvaluationReport:
        strong_cases = [item for item in fixtures if item.strong_triggered]
        resolved = tuple(
            item.model_copy(
                update={
                    "fast_decision": item.strong_decision or item.fast_decision,
                    "abstain": False,
                }
            )
            for item in fixtures
        )
        fast_metrics = _metrics(fixtures, "fast_decision")
        resolved_metrics = _metrics(resolved, "fast_decision")
        corrected = sum(
            item.fast_decision != item.gold_label and item.strong_decision == item.gold_label
            for item in strong_cases
        )
        harmed = sum(
            item.fast_decision == item.gold_label and item.strong_decision != item.gold_label
            for item in strong_cases
        )
        maintained = len(strong_cases) - corrected - harmed
        groups: dict[str, list[SemanticFixture]] = defaultdict(list)
        priorities: dict[str, int] = defaultdict(int)
        for item in fixtures:
            priorities[item.semantic_missed_priority] += 1
            for tag in item.difficulty_tags:
                groups[tag].append(item)
            groups[f"source_group:{item.source_group}"].append(item)
        return SemanticEvaluationReport(
            sample_count=len(fixtures),
            rule_baseline=_metrics(fixtures, "rule_decision"),
            fast_semantic=fast_metrics,
            strong_review_subset=(
                _metrics(tuple(strong_cases), "strong_decision") if strong_cases else None
            ),
            provisional_composite=_metrics(fixtures, "composite_decision"),
            subgroup_reports={
                tag: {
                    "count": len(items),
                    "fast_accuracy": _accuracy(items, "fast_decision"),
                    "composite_accuracy": _accuracy(items, "composite_decision"),
                }
                for tag, items in sorted(groups.items())
            },
            confidence_buckets=_confidence_buckets(fixtures),
            strong_review_incremental_value=StrongIncrementalValue(
                calls=len(strong_cases),
                trigger_ratio=_ratio(len(strong_cases), len(fixtures)) or 0.0,
                corrected=corrected,
                harmed=harmed,
                maintained=maintained,
                net_corrections=corrected - harmed,
                calls_per_correction=_ratio(len(strong_cases), corrected),
                trigger_per_100_papers=100 * (_ratio(len(strong_cases), len(fixtures)) or 0.0),
                net_recall_change=_difference(
                    resolved_metrics.positive_recall, fast_metrics.positive_recall
                ),
                net_precision_change=_difference(
                    resolved_metrics.positive_precision, fast_metrics.positive_precision
                ),
            ),
            semantic_missed_priority_counts=dict(sorted(priorities.items())),
            top_k_missed_priority=_top_k_missed_priority(fixtures),
            warnings=(
                "Synthetic fixtures are not production prevalence estimates.",
                "Confidence is self-reported model certainty, not a calibrated probability.",
                "Semantic score is ordinal and must not be interpreted as probability.",
            ),
        )


def _metrics(items: tuple[SemanticFixture, ...], field: str) -> DecisionMetrics:
    matrix = {
        gold: {pred: 0 for pred in ("include", "borderline", "exclude")}
        for gold in ("include", "borderline", "exclude")
    }
    for item in items:
        matrix[item.gold_label][getattr(item, field)] += 1
    predicted_positive = sum(getattr(item, field) in POSITIVE for item in items)
    actual_positive = sum(item.gold_label in POSITIVE for item in items)
    true_positive = sum(
        getattr(item, field) in POSITIVE and item.gold_label in POSITIVE for item in items
    )
    false_negative = sum(
        item.gold_label in POSITIVE and getattr(item, field) == "exclude" for item in items
    )
    actual_negative = sum(item.gold_label == "exclude" for item in items)
    false_positive = sum(
        item.gold_label == "exclude" and getattr(item, field) in POSITIVE for item in items
    )
    return DecisionMetrics(
        accuracy=_accuracy(items, field),
        positive_recall=_ratio(true_positive, actual_positive),
        positive_precision=_ratio(true_positive, predicted_positive),
        false_negative_count=false_negative,
        false_negative_rate=_ratio(false_negative, actual_positive),
        false_positive_count=false_positive,
        false_positive_rate=_ratio(false_positive, actual_negative),
        borderline_proportion=_ratio(
            sum(getattr(item, field) == "borderline" for item in items), len(items)
        )
        or 0.0,
        abstention_proportion=(
            _ratio(sum(item.abstain for item in items), len(items)) or 0.0
            if field == "fast_decision"
            else 0.0
        ),
        confusion_matrix=matrix,
    )


def _accuracy(items: list[SemanticFixture] | tuple[SemanticFixture, ...], field: str) -> float:
    return _ratio(sum(getattr(item, field) == item.gold_label for item in items), len(items)) or 0.0


def _confidence_buckets(items: tuple[SemanticFixture, ...]) -> dict[str, dict[str, float | int]]:
    buckets = {"0.00-0.49": [], "0.50-0.79": [], "0.80-1.00": []}
    for item in items:
        key = (
            "0.00-0.49"
            if item.confidence < 0.5
            else "0.50-0.79"
            if item.confidence < 0.8
            else "0.80-1.00"
        )
        buckets[key].append(item)
    return {
        key: {"count": len(members), "accuracy": _accuracy(members, "fast_decision")}
        for key, members in buckets.items()
    }


def _ratio(numerator: int, denominator: int) -> float | None:
    return numerator / denominator if denominator else None


def _difference(left: float | None, right: float | None) -> float | None:
    return left - right if left is not None and right is not None else None


def _top_k_missed_priority(
    fixtures: tuple[SemanticFixture, ...],
) -> dict[str, dict[str, int | float | None]]:
    order = {"urgent": 0, "high": 1, "normal": 2, "low": 3}
    ranked = sorted(
        fixtures,
        key=lambda item: (order.get(item.semantic_missed_priority, 4), item.case_id),
    )
    total_positive = sum(item.gold_label in POSITIVE for item in fixtures)
    report: dict[str, dict[str, int | float | None]] = {}
    for k in (5, 10, len(fixtures)):
        selected = ranked[:k]
        found = sum(item.gold_label in POSITIVE for item in selected)
        report[str(k)] = {
            "reviewed": len(selected),
            "positive_found": found,
            "positive_coverage": _ratio(found, total_positive),
        }
    return report
