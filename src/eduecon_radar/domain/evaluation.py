from __future__ import annotations

from collections import defaultdict
from typing import Any

from pydantic import BaseModel, ConfigDict

from eduecon_radar.domain.gold_set import GoldSet

CANDIDATE_TIERS = {"strong_positive", "moderate_positive"}
POSITIVE_LABELS = {"include", "borderline"}


class DomainEvaluationReport(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True)

    report_status: str = "UNCALIBRATED - SIMULATION ONLY"
    sample_count: int
    valid_label_count: int
    uncertain_count: int
    candidate_recall: float | None
    candidate_precision: float | None
    false_negative_count: int
    false_positive_count: int
    false_negative_rate: float | None
    evidence_coverage: float | None
    subgroup_reports: dict[str, dict[str, float | int | None]]
    top_k_discovery: dict[str, dict[str, Any]]
    warnings: tuple[str, ...]


class DomainEvaluator:
    def evaluate(self, gold_set: GoldSet) -> DomainEvaluationReport:
        labeled = [item for item in gold_set.items if item.gold_label is not None]
        uncertain = [item for item in labeled if item.gold_label == "uncertain"]
        core = [item for item in labeled if item.gold_label != "uncertain"]
        positives = [item for item in core if item.gold_label in POSITIVE_LABELS]
        predicted = [item for item in core if item.rule_evidence_tier in CANDIDATE_TIERS]
        true_positive = sum(item.gold_label in POSITIVE_LABELS for item in predicted)
        false_positive = sum(item.gold_label == "exclude" for item in predicted)
        false_negative = sum(
            item.gold_label in POSITIVE_LABELS and item.rule_evidence_tier not in CANDIDATE_TIERS
            for item in core
        )
        covered = sum(item.rule_evidence_tier is not None for item in core)
        warnings: list[str] = []
        if not core:
            warnings.append("no_core_human_labels")
        return DomainEvaluationReport(
            sample_count=len(gold_set.items),
            valid_label_count=len(core),
            uncertain_count=len(uncertain),
            candidate_recall=_ratio(true_positive, len(positives)),
            candidate_precision=_ratio(true_positive, len(predicted)),
            false_negative_count=false_negative,
            false_positive_count=false_positive,
            false_negative_rate=_ratio(false_negative, len(positives)),
            evidence_coverage=_ratio(covered, len(core)),
            subgroup_reports=self._subgroups(core),
            top_k_discovery=self._top_k(core),
            warnings=tuple(warnings),
        )

    @staticmethod
    def _subgroups(items: list) -> dict[str, dict[str, float | int | None]]:
        groups: dict[str, list] = defaultdict(list)
        for item in items:
            for tag in item.difficulty_tags:
                groups[tag].append(item)
            groups[f"journal:{item.source_group}"].append(item)
        report: dict[str, dict[str, float | int | None]] = {}
        for name, members in sorted(groups.items()):
            positives = sum(item.gold_label in POSITIVE_LABELS for item in members)
            discovered = sum(
                item.gold_label in POSITIVE_LABELS and item.rule_evidence_tier in CANDIDATE_TIERS
                for item in members
            )
            report[name] = {
                "count": len(members),
                "positive_count": positives,
                "candidate_recall": _ratio(discovered, positives),
            }
        return report

    @staticmethod
    def _top_k(items: list) -> dict[str, dict[str, float | int | None]]:
        ranked = sorted(
            items,
            key=lambda item: (
                -(item.priority_score if item.priority_score is not None else -1.0),
                item.item_id,
            ),
        )
        total_positive = sum(item.gold_label in POSITIVE_LABELS for item in items)
        result: dict[str, dict[str, float | int | None]] = {}
        for k in (5, 10, 20, 50):
            selected = ranked[:k]
            relevant = sum(item.gold_label in POSITIVE_LABELS for item in selected)
            tag_relevant: dict[str, int] = defaultdict(int)
            for item in selected:
                if item.gold_label in POSITIVE_LABELS:
                    for tag in item.difficulty_tags:
                        tag_relevant[tag] += 1
            result[str(k)] = {
                "reviewed": len(selected),
                "relevant_found": relevant,
                "positive_coverage": _ratio(relevant, total_positive),
                "papers_per_relevant": _ratio(len(selected), relevant),
                "relevant_by_difficulty_tag": dict(sorted(tag_relevant.items())),
            }
        return result


def _ratio(numerator: int, denominator: int) -> float | None:
    return numerator / denominator if denominator else None
