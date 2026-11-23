from __future__ import annotations

import random
from collections import defaultdict
from collections.abc import Callable
from dataclasses import dataclass

LABELS = ("include", "borderline", "exclude")
RELEVANT = {"include", "borderline"}


@dataclass(frozen=True)
class BootstrapInterval:
    estimate: float | None
    lower: float | None
    upper: float | None
    iterations: int
    cluster_aware: bool
    warning: str | None


def evaluate_gold_predictions(
    rows: list[dict],
    *,
    prediction_field: str = "prediction",
    score_field: str = "relevance_score",
    weight_field: str = "sampling_weight",
    estimated_minutes_per_review: float = 2.0,
) -> dict:
    usable = [
        row
        for row in rows
        if row.get("final_label") in LABELS and row.get(prediction_field) in LABELS
    ]
    missing_predictions = len(rows) - len(usable)
    matrix = {
        actual: {
            predicted: sum(
                row["final_label"] == actual and row[prediction_field] == predicted
                for row in usable
            )
            for predicted in LABELS
        }
        for actual in LABELS
    }
    per_class = {}
    for label in LABELS:
        tp = matrix[label][label]
        fp = sum(matrix[other][label] for other in LABELS if other != label)
        fn = sum(matrix[label][other] for other in LABELS if other != label)
        precision = _ratio(tp, tp + fp)
        recall = _ratio(tp, tp + fn)
        per_class[label] = {
            "precision": precision,
            "recall": recall,
            "f1": _f1(precision, recall),
            "support": sum(matrix[label].values()),
        }
    relevant = sum(row["final_label"] in RELEVANT for row in usable)
    missed = sum(
        row["final_label"] in RELEVANT and row[prediction_field] == "exclude" for row in usable
    )
    ai_excluded = [row for row in usable if row[prediction_field] == "exclude"]
    candidates = [row for row in usable if row[prediction_field] in RELEVANT]
    strict_positive = sum(row["final_label"] == "include" for row in usable)
    strict_missed = sum(
        row["final_label"] == "include" and row[prediction_field] != "include" for row in usable
    )
    ranked = sorted(
        usable,
        key=lambda row: (-float(row.get(score_field) or 0.0), str(row.get("gold_paper_id"))),
    )
    result = {
        "count": len(usable),
        "missing_prediction_count": missing_predictions,
        "failure_mapped_to_exclude": False,
        "three_class": {
            "confusion_matrix": matrix,
            "accuracy": _ratio(
                sum(row["final_label"] == row[prediction_field] for row in usable),
                len(usable),
            ),
            "macro_precision": _mean(
                [item["precision"] for item in per_class.values() if item["precision"] is not None]
            ),
            "macro_recall": _mean(
                [item["recall"] for item in per_class.values() if item["recall"] is not None]
            ),
            "macro_f1": _mean(
                [item["f1"] for item in per_class.values() if item["f1"] is not None]
            ),
            "per_class": per_class,
        },
        "safety_binary": {
            "false_negative_rate": _ratio(missed, relevant),
            "relevant_recall": _ratio(relevant - missed, relevant),
            "ai_exclude_precision": _ratio(
                sum(row["final_label"] == "exclude" for row in ai_excluded),
                len(ai_excluded),
            ),
            "review_load": _ratio(len(candidates), len(usable)),
        },
        "strict_binary": {
            "include_recall": _ratio(strict_positive - strict_missed, strict_positive),
            "include_false_negative_rate": _ratio(strict_missed, strict_positive),
        },
        "workload": {
            "automatic_exclusion_rate": _ratio(len(ai_excluded), len(usable)),
            "number_needed_to_review": (
                _ratio(len(candidates), sum(row["final_label"] in RELEVANT for row in candidates))
            ),
            "estimated_human_minutes": len(candidates) * estimated_minutes_per_review,
            "high_risk_exclude_recovery_rate": _high_risk_recovery(ai_excluded, score_field),
            "top_k_recall": {
                str(k): _top_k_recall(ranked, k) for k in (10, 25, 50, 100)
            },
        },
        "weighted_sampling": _weighted_safety(usable, prediction_field, weight_field),
        "subgroups": _subgroups(usable, prediction_field),
    }
    return result


def cluster_stratified_bootstrap(
    rows: list[dict],
    statistic: Callable[[list[dict]], float | None],
    *,
    cluster_field: str = "duplicate_cluster_uid",
    strata_fields: tuple[str, ...] = ("final_label",),
    iterations: int = 1000,
    random_seed: int = 20260726,
) -> BootstrapInterval:
    estimate = statistic(rows)
    if iterations <= 0:
        raise ValueError("bootstrap iterations must be positive")
    groups: dict[tuple[str, ...], dict[str, list[dict]]] = defaultdict(
        lambda: defaultdict(list)
    )
    for row in rows:
        stratum = tuple(str(row.get(field) or "unknown") for field in strata_fields)
        cluster = str(row.get(cluster_field) or row.get("gold_paper_id"))
        groups[stratum][cluster].append(row)
    cluster_count = sum(len(clusters) for clusters in groups.values())
    warning = "small_sample_interval_unstable" if cluster_count < 30 else None
    if not rows or not cluster_count:
        return BootstrapInterval(estimate, None, None, iterations, True, warning)
    rng = random.Random(random_seed)
    estimates: list[float] = []
    for _ in range(iterations):
        draw: list[dict] = []
        for clusters in groups.values():
            keys = sorted(clusters)
            for _ in keys:
                draw.extend(clusters[keys[rng.randrange(len(keys))]])
        value = statistic(draw)
        if value is not None:
            estimates.append(float(value))
    if not estimates:
        return BootstrapInterval(estimate, None, None, iterations, True, warning)
    estimates.sort()
    return BootstrapInterval(
        estimate=estimate,
        lower=estimates[_percentile_index(len(estimates), 0.025)],
        upper=estimates[_percentile_index(len(estimates), 0.975)],
        iterations=iterations,
        cluster_aware=True,
        warning=warning,
    )


def _subgroups(rows: list[dict], prediction_field: str) -> dict[str, dict]:
    fields = (
        "journal_family",
        "publication_year",
        "abstract_quality_status",
        "title_term_presence",
        "study_type",
        "education_centrality",
        "source_quality",
        "doi_presence",
    )
    groups: dict[str, list[dict]] = defaultdict(list)
    for row in rows:
        for field in fields:
            groups[f"{field}:{row.get(field, 'unknown')}"].append(row)
        for reason in row.get("reason_codes", []):
            groups[f"boundary_reason_code:{reason}"].append(row)
    report = {}
    for name, members in sorted(groups.items()):
        relevant = sum(row["final_label"] in RELEVANT for row in members)
        missed = sum(
            row["final_label"] in RELEVANT and row[prediction_field] == "exclude"
            for row in members
        )
        report[name] = {
            "count": len(members),
            "relevant_recall": _ratio(relevant - missed, relevant),
            "false_negative_rate": _ratio(missed, relevant),
        }
    return report


def _weighted_safety(rows: list[dict], prediction_field: str, weight_field: str) -> dict:
    relevant_weight = sum(
        float(row.get(weight_field) or 1.0) for row in rows if row["final_label"] in RELEVANT
    )
    missed_weight = sum(
        float(row.get(weight_field) or 1.0)
        for row in rows
        if row["final_label"] in RELEVANT and row[prediction_field] == "exclude"
    )
    return {
        "weighted_relevant_recall": (
            (relevant_weight - missed_weight) / relevant_weight if relevant_weight else None
        ),
        "weighted_false_negative_rate": (
            missed_weight / relevant_weight if relevant_weight else None
        ),
    }


def _top_k_recall(rows: list[dict], k: int) -> float | None:
    relevant = sum(row["final_label"] in RELEVANT for row in rows)
    found = sum(row["final_label"] in RELEVANT for row in rows[:k])
    return _ratio(found, relevant)


def _high_risk_recovery(rows: list[dict], score_field: str) -> float | None:
    if not rows:
        return None
    threshold = sorted(float(row.get(score_field) or 0.0) for row in rows)[
        max(0, round(0.8 * (len(rows) - 1)))
    ]
    high_risk = [row for row in rows if float(row.get(score_field) or 0.0) >= threshold]
    relevant = [row for row in high_risk if row["final_label"] in RELEVANT]
    return _ratio(len(relevant), len(high_risk))


def _f1(precision: float | None, recall: float | None) -> float | None:
    if precision is None or recall is None or precision + recall == 0:
        return None
    return 2 * precision * recall / (precision + recall)


def _mean(values: list[float]) -> float | None:
    return sum(values) / len(values) if values else None


def _ratio(numerator: int | float, denominator: int | float) -> float | None:
    return numerator / denominator if denominator else None


def _percentile_index(size: int, proportion: float) -> int:
    return min(size - 1, max(0, round((size - 1) * proportion)))
