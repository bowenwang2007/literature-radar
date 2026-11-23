from __future__ import annotations

from collections import Counter

from eduecon_radar.domain.validation_metrics import POSITIVE_LABELS


def analyze_screening_errors(rows: list[dict]) -> dict:
    false_positives: list[dict] = []
    false_negatives: list[dict] = []
    for row in rows:
        ai_positive = row.get("ai_decision") in POSITIVE_LABELS
        human_positive = row.get("human_label") in POSITIVE_LABELS
        if ai_positive and not human_positive:
            false_positives.append(_classify_false_positive(row))
        elif not ai_positive and human_positive:
            false_negatives.append(_classify_false_negative(row))
    fp_counts = Counter(
        category for item in false_positives for category in item["categories"]
    )
    fn_counts = Counter(
        category for item in false_negatives for category in item["categories"]
    )
    return {
        "false_positive": {
            "count": len(false_positives),
            "category_counts": dict(sorted(fp_counts.items())),
            "items": false_positives,
        },
        "false_negative": {
            "count": len(false_negatives),
            "category_counts": dict(sorted(fn_counts.items())),
            "items": false_negatives,
        },
    }


def _classify_false_positive(row: dict) -> dict:
    semantic = row.get("semantic") or {}
    evidence = _evidence(row)
    categories: list[str] = []
    adjacent = semantic.get("adjacent_domain")
    if adjacent:
        categories.append("adjacent economics")
    if semantic.get("education_is_core") is False:
        categories.append("education mentioned but not core")
    exclusions = _exclusion_ids(evidence)
    if any("theor" in item.lower() for item in exclusions):
        categories.append("pure theory")
    rule_tier = _rule_tier(evidence)
    journal_prior = _journal_prior(evidence)
    if journal_prior in {"verified_domain", "strong_positive"} and rule_tier in {
        "weak_or_mixed",
        "little_evidence",
    }:
        categories.append("journal prior bias")
    if float(row.get("confidence") or 0) >= 0.85 or float(
        row.get("semantic_score") or 0
    ) >= 85:
        categories.append("semantic overconfidence")
    if not categories:
        categories.append("unclassified false positive")
    return _error_item(row, categories)


def _classify_false_negative(row: dict) -> dict:
    semantic = row.get("semantic") or {}
    evidence = _evidence(row)
    categories: list[str] = []
    if not row.get("original_abstract"):
        categories.append("missing abstract")
    opacity = (
        evidence.get("evidence", {})
        .get("opaque_title_features", {})
        .get("opacity_level")
    )
    if opacity == "high":
        categories.append("title ambiguity")
    matched = evidence.get("evidence", {}).get("matched_concepts") or []
    if not matched and _rule_tier(evidence) == "little_evidence":
        categories.append("missing ontology concept")
    if semantic.get("title_evidence_sufficiency") in {"low", "insufficient"}:
        categories.append("weak education wording")
    if not categories:
        categories.append("semantic mistake")
    return _error_item(row, categories)


def _error_item(row: dict, categories: list[str]) -> dict:
    return {
        "validation_item_id": row.get("id"),
        "source_paper_id": row.get("source_paper_id"),
        "title": row.get("original_title"),
        "ai_decision": row.get("ai_decision"),
        "human_label": row.get("human_label"),
        "categories": list(dict.fromkeys(categories)),
    }


def _evidence(row: dict) -> dict:
    value = row.get("evidence")
    return value if isinstance(value, dict) else {}


def _rule_tier(evidence: dict) -> str | None:
    return evidence.get("scoring", {}).get("rule_evidence_tier")


def _journal_prior(evidence: dict) -> str | None:
    return (
        evidence.get("evidence", {})
        .get("journal_prior_signal", {})
        .get("prior_tier")
    )


def _exclusion_ids(evidence: dict) -> list[str]:
    values = evidence.get("evidence", {}).get("matched_exclusion_contexts") or []
    return [
        str(item.get("context_id", ""))
        for item in values
        if isinstance(item, dict)
    ]
