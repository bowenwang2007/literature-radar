from __future__ import annotations

import hashlib
import json
import math
import random
from collections import defaultdict
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Literal

from eduecon_radar.services.archive import sha256_file
from eduecon_radar.services.doi import canonicalize_doi

Decision = Literal["include", "borderline", "exclude"]
SplitName = Literal["development", "validation", "holdout"]
POSITIVE = {"include", "borderline"}


@dataclass(frozen=True)
class SplitManifest:
    snapshot_sha256: str
    random_seed: int
    split_version: str
    development: tuple[str, ...]
    validation: tuple[str, ...]
    holdout: tuple[str, ...]
    item_to_split: dict[str, SplitName]
    doi_group_count: int
    manifest_sha256: str


@dataclass(frozen=True)
class MetricSet:
    count: int
    accuracy: float | None
    recall: float | None
    precision: float | None
    false_negative_rate: float | None
    false_positive_rate: float | None
    borderline_rate: float | None
    abstention_rate: float | None
    confusion_matrix: dict[str, dict[str, int]]


@dataclass(frozen=True)
class ConfidenceInterval:
    estimate: float | None
    lower: float | None
    upper: float | None
    samples: int


@dataclass(frozen=True)
class PolicyCandidate:
    name: str
    semantic_score_min: float
    confidence_min: float
    exclude_review_fraction: float
    strong_review_enabled: bool


def load_human_gold_snapshot(path: Path, expected_sha256: str | None = None) -> dict:
    path = path.resolve()
    actual_hash = sha256_file(path)
    if expected_sha256 and actual_hash != expected_sha256.upper():
        raise ValueError("Gold Set snapshot SHA-256 does not match --gold-set-sha256")
    payload = json.loads(path.read_text(encoding="utf-8"))
    if payload.get("snapshot_status") != "IMMUTABLE HUMAN GOLD TRUTH":
        raise ValueError("A real immutable human Gold Set snapshot is required")
    items = payload.get("items")
    if not isinstance(items, list) or not items:
        raise ValueError("Gold Set snapshot has no items")
    seen: set[str] = set()
    for item in items:
        item_id = str(item.get("item_id", "")).strip()
        label = item.get("final_label")
        if not item_id or item_id in seen:
            raise ValueError("Gold Set snapshot item IDs must be non-empty and unique")
        if label not in {"include", "borderline", "exclude", "uncertain"}:
            raise ValueError(f"Gold Set item {item_id} has no valid final human label")
        seen.add(item_id)
    payload["snapshot_sha256"] = actual_hash
    return payload


def create_split_manifest(
    snapshot: dict,
    *,
    random_seed: int = 20260719,
    split_version: str = "gold-split-v1",
) -> SplitManifest:
    eligible = [item for item in snapshot["items"] if item["final_label"] != "uncertain"]
    if not eligible:
        raise ValueError("Gold Set has no resolved labels available for calibration")
    groups: dict[str, list[dict]] = defaultdict(list)
    for item in eligible:
        doi = canonicalize_doi(item.get("doi"))
        groups[f"doi:{doi}" if doi else f"item:{item['item_id']}"].append(item)
    strata: dict[tuple[str, str, bool, str], list[list[dict]]] = defaultdict(list)
    for members in groups.values():
        item = members[0]
        tags = tuple(sorted(str(tag) for tag in item.get("difficulty_tags", [])))
        strata[
            (
                str(item["final_label"]),
                str(item.get("journal_group") or "unknown"),
                bool(item.get("original_abstract")),
                "|".join(tags),
            )
        ].append(members)
    allocations: dict[str, SplitName] = {}
    for key, member_groups in sorted(strata.items(), key=lambda pair: repr(pair[0])):
        ordered = sorted(
            member_groups,
            key=lambda members: _stable_rank(random_seed, key, members[0]["item_id"]),
        )
        for index, members in enumerate(ordered):
            split = _stratified_split(index, len(ordered))
            for item in members:
                allocations[str(item["item_id"])] = split
    by_split: dict[SplitName, list[str]] = {
        "development": [],
        "validation": [],
        "holdout": [],
    }
    for item_id, split in sorted(allocations.items()):
        by_split[split].append(item_id)
    core = {
        "snapshot_sha256": snapshot["snapshot_sha256"],
        "random_seed": random_seed,
        "split_version": split_version,
        "development": by_split["development"],
        "validation": by_split["validation"],
        "holdout": by_split["holdout"],
        "item_to_split": allocations,
        "doi_group_count": len(groups),
    }
    return SplitManifest(
        **core,
        manifest_sha256=_canonical_hash(core),
    )


def save_split_manifest(manifest: SplitManifest, output: Path) -> None:
    if output.exists():
        raise FileExistsError(f"Split manifest already exists: {output}")
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(
        json.dumps(asdict(manifest), ensure_ascii=False, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )


def evaluate_prediction_layer(
    items: list[dict],
    prediction_field: str,
    *,
    abstention_field: str | None = None,
) -> MetricSet:
    usable = [
        item
        for item in items
        if item.get("final_label") in {"include", "borderline", "exclude"}
        and item.get(prediction_field) in {"include", "borderline", "exclude"}
    ]
    labels = ("include", "borderline", "exclude")
    matrix = {actual: {predicted: 0 for predicted in labels} for actual in labels}
    for item in usable:
        matrix[item["final_label"]][item[prediction_field]] += 1
    actual_positive = sum(item["final_label"] in POSITIVE for item in usable)
    actual_negative = sum(item["final_label"] == "exclude" for item in usable)
    predicted_positive = sum(item[prediction_field] in POSITIVE for item in usable)
    true_positive = sum(
        item["final_label"] in POSITIVE and item[prediction_field] in POSITIVE for item in usable
    )
    false_positive = sum(
        item["final_label"] == "exclude" and item[prediction_field] in POSITIVE for item in usable
    )
    false_negative = sum(
        item["final_label"] in POSITIVE and item[prediction_field] == "exclude" for item in usable
    )
    return MetricSet(
        count=len(usable),
        accuracy=_ratio(
            sum(item["final_label"] == item[prediction_field] for item in usable), len(usable)
        ),
        recall=_ratio(true_positive, actual_positive),
        precision=_ratio(true_positive, predicted_positive),
        false_negative_rate=_ratio(false_negative, actual_positive),
        false_positive_rate=_ratio(false_positive, actual_negative),
        borderline_rate=_ratio(
            sum(item[prediction_field] == "borderline" for item in usable), len(usable)
        ),
        abstention_rate=_ratio(
            sum(bool(item.get(abstention_field)) for item in usable), len(usable)
        )
        if abstention_field
        else 0.0,
        confusion_matrix=matrix,
    )


def bootstrap_interval(
    items: list[dict],
    metric: str,
    prediction_field: str,
    *,
    samples: int = 1000,
    random_seed: int = 20260719,
) -> ConfidenceInterval:
    if not items:
        return ConfidenceInterval(None, None, None, samples)
    rng = random.Random(random_seed)
    estimates: list[float] = []
    for _ in range(samples):
        draw = [items[rng.randrange(len(items))] for _ in items]
        value = getattr(evaluate_prediction_layer(draw, prediction_field), metric)
        if value is not None:
            estimates.append(float(value))
    estimate = getattr(evaluate_prediction_layer(items, prediction_field), metric)
    if not estimates:
        return ConfidenceInterval(estimate, None, None, samples)
    estimates.sort()
    return ConfidenceInterval(
        estimate,
        estimates[_percentile_index(len(estimates), 0.025)],
        estimates[_percentile_index(len(estimates), 0.975)],
        samples,
    )


def strong_review_incremental_value(items: list[dict]) -> dict[str, int | float | None]:
    triggered = [item for item in items if item.get("strong_triggered")]
    corrected = sum(
        item.get("fast_decision") != item.get("final_label")
        and item.get("strong_decision") == item.get("final_label")
        for item in triggered
    )
    harmed = sum(
        item.get("fast_decision") == item.get("final_label")
        and item.get("strong_decision") != item.get("final_label")
        for item in triggered
    )
    net = corrected - harmed
    return {
        "calls": len(triggered),
        "call_rate": _ratio(len(triggered), len(items)),
        "corrected": corrected,
        "harmed": harmed,
        "net_corrections": net,
        "calls_per_net_correction": _ratio(len(triggered), net) if net > 0 else None,
        "recommend_enable": net > 0,
    }


def search_decision_policies(
    snapshot: dict,
    predictions: list[dict],
    split: SplitManifest,
    *,
    minimum_validation_recall: float = 0.9,
    bootstrap_samples: int = 1000,
    random_seed: int = 20260719,
    evaluate_holdout: bool = True,
) -> dict:
    joined = _join_gold_predictions(snapshot, predictions)
    dev = [item for item in joined if split.item_to_split[item["item_id"]] == "development"]
    validation = [item for item in joined if split.item_to_split[item["item_id"]] == "validation"]
    holdout = [item for item in joined if split.item_to_split[item["item_id"]] == "holdout"]
    candidates = _policy_grid()
    attempted: list[dict] = []
    for policy in candidates:
        dev_rows = _apply_policy(dev, policy)
        validation_rows = _apply_policy(validation, policy)
        attempted.append(
            {
                "policy": asdict(policy),
                "development": _policy_metrics(dev_rows),
                "validation": _policy_metrics(validation_rows),
            }
        )
    viable = [
        row
        for row in attempted
        if (row["validation"]["metrics"]["recall"] or 0.0) >= minimum_validation_recall
    ]
    pool = viable or attempted
    selected = max(
        pool,
        key=lambda row: (
            row["validation"]["metrics"]["recall"] or -1.0,
            row["validation"]["metrics"]["precision"] or -1.0,
            -row["validation"]["review_load_per_100"],
            row["policy"]["name"],
        ),
    )
    chosen = PolicyCandidate(**selected["policy"])
    evaluation_items = holdout if evaluate_holdout else validation
    evaluation_rows = _apply_policy(evaluation_items, chosen)
    holdout_metrics = _policy_metrics(evaluation_rows) if evaluate_holdout else None
    if holdout_metrics is not None:
        holdout_metrics["confidence_intervals"] = {
            name: asdict(
                bootstrap_interval(
                    evaluation_rows,
                    name,
                    "policy_decision",
                    samples=bootstrap_samples,
                    random_seed=random_seed,
                )
            )
            for name in ("recall", "precision", "false_negative_rate")
        }
        holdout_metrics["confidence_intervals"]["top_k_discovery"] = _bootstrap_custom(
            evaluation_rows,
            _top_k_discovery,
            samples=bootstrap_samples,
            random_seed=random_seed,
        )
        holdout_metrics["confidence_intervals"]["strong_net_corrections"] = _bootstrap_custom(
            evaluation_rows,
            lambda rows: float(strong_review_incremental_value(rows)["net_corrections"]),
            samples=bootstrap_samples,
            random_seed=random_seed,
        )
    report = {
        "report_status": "REAL GOLD CALIBRATION - CANDIDATE ONLY",
        "gold_set_snapshot_sha256": snapshot["snapshot_sha256"],
        "semantic_score_interpretation": "ordinal score, not probability",
        "confidence_interpretation": "model certainty, not calibrated probability",
        "split_manifest_sha256": split.manifest_sha256,
        "minimum_validation_recall": minimum_validation_recall,
        "attempted_policies": attempted,
        "selected_policy": asdict(chosen),
        "selected_by": "development/validation only; holdout was not consulted",
        "evaluation_split": "holdout" if evaluate_holdout else "validation",
        "holdout_evaluation_count": int(evaluate_holdout),
        "holdout": holdout_metrics,
        "layer_metrics": {
            layer: asdict(
                evaluate_prediction_layer(evaluation_items, field, abstention_field="abstain")
            )
            for layer, field in (
                ("rule", "rule_decision"),
                ("fast", "fast_decision"),
                ("strong", "strong_decision"),
                ("composite", "composite_decision"),
            )
        },
        "subgroups": _subgroup_metrics(evaluation_rows),
        "strong_review_incremental_value": strong_review_incremental_value(evaluation_items),
        "strategy_candidates": _strategy_candidates(attempted),
        "warnings": _readiness_warnings(snapshot, holdout_metrics)
        if evaluate_holdout
        else ["holdout_not_evaluated_during_policy_search"],
        "random_seed": random_seed,
    }
    report["calibration_report_sha256"] = _canonical_hash(report)
    return report


def save_calibration_report(report: dict, output: Path) -> None:
    if output.exists():
        raise FileExistsError(f"Calibration report already exists: {output}")
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(
        json.dumps(report, ensure_ascii=False, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )


def _join_gold_predictions(snapshot: dict, predictions: list[dict]) -> list[dict]:
    prediction_map = {str(item["item_id"]): item for item in predictions}
    joined: list[dict] = []
    for gold in snapshot["items"]:
        if gold["final_label"] == "uncertain":
            continue
        prediction = prediction_map.get(str(gold["item_id"]))
        if prediction is None:
            raise ValueError(f"Missing pilot prediction for Gold item {gold['item_id']}")
        row = {**gold, **prediction}
        row["item_id"] = str(gold["item_id"])
        joined.append(row)
    return joined


def _policy_grid() -> list[PolicyCandidate]:
    names = {
        (20.0, 0.4, 0.2, True): "high_recall",
        (50.0, 0.6, 0.1, True): "balanced",
        (80.0, 0.8, 0.05, False): "low_cost",
    }
    values: list[PolicyCandidate] = []
    for score in (20.0, 35.0, 50.0, 65.0, 80.0):
        for confidence in (0.4, 0.6, 0.8):
            for fraction in (0.05, 0.1, 0.2):
                for strong in (False, True):
                    key = (score, confidence, fraction, strong)
                    values.append(
                        PolicyCandidate(
                            names.get(key, f"grid_{score:g}_{confidence:g}_{fraction:g}_{strong}"),
                            score,
                            confidence,
                            fraction,
                            strong,
                        )
                    )
    return values


def _apply_policy(items: list[dict], policy: PolicyCandidate) -> list[dict]:
    ordered_excludes = sorted(
        [item for item in items if item.get("composite_decision") == "exclude"],
        key=lambda item: (-float(item.get("missed_priority_score") or 0.0), item["item_id"]),
    )
    review_count = math.ceil(len(items) * policy.exclude_review_fraction)
    review_ids = {item["item_id"] for item in ordered_excludes[:review_count]}
    result: list[dict] = []
    for item in items:
        semantic_score = float(item.get("semantic_score") or 0.0)
        confidence = float(item.get("confidence") or 0.0)
        decision = str(item.get("composite_decision") or "borderline")
        if not policy.strong_review_enabled and item.get("fast_decision"):
            decision = str(item["fast_decision"])
        if semantic_score >= policy.semantic_score_min and confidence >= policy.confidence_min:
            decision = "include" if decision == "include" else "borderline"
        reviewed_exclude = item["item_id"] in review_ids
        if reviewed_exclude and item["final_label"] in POSITIVE:
            decision = "borderline"
        result.append(
            {
                **item,
                "policy_decision": decision,
                "exclude_human_review": reviewed_exclude,
            }
        )
    return result


def _policy_metrics(items: list[dict]) -> dict:
    metrics = asdict(evaluate_prediction_layer(items, "policy_decision"))
    candidate_count = sum(item["policy_decision"] in POSITIVE for item in items)
    exclude_reviews = sum(bool(item["exclude_human_review"]) for item in items)
    return {
        "metrics": metrics,
        "review_load_per_100": 100 * _ratio(candidate_count + exclude_reviews, len(items))
        if items
        else 0.0,
        "candidate_papers_per_100": 100 * _ratio(candidate_count, len(items)) if items else 0.0,
        "exclude_checks_per_100": 100 * _ratio(exclude_reviews, len(items)) if items else 0.0,
        "estimated_translation_candidates_per_100": 100 * _ratio(candidate_count, len(items))
        if items
        else 0.0,
    }


def _subgroup_metrics(items: list[dict]) -> dict[str, dict]:
    expected = (
        "abstract:available",
        "abstract:missing",
        "difficulty:opaque_title",
        "difficulty:false_friend",
        "difficulty:hidden_topic",
        "difficulty:adjacent_domain",
        "difficulty:metadata_sparse",
        "difficulty:cross_domain_intersection",
        "journal_group:domain_journal_candidate",
        "journal_group:general_economics_journal",
        "journal_group:applied_labor_public_journal",
    )
    groups: dict[str, list[dict]] = defaultdict(list, {name: [] for name in expected})
    for item in items:
        abstract_group = (
            "abstract:available" if item.get("original_abstract") else "abstract:missing"
        )
        groups[abstract_group].append(item)
        groups[f"journal_group:{item.get('journal_group', 'unknown')}"].append(item)
        for tag in item.get("difficulty_tags", []):
            groups[f"difficulty:{tag}"].append(item)
    return {
        name: asdict(evaluate_prediction_layer(members, "policy_decision"))
        for name, members in sorted(groups.items())
    }


def _strategy_candidates(attempted: list[dict]) -> dict[str, dict]:
    result: dict[str, dict] = {}
    for strategy in ("high_recall", "balanced", "low_cost"):
        match = next((item for item in attempted if item["policy"]["name"] == strategy), None)
        if match:
            result[strategy] = match
    return result


def _readiness_warnings(snapshot: dict, holdout: dict) -> list[str]:
    warnings: list[str] = []
    if len(snapshot["items"]) < 200:
        warnings.append("gold_set_below_200")
    recall_interval = holdout.get("confidence_intervals", {}).get("recall", {})
    lower = recall_interval.get("lower")
    upper = recall_interval.get("upper")
    if lower is None or upper is None:
        warnings.append("holdout_recall_interval_unavailable")
    elif upper - lower > 0.2:
        warnings.append("holdout_recall_interval_too_wide")
    return warnings


def _stratified_split(index: int, size: int) -> SplitName:
    if size == 1:
        return "development"
    position = index / size
    if position < 0.6:
        return "development"
    if position < 0.8:
        return "validation"
    return "holdout"


def _stable_rank(seed: int, key: object, item_id: object) -> str:
    return hashlib.sha256(f"{seed}|{key!r}|{item_id}".encode()).hexdigest()


def _canonical_hash(value: object) -> str:
    raw = json.dumps(value, ensure_ascii=False, sort_keys=True, separators=(",", ":"))
    return hashlib.sha256(raw.encode("utf-8")).hexdigest().upper()


def _ratio(numerator: int, denominator: int) -> float | None:
    return numerator / denominator if denominator else None


def _percentile_index(size: int, proportion: float) -> int:
    return min(size - 1, max(0, round((size - 1) * proportion)))


def _top_k_discovery(items: list[dict]) -> float | None:
    actual_positive = sum(item["final_label"] in POSITIVE for item in items)
    found = sum(
        item["final_label"] in POSITIVE and item.get("exclude_human_review") for item in items
    )
    return _ratio(found, actual_positive)


def _bootstrap_custom(
    items: list[dict],
    statistic,
    *,
    samples: int,
    random_seed: int,
) -> dict[str, float | int | None]:
    estimate = statistic(items)
    if not items:
        return {"estimate": estimate, "lower": None, "upper": None, "samples": samples}
    rng = random.Random(random_seed)
    values: list[float] = []
    for _ in range(samples):
        draw = [items[rng.randrange(len(items))] for _ in items]
        value = statistic(draw)
        if value is not None:
            values.append(float(value))
    values.sort()
    return {
        "estimate": estimate,
        "lower": values[_percentile_index(len(values), 0.025)] if values else None,
        "upper": values[_percentile_index(len(values), 0.975)] if values else None,
        "samples": samples,
    }
