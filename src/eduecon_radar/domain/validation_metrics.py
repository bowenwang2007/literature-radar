from __future__ import annotations

from collections.abc import Iterable
from dataclasses import asdict, dataclass

VALID_LABELS = {"include", "borderline", "exclude"}
POSITIVE_LABELS = {"include", "borderline"}


@dataclass(frozen=True)
class ScreeningMetrics:
    total: int
    accuracy: float
    precision: float
    recall: float
    f1: float
    false_positive_rate: float
    false_negative_rate: float
    true_positive: int
    true_negative: int
    false_positive: int
    false_negative: int
    exact_matches: int

    def as_dict(self) -> dict[str, int | float]:
        return asdict(self)


def evaluate_screening(
    rows: Iterable[dict],
    *,
    ai_field: str = "ai_decision",
    human_field: str = "human_label",
) -> ScreeningMetrics:
    values = list(rows)
    if not values:
        raise ValueError("At least one annotated validation item is required")
    for row in values:
        if row.get(ai_field) not in VALID_LABELS:
            raise ValueError(f"Invalid AI decision: {row.get(ai_field)!r}")
        if row.get(human_field) not in VALID_LABELS:
            raise ValueError(f"Invalid human label: {row.get(human_field)!r}")
    tp = tn = fp = fn = exact = 0
    for row in values:
        ai = str(row[ai_field])
        human = str(row[human_field])
        exact += ai == human
        ai_positive = ai in POSITIVE_LABELS
        human_positive = human in POSITIVE_LABELS
        if ai_positive and human_positive:
            tp += 1
        elif ai_positive:
            fp += 1
        elif human_positive:
            fn += 1
        else:
            tn += 1
    precision = _ratio(tp, tp + fp)
    recall = _ratio(tp, tp + fn)
    return ScreeningMetrics(
        total=len(values),
        accuracy=_ratio(exact, len(values)),
        precision=precision,
        recall=recall,
        f1=_ratio(2 * precision * recall, precision + recall),
        false_positive_rate=_ratio(fp, fp + tn),
        false_negative_rate=_ratio(fn, fn + tp),
        true_positive=tp,
        true_negative=tn,
        false_positive=fp,
        false_negative=fn,
        exact_matches=exact,
    )


def _ratio(numerator: float, denominator: float) -> float:
    return numerator / denominator if denominator else 0.0
