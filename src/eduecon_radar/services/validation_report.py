from __future__ import annotations

import json
from dataclasses import asdict
from pathlib import Path

from eduecon_radar.domain.validation_metrics import evaluate_screening
from eduecon_radar.services.error_analysis import analyze_screening_errors
from eduecon_radar.services.ontology_suggestion import suggest_ontology_candidates
from eduecon_radar.services.validation_batch import ValidationRepository


def build_validation_report(
    repository: ValidationRepository, batch_id: int
) -> dict[str, object]:
    batch = repository.get_batch(batch_id)
    rows = repository.all_annotated_items(batch_id)
    if not rows:
        raise ValueError("Validation batch has no human annotations")
    metrics = evaluate_screening(rows)
    errors = analyze_screening_errors(rows)
    suggestions = suggest_ontology_candidates(
        errors["false_negative"]["items"], minimum_frequency=2
    )
    return {
        "report_type": "EduEcon Radar Validation Report",
        "batch": asdict(batch),
        "annotation_progress": {
            "annotated": len(rows),
            "total": batch.total_papers,
            "complete": len(rows) == batch.total_papers,
        },
        "metric_definition": {
            "exact_accuracy": "Three-class exact agreement",
            "positive_labels": ["include", "borderline"],
            "negative_labels": ["exclude"],
            "priority_metric": "false_negative_rate",
        },
        "performance": metrics.as_dict(),
        "errors": errors,
        "ontology_candidates": suggestions,
        "recommendations": _recommendations(errors),
    }


def validation_report_markdown(report: dict[str, object]) -> str:
    batch = report["batch"]
    progress = report["annotation_progress"]
    performance = report["performance"]
    errors = report["errors"]
    assert isinstance(batch, dict)
    assert isinstance(progress, dict)
    assert isinstance(performance, dict)
    assert isinstance(errors, dict)
    fp = errors["false_positive"]
    fn = errors["false_negative"]
    assert isinstance(fp, dict) and isinstance(fn, dict)
    lines = [
        "# EduEcon Radar Validation Report",
        "",
        "## Dataset",
        "",
        f"- Validation batch: {batch['batch_id']}",
        f"- Journals: {', '.join(batch['source_journals'])}",
        f"- Period: {batch['date_range'][0] or '—'} to "
        f"{batch['date_range'][1] or '—'}",
        f"- N: {batch['total_papers']}",
        f"- Annotated: {progress['annotated']} / {progress['total']}",
        "",
        "## Performance",
        "",
        f"- Accuracy: {_percent(performance['accuracy'])}",
        f"- Precision: {_percent(performance['precision'])}",
        f"- Recall: {_percent(performance['recall'])}",
        f"- F1: {_percent(performance['f1'])}",
        f"- False Positive Rate: {_percent(performance['false_positive_rate'])}",
        f"- False Negative Rate: {_percent(performance['false_negative_rate'])}",
        "",
        "> Primary safety metric: False Negative Rate. Include and borderline are treated "
        "as retrieved papers; exclude is treated as not retrieved.",
        "",
        "## Errors",
        "",
        f"- False Positive: {fp['count']}",
        f"- False Negative: {fn['count']}",
        "",
        "### Most common failure reasons",
        "",
    ]
    counts = {
        **{
            f"FP — {key}": value
            for key, value in dict(fp["category_counts"]).items()
        },
        **{
            f"FN — {key}": value
            for key, value in dict(fn["category_counts"]).items()
        },
    }
    ordered_counts = sorted(counts.items(), key=lambda item: -item[1])
    lines.extend(
        [f"- {name}: {count}" for name, count in ordered_counts]
        or ["- No classified errors."]
    )
    lines.extend(["", "## Recommendations", ""])
    lines.extend(
        [f"- {item}" for item in report["recommendations"]]
        or ["- No automatic change is recommended from the current annotations."]
    )
    lines.extend(
        [
            "",
            "Ontology candidates are diagnostic suggestions only and require human verification.",
            "",
        ]
    )
    return "\n".join(lines)


def write_validation_report(
    report: dict[str, object],
    *,
    output_dir: Path,
    overwrite: bool = False,
) -> dict[str, str]:
    output_dir.mkdir(parents=True, exist_ok=True)
    batch = report["batch"]
    assert isinstance(batch, dict)
    stem = f"validation_batch_{batch['batch_id']}_report"
    json_path = output_dir / f"{stem}.json"
    markdown_path = output_dir / f"{stem}.md"
    if not overwrite and (json_path.exists() or markdown_path.exists()):
        raise FileExistsError(f"Validation report already exists in {output_dir}")
    json_path.write_text(
        json.dumps(report, ensure_ascii=False, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    markdown_path.write_text(validation_report_markdown(report), encoding="utf-8")
    return {"json": str(json_path.resolve()), "markdown": str(markdown_path.resolve())}


def _recommendations(errors: dict) -> list[str]:
    categories = {
        **errors["false_positive"]["category_counts"],
        **errors["false_negative"]["category_counts"],
    }
    recommendations: list[str] = []
    if categories.get("missing ontology concept"):
        recommendations.append(
            "Review ontology candidate phrases; do not add without verification."
        )
    if categories.get("journal prior bias"):
        recommendations.append(
            "Review journal prior evidence without hard-coding a decision score."
        )
    if categories.get("education mentioned but not core") or categories.get("pure theory"):
        recommendations.append("Review exclusion-context evidence.")
    if categories.get("semantic mistake") or categories.get("semantic overconfidence"):
        recommendations.append("Review semantic prompt examples; do not auto-edit the prompt.")
    if categories.get("missing abstract"):
        recommendations.append("Prioritize abstract enrichment before judging affected papers.")
    return recommendations


def _percent(value: object) -> str:
    return f"{float(value):.2%}"
