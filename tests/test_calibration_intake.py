"""Human calibration intake and diagnosis.

The first calibration round has two ways to produce a number that cannot be
trusted, and both are silent:

* a sheet that was never actually annotated, or was annotated *after* the
  automatic judgement was visible;
* a "systematic problem" claimed from two mistakes that share a journal.

These tests pin the guards, and pin that the diagnosis reports findings without
ever touching a prompt, a threshold, or a profile file.
"""

from __future__ import annotations

import csv
import io
import json

import pytest

from eduecon_radar.services.profile_quality import (
    DOMINANT_SHARE,
    MIN_LABELS_FOR_RATE,
    MIN_PATTERN_COUNT,
    STRATUM_BOUNDARY,
    STRATUM_REPRESENTATIVE,
    AssessmentRow,
    compare_sheet,
    diagnose,
    format_diagnosis,
    format_intake,
    intake_sheet,
)

SHEET_COLUMNS = (
    "profile_id",
    "paper_id",
    "journal",
    "title",
    "human_decision",
    "human_note",
    "auto_decision",
    "auto_basis",
    "auto_confidence",
    "sample_stratum",
    "sample_reason",
)


def _row(
    paper_id: int,
    decision: str = "relevant",
    *,
    profile: str = "behavioral_economics",
    basis: str = "title_and_abstract",
    confidence: str = "high",
) -> AssessmentRow:
    return AssessmentRow(
        profile_id=profile,
        paper_id=paper_id,
        decision=decision,
        screening_basis=basis,
        confidence=confidence,
        prompt_version="v1",
        model_name="fixture",
        title=f"Paper {paper_id}",
        abstract="An abstract.",
        journal="JEBO",
        publication_date="2026-03-01",
        classification={"confidence": confidence},
    )


def _sheet(rows: list[dict[str, str]]) -> str:
    buffer = io.StringIO()
    writer = csv.DictWriter(buffer, fieldnames=list(SHEET_COLUMNS), lineterminator="\n")
    writer.writeheader()
    for row in rows:
        writer.writerow({column: row.get(column, "") for column in SHEET_COLUMNS})
    return buffer.getvalue()


# --------------------------------------------------------------------------- #
# Intake
# --------------------------------------------------------------------------- #


def test_a_fully_unlabelled_sheet_is_blocked_not_compared():
    text = _sheet([{"profile_id": "behavioral_economics", "paper_id": "1"}])
    intake = intake_sheet(text)
    assert intake.labelled == 0
    assert intake.unlabelled == 1
    assert intake.usable is False
    assert any("has not been annotated" in item for item in intake.blocking)
    assert "BLOCKING" in format_intake(intake)


def test_intake_reports_completion_rate_and_per_profile_counts():
    text = _sheet(
        [
            {"profile_id": "behavioral_economics", "paper_id": "1",
             "human_decision": "relevant"},
            {"profile_id": "behavioral_economics", "paper_id": "2",
             "human_decision": "irrelevant"},
            {"profile_id": "labor_economics", "paper_id": "3",
             "human_decision": "relevant"},
            {"profile_id": "labor_economics", "paper_id": "4"},
        ]
    )
    intake = intake_sheet(text)
    assert intake.rows == 4
    assert intake.labelled == 3
    assert intake.unlabelled == 1
    assert intake.as_dict()["completion_rate"] == pytest.approx(0.75)
    assert intake.per_profile == {"behavioral_economics": 2, "labor_economics": 1}


def test_a_sheet_with_a_visible_auto_decision_is_not_treated_as_blind():
    """If the annotator could see the answer, agreement measures the wrong thing."""
    text = _sheet(
        [
            {"profile_id": "behavioral_economics", "paper_id": "1",
             "human_decision": "relevant", "auto_decision": "relevant"},
        ]
    )
    intake = intake_sheet(text)
    assert intake.usable is False
    assert any("not blind" in item for item in intake.blocking)
    # The escape hatch exists for deliberately unblinded audits, and says so.
    allowed = intake_sheet(text, require_blind=False)
    assert allowed.usable is True


def test_skip_is_counted_separately_from_a_decision():
    text = _sheet(
        [
            {"profile_id": "behavioral_economics", "paper_id": "1",
             "human_decision": "relevant"},
            {"profile_id": "behavioral_economics", "paper_id": "2",
             "human_decision": "skip", "human_note": "no abstract available"},
            {"profile_id": "behavioral_economics", "paper_id": "3",
             "human_decision": "skip"},
        ]
    )
    intake = intake_sheet(text)
    assert intake.skipped == 2
    assert intake.labelled == 3  # skips are labels, just not judgements
    assert intake.per_decision == {"relevant": 1, "skip": 2}
    assert intake.notes_provided == 1


def test_notes_are_preserved_through_intake():
    text = _sheet(
        [
            {"profile_id": "behavioral_economics", "paper_id": "1",
             "human_decision": "relevant", "human_note": "belief updating is the object"},
        ]
    )
    intake = intake_sheet(text)
    assert intake.notes_provided == 1


def test_unknown_decisions_are_reported_but_do_not_block_the_whole_sheet():
    text = _sheet(
        [
            {"profile_id": "behavioral_economics", "paper_id": "1",
             "human_decision": "relevant"},
            {"profile_id": "behavioral_economics", "paper_id": "2",
             "human_decision": "probably"},
        ]
    )
    intake = intake_sheet(text)
    assert intake.labelled == 1
    assert any("not one of" in problem for problem in intake.problems)
    assert intake.usable is True


def test_a_sheet_missing_required_columns_is_blocked():
    buffer = io.StringIO()
    writer = csv.DictWriter(buffer, fieldnames=["paper_id", "human_decision"],
                            lineterminator="\n")
    writer.writeheader()
    writer.writerow({"paper_id": "1", "human_decision": "relevant"})
    intake = intake_sheet(buffer.getvalue())
    assert intake.usable is False
    assert any("missing required columns" in item for item in intake.blocking)


def test_an_empty_sheet_is_blocked():
    intake = intake_sheet("profile_id,paper_id,human_decision\n")
    assert intake.rows == 0
    assert intake.usable is False


def test_stratum_counts_survive_intake():
    text = _sheet(
        [
            {"profile_id": "behavioral_economics", "paper_id": "1",
             "human_decision": "relevant", "sample_stratum": STRATUM_BOUNDARY},
            {"profile_id": "behavioral_economics", "paper_id": "2",
             "human_decision": "relevant", "sample_stratum": STRATUM_REPRESENTATIVE},
        ]
    )
    intake = intake_sheet(text)
    assert intake.per_stratum == {STRATUM_BOUNDARY: 1, STRATUM_REPRESENTATIVE: 1}


# --------------------------------------------------------------------------- #
# Diagnosis
# --------------------------------------------------------------------------- #


def _errors(specs: list[tuple[int, str, dict]]) -> list[dict]:
    out = []
    for paper_id, kind, extra in specs:
        out.append({"paper_id": paper_id, "error_kind": kind, **extra})
    return out


def _empty_pairs(count: int = 0) -> list:
    return []


def test_diagnosis_refuses_to_characterise_an_unlabelled_sheet():
    result = diagnose(_empty_pairs(), profile_id="x", errors=[], total_errors=0, labelled=0)
    assert result.direction == "insufficient"
    assert result.conclusions_permitted is False
    assert "No labels yet" in result.statement


def test_diagnosis_refuses_patterns_from_a_small_sample():
    errors = _errors(
        [
            (1, "false_negative", {"auto_basis": "title_only", "journal": "JEBO",
                                   "auto_confidence": "low", "human_note": "a"}),
            (2, "false_negative", {"auto_basis": "title_only", "journal": "JEBO",
                                   "auto_confidence": "low", "human_note": "b"}),
        ]
    )
    result = diagnose(_empty_pairs(), profile_id="x", errors=errors, total_errors=2,
                      labelled=5)
    assert result.patterns == []
    assert result.sample_adequate is False
    assert any("no cluster can reach" in caveat for caveat in result.caveats)
    assert "not enough to characterise the boundary" in result.statement


def test_diagnosis_names_a_systematic_false_negative_direction():
    errors = _errors(
        [(index, "false_negative", {}) for index in range(1, 9)]
        + [(99, "false_positive", {})]
    )
    result = diagnose(_empty_pairs(), profile_id="x", errors=errors, total_errors=9,
                      labelled=40)
    assert result.false_negatives == 8
    assert result.direction == "systematic_false_negative"
    assert result.conclusions_permitted is True
    assert "missing papers the human wants" in result.statement


def test_diagnosis_names_a_systematic_false_positive_direction():
    errors = _errors(
        [(index, "false_positive", {}) for index in range(1, 9)]
        + [(99, "false_negative", {})]
    )
    result = diagnose(_empty_pairs(), profile_id="x", errors=errors, total_errors=9,
                      labelled=40)
    assert result.direction == "systematic_false_positive"
    assert "costs attention" in result.statement


def test_diagnosis_calls_a_balanced_error_mix_mixed():
    errors = _errors(
        [(index, "false_negative", {}) for index in range(1, 6)]
        + [(index, "false_positive", {}) for index in range(100, 105)]
    )
    result = diagnose(_empty_pairs(), profile_id="x", errors=errors, total_errors=10,
                      labelled=50)
    assert result.direction == "mixed"
    assert result.false_negatives == 5 and result.false_positives == 5


def test_diagnosis_reports_a_cluster_only_at_the_threshold():
    below = _errors(
        [(index, "false_negative", {"auto_basis": "title_only"})
         for index in range(1, MIN_PATTERN_COUNT)]
    )
    at = _errors(
        [(index, "false_negative", {"auto_basis": "title_only"})
         for index in range(1, MIN_PATTERN_COUNT + 1)]
    )
    assert diagnose(_empty_pairs(), profile_id="x", errors=below,
                    total_errors=len(below), labelled=40).patterns == []
    patterns = diagnose(_empty_pairs(), profile_id="x", errors=at,
                        total_errors=len(at), labelled=40).patterns
    by_attribute = {pattern.attribute: pattern for pattern in patterns}
    assert by_attribute["auto_basis"].value == "title_only"
    assert by_attribute["auto_basis"].count == MIN_PATTERN_COUNT
    assert by_attribute["auto_basis"].share == pytest.approx(1.0)
    # Attributes the fixture never populated are reported as unrecorded rather
    # than silently dropped, so "no cluster here" cannot be confused with
    # "the field was missing".
    assert by_attribute["journal"].value == "(unrecorded)"


def test_cluster_notes_come_from_the_annotator_not_the_model():
    errors = _errors(
        [
            (index, "false_negative",
             {"auto_basis": "title_only", "human_note": f"reason {index}"})
            for index in range(1, MIN_PATTERN_COUNT + 1)
        ]
    )
    patterns = diagnose(_empty_pairs(), profile_id="x", errors=errors,
                        total_errors=len(errors), labelled=40).patterns
    basis_pattern = next(item for item in patterns if item.attribute == "auto_basis")
    assert basis_pattern.notes == tuple(
        f"reason {index}" for index in range(1, MIN_PATTERN_COUNT + 1)
    )


def test_diagnosis_counts_human_abstentions_separately():
    pairs = [
        ({"paper_id": "1", "human_decision": "skip"}, _row(1)),
        ({"paper_id": "2", "human_decision": "relevant"}, _row(2)),
    ]
    result = diagnose(pairs, profile_id="x", errors=[], total_errors=0, labelled=2)
    assert result.abstentions_from_human == 1


def test_diagnosis_never_claims_a_cause():
    errors = _errors([(index, "false_negative", {}) for index in range(1, 6)])
    result = diagnose(_empty_pairs(), profile_id="x", errors=errors, total_errors=5,
                      labelled=40)
    assert "not a diagnosis of cause" in result.statement
    rendered = format_diagnosis(result)
    assert "No prompt, threshold or profile file has been changed" in rendered


def test_diagnosis_json_is_serializable():
    errors = [
        {"paper_id": index, "error_kind": "false_negative", "journal": "JEBO",
         "auto_basis": "title_and_abstract"}
        for index in range(1, 5)
    ]
    payload = json.loads(json.dumps(
        diagnose(_empty_pairs(), profile_id="x", errors=errors, total_errors=4,
                 labelled=40).as_dict(), ensure_ascii=False
    ))
    assert payload["direction"] == "systematic_false_negative"
    journals = [item for item in payload["patterns"] if item["attribute"] == "journal"]
    assert journals and journals[0]["value"] == "JEBO"


def test_dominant_share_is_a_threshold_not_an_assumption():
    assert 0.0 < DOMINANT_SHARE <= 1.0
    # Exactly at the boundary counts as dominant; just under does not.
    total = 100
    at = int(total * DOMINANT_SHARE)
    errors = _errors(
        [(index, "false_negative", {}) for index in range(at)]
        + [(index, "false_positive", {}) for index in range(total - at)]
    )
    result = diagnose(_empty_pairs(), profile_id="x", errors=errors,
                      total_errors=total, labelled=total)
    assert result.direction == "systematic_false_negative"


# --------------------------------------------------------------------------- #
# End to end through compare_sheet
# --------------------------------------------------------------------------- #


def test_compare_produces_a_diagnosis_per_profile():
    assessments = [
        _row(1, "irrelevant", profile="behavioral_economics"),
        _row(2, "irrelevant", profile="labor_economics"),
    ]
    sheet = _sheet(
        [
            {"profile_id": "behavioral_economics", "paper_id": "1",
             "human_decision": "relevant", "human_note": "mechanism is central",
             "sample_stratum": STRATUM_REPRESENTATIVE},
            {"profile_id": "labor_economics", "paper_id": "2",
             "human_decision": "irrelevant", "sample_stratum": STRATUM_REPRESENTATIVE},
        ]
    )
    report = compare_sheet(sheet, assessments)
    assert set(report.diagnoses) == {"behavioral_economics", "labor_economics"}
    assert report.diagnoses["behavioral_economics"]["false_negatives"] == 1
    assert report.diagnoses["labor_economics"]["false_negatives"] == 0
    rendered = format_comparison_text(report)
    assert "DIAGNOSIS" in rendered
    assert "nothing has been changed" in rendered


def test_error_rows_carry_their_kind_for_clustering():
    assessments = [_row(1, "relevant")]
    sheet = _sheet(
        [
            {"profile_id": "behavioral_economics", "paper_id": "1",
             "human_decision": "irrelevant", "human_note": "only mentions behaviour",
             "sample_stratum": STRATUM_REPRESENTATIVE},
        ]
    )
    report = compare_sheet(sheet, assessments)
    assert report.auto_positive_human_negative[0]["error_kind"] == "false_positive"


def format_comparison_text(report) -> str:
    from eduecon_radar.services.profile_quality import format_comparison

    return format_comparison(report)


def test_min_labels_threshold_is_shared_with_the_metrics_layer():
    """One threshold, so the report cannot say 'rate ok' and 'sample too small'."""
    assert MIN_LABELS_FOR_RATE >= 20
    errors = _errors([(1, "false_negative", {})])
    result = diagnose(_empty_pairs(), profile_id="x", errors=errors, total_errors=1,
                      labelled=MIN_LABELS_FOR_RATE - 1)
    assert result.sample_adequate is False
    result = diagnose(_empty_pairs(), profile_id="x", errors=errors, total_errors=1,
                      labelled=MIN_LABELS_FOR_RATE)
    assert result.sample_adequate is True
