"""Profile quality evaluation: sampling, blind annotation, and comparison.

The workflow promises four things, and each is tested here:

* it can pull a sample worth annotating and hand out a sheet a human can fill in;
* blinded annotation is genuinely blind (no automatic judgement in the sheet);
* a returned sheet is joined against stored judgements and yields false
  positives, false negatives and boundary cases;
* a title-only judgement is never silently averaged together with a full one.
"""

from __future__ import annotations

import csv
import io
from pathlib import Path

import pytest

from eduecon_radar.services.profile_quality import (
    HUMAN_DECISIONS,
    AssessmentRow,
    compare_sheet,
    format_comparison,
    format_screening_basis,
    load_human_labels,
    measure_screening_basis,
    render_sheet,
    sample_assessments,
    sample_reason,
    write_sheet,
)


def _row(
    paper_id: int,
    decision: str,
    *,
    basis: str = "title_and_abstract",
    confidence: str | None = "high",
    profile: str = "behavioral_economics",
    abstract: str | None = "An abstract.",
    title: str | None = None,
) -> AssessmentRow:
    return AssessmentRow(
        profile_id=profile,
        paper_id=paper_id,
        decision=decision,
        screening_basis=basis,
        confidence=confidence,
        prompt_version="2026-09-24-behavioral-v2",
        model_name="deepseek-chat",
        title=title or f"Paper {paper_id} on decision making",
        abstract=abstract,
        journal="JEBO",
        publication_date="2026-03-01",
        classification={"confidence": confidence},
    )


def _corpus() -> list[AssessmentRow]:
    rows: list[AssessmentRow] = []
    for index in range(1, 13):
        rows.append(_row(index, "irrelevant"))
    for index in range(20, 28):
        rows.append(_row(index, "relevant"))
    rows.append(_row(40, "uncertain", basis="title_only", confidence="low"))
    rows.append(_row(41, "uncertain", basis="title_only", confidence="low"))
    rows.append(_row(50, "irrelevant", basis="deterministic_metadata_rule"))
    return rows


def _sheet(rows: list[tuple[int, str, str]], *, profile: str = "behavioral_economics") -> str:
    buffer = io.StringIO()
    writer = csv.DictWriter(
        buffer,
        fieldnames=["profile_id", "paper_id", "title", "human_decision", "human_note"],
        lineterminator="\n",
    )
    writer.writeheader()
    for paper_id, decision, note in rows:
        writer.writerow(
            {
                "profile_id": profile,
                "paper_id": paper_id,
                "title": f"Paper {paper_id}",
                "human_decision": decision,
                "human_note": note,
            }
        )
    return buffer.getvalue()


# --------------------------------------------------------------------------- #
# Sampling
# --------------------------------------------------------------------------- #


def test_sampling_draws_from_every_automatic_decision_including_uncertain():
    sample = sample_assessments(_corpus(), per_decision=2, seed=1)
    decisions = {item.decision for item in sample}
    assert decisions == {"irrelevant", "relevant", "uncertain"}
    assert len(sample) == 6


def test_sampling_is_deterministic_for_a_seed_and_differs_across_seeds():
    first = [item.paper_id for item in sample_assessments(_corpus(), per_decision=3, seed=7)]
    again = [item.paper_id for item in sample_assessments(_corpus(), per_decision=3, seed=7)]
    other = [item.paper_id for item in sample_assessments(_corpus(), per_decision=3, seed=8)]
    assert first == again
    assert first != other


def test_sampling_can_target_a_screening_basis():
    sample = sample_assessments(_corpus(), per_decision=0, per_basis=1, seed=3)
    counts: dict[str, int] = {}
    for item in sample:
        counts[item.basis] = counts.get(item.basis, 0) + 1
    assert counts == {
        "title_and_abstract": 1,
        "title_only": 1,
        "deterministic_metadata_rule": 1,
    }
    # The rows actually drawn belong to the basis they were drawn for.
    by_basis = {item.basis: item for item in sample}
    assert by_basis["title_only"].decision == "uncertain"
    assert by_basis["deterministic_metadata_rule"].decision == "irrelevant"


def test_sampling_respects_an_explicit_include_and_only_lists():
    included = sample_assessments(_corpus(), per_decision=0, include_ids=[41], seed=1)
    assert [item.paper_id for item in included] == [41]
    restricted = sample_assessments(
        _corpus(), per_decision=5, only_ids=[1, 2, 20, 21], seed=1
    )
    assert {item.paper_id for item in restricted} <= {1, 2, 20, 21}


def test_sampling_never_invents_rows_for_absent_decisions():
    only_relevant = [item for item in _corpus() if item.decision == "relevant"]
    sample = sample_assessments(only_relevant, per_decision=100, seed=1)
    assert {item.paper_id for item in sample} == {item.paper_id for item in only_relevant}


@pytest.mark.parametrize(
    ("kwargs", "expected"),
    [
        ({"decision": "uncertain"}, "auto_uncertain"),
        ({"basis": "title_only"}, "title_only_judgement"),
        ({"basis": "deterministic_metadata_rule"}, "rule_based_judgement"),
        ({"confidence": "low"}, "low_confidence"),
        ({"abstract": None}, "no_abstract"),
    ],
)
def test_sample_reason_names_the_boundary_risk(kwargs, expected):
    row = _row(1, kwargs.pop("decision", "relevant"), **kwargs)
    assert expected in sample_reason(row)


def test_sample_reason_defaults_to_random_within_decision():
    assert sample_reason(_row(1, "relevant")) == "random_within_decision"


# --------------------------------------------------------------------------- #
# The sheet
# --------------------------------------------------------------------------- #


def test_sheet_has_one_blank_row_per_sample_and_the_expected_columns():
    rows = sample_assessments(_corpus(), per_decision=2, seed=1)
    parsed = list(csv.DictReader(io.StringIO(render_sheet(rows))))
    assert len(parsed) == len(rows)
    assert all(row["human_decision"] == "" for row in parsed)
    assert all(row["human_note"] == "" for row in parsed)
    assert {"paper_id", "title", "abstract", "auto_decision", "auto_basis"} <= set(parsed[0])


def test_blind_sheet_withholds_the_automatic_judgement():
    rows = sample_assessments(_corpus(), per_decision=2, seed=1)
    blind = list(csv.DictReader(io.StringIO(render_sheet(rows, blind=True))))
    assert all(row["auto_decision"] == "" for row in blind)
    assert all(row["auto_basis"] == "" for row in blind)
    assert all(row["auto_confidence"] == "" for row in blind)
    # ...while the sample reason stays, because it is a reading hint, not a label.
    assert any(row["sample_reason"] for row in blind)


def test_sheet_can_omit_abstracts_for_title_only_evaluation():
    rows = [_row(1, "relevant")]
    parsed = list(csv.DictReader(io.StringIO(render_sheet(rows, include_abstract=False))))
    assert parsed[0]["abstract"] == ""
    assert parsed[0]["title"]


def test_write_sheet_records_a_sidecar_manifest(tmp_path: Path):
    rows = sample_assessments(_corpus(), per_decision=2, seed=1)
    path = tmp_path / "sheet.csv"
    manifest = write_sheet(path, rows, blind=True)
    assert path.is_file()
    sidecar = path.with_suffix(path.suffix + ".manifest.json")
    assert sidecar.is_file()
    assert manifest["rows"] == len(rows)
    assert manifest["blind"] is True
    assert manifest["profiles"] == ["behavioral_economics"]
    assert set(manifest["human_decisions_allowed"]) == set(HUMAN_DECISIONS)
    # The UTF-8 BOM is there so Excel opens Chinese notes correctly.
    assert path.read_bytes().startswith(b"\xef\xbb\xbf")


# --------------------------------------------------------------------------- #
# Reading annotations back
# --------------------------------------------------------------------------- #


def test_human_labels_require_only_paper_id_and_decision():
    labels, problems = load_human_labels("paper_id,human_decision\n7,relevant\n")
    assert problems == []
    assert labels[0].paper_id == 7
    assert labels[0].decision == "relevant"
    assert labels[0].profile_id == ""


def test_blank_rows_are_reported_as_unlabelled_not_as_errors():
    labels, problems = load_human_labels("paper_id,human_decision\n7,\n")
    assert labels == []
    assert any("no human_decision" in problem for problem in problems)


def test_unknown_human_decisions_are_rejected_loudly():
    labels, problems = load_human_labels("paper_id,human_decision\n7,probably\n")
    assert labels == []
    assert any("not one of" in problem for problem in problems)


def test_missing_or_non_numeric_paper_id_is_reported():
    _, problems = load_human_labels("paper_id,human_decision\n,relevant\n")
    assert any("paper_id" in problem for problem in problems)


def test_decision_matching_is_case_and_whitespace_insensitive():
    labels, problems = load_human_labels("paper_id,human_decision\n7,  RELEVANT \n")
    assert problems == []
    assert labels[0].decision == "relevant"


def test_bom_prefixed_sheets_parse():
    labels, problems = load_human_labels("\ufeffpaper_id,human_decision\n7,relevant\n")
    assert problems == []
    assert labels[0].paper_id == 7


# --------------------------------------------------------------------------- #
# Comparison
# --------------------------------------------------------------------------- #


def test_compare_counts_agreement_and_the_two_error_directions():
    assessments = _corpus()
    text = _sheet(
        [
            (1, "irrelevant", "clearly out of scope"),  # agree
            (20, "relevant", "core mechanism"),  # agree
            (2, "relevant", "behavioural mechanism is the object"),  # false negative
            (21, "irrelevant", "only mentions behaviour in passing"),  # false positive
        ]
    )
    report = compare_sheet(text, assessments, profile_id="behavioral_economics")
    assert report.labelled_rows == 4
    assert report.agreement == 2
    assert report.agreement_rate == pytest.approx(0.5)
    assert [item["paper_id"] for item in report.auto_negative_human_positive] == [2]
    assert [item["paper_id"] for item in report.auto_positive_human_negative] == [21]
    assert report.problems == []


def test_false_positive_and_false_negative_carry_the_human_reason():
    text = _sheet([(2, "relevant", "belief updating is the mechanism")])
    report = compare_sheet(text, _corpus(), profile_id="behavioral_economics")
    assert report.auto_negative_human_positive[0]["human_note"] == (
        "belief updating is the mechanism"
    )


def test_uncertain_on_either_side_is_a_boundary_case_not_an_error():
    text = _sheet(
        [
            (40, "relevant", "from the title this looks behavioural"),  # auto uncertain
            (1, "uncertain", "cannot tell without the mechanism"),  # human uncertain
        ]
    )
    report = compare_sheet(text, _corpus(), profile_id="behavioral_economics")
    assert len(report.boundary_cases) == 2
    assert report.auto_positive_human_negative == []
    assert report.auto_negative_human_positive == []
    # Neither row agrees, but neither is counted as a directional error either:
    # "the screen could not call it" is a different finding from "it called it wrong".
    assert report.agreement == 0
    assert [item["paper_id"] for item in report.ambiguous_human_calls] == [1]


def test_skip_is_honest_abstention_and_never_counted_as_agreement():
    text = _sheet([(1, "irrelevant", ""), (2, "skip", "no abstract available")])
    report = compare_sheet(text, _corpus())
    assert report.abstentions == 1
    assert report.labelled_rows == 1
    assert report.agreement == 1


def test_unlabelled_rows_are_counted_separately_from_labelled_ones():
    text = _sheet([(1, "irrelevant", ""), (2, "", "")])
    report = compare_sheet(text, _corpus())
    assert report.sampled_rows == 2
    assert report.labelled_rows == 1
    assert report.unlabelled_rows == 1


def test_title_only_and_full_screenings_are_reported_separately():
    assessments = [
        _row(1, "relevant", basis="title_and_abstract"),
        _row(2, "irrelevant", basis="title_and_abstract"),
        _row(40, "relevant", basis="title_only"),
        _row(41, "irrelevant", basis="title_only"),
    ]
    text = _sheet(
        [
            (1, "relevant", ""),
            (2, "relevant", ""),
            (40, "irrelevant", ""),
            (41, "irrelevant", ""),
        ]
    )
    report = compare_sheet(text, assessments)
    strata = report.stratification
    assert strata["title_and_abstract"]["considered"] == 2
    assert strata["title_and_abstract"]["agreement"] == 1
    assert strata["title_only"]["considered"] == 2
    assert strata["title_only"]["agreement"] == 1
    # The aggregate is reported but the split is what makes it interpretable.
    assert report.agreement == 2


def test_confusion_matrix_lists_the_paper_ids_behind_each_cell():
    text = _sheet([(1, "irrelevant", ""), (2, "irrelevant", "")])
    report = compare_sheet(text, _corpus())
    cell = next(
        entry for entry in report.confusion if entry.auto_decision == "irrelevant"
        and entry.human_decision == "irrelevant"
    )
    assert cell.count == 2
    assert set(cell.paper_ids) == {1, 2}


def test_rows_for_papers_without_an_assessment_are_reported_not_silently_dropped():
    text = _sheet([(999, "relevant", "")])
    report = compare_sheet(text, _corpus(), profile_id="behavioral_economics")
    assert report.labelled_rows == 0
    assert any("no stored assessment" in problem for problem in report.problems)


def test_a_sheet_without_profile_id_is_rejected_when_a_paper_is_judged_twice():
    assessments = [
        _row(1, "relevant", profile="behavioral_economics"),
        _row(1, "irrelevant", profile="labor_economics"),
    ]
    # A sheet with no profile column cannot say which queue the human judged.
    ambiguous_sheet = "paper_id,title,human_decision,human_note\n1,Paper 1,relevant,x\n"
    report = compare_sheet(ambiguous_sheet, assessments)
    assert report.labelled_rows == 0
    assert any("several profiles" in problem for problem in report.problems)


def test_an_explicit_profile_column_is_honoured():
    assessments = [
        _row(1, "relevant", profile="behavioral_economics"),
        _row(1, "irrelevant", profile="labor_economics"),
    ]
    text = _sheet([(1, "irrelevant", "")], profile="labor_economics")
    report = compare_sheet(text, assessments)
    assert report.labelled_rows == 1
    assert report.agreement == 1
    assert report.profile_id == "labor_economics"


def test_report_states_the_sampling_mode_so_a_balanced_sample_is_not_mistaken_for_precision():
    report = compare_sheet(
        _sheet([(1, "irrelevant", "")]),
        _corpus(),
        sampling_mode="balanced_per_decision",
    )
    assert report.sampling_mode == "balanced_per_decision"
    assert "balanced_per_decision" in report.as_dict()["sampling_mode"]


def test_small_samples_are_flagged_as_insufficient_for_prompt_changes():
    report = compare_sheet(_sheet([(1, "irrelevant", "")]), _corpus())
    text = format_comparison(report)
    assert "cannot support changing a prompt" in text


def test_formatting_lists_the_error_cases_with_titles():
    text = _sheet([(2, "relevant", "mechanism is central")])
    report = compare_sheet(text, _corpus(), profile_id="behavioral_economics")
    rendered = format_comparison(report)
    assert "automatic negative / human positive" in rendered
    assert "#2" in rendered
    assert "mechanism is central" in rendered


def test_json_payload_is_serializable_and_complete():
    import json

    report = compare_sheet(_sheet([(2, "relevant", "x")]), _corpus())
    payload = json.loads(json.dumps(report.as_dict(), ensure_ascii=False))
    assert payload["false_negatives"] == 1
    assert payload["boundary_case_count"] == 0
    assert payload["agreement_rate"] == 0.0
    assert isinstance(payload["confusion"], list)


def test_comparison_with_no_labels_does_not_divide_by_zero():
    report = compare_sheet(_sheet([(1, "", "")]), _corpus())
    assert report.agreement_rate == 0.0
    assert report.labelled_rows == 0
    assert format_comparison(report)


def test_the_workflow_never_writes_any_decision_back():
    """Compare is a read-only analysis: the module exposes no writer for labels."""
    import eduecon_radar.services.profile_quality as module

    exported = {name for name in dir(module) if not name.startswith("_")}
    assert not {name for name in exported if "write_review" in name or "save_" in name}
    assert "write_sheet" in exported  # the only writer, and it writes the request


# --------------------------------------------------------------------------- #
# Screening basis: title-only and full screening are different claims
# --------------------------------------------------------------------------- #


def _database_with(tmp_path: Path, rows: list[tuple]) -> Path:
    """A minimal production-shaped database for the read-only measurements."""
    import sqlite3

    path = tmp_path / "quality.db"
    with sqlite3.connect(path) as connection:
        connection.executescript(
            """
            CREATE TABLE journals (id INTEGER PRIMARY KEY, short_name TEXT NOT NULL);
            CREATE TABLE papers (
                id INTEGER PRIMARY KEY, title TEXT NOT NULL, abstract TEXT,
                jel_codes_json TEXT NOT NULL DEFAULT '[]', journal_id INTEGER NOT NULL);
            CREATE TABLE paper_profile_assessments (
                id INTEGER PRIMARY KEY, profile_id TEXT NOT NULL, paper_id INTEGER NOT NULL,
                decision TEXT, screening_basis TEXT,
                classification_json TEXT NOT NULL DEFAULT '{}',
                model_name TEXT, prompt_version TEXT, input_fingerprint TEXT,
                source_run_id INTEGER, assessed_at TEXT, updated_at TEXT NOT NULL);
            INSERT INTO journals (id, short_name) VALUES (1, 'JEBO');
            """
        )
        for paper_id, profile_id, decision, basis, abstract in rows:
            connection.execute(
                "INSERT INTO papers (id,title,abstract,journal_id) VALUES (?,?,?,1)",
                (paper_id, f"Paper {paper_id}", abstract),
            )
            connection.execute(
                """INSERT INTO paper_profile_assessments
                (profile_id,paper_id,decision,screening_basis,classification_json,
                 prompt_version,updated_at) VALUES (?,?,?,?,'{}','v1','now')""",
                (profile_id, paper_id, decision, basis),
            )
    return path


def test_screening_basis_separates_title_only_from_full(tmp_path):
    database = _database_with(
        tmp_path,
        [
            (1, "behavioral_economics", "relevant", "title_and_abstract", "abstract text"),
            (2, "behavioral_economics", "irrelevant", "title_and_abstract", "abstract text"),
            (3, "behavioral_economics", "uncertain", "title_only", None),
            (4, "behavioral_economics", "uncertain", "title_only", None),
        ],
    )
    report = measure_screening_basis(database, "behavioral_economics")
    assert report.total == 4
    assert report.by_basis == {"title_and_abstract": 2, "title_only": 2}
    assert report.title_only_share == pytest.approx(0.5)
    assert report.by_basis_decision["title_only"] == {"uncertain": 2}
    assert report.by_basis_decision["title_and_abstract"] == {
        "relevant": 1,
        "irrelevant": 1,
    }
    assert report.papers_without_abstract == 2
    assert "missing data" in report.note


def test_screening_basis_explains_why_title_only_is_not_evidence(tmp_path):
    """The note must say *why* title-only rows cannot speak to the boundary."""
    database = _database_with(
        tmp_path, [(1, "behavioral_economics", "uncertain", "title_only", None)]
    )
    report = measure_screening_basis(database, "behavioral_economics")
    assert "cannot be positive" in report.note
    assert "missing data" in report.note
    assert "title-only share" in format_screening_basis(report)


def test_screening_basis_is_silent_when_everything_was_fully_screened(tmp_path):
    database = _database_with(
        tmp_path,
        [(1, "labor_economics", "relevant", "title_and_abstract", "abstract text")],
    )
    report = measure_screening_basis(database, "labor_economics")
    assert report.title_only_share == 0.0
    assert report.note == ""
    assert "NOTE" not in format_screening_basis(report)


def test_screening_basis_reports_unrecorded_basis_honestly(tmp_path):
    """Historical rows predate the basis column; they must not count as full."""
    database = _database_with(
        tmp_path, [(1, "education_economics", "relevant", "", "abstract text")]
    )
    report = measure_screening_basis(database, "education_economics")
    assert report.by_basis == {"unrecorded": 1}
    assert report.title_only_share == 0.0


def test_screening_basis_json_is_serializable(tmp_path):
    import json

    database = _database_with(
        tmp_path, [(1, "behavioral_economics", "relevant", "title_and_abstract", "x")]
    )
    payload = json.loads(
        json.dumps(measure_screening_basis(database, "behavioral_economics").as_dict())
    )
    assert payload["profile_id"] == "behavioral_economics"
    assert payload["title_only_share"] == 0.0
