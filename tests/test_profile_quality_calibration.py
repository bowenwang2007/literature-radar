"""Corpus-scale annotation: sampling design, metrics, and honest suppression.

The first version of this workflow could only compare a small balanced sample
against itself.  A 130-row cross-profile sheet needs three things it did not have,
and each is pinned here:

* a **stratified design** that separates a probability sample (which can estimate
  precision/recall) from a boundary oversample (which exists to find failures);
* **real metrics** — precision, recall, coverage, confusion — with Wilson
  intervals, because annotation counts are small and precision sits near 1;
* **refusal to print a rate** from too few labels, so a 3-row cell cannot be read
  as a performance measurement, and no prompt or threshold is ever touched.
"""

from __future__ import annotations

import csv
import io
import json
from pathlib import Path

import pytest

from eduecon_radar.services.profile_quality import (
    MIN_LABELS_FOR_RATE,
    STRATUM_BOUNDARY,
    STRATUM_REPRESENTATIVE,
    AssessmentRow,
    compare_sheet,
    compute_metrics,
    design_sample,
    format_comparison,
    is_boundary_candidate,
    render_sheet,
    wilson_interval,
    write_sheet,
)


def _row(
    paper_id: int,
    decision: str,
    *,
    profile: str = "education_economics",
    basis: str = "title_and_abstract",
    confidence: str = "high",
    abstract: str | None = "An abstract.",
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
        abstract=abstract,
        journal="EER",
        publication_date="2026-03-01",
        classification={"confidence": confidence},
    )


def _corpus() -> list[AssessmentRow]:
    """A corpus shaped like the real one: one big profile and three small ones."""
    rows: list[AssessmentRow] = []
    for index in range(1, 121):
        rows.append(_row(index, "irrelevant" if index % 4 else "relevant"))
    for index in range(200, 240):
        rows.append(_row(index, "relevant", profile="labor_economics"))
    for index in range(300, 340):
        rows.append(_row(index, "uncertain", profile="behavioral_economics",
                         basis="title_only", confidence="low", abstract=None))
    for index in range(400, 415):
        rows.append(_row(index, "relevant", profile="political_economy"))
    return rows


# --------------------------------------------------------------------------- #
# Stratified design
# --------------------------------------------------------------------------- #


def test_design_hits_the_requested_total():
    plan = design_sample(_corpus(), total=130, seed=1)
    assert len(plan.rows) <= 130
    assert len({(item.profile_id, item.paper_id) for item in plan.rows}) == len(plan.rows)


def test_representative_seats_are_proportional_to_stratum_size():
    """The sample must mirror the library, not give every queue equal weight."""
    plan = design_sample(_corpus(), total=100, boundary_share=0.0, seed=1)
    weights = plan.per_profile
    assert weights["education_economics"] > weights["labor_economics"]
    assert weights["labor_economics"] > weights["political_economy"]
    assert sum(weights.values()) == 100
    # Seats track each profile's share of the judgements, within rounding.
    shares = {
        "education_economics": 120 / 175,
        "labor_economics": 40 / 175,
        "political_economy": 15 / 175,
    }
    for profile_id, share in shares.items():
        assert weights[profile_id] == pytest.approx(100 * share, abs=1.5)


def test_boundary_rows_are_never_counted_as_representative():
    plan = design_sample(_corpus(), total=100, boundary_share=0.4, seed=3)
    strata = {plan.stratum_of[(item.profile_id, item.paper_id)] for item in plan.rows}
    assert STRATUM_BOUNDARY in strata
    boundary = [
        item
        for item in plan.rows
        if plan.stratum_of[(item.profile_id, item.paper_id)] == STRATUM_BOUNDARY
    ]
    # Every boundary row must actually be one the system flagged.
    assert boundary
    assert all(is_boundary_candidate(item) for item in boundary)


def test_a_boundary_row_cannot_also_be_a_representative_row():
    plan = design_sample(_corpus(), total=130, boundary_share=0.5, seed=5)
    seen: dict[tuple[str, int], str] = {}
    for item in plan.rows:
        key = (item.profile_id, item.paper_id)
        assert key not in seen, "a row was drawn into two strata"
        seen[key] = plan.stratum_of[key]


def test_boundary_share_zero_yields_a_pure_probability_sample():
    plan = design_sample(_corpus(), total=60, boundary_share=0.0, seed=1)
    assert set(plan.stratum_of.values()) == {STRATUM_REPRESENTATIVE}
    # A pure probability draw must never contain a flagged row, because those
    # are excluded from the representative pool by construction.
    assert not any(is_boundary_candidate(item) for item in plan.rows)


def test_design_is_reproducible_for_a_seed():
    first = [item.paper_id for item in design_sample(_corpus(), total=50, seed=11).rows]
    again = [item.paper_id for item in design_sample(_corpus(), total=50, seed=11).rows]
    other = [item.paper_id for item in design_sample(_corpus(), total=50, seed=12).rows]
    assert first == again
    assert first != other


def test_design_reports_per_profile_stratum_counts():
    plan = design_sample(_corpus(), total=100, boundary_share=0.4, seed=2)
    payload = plan.as_dict()
    assert payload["rows"] == len(plan.rows)
    total = sum(sum(counts.values()) for counts in payload["per_profile_stratum"].values())
    assert total == len(plan.rows)
    assert payload["per_profile"] == plan.per_profile


def test_design_can_be_limited_to_selected_profiles():
    plan = design_sample(
        _corpus(), total=40, boundary_share=0.0, only_profiles=["labor_economics"], seed=1
    )
    assert {item.profile_id for item in plan.rows} == {"labor_economics"}


def test_design_handles_an_empty_pool():
    plan = design_sample([], total=100)
    assert plan.rows == []
    assert plan.as_dict()["rows"] == 0


def test_design_exhausts_small_pools_without_inventing_rows():
    """A profile with fewer eligible rows yields fewer, never duplicates."""
    small = [_row(1, "relevant", profile="political_economy"),
             _row(2, "irrelevant", profile="political_economy")]
    plan = design_sample(small, total=50, boundary_share=0.0, seed=1)
    assert len({item.paper_id for item in plan.rows}) == len(plan.rows) <= 2


@pytest.mark.parametrize(
    ("kwargs", "message"),
    [({"total": 0}, "total"), ({"total": 10, "boundary_share": 1.5}, "boundary_share")],
)
def test_design_rejects_impossible_parameters(kwargs, message):
    with pytest.raises(ValueError, match=message):
        design_sample(_corpus(), **kwargs)


def test_a_per_profile_floor_gives_every_queue_a_measurable_slice():
    """Strict proportion would starve the small queues; the floor prevents it."""
    without = design_sample(_corpus(), total=130, boundary_share=0.35, seed=1)
    with_floor = design_sample(
        _corpus(), total=130, boundary_share=0.35, min_per_profile=15, seed=1
    )
    assert without.per_profile["political_economy"] < 15
    assert with_floor.per_profile["political_economy"] >= 15
    assert len(with_floor.rows) == 130
    # The cost is paid by the largest queue, not by shrinking the sheet.
    assert (
        with_floor.per_profile["education_economics"]
        < without.per_profile["education_economics"]
    )


def test_a_queue_smaller_than_the_floor_has_all_of_its_rows_reserved():
    """A profile with eight judgements should have all eight looked at."""
    small = [_row(index, "relevant", profile="labor_economics") for index in range(1, 9)]
    big = [
        _row(index, "irrelevant", profile="education_economics")
        for index in range(100, 400)
    ]
    plan = design_sample(
        big + small, total=100, boundary_share=0.0, min_per_profile=15, seed=1
    )
    assert plan.per_profile["labor_economics"] == 8
    assert len(plan.rows) == 100


def test_the_floor_is_abandoned_rather_than_overrunning_a_tiny_sheet():
    plan = design_sample(_corpus(), total=5, boundary_share=0.0, min_per_profile=15, seed=1)
    assert len(plan.rows) <= 5
    assert len({(item.profile_id, item.paper_id) for item in plan.rows}) == len(plan.rows)


def test_plan_reports_seats_and_capacity_so_a_thin_slice_is_explainable():
    plan = design_sample(
        _corpus(), total=100, boundary_share=0.0, min_per_profile=15, seed=1
    )
    payload = plan.as_dict()
    assert payload["capacity"]["political_economy"] == 15
    assert payload["seats"]["political_economy"] == 15
    assert payload["rows"] == len(plan.rows)


def test_zero_floor_reproduces_the_pure_proportional_design():
    floor = design_sample(
        _corpus(), total=100, boundary_share=0.0, min_per_profile=0, seed=1
    )
    none = design_sample(_corpus(), total=100, boundary_share=0.0, seed=1)
    assert floor.per_profile == none.per_profile


def test_a_negative_floor_is_rejected():
    with pytest.raises(ValueError, match="min_per_profile"):
        design_sample(_corpus(), total=10, min_per_profile=-1)


def test_sheet_carries_the_stratum_even_when_blind():
    plan = design_sample(_corpus(), total=60, boundary_share=0.5, seed=4)
    text = render_sheet(plan.rows, blind=True, stratum_of=plan.stratum_of)
    parsed = list(csv.DictReader(io.StringIO(text)))
    assert all(row["auto_decision"] == "" for row in parsed)
    assert {row["sample_stratum"] for row in parsed} == {
        STRATUM_REPRESENTATIVE,
        STRATUM_BOUNDARY,
    }


def test_manifest_records_the_design_so_the_numbers_stay_interpretable(tmp_path: Path):
    plan = design_sample(_corpus(), total=60, boundary_share=0.4, seed=1)
    path = tmp_path / "sheet.csv"
    manifest = write_sheet(
        path,
        plan.rows,
        blind=True,
        stratum_of=plan.stratum_of,
        design={"kind": "stratified_representative_plus_boundary", "total_requested": 60},
    )
    assert manifest["strata"][STRATUM_BOUNDARY] > 0
    assert manifest["design"]["kind"].startswith("stratified")
    sidecar = json.loads(
        path.with_suffix(path.suffix + ".manifest.json").read_text(encoding="utf-8")
    )
    assert sidecar["strata"] == manifest["strata"]


# --------------------------------------------------------------------------- #
# Metrics
# --------------------------------------------------------------------------- #


def _pairs(spec: list[tuple[str, str]]) -> list[tuple[dict[str, str], AssessmentRow]]:
    out = []
    for index, (auto, human) in enumerate(spec, start=1):
        row = {"paper_id": str(index), "human_decision": human, "sample_stratum":
               STRATUM_REPRESENTATIVE, "title": f"P{index}", "journal": "EER",
               "auto_decision": auto, "auto_basis": "", "auto_confidence": "",
               "human_note": "", "sample_reason": ""}
        out.append((row, _row(index, auto)))
    return out


def test_precision_and_recall_are_computed_over_decisive_rows():
    pairs = _pairs(
        [
            ("relevant", "relevant"),      # TP
            ("relevant", "relevant"),      # TP
            ("relevant", "irrelevant"),    # FP
            ("irrelevant", "relevant"),    # FN
            ("irrelevant", "irrelevant"),  # TN
        ]
    )
    cell = compute_metrics(pairs, slice_name="x")
    assert (cell.true_positive, cell.false_positive) == (2, 1)
    assert (cell.false_negative, cell.true_negative) == (1, 1)
    assert cell.precision.value == pytest.approx(2 / 3)
    assert cell.recall.value == pytest.approx(2 / 3)
    assert cell.coverage.value == pytest.approx(1.0)
    assert cell.agreement.value == pytest.approx(3 / 5)


def test_uncertain_is_excluded_from_precision_and_recall_denominators():
    """An abstention is neither a positive nor a negative claim."""
    pairs = _pairs(
        [
            ("relevant", "relevant"),
            ("relevant", "uncertain"),
            ("uncertain", "relevant"),
            ("uncertain", "irrelevant"),
        ]
    )
    cell = compute_metrics(pairs, slice_name="x")
    assert cell.auto_uncertain == 2
    assert cell.human_uncertain == 1
    # Only one row had both sides decisive.
    assert cell.precision.total == 1
    assert cell.recall.total == 1
    assert cell.precision.value == pytest.approx(1.0)
    # Recall denominator excludes the human-uncertain row, so the human-relevant
    # rows that count are the one agree case and the auto-uncertain one is dropped.
    assert cell.recall.successes == 1
    # Coverage counts how often the queue was willing to decide at all.
    assert cell.coverage.value == pytest.approx(0.5)


def test_coverage_exposes_a_queue_that_abstains_its_way_to_high_precision():
    pairs = _pairs([("relevant", "relevant")] + [("uncertain", "irrelevant")] * 9)
    cell = compute_metrics(pairs, slice_name="x")
    assert cell.precision.value == pytest.approx(1.0)
    assert cell.coverage.value == pytest.approx(0.1)
    assert cell.coverage.total == 10


def test_agreement_counts_uncertain_matching_uncertain():
    pairs = _pairs([("uncertain", "uncertain"), ("uncertain", "relevant")])
    cell = compute_metrics(pairs, slice_name="x")
    assert cell.agreement.value == pytest.approx(0.5)


def test_rates_are_suppressed_below_the_minimum_sample():
    pairs = _pairs([("relevant", "relevant")] * (MIN_LABELS_FOR_RATE - 1))
    cell = compute_metrics(pairs, slice_name="x")
    assert cell.precision.sufficient is False
    assert cell.precision.value is not None  # computed, but not to be trusted
    assert "insufficient" in cell.precision.render("precision")


def test_rates_are_reported_once_the_sample_is_large_enough():
    pairs = _pairs([("relevant", "relevant")] * MIN_LABELS_FOR_RATE)
    cell = compute_metrics(pairs, slice_name="x")
    assert cell.precision.sufficient is True
    assert "[" in cell.precision.render("precision")


def test_empty_metrics_do_not_divide_by_zero():
    cell = compute_metrics([], slice_name="empty")
    assert cell.labelled == 0
    assert cell.precision.value is None
    assert "no labelled rows" in cell.precision.render("precision")


# --------------------------------------------------------------------------- #
# Wilson interval
# --------------------------------------------------------------------------- #


def test_wilson_interval_brackets_the_point_estimate():
    low, high = wilson_interval(8, 10)
    assert low < 0.8 < high
    assert low >= 0.0 and high <= 1.0


def test_wilson_interval_stays_inside_the_unit_range_at_the_boundaries():
    """The normal approximation produces intervals outside [0,1] here."""
    low, high = wilson_interval(10, 10)
    assert low > 0.0
    assert high == pytest.approx(1.0, abs=1e-9)
    low, high = wilson_interval(0, 10)
    assert low == pytest.approx(0.0, abs=1e-9)
    assert high < 1.0


def test_wilson_interval_narrows_as_the_sample_grows():
    narrow = wilson_interval(80, 100)
    wide = wilson_interval(8, 10)
    assert (narrow[1] - narrow[0]) < (wide[1] - wide[0])


def test_wilson_interval_on_an_empty_sample_is_maximally_uncertain():
    assert wilson_interval(0, 0) == (0.0, 1.0)


# --------------------------------------------------------------------------- #
# Comparison end to end
# --------------------------------------------------------------------------- #


def _sheet(rows: list[dict[str, str]]) -> str:
    columns = [
        "profile_id", "paper_id", "journal", "title", "human_decision", "human_note",
        "auto_decision", "auto_basis", "auto_confidence", "sample_stratum",
        "sample_reason",
    ]
    buffer = io.StringIO()
    writer = csv.DictWriter(buffer, fieldnames=columns, lineterminator="\n")
    writer.writeheader()
    for row in rows:
        writer.writerow({column: row.get(column, "") for column in columns})
    return buffer.getvalue()


def test_metrics_separate_the_representative_and_boundary_strata():
    assessments = [
        _row(1, "relevant", profile="behavioral_economics"),
        _row(2, "irrelevant", profile="behavioral_economics"),
        _row(3, "uncertain", profile="behavioral_economics", basis="title_only",
             confidence="low", abstract=None),
        _row(4, "uncertain", profile="behavioral_economics", basis="title_only",
             confidence="low", abstract=None),
    ]
    sheet = _sheet(
        [
            {"profile_id": "behavioral_economics", "paper_id": "1",
             "human_decision": "relevant", "sample_stratum": STRATUM_REPRESENTATIVE},
            {"profile_id": "behavioral_economics", "paper_id": "2",
             "human_decision": "relevant", "sample_stratum": STRATUM_REPRESENTATIVE},
            {"profile_id": "behavioral_economics", "paper_id": "3",
             "human_decision": "relevant", "sample_stratum": STRATUM_BOUNDARY},
            {"profile_id": "behavioral_economics", "paper_id": "4",
             "human_decision": "irrelevant", "sample_stratum": STRATUM_BOUNDARY},
        ]
    )
    report = compare_sheet(sheet, assessments, sampling_mode="stratified")
    assert report.representative_metrics is not None
    assert report.representative_metrics.labelled == 2
    assert report.boundary_metrics is not None
    assert report.boundary_metrics.labelled == 2
    # Representative: auto said relevant (paper 1) and the human agreed (TP), and
    # auto said irrelevant (paper 2) while the human said relevant (FN).
    assert report.representative_metrics.true_positive == 1
    assert report.representative_metrics.false_negative == 1
    assert report.representative_metrics.false_positive == 0
    # Boundary: auto abstained on both, so neither is a directional error; the
    # human still called one relevant and one irrelevant.
    assert report.boundary_metrics.false_positive == 0
    assert report.boundary_metrics.false_negative == 0
    assert report.boundary_metrics.auto_uncertain == 2
    # Pooling the two strata would have produced a meaningless 50% precision.
    assert report.representative_metrics.precision.value == pytest.approx(1.0)


def test_metrics_are_broken_down_by_profile_and_by_basis():
    assessments = [
        _row(1, "relevant", profile="education_economics"),
        _row(2, "irrelevant", profile="labor_economics"),
        _row(3, "uncertain", profile="labor_economics", basis="title_only",
             confidence="low", abstract=None),
    ]
    sheet = _sheet(
        [
            {"profile_id": "education_economics", "paper_id": "1",
             "human_decision": "relevant", "sample_stratum": STRATUM_REPRESENTATIVE},
            {"profile_id": "labor_economics", "paper_id": "2",
             "human_decision": "irrelevant", "sample_stratum": STRATUM_REPRESENTATIVE},
            {"profile_id": "labor_economics", "paper_id": "3",
             "human_decision": "irrelevant", "sample_stratum": STRATUM_BOUNDARY},
        ]
    )
    report = compare_sheet(sheet, assessments)
    assert set(report.by_profile) == {"education_economics", "labor_economics"}
    assert report.by_profile["education_economics"]["labelled"] == 1
    assert report.by_profile["labor_economics"]["labelled"] == 2
    assert set(report.by_basis) == {"title_and_abstract", "title_only"}
    assert report.by_basis["title_only"]["labelled"] == 1
    assert report.by_basis["title_only"]["auto"]["uncertain"] == 1


def test_full_screening_slice_excludes_title_only_rows():
    assessments = [
        _row(1, "relevant", basis="title_and_abstract"),
        _row(2, "uncertain", basis="title_only", confidence="low", abstract=None),
    ]
    sheet = _sheet(
        [
            {"profile_id": "education_economics", "paper_id": "1",
             "human_decision": "relevant", "sample_stratum": STRATUM_REPRESENTATIVE},
            {"profile_id": "education_economics", "paper_id": "2",
             "human_decision": "relevant", "sample_stratum": STRATUM_BOUNDARY},
        ]
    )
    report = compare_sheet(sheet, assessments)
    assert report.full_screening_metrics is not None
    assert report.full_screening_metrics.labelled == 1
    assert report.full_screening_metrics.precision.value == pytest.approx(1.0)


def test_rendered_report_states_the_slice_and_refuses_a_rate_too_early():
    assessments = [_row(1, "relevant")]
    sheet = _sheet(
        [
            {"profile_id": "education_economics", "paper_id": "1",
             "human_decision": "irrelevant", "sample_stratum": STRATUM_REPRESENTATIVE},
        ]
    )
    rendered = format_comparison(compare_sheet(sheet, assessments))
    assert "representative stratum" in rendered
    assert "insufficient" in rendered
    assert "cannot support changing a prompt" in rendered


def test_rendered_report_with_no_labels_makes_no_performance_claim():
    assessments = [_row(1, "relevant")]
    sheet = _sheet(
        [
            {"profile_id": "education_economics", "paper_id": "1",
             "human_decision": "", "sample_stratum": STRATUM_REPRESENTATIVE},
        ]
    )
    rendered = format_comparison(compare_sheet(sheet, assessments))
    assert "No performance claim can be made" in rendered
    assert "precision (auto=relevant)" not in rendered


def test_json_output_carries_the_metrics_block():
    assessments = [_row(1, "relevant")]
    sheet = _sheet(
        [
            {"profile_id": "education_economics", "paper_id": "1",
             "human_decision": "relevant", "sample_stratum": STRATUM_REPRESENTATIVE},
        ]
    )
    payload = json.loads(
        json.dumps(compare_sheet(sheet, assessments).as_dict(), ensure_ascii=False)
    )
    metrics = payload["metrics"]
    assert set(metrics) == {
        "representative",
        "boundary",
        "full_screening",
        "by_profile",
        "by_basis",
    }
    assert metrics["representative"]["confusion"]["true_positive"] == 1
    assert metrics["representative"]["precision"]["wilson_95"] is not None


def test_rows_without_a_stratum_are_treated_as_representative():
    """Older sheets have no stratum column; they must still be comparable."""
    assessments = [_row(1, "relevant")]
    sheet = _sheet(
        [{"profile_id": "education_economics", "paper_id": "1",
          "human_decision": "relevant"}]
    )
    report = compare_sheet(sheet, assessments)
    assert report.representative_metrics is not None
    assert report.representative_metrics.labelled == 1


def test_the_workflow_still_exposes_no_way_to_change_a_prompt_or_threshold():
    """The calibration phase must not quietly grow a tuning path."""
    import eduecon_radar.services.profile_quality as module

    exported = {name for name in dir(module) if not name.startswith("_")}
    forbidden = {
        name
        for name in exported
        if any(
            token in name
            for token in ("tune", "threshold", "rewrite_prompt", "apply_", "calibrate")
        )
    }
    assert forbidden == set()
    # No SQL that writes: the module reads the product tables and writes only the
    # annotation sheet and its sidecar.  Checking the statements rather than the
    # prose, because the docstring legitimately names the tables it refuses to touch.
    source = Path(module.__file__).read_text(encoding="utf-8")
    for statement in ("INSERT INTO", "UPDATE ", "DELETE FROM", "DROP "):
        assert statement not in source, f"{statement!r} appears in a read-only module"
