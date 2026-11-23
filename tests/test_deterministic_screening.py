from __future__ import annotations

import json
from datetime import date
from pathlib import Path

import pytest

from eduecon_radar.db import Database
from eduecon_radar.schemas import PaperMetadata
from eduecon_radar.services.deepseek import DeepSeekError, DeepSeekResponse
from eduecon_radar.services.deterministic_screening import (
    DeterministicScreeningService,
    EvidenceItem,
    PrimaryScreeningOutput,
    ReviewScreeningOutput,
    input_hash_for,
    normalize_evidence_text,
    parse_primary_text,
    parse_review_text,
)
from eduecon_radar.services.migrations import (
    apply_overnight_migration,
    apply_p0_hardening_migration,
    apply_profile_migration,
    apply_translation_migration,
    apply_workflow_migration,
)
from eduecon_radar.services.profile_registry import default_profile_registry

ROOT = Path(__file__).resolve().parents[1]

ABSTRACT = (
    "This paper studies how performance-based teacher incentives affect student "
    "achievement and teacher effort. Using administrative panel data and a sharp "
    "regression discontinuity design, we estimate that incentive pay significantly "
    "improves mathematics test scores, with larger effects for low-performing schools."
)
QUOTE = "performance-based teacher incentives affect student achievement"


def _database(tmp_path: Path) -> Database:
    database = Database(tmp_path / "screening.db")
    database.init_schema()
    apply_translation_migration(database, ROOT / "migrations/0001_add_paper_translations.sql")
    apply_workflow_migration(database, ROOT / "migrations/0002_add_workflow_batches.sql")
    apply_p0_hardening_migration(
        database, ROOT / "migrations/0007_add_runtime_leases_and_task_reviews.sql"
    )
    apply_overnight_migration(
        database, ROOT / "migrations/0010_add_overnight_pipeline.sql"
    )
    journal = default_profile_registry().by_key("economics_of_education_review")
    journal_id = database.upsert_journal(journal.as_journal_config())
    paper_id, _ = database.upsert_paper(
        PaperMetadata(
            doi="10.1000/screening",
            title="Teacher incentives and student achievement",
            abstract=ABSTRACT,
            authors=["Researcher"],
            issue_date=date(2026, 6, 1),
            source_metadata={"DOI": "10.1000/screening"},
        ),
        journal_id,
    )
    apply_profile_migration(
        database, ROOT / "migrations/0013_add_profile_scoped_assessment.sql"
    )
    return database


def _primary(
    *,
    decision: str = "relevant",
    centrality: str = "central",
    economic: str = "substantive",
    sufficiency: str = "sufficient",
    confidence: str = "high",
    evidence: list[EvidenceItem] | None = None,
) -> PrimaryScreeningOutput:
    return PrimaryScreeningOutput(
        decision=decision,
        education_centrality=centrality,
        economic_relevance=economic,
        evidence_sufficiency=sufficiency,
        confidence=confidence,
        evidence=evidence
        or [EvidenceItem(quote=QUOTE, supports="final_decision")],
        reason="Teacher incentives are a core education-economics mechanism.",
        topic_tags=["teacher incentives"],
        review_priority="high",
    )


def _review(
    *,
    review_decision: str = "confirm",
    final: str = "relevant",
    evidence: list[EvidenceItem] | None = None,
) -> ReviewScreeningOutput:
    return ReviewScreeningOutput(
        review_decision=review_decision,
        final_model_decision=final,
        disagreement_reason="",
        validated_evidence=evidence or [EvidenceItem(quote=QUOTE, supports="final_decision")],
        reason="fixture review",
    )


class CallRecorder:
    def __init__(
        self,
        primary: PrimaryScreeningOutput,
        review: ReviewScreeningOutput | None = None,
    ):
        self.primary = primary
        self.review = review
        self.primary_calls = 0
        self.review_calls = 0

    def primary_call(self, paper):
        self.primary_calls += 1
        return DeepSeekResponse(
            data=self.primary,
            model_name="fixture-primary",
            input_tokens=5,
            output_tokens=5,
        )

    def review_call(self, paper):
        self.review_calls += 1
        assert self.review is not None
        return DeepSeekResponse(
            data=self.review,
            model_name="fixture-review",
            input_tokens=5,
            output_tokens=5,
        )


def _service(
    database: Database,
    recorder: CallRecorder,
) -> DeterministicScreeningService:
    return DeterministicScreeningService(
        database,
        primary_caller=recorder.primary_call,
        review_caller=recorder.review_call,
        model="fixture-model",
    )


def _latest_screening(database: Database, paper_id: int = 1) -> dict:
    with database.connect() as connection:
        row = connection.execute(
            """SELECT * FROM paper_screenings WHERE paper_id=?
            ORDER BY id DESC LIMIT 1""",
            (paper_id,),
        ).fetchone()
    return dict(row) if row else {}


def test_valid_relevant_high_confidence_no_recheck(tmp_path) -> None:
    database = _database(tmp_path)
    recorder = CallRecorder(_primary())
    outcome = _service(database, recorder).screen_paper(1)
    assert recorder.primary_calls == 1
    assert recorder.review_calls == 0
    assert outcome.status == "classified"
    assert outcome.final_model_decision == "relevant"
    row = _latest_screening(database)
    assert row["validation_status"] == "valid"
    assert json.loads(row["evidence_json"])[0]["quote"] == QUOTE


def test_relevant_medium_confidence_triggers_recheck_and_confirm(tmp_path) -> None:
    database = _database(tmp_path)
    recorder = CallRecorder(_primary(confidence="medium"), _review())
    outcome = _service(database, recorder).screen_paper(1)
    assert recorder.review_calls == 1
    assert outcome.final_model_decision == "relevant"
    assert outcome.review_decision == "confirm"
    assert outcome.status == "classified"


def test_outcome_reports_the_tokens_the_provider_actually_charged(tmp_path) -> None:
    """The provider returns token usage on every response; it must not be dropped.

    Recording an AI call that consumed nothing would put a half-truth in the run's
    audit row, so the call count and the token counts have to travel together.
    """
    database = _database(tmp_path)
    recorder = CallRecorder(_primary())
    outcome = _service(database, recorder).screen_paper(1)
    assert recorder.primary_calls == 1
    assert outcome.ai_calls == 1
    assert (outcome.input_tokens, outcome.output_tokens) == (5, 5)


def test_recheck_tokens_are_added_not_replaced(tmp_path) -> None:
    database = _database(tmp_path)
    recorder = CallRecorder(_primary(confidence="medium"), _review())
    outcome = _service(database, recorder).screen_paper(1)
    assert recorder.primary_calls == 1 and recorder.review_calls == 1
    assert outcome.ai_calls == 2
    # primary (5/5) + review (5/5)
    assert (outcome.input_tokens, outcome.output_tokens) == (10, 10)


def test_uncertain_triggers_recheck_and_revise(tmp_path) -> None:
    database = _database(tmp_path)
    recorder = CallRecorder(
        _primary(decision="uncertain", centrality="secondary", economic="incidental"),
        _review(review_decision="revise", final="irrelevant"),
    )
    outcome = _service(database, recorder).screen_paper(1)
    assert recorder.review_calls == 1
    assert outcome.final_model_decision == "irrelevant"
    assert outcome.status == "classified"


def test_fabricated_evidence_rejected_and_review_recovers(tmp_path) -> None:
    database = _database(tmp_path)
    recorder = CallRecorder(
        _primary(
            evidence=[
                EvidenceItem(
                    quote="This sentence was invented by the model and never appeared.",
                    supports="final_decision",
                )
            ]
        ),
        _review(),
    )
    outcome = _service(database, recorder).screen_paper(1)
    assert recorder.review_calls == 1
    assert outcome.final_model_decision == "relevant"
    row = _latest_screening(database)
    assert row["validation_status"] == "needs_recheck"


def test_confirm_with_invalid_evidence_never_saves_reliable_relevant(tmp_path) -> None:
    database = _database(tmp_path)
    recorder = CallRecorder(
        _primary(
            evidence=[
                EvidenceItem(
                    quote="Entirely fabricated quote that is not present in the source text.",
                    supports="final_decision",
                )
            ]
        ),
        _review(
            review_decision="confirm",
            evidence=[
                EvidenceItem(
                    quote="Another fabricated sentence that does not appear anywhere.",
                    supports="final_decision",
                )
            ],
        ),
    )
    outcome = _service(database, recorder).screen_paper(1)
    assert outcome.final_model_decision == "uncertain"
    assert outcome.status == "needs_retry"
    row = _latest_screening(database)
    assert row["validation_status"] == "recheck_failed"


def test_parse_failure_marks_needs_retry_without_recheck(tmp_path) -> None:
    database = _database(tmp_path)

    class BrokenPrimary:
        def __call__(self, paper):
            raise DeepSeekError("invalid json", raw_response="not json at all")

    service = DeterministicScreeningService(
        database,
        primary_caller=BrokenPrimary(),
        model="fixture-model",
    )
    outcome = service.screen_paper(1)
    assert outcome.status == "needs_retry"
    assert outcome.validation_status == "parse_failed"
    row = _latest_screening(database)
    assert row["raw_response_json"]


def test_no_abstract_produces_title_only_screening(tmp_path) -> None:
    database = _database(tmp_path)
    with database.connect() as connection:
        connection.execute(
            "UPDATE papers SET abstract=NULL WHERE id=1"
        )
    recorder = CallRecorder(
        _primary(
            evidence=[
                EvidenceItem(
                    quote="Teacher incentives and student achievement",
                    supports="final_decision",
                )
            ]
        )
    )
    outcome = _service(database, recorder).screen_paper(1)
    assert recorder.primary_calls == 1
    assert outcome.status == "classified"
    assert outcome.final_model_decision == "relevant"
    assert outcome.screening_basis == "title_only"
    row = _latest_screening(database)
    assert json.loads(row["raw_response_json"])["screening_basis"] == "title_only"


def test_title_only_model_requesting_abstract_becomes_uncertain(tmp_path) -> None:
    database = _database(tmp_path)
    with database.connect() as connection:
        connection.execute("UPDATE papers SET abstract=NULL WHERE id=1")
    recorder = CallRecorder(_primary(decision="needs_abstract"))
    outcome = _service(database, recorder).screen_paper(1)
    assert outcome.status == "classified"
    assert outcome.final_model_decision == "uncertain"
    assert outcome.screening_basis == "title_only"


class _SequenceRecorder:
    def __init__(self, primaries):
        self.primaries = list(primaries)
        self.calls = 0

    def primary_call(self, paper):
        output = self.primaries[min(self.calls, len(self.primaries) - 1)]
        self.calls += 1
        return DeepSeekResponse(
            data=output,
            model_name="fixture-primary",
            input_tokens=5,
            output_tokens=5,
        )

    def review_call(self, paper):
        raise AssertionError("title-only upgrade must not call the review model")


def test_title_only_upgrades_to_full_when_abstract_arrives(tmp_path) -> None:
    database = _database(tmp_path)
    with database.connect() as connection:
        connection.execute("UPDATE papers SET abstract=NULL WHERE id=1")
    recorder = _SequenceRecorder(
        [
            _primary(
                decision="irrelevant",
                evidence=[
                    EvidenceItem(
                        quote="Teacher incentives and student achievement",
                        supports="final_decision",
                    )
                ],
            ),
            _primary(),
        ]
    )
    service = _service(database, recorder)
    first = service.screen_paper(1)
    assert first.screening_basis == "title_only"
    assert first.final_model_decision == "irrelevant"
    with database.connect() as connection:
        connection.execute(
            "UPDATE papers SET abstract=? WHERE id=1", (ABSTRACT,)
        )
    second = service.screen_paper(1)
    assert second.screening_basis == "title_and_abstract"
    assert second.final_model_decision == "relevant"
    with database.connect() as connection:
        stale = connection.execute(
            "SELECT COUNT(*) FROM paper_screenings WHERE paper_id=1 AND stale=1"
        ).fetchone()[0]
    assert stale == 1
    reused = service.screen_paper(1)
    assert reused.reused is True
    assert reused.screening_basis == "title_and_abstract"


def test_needs_abstract_with_abstract_present_is_invalid_and_rechecked(tmp_path) -> None:
    database = _database(tmp_path)
    recorder = CallRecorder(
        _primary(decision="needs_abstract"),
        _review(review_decision="revise", final="relevant"),
    )
    outcome = _service(database, recorder).screen_paper(1)
    assert recorder.review_calls == 1
    assert outcome.final_model_decision == "relevant"
    assert outcome.status == "classified"


def test_insufficient_evidence_high_confidence_invalid(tmp_path) -> None:
    database = _database(tmp_path)
    recorder = CallRecorder(
        _primary(decision="insufficient_evidence", confidence="high"),
        _review(review_decision="revise", final="uncertain"),
    )
    outcome = _service(database, recorder).screen_paper(1)
    assert recorder.review_calls == 1
    assert outcome.final_model_decision == "uncertain"


def test_unchanged_input_reuses_screening_without_calls(tmp_path) -> None:
    database = _database(tmp_path)
    recorder = CallRecorder(_primary())
    service = _service(database, recorder)
    service.screen_paper(1)
    outcome = service.screen_paper(1)
    assert recorder.primary_calls == 1
    assert outcome.reused is True
    assert outcome.status == "classified"


def test_force_rescreen_creates_new_row_and_stales_old(tmp_path) -> None:
    database = _database(tmp_path)
    recorder = CallRecorder(_primary())
    service = _service(database, recorder)
    service.screen_paper(1)
    recorder.primary = _primary(decision="irrelevant", confidence="high")
    recorder.review = _review(review_decision="confirm", final="irrelevant")
    outcome = service.screen_paper(1, force=True)
    assert recorder.primary_calls == 2
    assert outcome.final_model_decision == "irrelevant"
    with database.connect() as connection:
        rows = connection.execute(
            "SELECT stale,model_decision FROM paper_screenings ORDER BY id"
        ).fetchall()
    assert [tuple(row) for row in rows] == [(1, "relevant"), (0, "irrelevant")]


def test_screening_never_touches_human_review_fields(tmp_path) -> None:
    database = _database(tmp_path)
    with database.connect() as connection:
        connection.execute(
            """UPDATE reviews SET status='pending',reviewer_note='keep me'
            WHERE paper_id=1"""
        )
    recorder = CallRecorder(_primary())
    _service(database, recorder).screen_paper(1)
    with database.connect() as connection:
        row = connection.execute(
            "SELECT status,reviewer_note FROM reviews WHERE paper_id=1"
        ).fetchone()
    assert tuple(row) == ("pending", "keep me")
    assert recorder.primary_calls == 1


def test_parser_handles_fences_prose_casing_string_evidence_and_extra_fields() -> None:
    raw = """
    Here is the model response:
    ```json
    {"decision": "RELEVANT", "education_centrality": "Central",
     "economic_relevance": "substantive", "evidence_sufficiency": "sufficient",
     "confidence": "High",
     "evidence": "performance-based teacher incentives affect student achievement",
     "reason": "clear", "topic_tags": ["x"], "review_priority": "medium",
     "unexpected_extra_field": true}
    ```
    End of response.
    """
    parsed = parse_primary_text(raw)
    assert parsed.decision == "relevant"
    assert parsed.education_centrality == "central"
    assert parsed.confidence == "high"
    assert parsed.evidence[0].quote == QUOTE
    assert parsed.evidence[0].supports == "final_decision"


def test_parser_rejects_truncated_and_non_json() -> None:
    with pytest.raises(ValueError):
        parse_primary_text('{"decision": "relevant", ')
    with pytest.raises(ValueError):
        parse_primary_text("this is not json")


def test_review_parser_normalizes_enums() -> None:
    parsed = parse_review_text(
        '{"review_decision": "REVISE", "final_model_decision": "irrelevant",'
        ' "disagreement_reason": "education is background", "validated_evidence": []}'
    )
    assert parsed.review_decision == "revise"
    assert parsed.final_model_decision == "irrelevant"


def test_normalize_evidence_text_handles_punctuation_and_case() -> None:
    source = "Teacher Incentives, and Student Achievement!"
    quote = "teacher incentives and student achievement"
    assert normalize_evidence_text(quote) in normalize_evidence_text(source)
    assert input_hash_for("a", "b") == input_hash_for("a", "b")
    assert input_hash_for("a", "b") != input_hash_for("a", "c")


def test_parser_repairs_trailing_commas() -> None:
    parsed = parse_primary_text(
        '{"decision": "relevant", "education_centrality": "central",'
        ' "economic_relevance": "substantive", "evidence_sufficiency": "sufficient",'
        ' "confidence": "high", "evidence": [{"quote": "'
        + QUOTE
        + '", "supports": "final_decision",},],'
        ' "reason": "clear", "topic_tags": [], "review_priority": "medium",}'
    )
    assert parsed.decision == "relevant"
    assert parsed.evidence[0].quote == QUOTE


def test_short_two_word_quote_accepted_and_single_keyword_rejected(tmp_path) -> None:
    database = _database(tmp_path)
    recorder = CallRecorder(
        _primary(
            evidence=[
                EvidenceItem(quote="teacher incentives", supports="final_decision")
            ]
        )
    )
    outcome = _service(database, recorder).screen_paper(1)
    assert recorder.review_calls == 0
    assert outcome.status == "classified"

    recorder2 = CallRecorder(
        _primary(
            evidence=[EvidenceItem(quote="teacher", supports="final_decision")]
        ),
        _review(),
    )
    outcome2 = _service(database, recorder2).screen_paper(1, force=True)
    assert recorder2.review_calls == 1
    assert outcome2.validation_status == "needs_recheck"


def test_prompt_version_upgrade_invalidates_reuse(tmp_path) -> None:
    database = _database(tmp_path)
    recorder = CallRecorder(_primary())
    service = _service(database, recorder)
    service.screen_paper(1)
    with database.connect() as connection:
        connection.execute(
            "UPDATE paper_screenings SET prompt_version='legacy-v0' WHERE paper_id=1"
        )
    outcome = service.screen_paper(1)
    assert recorder.primary_calls == 2
    assert outcome.reused is False
    with database.connect() as connection:
        rows = connection.execute(
            "SELECT prompt_version,stale FROM paper_screenings ORDER BY id"
        ).fetchall()
    assert [tuple(row) for row in rows] == [
        ("legacy-v0", 1),
        ("2026-07-31-screening-v1", 0),
    ]
