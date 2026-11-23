from eduecon_radar.db import Database, utcnow
from eduecon_radar.schemas import JournalConfig, PaperMetadata
from eduecon_radar.services.ai_screening import ScreeningService
from eduecon_radar.services.web_screening import (
    ScreeningParameters,
    claim_screening_run,
    finish_screening_run,
    screening_preview,
    screening_preview_is_valid,
)


def _database(tmp_path):
    db = Database(tmp_path / "screening.db")
    db.init_schema()
    journal_id = db.upsert_journal(
        JournalConfig(name="Journal", short_name="J", issn="0000-0000")
    )
    awaiting, _ = db.upsert_paper(PaperMetadata(title="Awaiting", abstract="school"), journal_id)
    failed, _ = db.upsert_paper(PaperMetadata(title="Failed", abstract="school"), journal_id)
    with db.connect() as connection:
        connection.execute(
            """INSERT INTO screenings
            (paper_id,ai_decision,ai_reason,model_name,prompt_version,screened_at)
            VALUES (?,'unprocessed','validation_error','fake','old',?)""",
            (failed, utcnow()),
        )
    return db, awaiting, failed


def test_screening_parameters_are_deterministic_and_bounded():
    assert ScreeningParameters().signature == ("awaiting_screening", 10, True)


def test_preview_selects_awaiting_or_failed_without_ai(tmp_path):
    db, awaiting, failed = _database(tmp_path)

    awaiting_preview = screening_preview(
        db, ScreeningParameters(source="awaiting_screening", limit=10)
    )
    failed_preview = screening_preview(
        db, ScreeningParameters(source="screening_failed", limit=10)
    )

    assert [row["id"] for row in awaiting_preview["candidates"]] == [awaiting]
    assert [row["id"] for row in failed_preview["candidates"]] == [failed]
    assert failed_preview["retry_failed"] is True


def test_preview_signature_and_confirmation_are_invalidated_by_parameter_changes():
    parameters = ScreeningParameters()
    state = {
        "screen_preview_result": {"planned_count": 1},
        "screen_preview_signature": parameters.signature,
        "screen_confirm": True,
    }
    assert screening_preview_is_valid(state, parameters.signature)
    assert not screening_preview_is_valid(
        state, ScreeningParameters(limit=11).signature
    )


def test_screening_run_can_only_be_claimed_once():
    parameters = ScreeningParameters()
    state = {
        "screen_preview_result": {"planned_count": 1},
        "screen_preview_signature": parameters.signature,
        "screen_confirm": True,
    }

    assert claim_screening_run(state, parameters.signature)
    finish_screening_run(state)
    state["screen_confirm"] = True
    state["screen_preview_signature"] = parameters.signature
    assert not claim_screening_run(state, parameters.signature)


def test_failed_source_does_not_process_awaiting_and_preserves_human_review(tmp_path):
    db, awaiting, failed = _database(tmp_path)
    with db.connect() as connection:
        connection.execute(
            """UPDATE reviews SET status='approved',reviewer_note='keep',
            edited_chinese_title='keep title',edited_chinese_summary='keep summary'
            WHERE paper_id=?""",
            (failed,),
        )
    service = ScreeningService(db, tmp_path, None, "fast", "strong")

    stats = service.run(limit=10, no_ai=True, source="screening_failed")

    assert stats.considered == 1
    with db.connect() as connection:
        assert connection.execute(
            "SELECT COUNT(*) FROM screenings WHERE paper_id=?", (awaiting,)
        ).fetchone()[0] == 0
        review = connection.execute(
            """SELECT status,reviewer_note,edited_chinese_title,edited_chinese_summary
            FROM reviews WHERE paper_id=?""",
            (failed,),
        ).fetchone()
    assert tuple(review) == ("approved", "keep", "keep title", "keep summary")
