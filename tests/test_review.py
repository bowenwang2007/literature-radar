import pytest

from eduecon_radar.db import Database
from eduecon_radar.schemas import JournalConfig, PaperMetadata
from eduecon_radar.services.review import (
    PaperFilters,
    dashboard_stats,
    get_paper_detail,
    list_all_papers,
    save_review,
    set_review_status,
)


def seeded_db(tmp_path):
    db = Database(tmp_path / "radar.db")
    db.init_schema()
    journal_id = db.upsert_journal(JournalConfig(name="Journal", short_name="J", issn="0000-0000"))
    paper_id, _ = db.upsert_paper(
        PaperMetadata(title="Teacher effects", abstract="Education research"), journal_id
    )
    return db, paper_id


def add_screening(
    db: Database,
    paper_id: int,
    decision: str,
    *,
    version: str = "v1",
    reason: str | None = None,
    screened_at: str = "2026-07-19T01:00:00+00:00",
) -> None:
    with db.connect() as connection:
        connection.execute(
            """INSERT INTO screenings
            (paper_id,ai_decision,ai_reason,model_name,prompt_version,screened_at)
            VALUES (?,?,?,?,?,?)""",
            (paper_id, decision, reason, "test-model", version, screened_at),
        )


def test_approval_requires_both_verifications(tmp_path):
    db, paper_id = seeded_db(tmp_path)
    with pytest.raises(ValueError, match="批准前"):
        save_review(
            db,
            paper_id,
            status="approved",
            note="",
            chinese_title="标题",
            chinese_summary="摘要",
            verified_numbers=True,
            verified_causal_language=False,
        )


def test_manual_edits_are_saved_and_returned(tmp_path):
    db, paper_id = seeded_db(tmp_path)
    save_review(
        db,
        paper_id,
        status="approved",
        note="checked",
        chinese_title="人工标题",
        chinese_summary="人工摘要",
        verified_numbers=True,
        verified_causal_language=True,
    )
    detail = get_paper_detail(db, paper_id)
    assert detail["edited_chinese_title"] == "人工标题"
    assert detail["edited_chinese_summary"] == "人工摘要"
    assert detail["status"] == "approved"


def test_dashboard_stats_handles_empty_database(tmp_path):
    db = Database(tmp_path / "empty.db")
    db.init_schema()
    stats = dashboard_stats(db)
    expected = {
        "awaiting_screening": 0,
        "screening_failed": 0,
        "ai_excluded": 0,
        "ai_included": 0,
        "ai_borderline": 0,
        "review_pending": 0,
        "approved": 0,
        "rejected": 0,
        "needs_revision": 0,
        "total": 0,
    }
    assert {key: stats[key] for key in expected} == expected
    assert stats["screening_failure_reasons"] == {}
    assert "pending" not in stats


def test_dashboard_stats_separates_latest_ai_and_review_statuses(tmp_path):
    db = Database(tmp_path / "stats.db")
    db.init_schema()
    journal_id = db.upsert_journal(
        JournalConfig(name="Journal", short_name="J", issn="0000-0000")
    )

    def add_paper(name: str, review_status: str = "pending") -> int:
        paper_id, _ = db.upsert_paper(PaperMetadata(title=name), journal_id)
        if review_status != "pending":
            with db.connect() as connection:
                connection.execute(
                    "UPDATE reviews SET status=? WHERE paper_id=?",
                    (review_status, paper_id),
                )
        return paper_id

    def add_screening(paper_id: int, decision: str, version: str, screened_at: str) -> None:
        with db.connect() as connection:
            connection.execute(
                """INSERT INTO screenings
                (paper_id,ai_decision,model_name,prompt_version,screened_at)
                VALUES (?,?,?,?,?)""",
                (paper_id, decision, "test-model", version, screened_at),
            )

    add_paper("Awaiting screening")
    failed = add_paper("Failed screening")
    excluded = add_paper("Excluded pending")
    included = add_paper("Included pending")
    borderline = add_paper("Borderline pending")
    approved = add_paper("Included approved", "approved")
    rejected = add_paper("Excluded rejected", "rejected")
    revision = add_paper("Borderline revision", "needs_revision")

    add_screening(failed, "unprocessed", "v1", "2026-07-18T01:00:00+00:00")
    add_screening(excluded, "exclude", "v1", "2026-07-18T01:00:00+00:00")
    add_screening(included, "exclude", "v1", "2026-07-18T00:00:00+00:00")
    add_screening(included, "include", "v2", "2026-07-18T01:00:00+00:00")
    add_screening(borderline, "borderline", "v1", "2026-07-18T01:00:00+00:00")
    add_screening(approved, "include", "v1", "2026-07-18T01:00:00+00:00")
    add_screening(rejected, "exclude", "v1", "2026-07-18T01:00:00+00:00")
    add_screening(revision, "borderline", "v1", "2026-07-18T01:00:00+00:00")

    run_id = db.start_sync_run()
    db.finish_sync_run(
        run_id,
        status="success",
        stats={"records_inserted": 8, "ai_calls": 7, "input_tokens": 100},
    )

    stats = dashboard_stats(db)

    assert stats["awaiting_screening"] == 1
    assert stats["screening_failed"] == 1
    assert stats["screening_failure_reasons"] == {"unknown": 1}
    assert stats["ai_excluded"] == 2
    assert stats["ai_included"] == 2
    assert stats["ai_borderline"] == 2
    assert sum(
        stats[key]
        for key in (
            "awaiting_screening",
            "screening_failed",
            "ai_excluded",
            "ai_included",
            "ai_borderline",
        )
    ) == stats["total"] == 8
    assert stats["review_pending"] == 2
    assert stats["approved"] == 1
    assert stats["rejected"] == 1
    assert stats["needs_revision"] == 1
    assert stats["records_inserted"] == 8
    assert stats["ai_calls"] == 7
    assert stats["input_tokens"] == 100
    assert "pending" not in stats


def test_all_papers_query_does_not_omit_unscreened_or_reviewed_papers(tmp_path):
    db = Database(tmp_path / "all-papers.db")
    db.init_schema()
    journal_id = db.upsert_journal(
        JournalConfig(name="Journal", short_name="J", issn="0000-0000")
    )
    paper_ids = []
    for index in range(5):
        paper_id, _ = db.upsert_paper(PaperMetadata(title=f"Paper {index}"), journal_id)
        paper_ids.append(paper_id)
    add_screening(db, paper_ids[1], "unprocessed")
    add_screening(db, paper_ids[2], "exclude")
    add_screening(db, paper_ids[3], "include")
    add_screening(db, paper_ids[4], "borderline")
    set_review_status(db, paper_ids[3], "approved")

    result = list_all_papers(db, page_size=20)

    assert result.total == 5
    assert {item["id"] for item in result.items} == set(paper_ids)
    assert {item["ai_status"] for item in result.items} == {
        "awaiting_screening",
        "screening_failed",
        "ai_excluded",
        "ai_included",
        "ai_borderline",
    }


def test_ai_excluded_paper_remains_queryable(tmp_path):
    db, paper_id = seeded_db(tmp_path)
    add_screening(db, paper_id, "include", version="old", screened_at="2026-07-18T01:00:00+00:00")
    add_screening(db, paper_id, "exclude", version="new")

    result = list_all_papers(
        db,
        PaperFilters(ai_statuses=("ai_excluded",)),
    )

    assert result.total == 1
    assert result.items[0]["id"] == paper_id
    assert result.items[0]["ai_decision"] == "exclude"


def test_human_approval_does_not_change_screening(tmp_path):
    db, paper_id = seeded_db(tmp_path)
    add_screening(db, paper_id, "exclude", reason="AI exclusion")
    with db.connect() as connection:
        before = dict(connection.execute("SELECT * FROM screenings").fetchone())

    set_review_status(db, paper_id, "approved")

    with db.connect() as connection:
        after = dict(connection.execute("SELECT * FROM screenings").fetchone())
        review = connection.execute(
            "SELECT status FROM reviews WHERE paper_id=?", (paper_id,)
        ).fetchone()
    assert after == before
    assert review["status"] == "approved"


def test_review_decision_saves_note_without_changing_screening(tmp_path):
    db, paper_id = seeded_db(tmp_path)
    add_screening(db, paper_id, "include", reason="AI inclusion")
    with db.connect() as connection:
        screening_before = dict(connection.execute("SELECT * FROM screenings").fetchone())

    set_review_status(db, paper_id, "approved", note="人工核验通过")

    with db.connect() as connection:
        screening_after = dict(connection.execute("SELECT * FROM screenings").fetchone())
        review = connection.execute(
            "SELECT status,reviewer_note FROM reviews WHERE paper_id=?", (paper_id,)
        ).fetchone()
    assert screening_after == screening_before
    assert dict(review) == {"status": "approved", "reviewer_note": "人工核验通过"}


def test_all_papers_can_filter_original_abstract_presence(tmp_path):
    db = Database(tmp_path / "abstract-filter.db")
    db.init_schema()
    journal_id = db.upsert_journal(
        JournalConfig(name="Journal", short_name="J", issn="0000-0000")
    )
    with_abstract, _ = db.upsert_paper(
        PaperMetadata(title="With abstract", abstract="Original abstract"), journal_id
    )
    without_abstract, _ = db.upsert_paper(
        PaperMetadata(title="Without abstract"), journal_id
    )

    present = list_all_papers(db, PaperFilters(has_abstract=True))
    absent = list_all_papers(db, PaperFilters(has_abstract=False))

    assert [item["id"] for item in present.items] == [with_abstract]
    assert [item["id"] for item in absent.items] == [without_abstract]


def test_all_papers_pagination_is_complete_and_non_overlapping(tmp_path):
    db = Database(tmp_path / "pagination.db")
    db.init_schema()
    journal_id = db.upsert_journal(
        JournalConfig(name="Journal", short_name="J", issn="0000-0000")
    )
    expected_ids = set()
    for index in range(45):
        paper_id, _ = db.upsert_paper(PaperMetadata(title=f"Paged paper {index:02d}"), journal_id)
        expected_ids.add(paper_id)

    pages = [list_all_papers(db, page=page, page_size=20) for page in (1, 2, 3)]
    page_ids = [{item["id"] for item in result.items} for result in pages]

    assert [len(items) for items in page_ids] == [20, 20, 5]
    assert pages[0].total == pages[1].total == pages[2].total == 45
    assert not (page_ids[0] & page_ids[1] or page_ids[0] & page_ids[2] or page_ids[1] & page_ids[2])
    assert set().union(*page_ids) == expected_ids


def test_screening_failure_reasons_use_only_explicit_evidence(tmp_path):
    db = Database(tmp_path / "failure-reasons.db")
    db.init_schema()
    journal_id = db.upsert_journal(
        JournalConfig(name="Journal", short_name="J", issn="0000-0000")
    )
    reasons = [
        "DeepSeek request failed: Invalid structured AI response: 1 validation error",
        "No abstract available from Crossref",
        "HTTP error: status code 500",
        "AI disabled; metadata retained for later screening",
        "Unexpected response",
    ]
    for index, reason in enumerate(reasons):
        paper_id, _ = db.upsert_paper(PaperMetadata(title=f"Failure {index}"), journal_id)
        add_screening(db, paper_id, "unprocessed", reason=reason)

    stats = dashboard_stats(db)

    assert stats["screening_failed"] == 5
    assert stats["screening_failure_reasons"] == {
        "ai_disabled": 1,
        "api_error": 1,
        "missing_abstract": 1,
        "unknown": 1,
        "validation_error": 1,
    }


def test_source_metadata_is_not_replaced_by_ai_fields(tmp_path):
    db, paper_id = seeded_db(tmp_path)
    with db.connect() as connection:
        connection.execute(
            "UPDATE papers SET title=?,abstract=?,doi=? WHERE id=?",
            ("Original title", "Original Crossref abstract", "10.1000/original", paper_id),
        )
    add_screening(
        db,
        paper_id,
        "exclude",
        reason="AI invented title and AI invented abstract",
    )

    item = list_all_papers(db).items[0]

    assert item["title"] == "Original title"
    assert item["abstract"] == "Original Crossref abstract"
    assert item["doi"] == "10.1000/original"
    assert item["ai_reason"] == "AI invented title and AI invented abstract"
    assert item["ai_reason"] not in item["abstract"]


def test_reject_changes_only_review_record(tmp_path):
    db, paper_id = seeded_db(tmp_path)
    add_screening(db, paper_id, "include", reason="AI reason")
    with db.connect() as connection:
        paper_before = dict(connection.execute("SELECT * FROM papers").fetchone())
        screening_before = dict(connection.execute("SELECT * FROM screenings").fetchone())

    set_review_status(db, paper_id, "rejected")

    with db.connect() as connection:
        paper_after = dict(connection.execute("SELECT * FROM papers").fetchone())
        screening_after = dict(connection.execute("SELECT * FROM screenings").fetchone())
    assert paper_after == paper_before
    assert screening_after == screening_before


def test_needs_revision_does_not_change_ai_assessment(tmp_path):
    db, paper_id = seeded_db(tmp_path)
    add_screening(db, paper_id, "borderline", reason="AI reason")
    with db.connect() as connection:
        before = dict(connection.execute("SELECT * FROM screenings").fetchone())

    set_review_status(db, paper_id, "needs_revision")

    with db.connect() as connection:
        after = dict(connection.execute("SELECT * FROM screenings").fetchone())
    assert after == before


def test_review_queue_filters_keep_only_pending_include_and_borderline(tmp_path):
    db = Database(tmp_path / "review-queue.db")
    db.init_schema()
    journal_id = db.upsert_journal(
        JournalConfig(name="Journal", short_name="J", issn="0000-0000")
    )
    paper_ids = {}
    for decision in ("include", "borderline", "exclude"):
        paper_id, _ = db.upsert_paper(PaperMetadata(title=f"Queue {decision}"), journal_id)
        paper_ids[decision] = paper_id
        add_screening(db, paper_id, decision)
    approved_id, _ = db.upsert_paper(PaperMetadata(title="Approved include"), journal_id)
    add_screening(db, approved_id, "include")
    set_review_status(db, approved_id, "approved")

    result = list_all_papers(
        db,
        PaperFilters(
            ai_statuses=("ai_included", "ai_borderline"),
            review_statuses=("pending",),
        ),
    )

    assert result.total == 2
    assert {item["id"] for item in result.items} == {
        paper_ids["include"],
        paper_ids["borderline"],
    }
