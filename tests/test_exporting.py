import csv
import io
import sqlite3
from datetime import date
from pathlib import Path

from eduecon_radar.db import Database, utcnow
from eduecon_radar.schemas import JournalConfig, PaperMetadata
from eduecon_radar.services.exporting import (
    MentorExportRow,
    PersonalResearchExportRow,
    export_csv_text,
    export_markdown,
    export_mentor_csv_text,
    export_mentor_markdown,
    export_personal_csv_text,
    export_personal_markdown,
    mentor_records,
    personal_research_records,
)
from eduecon_radar.services.migrations import (
    apply_translation_migration,
    apply_workflow_migration,
)
from eduecon_radar.services.review import save_review
from eduecon_radar.services.workflow import create_workflow_batch

ROOT = Path(__file__).resolve().parents[1]


def _database(tmp_path) -> tuple[Database, int]:
    db = Database(tmp_path / "radar.db")
    db.init_schema()
    apply_translation_migration(db, ROOT / "migrations" / "0001_add_paper_translations.sql")
    journal_id = db.upsert_journal(
        JournalConfig(name="Journal", short_name="J", issn="0000-0000")
    )
    return db, journal_id


def _approved_record(tmp_path, *, abstract: str | None = "Original Crossref abstract"):
    db, journal_id = _database(tmp_path)
    paper_id, _ = db.upsert_paper(
        PaperMetadata(
            doi="10.1000/export",
            title="Original English Title",
            authors=["A Author"],
            issue_date=date(2026, 1, 2),
            abstract=abstract,
            source_url="https://example.test/source",
            open_access_url="https://example.test/open",
        ),
        journal_id,
    )
    with db.connect() as connection:
        connection.execute(
            """INSERT INTO screenings
            (paper_id,ai_decision,ai_confidence,ai_reason,research_question,data_and_sample,
             identification_strategy,main_result,chinese_title,chinese_summary,topics_json,
             model_name,prompt_version,screened_at)
            VALUES (?,?,?,?,?,?,?,?,?,?,?,?,?,?)""",
            (
                paper_id,
                "include",
                0.9,
                "OLD AI REASON",
                "OLD AI RESEARCH QUESTION",
                "OLD AI DATA",
                "OLD AI STRATEGY",
                "OLD AI MAIN RESULT",
                "OLD AI TITLE",
                "OLD AI SUMMARY",
                '["OLD AI TOPIC"]',
                "fake-model",
                "old-prompt",
                utcnow(),
            ),
        )
    save_review(
        db,
        paper_id,
        status="approved",
        note="Human note",
        chinese_title="OLD HUMAN PERSONAL TITLE",
        chinese_summary="OLD HUMAN PERSONAL SUMMARY",
        verified_numbers=True,
        verified_causal_language=True,
    )
    with db.connect() as connection:
        connection.execute(
            """INSERT INTO paper_translations
            (paper_id,chinese_title,chinese_abstract,title_translation_status,
             abstract_translation_status,title_translation_source,abstract_translation_source,
             model_name,prompt_version,translated_at,updated_at)
            VALUES (?,?,?,?,?,?,?,?,?,?,?)""",
            (
                paper_id,
                "正式中文标题翻译",
                "正式中文摘要翻译" if abstract else None,
                "human_edited",
                "human_edited" if abstract else "unavailable",
                "human_edited",
                "human_edited" if abstract else None,
                "fake-model",
                "translation-v1",
                utcnow(),
                utcnow(),
            ),
        )
    return db, journal_id, paper_id


def test_mentor_export_object_has_only_source_review_and_translation_fields(tmp_path):
    db, _, _ = _approved_record(tmp_path)
    row = mentor_records(db)[0]

    assert isinstance(row, MentorExportRow)
    assert row.original_title == "Original English Title"
    assert row.original_abstract == "Original Crossref abstract"
    assert row.chinese_title == "正式中文标题翻译"
    assert row.chinese_abstract == "正式中文摘要翻译"
    forbidden = {
        "chinese_summary",
        "research_question",
        "data_and_sample",
        "identification_strategy",
        "main_result",
        "topics",
        "ai_reason",
        "ai_score",
    }
    assert forbidden.isdisjoint(MentorExportRow.__dataclass_fields__)


def test_mentor_files_contain_five_core_fields_and_no_old_ai_analysis(tmp_path):
    db, _, _ = _approved_record(tmp_path)
    markdown = export_mentor_markdown(db)
    csv_text = export_mentor_csv_text(db)

    for expected in (
        "Original English Title",
        "正式中文标题翻译",
        "A Author",
        "Original Crossref abstract",
        "正式中文摘要翻译",
    ):
        assert expected in markdown
        assert expected in csv_text
    for forbidden in (
        "OLD AI SUMMARY",
        "OLD AI RESEARCH QUESTION",
        "OLD AI STRATEGY",
        "OLD AI MAIN RESULT",
        "OLD AI TOPIC",
        "OLD HUMAN PERSONAL SUMMARY",
    ):
        assert forbidden not in markdown
        assert forbidden not in csv_text


def test_mentor_csv_has_explicit_original_and_translation_columns(tmp_path):
    db, _, _ = _approved_record(tmp_path)
    rows = list(csv.DictReader(io.StringIO(export_mentor_csv_text(db).lstrip("\ufeff"))))

    assert list(rows[0]) == [
        "id",
        "doi",
        "original_title",
        "chinese_title",
        "authors",
        "journal",
        "publication_date",
        "original_abstract",
        "chinese_abstract",
        "source_url",
        "open_access_url",
        "title_translation_source",
        "abstract_translation_source",
    ]
    assert "summary" not in rows[0]


def test_missing_original_abstract_never_falls_back_to_old_summaries(tmp_path):
    db, _, _ = _approved_record(tmp_path, abstract=None)
    row = mentor_records(db)[0]

    assert row.original_abstract is None
    assert row.chinese_abstract is None
    markdown = export_mentor_markdown(db)
    csv_text = export_mentor_csv_text(db)
    assert "Original abstract unavailable in current metadata." in markdown
    assert "OLD AI SUMMARY" not in markdown + csv_text
    assert "OLD HUMAN PERSONAL SUMMARY" not in markdown + csv_text


def test_human_note_is_separate_and_opt_in(tmp_path):
    db, _, _ = _approved_record(tmp_path)

    default_csv = export_mentor_csv_text(db)
    opted_in_csv = export_mentor_csv_text(db, include_human_note=True)
    assert "human_note" not in default_csv.splitlines()[0]
    assert "Human note" not in default_csv
    assert "human_note" in opted_in_csv.splitlines()[0]
    assert "Human note" in opted_in_csv


def test_personal_export_keeps_old_ai_content_with_explicit_provenance(tmp_path):
    db, _, _ = _approved_record(tmp_path)
    row = personal_research_records(db)[0]

    assert isinstance(row, PersonalResearchExportRow)
    assert row.ai_chinese_summary == "OLD AI SUMMARY"
    assert row.human_edited_chinese_summary == "OLD HUMAN PERSONAL SUMMARY"
    markdown = export_personal_markdown(db)
    csv_text = export_personal_csv_text(db)
    assert "不是论文原始摘要或其正式翻译" in markdown
    assert "ai_chinese_summary" in csv_text.splitlines()[0]
    assert "OLD AI SUMMARY" in markdown + csv_text


def test_topic_filter_only_changes_mentor_row_selection(tmp_path):
    db, _, _ = _approved_record(tmp_path)

    assert len(mentor_records(db, topic="OLD AI TOPIC")) == 1
    assert mentor_records(db, topic="not-present") == []
    assert "ai_topics" not in MentorExportRow.__dataclass_fields__


def test_default_legacy_function_names_are_now_safe_mentor_exports(tmp_path):
    db, _, _ = _approved_record(tmp_path)

    assert "Original Crossref abstract" in export_markdown(db)
    assert "original_abstract" in export_csv_text(db).splitlines()[0]
    assert "OLD AI SUMMARY" not in export_markdown(db) + export_csv_text(db)


def test_mentor_filters_status_journal_date_and_abstract(tmp_path):
    db, _, _ = _approved_record(tmp_path)

    assert len(mentor_records(db, journal="J")) == 1
    assert mentor_records(db, journal="OTHER") == []
    assert len(mentor_records(db, date_from=date(2026, 1, 1), has_abstract=True)) == 1
    assert mentor_records(db, date_to=date(2025, 1, 1)) == []


def test_mentor_export_can_filter_exact_workflow_batch_without_ai_leakage(tmp_path):
    db, journal_id, paper_id = _approved_record(tmp_path)
    outside_id, _ = db.upsert_paper(PaperMetadata(title="Approved outside batch"), journal_id)
    with db.connect() as connection:
        connection.execute(
            "UPDATE reviews SET status='approved' WHERE paper_id=?", (outside_id,)
        )
    apply_workflow_migration(
        db, ROOT / "migrations" / "0002_add_workflow_batches.sql"
    )
    batch_id = create_workflow_batch(
        db,
        name="Export batch",
        source_type="selected_papers",
        paper_ids=[paper_id],
    )

    records = mentor_records(db, batch_id=batch_id)
    markdown = export_mentor_markdown(db, batch_id=batch_id)

    assert [item.id for item in records] == [paper_id]
    assert "Original English Title" in markdown
    assert "Approved outside batch" not in markdown
    assert "OLD AI SUMMARY" not in markdown


def test_mentor_query_does_not_require_translation_migration_for_safe_empty_translation(tmp_path):
    db = Database(tmp_path / "legacy.db")
    db.init_schema()
    journal_id = db.upsert_journal(
        JournalConfig(name="Journal", short_name="J", issn="0000-0000")
    )
    paper_id, _ = db.upsert_paper(PaperMetadata(title="Legacy title"), journal_id)
    with db.connect() as connection:
        connection.execute("UPDATE reviews SET status='approved' WHERE paper_id=?", (paper_id,))

    row = mentor_records(db)[0]
    assert row.chinese_title is None
    assert row.chinese_abstract is None
    with sqlite3.connect(db.path) as connection:
        assert connection.execute(
            "SELECT 1 FROM sqlite_master WHERE name='paper_translations'"
        ).fetchone() is None
