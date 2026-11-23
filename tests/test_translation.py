import sqlite3
from pathlib import Path

import pytest

from eduecon_radar.db import Database, utcnow
from eduecon_radar.schemas import (
    AbstractTranslationResult,
    JournalConfig,
    PaperMetadata,
    TitleTranslationResult,
)
from eduecon_radar.services.deepseek import DeepSeekResponse
from eduecon_radar.services.migrations import (
    apply_translation_migration,
    translation_schema_ready,
)
from eduecon_radar.services.translation import (
    TranslationParameters,
    TranslationService,
    formal_translation_prefill,
    get_translation,
    preview_translations,
    save_human_translation,
)

ROOT = Path(__file__).resolve().parents[1]


class FakeTranslationClient:
    def __init__(self) -> None:
        self.calls: list[dict] = []

    def request_json(self, **kwargs):
        self.calls.append(kwargs)
        schema = kwargs["schema"]
        if schema is TitleTranslationResult:
            data = TitleTranslationResult(chinese_title="忠实标题翻译")
        elif schema is AbstractTranslationResult:
            data = AbstractTranslationResult(chinese_abstract="忠实摘要完整翻译")
        else:  # pragma: no cover - fails loudly if service crosses schemas
            raise AssertionError(f"Unexpected schema: {schema}")
        return DeepSeekResponse(data=data, model_name="fake", input_tokens=10, output_tokens=20)


def _database(tmp_path, *, abstracts=("Original abstract only", None)):
    db = Database(tmp_path / "translation.db")
    db.init_schema()
    journal_id = db.upsert_journal(
        JournalConfig(name="Journal", short_name="J", issn="0000-0000")
    )
    paper_ids = []
    for index, abstract in enumerate(abstracts):
        paper_id, _ = db.upsert_paper(
            PaperMetadata(
                title=f"Original title {index}",
                abstract=abstract,
                source_metadata={"abstract": abstract, "forbidden": "metadata only"},
            ),
            journal_id,
        )
        paper_ids.append(paper_id)
        with db.connect() as connection:
            connection.execute(
                "UPDATE reviews SET status='approved',reviewer_note='preserve me' WHERE paper_id=?",
                (paper_id,),
            )
            connection.execute(
                """INSERT INTO screenings
                (paper_id,ai_decision,ai_reason,research_question,main_result,chinese_summary,
                 model_name,prompt_version,screened_at)
                VALUES (?,?,?,?,?,?,?,?,?)""",
                (
                    paper_id,
                    "include",
                    "OLD SCREEN REASON",
                    "OLD RESEARCH QUESTION",
                    "OLD MAIN RESULT",
                    "OLD CHINESE SUMMARY",
                    "screen-model",
                    "screen-v1",
                    utcnow(),
                ),
            )
    apply_translation_migration(db, ROOT / "migrations" / "0001_add_paper_translations.sql")
    return db, paper_ids


def test_translation_migration_is_additive_and_preserves_old_rows(tmp_path):
    db = Database(tmp_path / "migration.db")
    db.init_schema()
    journal_id = db.upsert_journal(
        JournalConfig(name="Journal", short_name="J", issn="0000-0000")
    )
    paper_id, _ = db.upsert_paper(PaperMetadata(title="Keep me"), journal_id)
    with db.connect() as connection:
        before = {
            table: connection.execute(f"SELECT COUNT(*) FROM {table}").fetchone()[0]
            for table in ("papers", "reviews", "screenings")
        }

    apply_translation_migration(db, ROOT / "migrations" / "0001_add_paper_translations.sql")

    assert translation_schema_ready(db)
    with db.connect() as connection:
        after = {
            table: connection.execute(f"SELECT COUNT(*) FROM {table}").fetchone()[0]
            for table in ("papers", "reviews", "screenings")
        }
        assert connection.execute(
            "SELECT title FROM papers WHERE id=?", (paper_id,)
        ).fetchone()[0] == "Keep me"
        assert connection.execute("SELECT COUNT(*) FROM paper_translations").fetchone()[0] == 0
    assert before == after


def test_failed_migration_transaction_rolls_back(tmp_path):
    db = Database(tmp_path / "rollback.db")
    db.init_schema()
    broken = tmp_path / "broken.sql"
    broken.write_text(
        "BEGIN IMMEDIATE; CREATE TABLE should_rollback(id INTEGER); INVALID SQL; COMMIT;",
        encoding="utf-8",
    )

    with pytest.raises(sqlite3.OperationalError):
        apply_translation_migration(db, broken)

    with db.connect() as connection:
        assert connection.execute(
            "SELECT 1 FROM sqlite_master WHERE name='should_rollback'"
        ).fetchone() is None


def test_preview_is_read_only_and_does_not_call_ai(tmp_path):
    db, _ = _database(tmp_path)
    client = FakeTranslationClient()

    preview = preview_translations(db, TranslationParameters())

    assert preview.matching_total == 2
    assert preview.title_pending == 2
    assert preview.abstract_pending == 1
    assert preview.abstract_unavailable == 1
    assert client.calls == []
    with db.connect() as connection:
        assert connection.execute("SELECT COUNT(*) FROM paper_translations").fetchone()[0] == 0


def test_translation_inputs_only_original_title_and_original_abstract(tmp_path):
    db, _ = _database(tmp_path, abstracts=("ORIGINAL ABSTRACT INPUT",))
    client = FakeTranslationClient()
    service = TranslationService(db, ROOT / "prompts", client, "fake-model")

    stats = service.run(TranslationParameters())

    assert stats.ai_calls == 2
    assert stats.titles_translated == 1
    assert stats.abstracts_translated == 1
    title_call, abstract_call = client.calls
    assert "Original title 0" in title_call["user_prompt"]
    assert "ORIGINAL ABSTRACT INPUT" in abstract_call["user_prompt"]
    combined_prompts = "\n".join(call["user_prompt"] for call in client.calls)
    for forbidden in (
        "OLD SCREEN REASON",
        "OLD RESEARCH QUESTION",
        "OLD MAIN RESULT",
        "OLD CHINESE SUMMARY",
    ):
        assert forbidden not in combined_prompts


def test_missing_abstract_skips_abstract_ai_call_and_stores_unavailable(tmp_path):
    db, (paper_id,) = _database(tmp_path, abstracts=(None,))
    client = FakeTranslationClient()

    stats = TranslationService(db, ROOT / "prompts", client, "fake-model").run(
        TranslationParameters()
    )
    translation = get_translation(db, paper_id)

    assert stats.ai_calls == 1
    assert [call["schema"] for call in client.calls] == [TitleTranslationResult]
    assert translation["chinese_abstract"] is None
    assert translation["abstract_translation_status"] == "unavailable"
    assert translation["abstract_translation_source"] is None


def test_translation_does_not_modify_source_screening_or_review(tmp_path):
    db, (paper_id,) = _database(tmp_path, abstracts=("Source abstract",))
    client = FakeTranslationClient()
    with db.connect() as connection:
        before = dict(
            connection.execute(
                """SELECT p.title,p.abstract,s.ai_decision,s.ai_reason,s.chinese_summary,
                r.status,r.reviewer_note FROM papers p
                JOIN screenings s ON s.paper_id=p.id JOIN reviews r ON r.paper_id=p.id
                WHERE p.id=?""",
                (paper_id,),
            ).fetchone()
        )

    TranslationService(db, ROOT / "prompts", client, "fake-model").run(
        TranslationParameters()
    )

    with db.connect() as connection:
        after = dict(
            connection.execute(
                """SELECT p.title,p.abstract,s.ai_decision,s.ai_reason,s.chinese_summary,
                r.status,r.reviewer_note FROM papers p
                JOIN screenings s ON s.paper_id=p.id JOIN reviews r ON r.paper_id=p.id
                WHERE p.id=?""",
                (paper_id,),
            ).fetchone()
        )
    assert before == after


def test_human_translation_has_priority_and_is_not_overwritten_by_default(tmp_path):
    db, (paper_id,) = _database(tmp_path, abstracts=("Source abstract",))
    save_human_translation(
        db,
        paper_id,
        chinese_title="人工标题校订",
        chinese_abstract="人工摘要校订",
    )
    client = FakeTranslationClient()

    stats = TranslationService(db, ROOT / "prompts", client, "fake-model").run(
        TranslationParameters(missing_only=False)
    )
    translation = get_translation(db, paper_id)

    assert client.calls == []
    assert translation["chinese_title"] == "人工标题校订"
    assert translation["chinese_abstract"] == "人工摘要校订"
    assert translation["title_translation_source"] == "human_edited"
    assert translation["abstract_translation_source"] == "human_edited"
    assert stats.human_preserved == 1


def test_formal_translation_prefill_never_uses_legacy_or_screening_fields():
    values = {
        "formal_chinese_title": None,
        "formal_chinese_abstract": None,
        "title_translation_source": None,
        "abstract_translation_source": None,
        "title_translation_status": "pending",
        "abstract_translation_status": "pending",
        "edited_chinese_title": "旧人工标题",
        "edited_chinese_summary": "旧人工摘要",
        "ai_chinese_title": "旧AI标题",
        "ai_chinese_summary": "旧AI摘要",
    }

    assert formal_translation_prefill(values) == ("", "")


def test_formal_ai_translation_is_prefilled_for_human_correction():
    values = {
        "formal_chinese_title": "AI正式标题",
        "formal_chinese_abstract": "AI正式摘要",
        "title_translation_source": "ai_generated",
        "abstract_translation_source": "ai_generated",
        "title_translation_status": "ai_generated",
        "abstract_translation_status": "ai_generated",
    }

    assert formal_translation_prefill(values) == ("AI正式标题", "AI正式摘要")


def test_unchanged_human_correction_does_not_update_translation_row(tmp_path):
    db, (paper_id,) = _database(tmp_path, abstracts=("Source abstract",))
    TranslationService(db, ROOT / "prompts", FakeTranslationClient(), "fake-model").run(
        TranslationParameters()
    )
    before = get_translation(db, paper_id)

    result = save_human_translation(
        db,
        paper_id,
        chinese_title=before["chinese_title"],
        chinese_abstract=before["chinese_abstract"],
    )

    assert not result.updated
    assert get_translation(db, paper_id) == before


def test_title_only_correction_preserves_ai_abstract_provenance(tmp_path):
    db, (paper_id,) = _database(tmp_path, abstracts=("Source abstract",))
    TranslationService(db, ROOT / "prompts", FakeTranslationClient(), "fake-model").run(
        TranslationParameters()
    )

    result = save_human_translation(
        db,
        paper_id,
        chinese_title="人工修改标题",
        chinese_abstract="忠实摘要完整翻译",
    )
    translation = get_translation(db, paper_id)

    assert result.title_changed and not result.abstract_changed
    assert translation["title_translation_source"] == "human_edited"
    assert translation["title_translation_status"] == "human_edited"
    assert translation["abstract_translation_source"] == "ai_generated"
    assert translation["abstract_translation_status"] == "ai_generated"


def test_abstract_only_correction_preserves_ai_title_provenance(tmp_path):
    db, (paper_id,) = _database(tmp_path, abstracts=("Source abstract",))
    TranslationService(db, ROOT / "prompts", FakeTranslationClient(), "fake-model").run(
        TranslationParameters()
    )

    result = save_human_translation(
        db,
        paper_id,
        chinese_title="忠实标题翻译",
        chinese_abstract="人工修改摘要",
    )
    translation = get_translation(db, paper_id)

    assert not result.title_changed and result.abstract_changed
    assert translation["title_translation_source"] == "ai_generated"
    assert translation["title_translation_status"] == "ai_generated"
    assert translation["abstract_translation_source"] == "human_edited"
    assert translation["abstract_translation_status"] == "human_edited"


def test_explicit_overwrite_can_replace_human_translation(tmp_path):
    db, (paper_id,) = _database(tmp_path, abstracts=("Source abstract",))
    save_human_translation(
        db,
        paper_id,
        chinese_title="人工标题校订",
        chinese_abstract="人工摘要校订",
    )
    client = FakeTranslationClient()

    TranslationService(db, ROOT / "prompts", client, "fake-model").run(
        TranslationParameters(missing_only=False, overwrite_human=True)
    )
    translation = get_translation(db, paper_id)

    assert len(client.calls) == 2
    assert translation["chinese_title"] == "忠实标题翻译"
    assert translation["chinese_abstract"] == "忠实摘要完整翻译"
    assert translation["title_translation_source"] == "ai_generated"


def test_human_abstract_translation_is_rejected_without_original_abstract(tmp_path):
    db, (paper_id,) = _database(tmp_path, abstracts=(None,))

    with pytest.raises(ValueError, match="原始摘要缺失"):
        save_human_translation(
            db,
            paper_id,
            chinese_title="标题翻译",
            chinese_abstract="伪造摘要",
        )


def test_translation_table_enforces_one_current_row_per_paper(tmp_path):
    db, (paper_id,) = _database(tmp_path, abstracts=("Source abstract",))
    save_human_translation(db, paper_id, chinese_title="A", chinese_abstract="B")

    with pytest.raises(sqlite3.IntegrityError), db.connect() as connection:
        connection.execute(
            """INSERT INTO paper_translations
            (paper_id,title_translation_status,abstract_translation_status,updated_at)
            VALUES (?,'pending','pending',?)""",
            (paper_id, utcnow()),
        )
