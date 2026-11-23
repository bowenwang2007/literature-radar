from __future__ import annotations

import json
from datetime import date
from pathlib import Path

from eduecon_radar.db import Database
from eduecon_radar.schemas import PaperMetadata
from eduecon_radar.services.batch_translation import (
    BatchTitleItem,
    BatchTitleTranslationOutput,
    BatchTitleTranslationService,
)
from eduecon_radar.services.deepseek import DeepSeekError, DeepSeekResponse
from eduecon_radar.services.migrations import (
    apply_overnight_migration,
    apply_p0_hardening_migration,
    apply_profile_migration,
    apply_translation_migration,
    apply_workflow_migration,
)
from eduecon_radar.services.profile_registry import default_profile_registry

ROOT = Path(__file__).resolve().parents[1]


def _database(tmp_path: Path) -> Database:
    database = Database(tmp_path / "batch_title.db")
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
    ids = []
    for index in range(1, 11):
        paper_id, _ = database.upsert_paper(
            PaperMetadata(
                doi=f"10.1000/batch-{index}",
                title=f"Education policy and student outcomes number {index}",
                abstract=None,
                authors=["Researcher"],
                issue_date=date(2026, 7, index),
                source_metadata={"DOI": f"10.1000/batch-{index}"},
            ),
            journal_id,
        )
        ids.append(paper_id)
    apply_profile_migration(
        database, ROOT / "migrations/0013_add_profile_scoped_assessment.sql"
    )
    return database


class FakeClient:
    def __init__(
        self,
        *,
        fail_all: bool = False,
        fail_ids: set[int] | None = None,
        omit_ids: set[int] | None = None,
        copy_ids: set[int] | None = None,
        english_ids: set[int] | None = None,
    ) -> None:
        self.fail_all = fail_all
        self.fail_ids = fail_ids or set()
        self.omit_ids = omit_ids or set()
        self.copy_ids = copy_ids or set()
        self.english_ids = english_ids or set()
        self.calls = 0

    def request_json(self, **kwargs):
        self.calls += 1
        if self.fail_all:
            raise DeepSeekError("fixture provider failure")
        user = kwargs["user_prompt"]
        raw = user.split("ITEMS_START\n", 1)[1].split("\nITEMS_END", 1)[0]
        items = json.loads(raw)
        entries = []
        for item in items:
            paper_id = int(item["paper_id"])
            if paper_id in self.fail_ids:
                entries.append(
                    BatchTitleItem(
                        paper_id=paper_id,
                        validation_status="error",
                        error_category="translation_failed",
                        error_message="fixture item failure",
                    )
                )
                continue
            if paper_id in self.omit_ids:
                continue
            if paper_id in self.copy_ids:
                chinese_title = item["title"]
            elif paper_id in self.english_ids:
                chinese_title = "English only output"
            else:
                chinese_title = f"教育研究标题{paper_id}"
            entries.append(
                BatchTitleItem(
                    paper_id=paper_id,
                    chinese_title=chinese_title,
                )
            )
        return DeepSeekResponse(
            data=BatchTitleTranslationOutput(translations=entries),
            model_name="fixture-batch",
            input_tokens=10,
            output_tokens=10,
        )


def _ids(database: Database) -> tuple[int, ...]:
    with database.connect() as connection:
        return tuple(
            int(row[0])
            for row in connection.execute("SELECT id FROM papers ORDER BY id").fetchall()
        )


def _service(database: Database, client: FakeClient) -> BatchTitleTranslationService:
    return BatchTitleTranslationService(
        database, ROOT / "prompts", client, "fixture-model"
    )


def _row(database: Database, paper_id: int) -> dict:
    with database.connect() as connection:
        row = connection.execute(
            "SELECT * FROM paper_translations WHERE paper_id=?", (paper_id,)
        ).fetchone()
    return dict(row)


def test_batch_translates_ten_titles_in_one_call_with_hashes(tmp_path) -> None:
    database = _database(tmp_path)
    ids = _ids(database)
    client = FakeClient()
    result = _service(database, client).run(ids)
    assert client.calls == 1
    assert result.translated == 10
    assert result.failed == 0
    row = _row(database, ids[0])
    assert row["chinese_title"] == "教育研究标题" + str(ids[0])
    assert row["title_translation_status"] == "ai_generated"
    assert row["validation_status"] == "ok"
    assert row["source_title_hash"]


def test_batch_partial_item_failure_isolates_and_keeps_others(tmp_path) -> None:
    database = _database(tmp_path)
    ids = _ids(database)
    client = FakeClient(fail_ids={ids[2]})
    result = _service(database, client).run(ids)
    assert client.calls == 1
    assert result.translated == 9
    assert result.failed == 1
    assert _row(database, ids[0])["chinese_title"]
    failed = _row(database, ids[2])
    assert failed["chinese_title"] is None
    assert failed["validation_status"] == "error"
    assert failed["error_category"] == "translation_failed"


def test_batch_call_failure_splits_into_smaller_batches(tmp_path) -> None:
    database = _database(tmp_path)
    ids = _ids(database)[:2]

    class FailOnceClient(FakeClient):
        def request_json(self, **kwargs):
            if self.calls == 0:
                self.calls += 1
                raise DeepSeekError("fixture provider failure")
            return super().request_json(**kwargs)

    client = FailOnceClient()
    result = _service(database, client).run(ids)
    assert client.calls == 3  # 1 failed batch of 2, then 2 single-item batches
    assert result.translated == 2
    assert result.failed == 0


def test_batch_missing_paper_id_retries_smaller_and_marks_failure(tmp_path) -> None:
    database = _database(tmp_path)
    ids = _ids(database)[:3]
    client = FakeClient(omit_ids={ids[1]})
    result = _service(database, client).run(ids)
    assert result.translated == 2
    assert result.failed == 1
    failed = _row(database, ids[1])
    assert failed["chinese_title"] is None
    assert failed["error_category"] == "batch_response_missing_paper_id"


def test_batch_preserves_manual_title_and_skips_unchanged(tmp_path) -> None:
    database = _database(tmp_path)
    ids = _ids(database)
    client = FakeClient()
    _service(database, client).run(ids)
    first_calls = client.calls
    unchanged = _service(database, client).run(ids)
    assert client.calls == first_calls
    assert unchanged.translated == 10
    assert unchanged.ai_calls == 0

    with database.connect() as connection:
        connection.execute(
            """UPDATE paper_translations SET chinese_title='人工标题',
            title_translation_status='human_edited',
            title_translation_source='human_edited' WHERE paper_id=?""",
            (ids[0],),
        )
    result = _service(database, client).run(ids)
    assert result.translated == 10
    assert _row(database, ids[0])["chinese_title"] == "人工标题"


def test_batch_rejects_copied_and_non_chinese_titles(tmp_path) -> None:
    database = _database(tmp_path)
    ids = _ids(database)[:2]
    client = FakeClient(copy_ids={ids[0]}, english_ids={ids[1]})
    result = _service(database, client).run(ids)
    assert result.translated == 0
    assert result.failed == 2
    assert "copied" in result.failure_reasons[0]
    assert _row(database, ids[0])["validation_status"] == "error"
    assert _row(database, ids[1])["validation_status"] == "error"


def test_batch_without_api_key_marks_pending(tmp_path) -> None:
    database = _database(tmp_path)
    ids = _ids(database)[:3]
    result = BatchTitleTranslationService(
        database, ROOT / "prompts", None, "fixture-model"
    ).run(ids)
    assert result.pending == 3
    assert result.ai_calls == 0
    assert _row(database, ids[0])["title_translation_status"] == "pending"
