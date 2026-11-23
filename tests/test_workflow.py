from __future__ import annotations

from pathlib import Path

from eduecon_radar.db import Database, utcnow
from eduecon_radar.journals import JournalFile
from eduecon_radar.schemas import (
    AbstractTranslationResult,
    AIScreeningResult,
    JournalConfig,
    PaperMetadata,
    SummaryResult,
    TitleTranslationResult,
)
from eduecon_radar.services.ai_screening import ScreeningService
from eduecon_radar.services.deepseek import DeepSeekError, DeepSeekResponse
from eduecon_radar.services.ingestion import SyncService
from eduecon_radar.services.migrations import (
    apply_translation_migration,
    apply_workflow_migration,
    workflow_schema_ready,
)
from eduecon_radar.services.review import PaperFilters, list_all_papers, set_review_status
from eduecon_radar.services.translation import TranslationService, get_translation
from eduecon_radar.services.web_workflow import (
    claim_workflow_run,
    finish_workflow_run,
    workflow_preview_is_valid,
)
from eduecon_radar.services.workflow import (
    WorkflowOrchestrator,
    WorkflowPolicy,
    add_batch_items,
    batch_paper_ids,
    complete_batch_if_reviewed,
    create_workflow_batch,
    get_workflow_batch,
)

ROOT = Path(__file__).resolve().parents[1]


class WorkflowAI:
    def __init__(
        self,
        *,
        fail_title: str | None = None,
        fail_screen: str | None = None,
        fail_abstract: bool = False,
    ) -> None:
        self.events: list[tuple[str, str]] = []
        self.fail_title = fail_title
        self.fail_screen = fail_screen
        self.fail_abstract = fail_abstract

    def request_json(self, *, model, system_prompt, user_prompt, schema):
        self.events.append((schema.__name__, user_prompt))
        if schema is TitleTranslationResult:
            if self.fail_title and self.fail_title in user_prompt:
                raise DeepSeekError("title translation failed")
            original = user_prompt.split("Original title:", 1)[-1].strip()
            data = TitleTranslationResult(chinese_title=f"正式标题：{original}")
        elif schema is AbstractTranslationResult:
            if self.fail_abstract:
                raise DeepSeekError("abstract translation failed")
            data = AbstractTranslationResult(chinese_abstract="正式摘要翻译")
        elif schema is AIScreeningResult:
            if self.fail_screen and self.fail_screen in user_prompt:
                raise DeepSeekError("screening failed")
            if "Exclude" in user_prompt:
                decision = "exclude"
            elif "Borderline" in user_prompt:
                decision = "borderline"
            else:
                decision = "include"
            data = AIScreeningResult(
                decision=decision,
                confidence=0.95,
                reason="Test workflow classification",
                topics=["education"],
            )
        elif schema is SummaryResult:
            data = SummaryResult(
                chinese_title="旧筛选标题，不是正式翻译",
                chinese_summary="旧筛选摘要，不是正式翻译",
            )
        else:  # pragma: no cover
            raise AssertionError(schema)
        return DeepSeekResponse(data=data, model_name=model, input_tokens=5, output_tokens=3)


def workflow_db(tmp_path):
    database = Database(tmp_path / "workflow.db")
    database.init_schema()
    apply_translation_migration(
        database, ROOT / "migrations" / "0001_add_paper_translations.sql"
    )
    apply_workflow_migration(
        database, ROOT / "migrations" / "0002_add_workflow_batches.sql"
    )
    journal_id = database.upsert_journal(
        JournalConfig(name="Journal", short_name="J", issn="0000-0000")
    )
    return database, journal_id


def add_paper(database, journal_id, title, *, abstract="Education school student study"):
    paper_id, _ = database.upsert_paper(
        PaperMetadata(title=title, abstract=abstract), journal_id
    )
    return paper_id


def orchestrator(database, ai):
    return WorkflowOrchestrator(
        database,
        screening_service=ScreeningService(
            database, ROOT / "prompts", ai, "fast", "strong"
        ),
        translation_service=TranslationService(
            database, ROOT / "prompts", ai, "translation"
        ),
        model_name="fast",
    )


class WorkflowCrossref:
    def __init__(self, papers):
        self.papers = papers
        self.calls = 0

    def fetch(self, issn, since, limit=None):
        self.calls += 1
        return self.papers[:limit] if limit else self.papers


def workflow_journals():
    return JournalFile.model_validate(
        {
            "ranking": {"name": "test", "frozen_at": "2026-01-01"},
            "journals": [
                {"name": "Journal", "short_name": "J", "issn": "0000-0000"}
            ],
        }
    )


def test_workflow_batch_is_persistent_and_rejects_duplicate_items(tmp_path):
    database, journal_id = workflow_db(tmp_path)
    paper_id = add_paper(database, journal_id, "School persistence")
    batch_id = create_workflow_batch(
        database,
        name="Persistent batch",
        source_type="selected_papers",
        paper_ids=[paper_id, paper_id],
    )

    assert workflow_schema_ready(database)
    assert batch_paper_ids(Database(database.path), batch_id) == [paper_id]
    assert add_batch_items(database, batch_id, [paper_id]) == 0
    assert get_workflow_batch(Database(database.path), batch_id)["item_count"] == 1


def test_workflow_migration_is_additive_and_does_not_copy_paper_content(tmp_path):
    database = Database(tmp_path / "migration.db")
    database.init_schema()
    journal_id = database.upsert_journal(
        JournalConfig(name="Journal", short_name="J", issn="0000-0000")
    )
    add_paper(database, journal_id, "School preserved paper")
    with database.connect() as connection:
        before = {
            table: connection.execute(f"SELECT COUNT(*) FROM {table}").fetchone()[0]
            for table in ("papers", "reviews", "screenings")
        }

    apply_workflow_migration(
        database, ROOT / "migrations" / "0002_add_workflow_batches.sql"
    )

    with database.connect() as connection:
        after = {
            table: connection.execute(f"SELECT COUNT(*) FROM {table}").fetchone()[0]
            for table in ("papers", "reviews", "screenings")
        }
        assert connection.execute("SELECT COUNT(*) FROM workflow_batches").fetchone()[0] == 0
        assert connection.execute("SELECT COUNT(*) FROM workflow_batch_items").fetchone()[0] == 0
    assert before == after


def test_all_batch_papers_get_titles_before_screening_and_only_relevant_get_abstracts(
    tmp_path,
):
    database, journal_id = workflow_db(tmp_path)
    ids = {
        name: add_paper(database, journal_id, f"School {name} paper")
        for name in ("Include", "Borderline", "Exclude")
    }
    failed_id = add_paper(database, journal_id, "School screening failed paper")
    with database.connect() as connection:
        connection.execute(
            """INSERT INTO screenings
            (paper_id,ai_decision,ai_reason,model_name,prompt_version,screened_at)
            VALUES (?,'unprocessed','old API failure','old','old',?)""",
            (failed_id, utcnow()),
        )
    batch_id = create_workflow_batch(
        database,
        name="All title states",
        source_type="selected_papers",
        paper_ids=[*ids.values(), failed_id],
    )
    ai = WorkflowAI()

    result = orchestrator(database, ai).run(
        batch_id, WorkflowPolicy(limit=20, allow_strong=False)
    )

    assert result.status == "ready_for_review"
    for paper_id in (*ids.values(), failed_id):
        translation = get_translation(database, paper_id)
        assert translation["chinese_title"].startswith("正式标题：")
        assert "旧筛选标题" not in translation["chinese_title"]
    assert get_translation(database, ids["Include"])["chinese_abstract"] == "正式摘要翻译"
    assert get_translation(database, ids["Borderline"])["chinese_abstract"] == "正式摘要翻译"
    assert get_translation(database, ids["Exclude"])["chinese_abstract"] is None
    assert get_translation(database, failed_id)["chinese_abstract"] is None
    event_names = [event[0] for event in ai.events]
    assert max(i for i, name in enumerate(event_names) if name == "TitleTranslationResult") < min(
        i for i, name in enumerate(event_names) if name == "AIScreeningResult"
    )


def test_existing_ai_and_human_titles_are_reused_without_overwrite(tmp_path):
    database, journal_id = workflow_db(tmp_path)
    ai_id = add_paper(database, journal_id, "School AI title")
    human_id = add_paper(database, journal_id, "School human title")
    with database.connect() as connection:
        connection.executemany(
            """INSERT INTO paper_translations
            (paper_id,chinese_title,title_translation_status,title_translation_source,
             abstract_translation_status,updated_at)
            VALUES (?,?,?,?,'pending',?)""",
            [
                (ai_id, "已有AI正式标题", "ai_generated", "ai_generated", utcnow()),
                (human_id, "人工正式标题", "human_edited", "human_edited", utcnow()),
            ],
        )
    batch_id = create_workflow_batch(
        database,
        name="Protected titles",
        source_type="selected_papers",
        paper_ids=[ai_id, human_id],
    )
    ai = WorkflowAI()

    orchestrator(database, ai).run(
        batch_id, WorkflowPolicy(limit=10, allow_strong=False)
    )

    title_events = [event for event in ai.events if event[0] == "TitleTranslationResult"]
    assert title_events == []
    assert get_translation(database, ai_id)["chinese_title"] == "已有AI正式标题"
    assert get_translation(database, human_id)["chinese_title"] == "人工正式标题"
    assert get_translation(database, human_id)["title_translation_source"] == "human_edited"


def test_title_failure_does_not_block_screening_or_abstract_translation(tmp_path):
    database, journal_id = workflow_db(tmp_path)
    paper_id = add_paper(database, journal_id, "School Fail title Include paper")
    batch_id = create_workflow_batch(
        database,
        name="Title failure",
        source_type="selected_papers",
        paper_ids=[paper_id],
    )
    ai = WorkflowAI(fail_title="Fail title")

    result = orchestrator(database, ai).run(
        batch_id, WorkflowPolicy(limit=10, allow_strong=False)
    )

    translation = get_translation(database, paper_id)
    assert result.status == "ready_for_review"
    assert translation["chinese_title"] is None
    assert translation["title_translation_status"] == "failed"
    assert translation["chinese_abstract"] == "正式摘要翻译"
    assert any(event[0] == "AIScreeningResult" for event in ai.events)
    assert "标题翻译失败" in get_workflow_batch(database, batch_id)["last_error"]

    calls_before = len(ai.events)
    ai.fail_title = None
    orchestrator(database, ai).run(
        batch_id, WorkflowPolicy(limit=10, allow_strong=False)
    )
    retry_events = ai.events[calls_before:]
    assert [event[0] for event in retry_events] == ["TitleTranslationResult"]
    assert get_translation(database, paper_id)["chinese_title"]


def test_missing_abstract_still_translates_title_and_marks_unavailable(tmp_path):
    database, journal_id = workflow_db(tmp_path)
    paper_id = add_paper(database, journal_id, "School Include no abstract", abstract=None)
    batch_id = create_workflow_batch(
        database,
        name="No abstract",
        source_type="selected_papers",
        paper_ids=[paper_id],
    )
    ai = WorkflowAI()

    orchestrator(database, ai).run(
        batch_id, WorkflowPolicy(limit=10, allow_strong=False)
    )

    translation = get_translation(database, paper_id)
    assert translation["chinese_title"]
    assert translation["chinese_abstract"] is None
    assert translation["abstract_translation_status"] == "unavailable"
    assert not any(event[0] == "AbstractTranslationResult" for event in ai.events)


def test_borderline_abstract_can_be_disabled_without_affecting_title(tmp_path):
    database, journal_id = workflow_db(tmp_path)
    paper_id = add_paper(database, journal_id, "School Borderline paper")
    batch_id = create_workflow_batch(
        database,
        name="Include only abstracts",
        source_type="selected_papers",
        paper_ids=[paper_id],
    )
    ai = WorkflowAI()

    orchestrator(database, ai).run(
        batch_id,
        WorkflowPolicy(limit=10, allow_strong=False, translate_borderline=False),
    )

    translation = get_translation(database, paper_id)
    assert translation["chinese_title"]
    assert translation["chinese_abstract"] is None


def test_failed_screening_can_resume_without_retranslating_successful_title(tmp_path):
    database, journal_id = workflow_db(tmp_path)
    paper_id = add_paper(database, journal_id, "School Retry screening paper")
    batch_id = create_workflow_batch(
        database,
        name="Retry screening",
        source_type="selected_papers",
        paper_ids=[paper_id],
    )
    ai = WorkflowAI(fail_screen="Retry screening")
    service = orchestrator(database, ai)

    first = service.run(batch_id, WorkflowPolicy(limit=10, allow_strong=False))
    assert first.status == "ready_for_review"
    assert get_translation(database, paper_id)["chinese_title"]
    assert get_translation(database, paper_id)["chinese_abstract"] is None
    title_calls = sum(event[0] == "TitleTranslationResult" for event in ai.events)

    ai.fail_screen = None
    service.run(
        batch_id,
        WorkflowPolicy(limit=10, allow_strong=False, retry_failed=True),
    )

    assert sum(event[0] == "TitleTranslationResult" for event in ai.events) == title_calls
    assert get_translation(database, paper_id)["chinese_abstract"] == "正式摘要翻译"


def test_failed_abstract_can_resume_without_repeating_title_or_screening(tmp_path):
    database, journal_id = workflow_db(tmp_path)
    paper_id = add_paper(database, journal_id, "School Include abstract retry")
    batch_id = create_workflow_batch(
        database,
        name="Retry abstract",
        source_type="selected_papers",
        paper_ids=[paper_id],
    )
    ai = WorkflowAI(fail_abstract=True)
    service = orchestrator(database, ai)

    service.run(batch_id, WorkflowPolicy(limit=10, allow_strong=False))
    before = {
        name: sum(event[0] == name for event in ai.events)
        for name in ("TitleTranslationResult", "AIScreeningResult")
    }
    ai.fail_abstract = False
    service.run(batch_id, WorkflowPolicy(limit=10, allow_strong=False))

    assert sum(event[0] == "TitleTranslationResult" for event in ai.events) == before[
        "TitleTranslationResult"
    ]
    assert sum(event[0] == "AIScreeningResult" for event in ai.events) == before[
        "AIScreeningResult"
    ]
    assert get_translation(database, paper_id)["chinese_abstract"] == "正式摘要翻译"


def test_workflow_preview_is_read_only_for_papers_and_does_not_call_ai(tmp_path):
    database, journal_id = workflow_db(tmp_path)
    paper_id = add_paper(database, journal_id, "School preview")
    batch_id = create_workflow_batch(
        database,
        name="Preview",
        source_type="selected_papers",
        paper_ids=[paper_id],
    )
    ai = WorkflowAI()
    with database.connect() as connection:
        before = dict(connection.execute("SELECT * FROM papers WHERE id=?", (paper_id,)).fetchone())

    preview = orchestrator(database, ai).preview(batch_id, WorkflowPolicy())

    with database.connect() as connection:
        after = dict(connection.execute("SELECT * FROM papers WHERE id=?", (paper_id,)).fetchone())
        translations = connection.execute("SELECT COUNT(*) FROM paper_translations").fetchone()[0]
        screenings = connection.execute("SELECT COUNT(*) FROM screenings").fetchone()[0]
    assert preview.title_calls_max == 1
    assert preview.screening_planned == 1
    assert preview.max_ai_calls >= 1
    assert ai.events == []
    assert before == after
    assert translations == screenings == 0


def test_sync_workflow_adds_only_exact_inserted_paper_ids_by_default(tmp_path):
    database, journal_id = workflow_db(tmp_path)
    existing_id = add_paper(database, journal_id, "School existing paper")
    crossref = WorkflowCrossref(
        [
            PaperMetadata(
                doi=None,
                title="School existing paper",
                abstract="Education existing abstract",
            ),
            PaperMetadata(
                doi="10.1000/new-workflow",
                title="School Include new workflow paper",
                abstract="Education new abstract",
            ),
        ]
    )
    batch_id = create_workflow_batch(
        database,
        name="Sync workflow",
        source_type="sync",
        source_config={
            "journals": ["J"],
            "since": "2026-01-01",
            "limit": 10,
            "include_updated": False,
        },
    )
    ai = WorkflowAI()
    service = WorkflowOrchestrator(
        database,
        screening_service=ScreeningService(
            database, ROOT / "prompts", ai, "fast", "strong"
        ),
        translation_service=TranslationService(
            database, ROOT / "prompts", ai, "translation"
        ),
        model_name="fast",
        sync_service=SyncService(database, crossref),
        journal_file=workflow_journals(),
    )
    with database.connect() as connection:
        papers_before = connection.execute("SELECT COUNT(*) FROM papers").fetchone()[0]

    preview = service.preview(batch_id, WorkflowPolicy(limit=10, allow_strong=False))

    with database.connect() as connection:
        assert connection.execute("SELECT COUNT(*) FROM papers").fetchone()[0] == papers_before
    assert preview.sync_fetched == 2
    assert batch_paper_ids(database, batch_id) == []

    service.run(batch_id, WorkflowPolicy(limit=10, allow_strong=False))

    selected = batch_paper_ids(database, batch_id)
    assert existing_id not in selected
    assert len(selected) == 1
    assert get_translation(database, selected[0])["chinese_title"]


def test_batch_review_filter_and_completion_keep_batch_context(tmp_path):
    database, journal_id = workflow_db(tmp_path)
    in_batch = add_paper(database, journal_id, "School Include in batch")
    outside = add_paper(database, journal_id, "School Include outside batch")
    for paper_id in (in_batch, outside):
        with database.connect() as connection:
            connection.execute(
                """INSERT INTO screenings
                (paper_id,ai_decision,model_name,prompt_version,screened_at)
                VALUES (?,'include','fake','v1',?)""",
                (paper_id, utcnow()),
            )
    batch_id = create_workflow_batch(
        database,
        name="Review batch",
        source_type="selected_papers",
        paper_ids=[in_batch],
    )
    with database.connect() as connection:
        connection.execute(
            "UPDATE workflow_batches SET status='ready_for_review' WHERE id=?",
            (batch_id,),
        )

    result = list_all_papers(
        database,
        PaperFilters(
            batch_id=batch_id,
            ai_statuses=("ai_included",),
            review_statuses=("pending",),
        ),
    )
    assert [item["id"] for item in result.items] == [in_batch]
    assert result.items[0]["task_names"] == "Review batch"

    set_review_status(database, in_batch, "approved")
    assert complete_batch_if_reviewed(database, batch_id)
    assert get_workflow_batch(database, batch_id)["status"] == "completed"


def test_workflow_state_requires_one_valid_preview_and_one_confirmation():
    signature = (1, "policy")
    state = {"workflow_preview": object(), "workflow_preview_signature": signature}

    assert workflow_preview_is_valid(state, signature)
    assert not workflow_preview_is_valid(state, (2, "changed"))
    assert not claim_workflow_run(state, signature)
    state["workflow_confirm"] = True
    assert claim_workflow_run(state, signature)
    assert not claim_workflow_run(state, signature)
    finish_workflow_run(state)
    assert not state["workflow_running"]
    assert not state["workflow_confirm"]
