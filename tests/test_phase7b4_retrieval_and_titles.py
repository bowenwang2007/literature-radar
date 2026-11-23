from __future__ import annotations

import hashlib
import json
from datetime import date
from pathlib import Path

import httpx

from eduecon_radar.db import Database, utcnow
from eduecon_radar.schemas import PaperMetadata
from eduecon_radar.services.batch_translation import (
    BatchTitleItem,
    BatchTitleTranslationOutput,
    BatchTitleTranslationService,
)
from eduecon_radar.services.crossref import CrossrefClient
from eduecon_radar.services.deepseek import DeepSeekResponse
from eduecon_radar.services.migrations import (
    apply_overnight_migration,
    apply_p0_hardening_migration,
    apply_profile_migration,
    apply_translation_migration,
    apply_workflow_migration,
)
from eduecon_radar.services.personal_mode import (
    PersonalJournalSyncService,
    create_personal_task,
    load_personal_state,
    translate_personal_task_titles,
)
from eduecon_radar.services.profile_registry import default_profile_registry

ROOT = Path(__file__).resolve().parents[1]


def _database(tmp_path: Path) -> Database:
    database = Database(tmp_path / "phase7b4.db")
    database.init_schema()
    apply_translation_migration(database, ROOT / "migrations/0001_add_paper_translations.sql")
    apply_workflow_migration(database, ROOT / "migrations/0002_add_workflow_batches.sql")
    apply_p0_hardening_migration(
        database, ROOT / "migrations/0007_add_runtime_leases_and_task_reviews.sql"
    )
    apply_overnight_migration(
        database, ROOT / "migrations/0010_add_overnight_pipeline.sql"
    )
    apply_profile_migration(
        database, ROOT / "migrations/0013_add_profile_scoped_assessment.sql"
    )
    return database


def _registry():
    return default_profile_registry()


def _record(
    doi: str,
    title: str,
    *,
    issn: str = "0272-7757",
    container: str = "Economics of Education Review",
    year: int = 2026,
) -> PaperMetadata:
    return PaperMetadata(
        doi=doi,
        title=title,
        abstract=None,
        authors=["Researcher"],
        publisher="Fixture Publisher",
        issue_date=date(year, 7, 1),
        source_metadata={
            "DOI": doi,
            "ISSN": [issn],
            "container-title": [container],
        },
    )


class FakeCrossref:
    def __init__(self, records: list[PaperMetadata], stats: dict | None = None) -> None:
        self.records = records
        self.last_fetch_stats = stats

    def fetch(self, _issn, _since, limit, *, until=None):
        del until
        return list(self.records[:limit])


def _sync(database: Database, records: list[PaperMetadata], *, limit: int = 100):
    journal = _registry().by_key("economics_of_education_review")
    service = PersonalJournalSyncService(database, FakeCrossref(records))
    results = service.sync(
        (journal,),
        since=date(2026, 5, 1),
        until=date(2026, 7, 31),
        per_journal_limit=limit,
    )
    return journal, results


def _paper_ids(database: Database) -> tuple[int, ...]:
    with database.connect() as connection:
        return tuple(
            int(row[0])
            for row in connection.execute(
                "SELECT id FROM papers ORDER BY id"
            ).fetchall()
        )


def test_ui_limit_100_reaches_service_and_task_contains_all_matched(tmp_path) -> None:
    database = _database(tmp_path)
    records = [
        _record(f"10.1000/7b4-{index}", f"Education paper {index}")
        for index in range(1, 13)
    ]
    _, results = _sync(database, records, limit=100)
    result = results[0]
    assert result.found_count == 24  # two ISSNs, same 12 records each
    assert result.duplicates == 12
    assert len(result.matched_paper_ids) == 12
    created = create_personal_task(
        database,
        journal_results=results,
        selected_journals=(_registry().by_key("economics_of_education_review"),),
        since=date(2026, 5, 1),
        until=date(2026, 7, 31),
        per_journal_limit=100,
    )
    assert len(created.paper_ids) == 12
    assert len(load_personal_state(database, created.task_id)["items"]) == 12


def _crossref_item(doi: str, title: str) -> dict:
    return {
        "DOI": doi,
        "title": [title],
        "author": [{"given": "Researcher", "family": "One"}],
        "publisher": "Fixture",
        "type": "journal-article",
        "published-print": {"date-parts": [[2026, 7, 1]]},
        "volume": "1",
        "page": "1",
        "URL": f"https://doi.org/{doi}",
        "container-title": ["Economics of Education Review"],
        "ISSN": ["0272-7757"],
    }


def _paged_transport(
    page_sizes: list[int],
    *,
    fail_page: int | None = None,
    fail_status: int = 429,
) -> httpx.MockTransport:
    def handler(request: httpx.Request) -> httpx.Response:
        params = dict(request.url.params)
        cursor = params.get("cursor", "*")
        page_index = 0 if cursor == "*" else int(cursor)
        if fail_page is not None and page_index == fail_page:
            return httpx.Response(fail_status, request=request)
        size = page_sizes[page_index] if page_index < len(page_sizes) else 0
        start = sum(page_sizes[:page_index])
        items = [
            _crossref_item(
                f"10.1000/page-{start + index + 1}",
                f"Paged education paper {start + index + 1}",
            )
            for index in range(size)
        ]
        more = page_index + 1 < len(page_sizes)
        return httpx.Response(
            200,
            request=request,
            json={
                "message": {
                    "items": items,
                    "next-cursor": str(page_index + 1) if more else None,
                }
            },
        )

    return httpx.MockTransport(handler)


def test_crossref_paginates_two_pages_until_user_limit(tmp_path) -> None:
    client = CrossrefClient(
        client=httpx.Client(transport=_paged_transport([1000, 1000, 500]))
    )
    papers = client.fetch("0272-7757", date(2026, 5, 1), limit=2500)
    assert len(papers) == 2500
    stats = client.last_fetch_stats
    assert stats["pages_requested"] == 3
    assert stats["provider_returned_count"] == 2500
    assert stats["stopped_reason"] == "requested_limit_reached"


def test_crossref_short_last_page_reports_provider_exhausted(tmp_path) -> None:
    client = CrossrefClient(client=httpx.Client(transport=_paged_transport([1000, 400])))
    papers = client.fetch("0272-7757", date(2026, 5, 1), limit=2500)
    assert len(papers) == 1400
    assert client.last_fetch_stats["pages_requested"] == 2
    assert client.last_fetch_stats["stopped_reason"] == "provider_exhausted"


def test_crossref_page_two_429_preserves_first_page_results(tmp_path) -> None:
    client = CrossrefClient(
        client=httpx.Client(transport=_paged_transport([1000, 500], fail_page=1))
    )
    papers = client.fetch("0272-7757", date(2026, 5, 1), limit=2500)
    assert len(papers) == 1000
    stats = client.last_fetch_stats
    assert stats["stopped_reason"] == "rate_limited"
    assert stats["pages_requested"] == 2
    assert stats["provider_returned_count"] == 1000


def test_sync_stats_aggregate_pages_and_stopped_reason(tmp_path) -> None:
    database = _database(tmp_path)
    records = [_record(f"10.1000/agg-{index}", f"Agg paper {index}") for index in range(1, 13)]
    stats = {
        "requested_limit": 100,
        "pages_requested": 2,
        "provider_returned_count": 12,
        "raw_candidate_count": 12,
        "stopped_reason": "provider_exhausted",
        "error_summary": None,
    }
    journal = _registry().by_key("economics_of_education_review")
    service = PersonalJournalSyncService(
        database, FakeCrossref(records, stats=stats)
    )
    results = service.sync(
        (journal,),
        since=date(2026, 5, 1),
        until=date(2026, 7, 31),
        per_journal_limit=100,
    )
    result = results[0]
    assert result.pages_requested == 4  # two ISSNs, two pages each
    assert result.provider_returned_count == 24
    assert result.raw_candidate_count == 24
    assert result.stopped_reason == "provider_exhausted"
    with database.connect() as connection:
        row = connection.execute(
            "SELECT provider_returned_count,pages_requested,raw_candidate_count,"
            "normalized_count,stopped_reason FROM sync_runs ORDER BY id DESC LIMIT 1"
        ).fetchone()
    assert tuple(row) == (24, 4, 24, 24, "provider_exhausted")


class FakeBatchClient:
    def __init__(self) -> None:
        self.calls = 0

    def request_json(self, **kwargs):
        self.calls += 1
        user = kwargs["user_prompt"]
        raw = user.split("ITEMS_START\n", 1)[1].split("\nITEMS_END", 1)[0]
        items = json.loads(raw)
        entries = [
            BatchTitleItem(
                paper_id=int(item["paper_id"]),
                chinese_title=f"教育研究标题{item['paper_id']}",
            )
            for item in items
        ]
        return DeepSeekResponse(
            data=BatchTitleTranslationOutput(translations=entries),
            model_name="fixture-7b4",
            input_tokens=5,
            output_tokens=5,
        )


def test_title_translation_covers_unchanged_and_updated_matched_papers(tmp_path) -> None:
    database = _database(tmp_path)
    records = [
        _record(f"10.1000/existing-{index}", f"Existing paper {index}")
        for index in range(1, 6)
    ]
    _, first_results = _sync(database, records, limit=100)
    existing_ids = list(first_results[0].matched_paper_ids)
    records.extend(
        [
            _record("10.1000/new-1", "New paper one"),
            _record("10.1000/new-2", "New paper two"),
        ]
    )
    _, results = _sync(database, records, limit=100)
    result = results[0]
    assert result.inserted == 2
    assert result.updated == 0
    assert result.unchanged == 5
    assert len(result.matched_paper_ids) == 7
    created = create_personal_task(
        database,
        journal_results=results,
        selected_journals=(_registry().by_key("economics_of_education_review"),),
        since=date(2026, 5, 1),
        until=date(2026, 7, 31),
        per_journal_limit=100,
    )
    client = FakeBatchClient()
    service = BatchTitleTranslationService(
        database, ROOT / "prompts", client, "fixture-7b4"
    )
    result = translate_personal_task_titles(database, created.task_id, service)
    assert result.translated == 7
    assert result.failed == 0
    assert client.calls == 1
    with database.connect() as connection:
        missing = connection.execute(
            "SELECT COUNT(*) FROM papers p LEFT JOIN paper_translations pt "
            "ON pt.paper_id=p.id WHERE p.id IN (?,?,?,?,?,?,?) AND "
            "(pt.chinese_title IS NULL OR pt.chinese_title='')",
            [*existing_ids, *created.paper_ids[-2:]],
        ).fetchone()
    assert missing[0] == 0


def test_existing_machine_and_manual_titles_are_not_retranslated(tmp_path) -> None:
    database = _database(tmp_path)
    records = [
        _record(f"10.1000/existing-{index}", f"Existing paper {index}")
        for index in range(1, 3)
    ]
    _, first_results = _sync(database, records, limit=100)
    existing_ids = list(first_results[0].matched_paper_ids)
    now = utcnow()
    with database.connect() as connection:
        connection.execute(
            """INSERT INTO paper_translations
            (paper_id,chinese_title,title_translation_status,abstract_translation_status,
             title_translation_source,model_name,prompt_version,translated_at,updated_at,
             source_title_hash,validation_status)
            VALUES (?,?,?,?,?,?,?,?,?,?,?)""",
            (
                existing_ids[0],
                "已有机器译文",
                "ai_generated",
                "unavailable",
                "ai_generated",
                "fixture",
                "v1",
                now,
                now,
                hashlib.sha256(b"Existing paper 1").hexdigest(),
                "ok",
            ),
        )
        connection.execute(
            """INSERT INTO paper_translations
            (paper_id,chinese_title,title_translation_status,abstract_translation_status,
             title_translation_source,model_name,prompt_version,translated_at,updated_at,
             source_title_hash,validation_status)
            VALUES (?,?,?,?,?,?,?,?,?,?,?)""",
            (
                existing_ids[1],
                "人工保留标题",
                "human_edited",
                "unavailable",
                "human_edited",
                "fixture",
                "v1",
                now,
                now,
                hashlib.sha256(b"Existing paper 2").hexdigest(),
                "ok",
            ),
        )
    _, results = _sync(database, records, limit=100)
    created = create_personal_task(
        database,
        journal_results=results,
        selected_journals=(_registry().by_key("economics_of_education_review"),),
        since=date(2026, 5, 1),
        until=date(2026, 7, 31),
        per_journal_limit=100,
    )
    client = FakeBatchClient()
    service = BatchTitleTranslationService(
        database, ROOT / "prompts", client, "fixture-7b4"
    )
    result = translate_personal_task_titles(database, created.task_id, service)
    assert result.translated == 2
    assert client.calls == 0
    with database.connect() as connection:
        rows = connection.execute(
            "SELECT chinese_title FROM paper_translations "
            "WHERE paper_id IN (?,?) ORDER BY paper_id",
            (existing_ids[0], existing_ids[1]),
        ).fetchall()
    assert [row[0] for row in rows] == ["已有机器译文", "人工保留标题"]


def test_app_render_has_no_task_vocabulary(tmp_path, monkeypatch) -> None:
    from streamlit.testing.v1 import AppTest

    database = _database(tmp_path)
    records = [_record("10.1000/ui-1", "UI paper one")]
    _, results = _sync(database, records, limit=100)
    created = create_personal_task(
        database,
        journal_results=results,
        selected_journals=(_registry().by_key("economics_of_education_review"),),
        since=date(2026, 5, 1),
        until=date(2026, 7, 31),
        per_journal_limit=100,
    )
    before = hashlib.sha256(database.path.read_bytes()).hexdigest()
    monkeypatch.setenv("DATABASE_URL", f"sqlite:///{database.path.as_posix()}")
    monkeypatch.setenv("DEEPSEEK_API_KEY", "")

    monkeypatch.setenv("LITERATURE_RADAR_MODE", "local")
    monkeypatch.setattr(
        "eduecon_radar.services.product_access.local_request_allowed", lambda _host: True
    )

    def forbidden(*_args, **_kwargs):
        raise AssertionError("render must not call AI or an external provider")

    monkeypatch.setattr(CrossrefClient, "fetch", forbidden)
    from eduecon_radar.services.deepseek import DeepSeekClient

    monkeypatch.setattr(DeepSeekClient, "request_json", forbidden)
    app = AppTest.from_file(
        str(ROOT / "src/eduecon_radar/ui/app.py"), default_timeout=45
    ).run()
    app.segmented_control(key="edr_primary_view").set_value("文献工作台").run()
    assert not app.exception
    labels = {item.label for item in app.button}
    assert "获取论文" in labels
    rendered = "\n".join(
        str(item.value)
        for group in (
            app.title,
            app.caption,
            app.info,
            app.warning,
            app.error,
            app.markdown,
            app.subheader,
        )
        for item in group
    )
    for forbidden_text in ("个人任务", "task #", "本次最多进入任务", "当前本次同步"):
        assert forbidden_text not in rendered
    assert "一次专注一篇" not in rendered
    # The run summary moved into the top status bar: the acquisition date range
    # and its count are now shown against the research direction they belong to,
    # instead of a separate "最近一次获取" block with a duplicated caption.
    assert "当前研究方向" in rendered
    assert "本次获取" in rendered
    assert hashlib.sha256(database.path.read_bytes()).hexdigest() == before
    assert created.task_id is not None
