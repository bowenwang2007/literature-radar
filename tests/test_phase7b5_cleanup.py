from __future__ import annotations

import hashlib
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
from eduecon_radar.services.deepseek import DeepSeekResponse
from eduecon_radar.services.local_pipeline import (
    LocalPipelineServices,
    ensure_paper_statuses,
    latest_sync_run,
    process_batch,
    sync_run_papers,
    translate_titles,
)
from eduecon_radar.services.migrations import (
    apply_overnight_migration,
    apply_p0_hardening_migration,
    apply_profile_migration,
    apply_translation_migration,
    apply_workflow_migration,
)
from eduecon_radar.services.personal_mode import PersonalJournalSyncService
from eduecon_radar.services.profile_registry import default_profile_registry

ROOT = Path(__file__).resolve().parents[1]


def _database(tmp_path: Path) -> Database:
    database = Database(tmp_path / "phase7b5.db")
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


def _record(doi: str, title: str) -> PaperMetadata:
    return PaperMetadata(
        doi=doi,
        title=title,
        abstract=None,
        authors=["Researcher"],
        publisher="Fixture Publisher",
        issue_date=date(2026, 7, 1),
        source_metadata={
            "DOI": doi,
            "ISSN": ["0272-7757"],
            "container-title": ["Economics of Education Review"],
        },
    )


class FakeCrossref:
    def __init__(self, records: list[PaperMetadata]) -> None:
        self.records = records

    def fetch(self, _issn, _since, limit, *, until=None):
        del until
        return list(self.records[:limit])


class NoopEnrichment:
    def enrich_papers(
        self, paper_ids, *, workflow_batch_id=None, force_refresh=False
    ):
        class Stats:
            total = len(paper_ids)
            enriched = 0
            failed = len(paper_ids)

        return Stats()


class FakeBatchClient:
    def __init__(self) -> None:
        self.calls = 0

    def request_json(self, **kwargs):
        self.calls += 1
        user = kwargs["user_prompt"]
        raw = user.split("ITEMS_START\n", 1)[1].split("\nITEMS_END", 1)[0]
        items = json.loads(raw)
        return DeepSeekResponse(
            data=BatchTitleTranslationOutput(
                translations=[
                    BatchTitleItem(
                        paper_id=int(item["paper_id"]),
                        chinese_title=f"教育研究标题{item['paper_id']}",
                    )
                    for item in items
                ]
            ),
            model_name="fixture-7b5",
            input_tokens=5,
            output_tokens=5,
        )


def _sync(database: Database, count: int = 12):
    records = [_record(f"10.1000/7b5-{index}", f"Paper {index}") for index in range(1, count + 1)]
    journal = _registry().by_key("economics_of_education_review")
    PersonalJournalSyncService(database, FakeCrossref(records)).sync(
        (journal,),
        since=date(2026, 5, 1),
        until=date(2026, 7, 31),
        per_journal_limit=100,
    )
    run = latest_sync_run(database)
    assert run is not None
    return int(run["id"]), run


def test_product_flow_uses_sync_run_and_never_creates_task(tmp_path) -> None:
    database = _database(tmp_path)
    with database.connect() as connection:
        tasks_before = connection.execute(
            "SELECT COUNT(*) FROM workflow_batches"
        ).fetchone()[0]
    run_id, run = _sync(database, count=12)
    matched = json.loads(run["matched_paper_ids"])
    assert len(matched) == 12
    ensure_paper_statuses(database, run_id)
    services = LocalPipelineServices(enrichment=NoopEnrichment())
    result = process_batch(database, run_id, tuple(matched[:5]), services)
    assert len(result["needs_abstract"]) == 5
    assert len(result["completed"]) == 0
    with database.connect() as connection:
        tasks_after = connection.execute(
            "SELECT COUNT(*) FROM workflow_batches"
        ).fetchone()[0]
        statuses = connection.execute(
            """SELECT status FROM paper_processing_status
            WHERE sync_run_id=? ORDER BY paper_id""",
            (run_id,),
        ).fetchall()
    assert tasks_after == tasks_before
    assert [row[0] for row in statuses] == (
        ["needs_manual_abstract"] * 5 + ["pending"] * 7
    )
    assert len(sync_run_papers(database, run_id)) == 12


def test_title_translation_covers_all_matched_without_task(tmp_path) -> None:
    database = _database(tmp_path)
    run_id, run = _sync(database, count=12)
    ensure_paper_statuses(database, run_id)
    client = FakeBatchClient()
    services = LocalPipelineServices(
        enrichment=NoopEnrichment(),
        batch_titles=BatchTitleTranslationService(
            database, ROOT / "prompts", client, "fixture-7b5"
        ),
    )
    result = translate_titles(database, run_id, services)
    assert result["translated"] == 12
    assert result["pending"] == 0
    assert client.calls == 2
    with database.connect() as connection:
        missing = connection.execute(
            """SELECT COUNT(*) FROM papers p LEFT JOIN paper_translations pt
            ON pt.paper_id=p.id WHERE p.id IN ("""
            + ",".join("?" * 12)
            + """) AND (pt.chinese_title IS NULL OR pt.chinese_title='')""",
            list(json.loads(run["matched_paper_ids"])),
        ).fetchone()[0]
    assert missing == 0
    with database.connect() as connection:
        task_rows = connection.execute(
            "SELECT COUNT(*) FROM workflow_batches"
        ).fetchone()[0]
    assert task_rows == 0


def test_product_view_source_has_no_task_vocabulary_or_task_dependency() -> None:
    source = (ROOT / "src/eduecon_radar/ui/views/personal_mode.py").read_text(
        encoding="utf-8"
    )
    for forbidden in (
        "create_personal_task",
        "personal_task",
        "max_task_papers",
        "workflow_batches",
        "list_personal_tasks",
        "personal_task_snapshot",
        "current_task_id",
        "任务",
    ):
        assert forbidden not in source, forbidden
    assert "获取并分析论文" in source
    assert "获取所选时间范围内" not in source
    assert "一次专注一篇" not in source


def test_shipped_app_source_has_no_legacy_imports() -> None:
    app_source = (ROOT / "src/eduecon_radar/ui/app.py").read_text(encoding="utf-8")
    for forbidden in (
        "legacy_app",
        "literature_workflow",
        "pilot_runtime",
        "PilotAuthorizationService",
        "TaskLeaseManager",
        "RuntimeBundleResolver",
        "import tests",
    ):
        assert forbidden not in app_source, forbidden
    package_source = "\n".join(
        path.read_text(encoding="utf-8")
        for path in (ROOT / "src/eduecon_radar/ui").rglob("*.py")
        if path.name != "__pycache__"
    )
    for forbidden in ("literature_workflow", "legacy_app", "pilot_runtime"):
        assert forbidden not in package_source, forbidden


def test_run_web_prints_real_paths_and_rejects_second_instance() -> None:
    script = (ROOT / "run_web.ps1").read_text(encoding="utf-8")
    assert '"src\\eduecon_radar\\ui\\app.py"' in script
    assert ".venv\\Scripts\\python.exe" in script
    assert "Python: $Python" in script
    assert "Entry: $App" in script
    assert "http://localhost:8501/" in script
    assert "Get-NetTCPConnection" in script
    assert "继续阅读" in script
    assert "Stop-Process" not in script
    for forbidden in ("legacy", "pilot", "release", "lease", "task"):
        assert forbidden not in script.casefold(), forbidden


def test_app_render_shows_only_product_text(tmp_path, monkeypatch) -> None:
    from streamlit.testing.v1 import AppTest

    from eduecon_radar.services.crossref import CrossrefClient
    from eduecon_radar.services.deepseek import DeepSeekClient

    database = _database(tmp_path)
    run_id, _run = _sync(database, count=6)
    ensure_paper_statuses(database, run_id)
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
    assert 'href="https://doi.org/10.1000/7b5-1"' in rendered
    for forbidden_text in (
        "个人任务",
        "建立个人任务",
        "进入任务",
        "当前任务",
        "任务容量",
        "task #",
        "ready",
        "running",
        "completed",
        "本次最多进入任务",
        "受控试运行",
        "NOT RELEASED",
        "authorization",
        "lease",
        "heartbeat",
        "fixture",
        "demo mode",
    ):
        assert forbidden_text not in rendered, forbidden_text
    assert "一次专注一篇" not in rendered
    # Run provenance is still on the page, now as the status bar above the
    # review queue rather than as a standalone acquisition block.
    assert "当前研究方向" in rendered
    assert "本次获取" in rendered
    assert hashlib.sha256(database.path.read_bytes()).hexdigest() == before
