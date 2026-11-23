from __future__ import annotations

import json
import sqlite3
from dataclasses import replace
from datetime import date
from pathlib import Path

import httpx
import pytest

from eduecon_radar.db import Database
from eduecon_radar.schemas import PaperMetadata
from eduecon_radar.services.local_pipeline import ensure_paper_statuses, latest_sync_run
from eduecon_radar.services.migrations import (
    apply_overnight_migration,
    apply_product_safety_migration,
    apply_profile_migration,
    apply_translation_migration,
    apply_workflow_migration,
)
from eduecon_radar.services.paper_lifecycle import (
    preview_paper_removal,
    remove_paper,
)
from eduecon_radar.services.personal_mode import PersonalJournalSyncService
from eduecon_radar.services.retrieval_verification import (
    IndependentWork,
    cached_verification,
    verify_sync_run,
)
from eduecon_radar.services.source_traceability import build_doi_url
from eduecon_radar.ui.views.paper_library import _library_papers
from eduecon_radar.ui.views.personal_mode import _confirmed_overlap

ROOT = Path(__file__).resolve().parents[1]


def _database(tmp_path: Path) -> Database:
    database = Database(tmp_path / "phase7c2.db")
    database.init_schema()
    apply_translation_migration(database, ROOT / "migrations/0001_add_paper_translations.sql")
    apply_workflow_migration(database, ROOT / "migrations/0002_add_workflow_batches.sql")
    apply_overnight_migration(database, ROOT / "migrations/0010_add_overnight_pipeline.sql")
    apply_product_safety_migration(
        database, ROOT / "migrations/0011_add_product_safety_and_verification.sql"
    )
    apply_profile_migration(
        database, ROOT / "migrations/0013_add_profile_scoped_assessment.sql"
    )
    return database


def _paper(doi: str = " DOI:10.1000/PHASE7C2 ") -> PaperMetadata:
    return PaperMetadata(
        doi=doi,
        title="A safely removable paper",
        abstract="Evidence",
        authors=["Researcher"],
        issue_date=date(2026, 7, 1),
        source_metadata={"DOI": doi, "ISSN": ["0272-7757"]},
    )


class _Journal:
    journal_key = "economics-of-education-review"
    canonical_name = "Economics of Education Review"
    print_issn = "0272-7757"
    online_issn = "1873-7382"
    issns = (print_issn, online_issn)

    def as_journal_config(self):
        from eduecon_radar.schemas import JournalConfig

        return JournalConfig(
            name=self.canonical_name,
            short_name="EER",
            issn=self.print_issn,
            eissn=self.online_issn,
            priority="high",
        )


class _Crossref:
    def __init__(self, records):
        self.records = records
        self.last_fetch_stats = {
            "provider_returned_count": len(records),
            "pages_requested": 1,
            "raw_candidate_count": len(records),
            "stopped_reason": "provider_exhausted",
        }

    def fetch(self, _issn, _since, limit, *, until=None):
        del until
        return list(self.records[:limit])


def _sync(database: Database, paper: PaperMetadata) -> tuple[int, int]:
    PersonalJournalSyncService(database, _Crossref([paper])).sync(
        (_Journal(),),
        since=date(2026, 5, 1),
        until=date(2026, 8, 1),
        per_journal_limit=100,
    )
    run = latest_sync_run(database)
    assert run is not None
    run_id = int(run["id"])
    paper_id = int(json.loads(run["matched_paper_ids"])[0])
    ensure_paper_statuses(database, run_id)
    return run_id, paper_id


def test_doi_links_are_strict_and_normalized() -> None:
    assert build_doi_url(" HTTPS://DOI.ORG/10.1000/ABC ") == "https://doi.org/10.1000/abc"
    assert build_doi_url("doi:10.1000/a b") is None
    assert build_doi_url("not-a-doi") is None
    assert build_doi_url(None) is None


def test_removal_archives_human_data_cascades_and_suppresses_resync(tmp_path: Path) -> None:
    database = _database(tmp_path)
    paper = _paper()
    run_id, paper_id = _sync(database, paper)
    with database.connect() as connection:
        connection.execute(
            """UPDATE paper_processing_status SET human_decision='related',human_note='keep'
            WHERE sync_run_id=? AND paper_id=?""",
            (run_id, paper_id),
        )
        connection.execute(
            """INSERT INTO paper_translations
            (paper_id,chinese_title,title_translation_status,title_translation_source,
             abstract_translation_status,updated_at)
            VALUES (?,?,'human_edited','human_edited','pending','now')""",
            (paper_id, "人工标题"),
        )

    preview = preview_paper_removal(database, paper_id)
    assert preview.has_human_data is True
    assert preview.related_rows >= 2
    verify_sync_run(
        database,
        run_id,
        journal_issns=(("0272-7757",),),
        source=_VerificationSource(),
    )
    assert cached_verification(database, run_id) is not None
    remove_paper(database, paper_id)

    with database.connect() as connection:
        assert connection.execute("SELECT COUNT(*) FROM papers").fetchone()[0] == 0
        assert connection.execute(
            "SELECT COUNT(*) FROM paper_processing_status"
        ).fetchone()[0] == 0
        assert connection.execute("SELECT COUNT(*) FROM paper_translations").fetchone()[0] == 0
        tombstone = connection.execute("SELECT * FROM paper_tombstones").fetchone()
        assert tombstone["doi"] == "10.1000/phase7c2"
        assert tombstone["has_human_data"] == 1
        archive = json.loads(tombstone["archived_json"])
        assert archive["related"]["paper_processing_status"][0]["human_note"] == "keep"
        assert connection.execute("PRAGMA foreign_key_check").fetchall() == []
    assert cached_verification(database, run_id) is None

    results = PersonalJournalSyncService(database, _Crossref([paper])).sync(
        (_Journal(),),
        since=date(2026, 5, 1),
        until=date(2026, 8, 1),
        per_journal_limit=100,
    )
    assert results[0].ignored == 1
    with database.connect() as connection:
        assert connection.execute("SELECT COUNT(*) FROM papers").fetchone()[0] == 0
    assert _library_papers(
        database,
        journal="全部",
        abstract_state="全部",
        decision="全部",
        auto_decision="全部",
        year="全部",
        search="",
    ) == []


def test_failed_delete_rolls_back_archive_and_paper(tmp_path: Path) -> None:
    database = _database(tmp_path)
    _, paper_id = _sync(database, _paper())
    with database.connect() as connection:
        connection.execute(
            """CREATE TRIGGER reject_paper_delete BEFORE DELETE ON papers
            BEGIN SELECT RAISE(ABORT, 'blocked'); END"""
        )
    with pytest.raises(sqlite3.IntegrityError, match="blocked"):
        remove_paper(database, paper_id)
    with database.connect() as connection:
        assert connection.execute("SELECT COUNT(*) FROM papers").fetchone()[0] == 1
        assert connection.execute("SELECT COUNT(*) FROM paper_tombstones").fetchone()[0] == 0


class _VerificationSource:
    source_name = "Fixture index"
    query_count = 1

    def fetch(self, _issns, _since, _until):
        return (
            IndependentWork(
                "w1", "10.1000/phase7c2", "A safely removable paper",
                "2026-07-01", "article", False,
            ),
            IndependentWork(
                "w2", "10.1000/front", "Editorial Board",
                "2026-07-01", "paratext", True,
            ),
            IndependentWork(
                "w3", "10.1000/missing", "A possible missing paper",
                "2026-07-02", "article", False,
            ),
        )


class _FailedVerificationSource:
    source_name = "Unavailable index"
    query_count = 1

    def fetch(self, _issns, _since, _until):
        raise TimeoutError("provider timeout")


class _RateLimitedVerificationSource:
    source_name = "Rate-limited index"
    query_count = 1

    def fetch(self, _issns, _since, _until):
        raise RuntimeError("429 Too Many Requests")


class _TitleFallbackVerificationSource:
    source_name = "Fixture index"
    query_count = 1

    def fetch(self, _issns, _since, _until):
        return (
            IndependentWork(
                "w-title",
                None,
                "A safely removable paper",
                "2026-07-01",
                "article",
                False,
            ),
        )


def test_verification_uses_real_records_and_caches_without_blocking_failure(tmp_path: Path) -> None:
    database = _database(tmp_path)
    run_id, _ = _sync(database, _paper())
    result = verify_sync_run(
        database,
        run_id,
        journal_issns=(("0272-7757", "1873-7382"),),
        source=_VerificationSource(),
    )
    assert result.status == "possible_gaps"
    assert result.local_count == 1
    assert result.independent_count == 3
    assert result.comparable_independent_count == 2
    assert result.exact_doi_matches == 1
    assert result.possible_missing_count == 1
    assert result.details["confirmed_overlap_count"] == 1
    legacy_details = dict(result.details)
    legacy_details.pop("confirmed_overlap_count")
    assert _confirmed_overlap(replace(result, details=legacy_details)) == 1
    assert cached_verification(database, run_id) == result

    unavailable = verify_sync_run(
        database,
        run_id,
        journal_issns=(("0272-7757",),),
        source=_FailedVerificationSource(),
    )
    assert unavailable.status == "unavailable"
    assert "provider timeout" in (unavailable.error_message or "")
    assert cached_verification(database, run_id) == unavailable

    rate_limited = verify_sync_run(
        database,
        run_id,
        journal_issns=(("0272-7757",),),
        source=_RateLimitedVerificationSource(),
    )
    assert rate_limited.status == "unavailable"
    assert "429" in (rate_limited.error_message or "")
    assert rate_limited.local_count == 1


def test_verification_uses_title_fallback_and_explains_date_basis(tmp_path: Path) -> None:
    database = _database(tmp_path)
    run_id, _ = _sync(database, _paper())
    result = verify_sync_run(
        database,
        run_id,
        journal_issns=(("0272-7757",),),
        source=_TitleFallbackVerificationSource(),
    )
    assert result.status == "broadly_consistent"
    assert result.exact_doi_matches == 0
    assert result.source_only_count == 0
    assert result.local_only_count == 0
    assert result.possible_missing_count == 0
    assert result.details["title_matches"][0]["local_doi"] == "10.1000/phase7c2"
    assert result.details["confirmed_overlap_count"] == 1
    assert "OpenAlex publication_date" in result.details["date_basis"]
    assert "Crossref pub-date" in result.details["date_basis"]


def test_product_render_exposes_doi_and_does_not_auto_verify(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    from streamlit.testing.v1 import AppTest

    from eduecon_radar.services.deepseek import DeepSeekClient
    from eduecon_radar.services.retrieval_verification import OpenAlexVerificationSource

    database = _database(tmp_path)
    _sync(database, _paper())
    monkeypatch.setenv("DATABASE_URL", f"sqlite:///{database.path.as_posix()}")
    monkeypatch.setenv("DEEPSEEK_API_KEY", "")
    monkeypatch.setenv("LITERATURE_RADAR_MODE", "local")
    monkeypatch.setattr(
        "eduecon_radar.services.product_access.local_request_allowed", lambda _host: True
    )

    def forbidden(*_args, **_kwargs):
        raise AssertionError("ordinary render must not query verification providers")

    monkeypatch.setattr(OpenAlexVerificationSource, "fetch", forbidden)
    monkeypatch.setattr(DeepSeekClient, "request_json", forbidden)
    monkeypatch.setattr(httpx.Client, "send", forbidden)
    app = AppTest.from_file(
        str(ROOT / "src/eduecon_radar/ui/app.py"), default_timeout=45
    ).run()
    app.segmented_control(key="edr_primary_view").set_value("文献工作台").run()
    assert not app.exception
    assert "开始核验" not in {item.label for item in app.button}
    rendered = "\n".join(str(item.value) for item in app.markdown)
    assert 'href="https://doi.org/10.1000/phase7c2"' in rendered
    assert 'target="_blank"' in rendered

    app.segmented_control[1].set_value("检索核验").run()
    assert "开始核验" in {item.label for item in app.button}
    app.segmented_control[0].set_value("论文库").run()
    assert not app.exception
    next(item for item in app.toggle if item.label == "查看摘要与详情").set_value(
        True
    ).run()
    assert "移除论文" in {item.label for item in app.button}
    next(item for item in app.button if item.label == "移除论文").click().run()
    confirmation = next(
        item for item in app.checkbox if item.label == "确认移除这篇论文"
    )
    confirmation.check().run()
    commit = next(item for item in app.button if item.label == "确认移除")
    assert not commit.disabled
    commit.click().run()
    assert not app.exception
    assert "移除论文" not in {item.label for item in app.button}
    with database.connect() as connection:
        assert connection.execute("SELECT COUNT(*) FROM papers").fetchone()[0] == 0
        assert connection.execute("SELECT COUNT(*) FROM paper_tombstones").fetchone()[0] == 1


def test_verification_render_leads_with_plain_language_and_missing_paper(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    from streamlit.testing.v1 import AppTest


    database = _database(tmp_path)
    run_id, _ = _sync(database, _paper())
    verify_sync_run(
        database,
        run_id,
        journal_issns=(("0272-7757",),),
        source=_VerificationSource(),
    )
    monkeypatch.setenv("DATABASE_URL", f"sqlite:///{database.path.as_posix()}")
    monkeypatch.setenv("DEEPSEEK_API_KEY", "")
    monkeypatch.setenv("LITERATURE_RADAR_MODE", "local")
    monkeypatch.setattr(
        "eduecon_radar.services.product_access.local_request_allowed", lambda _host: True
    )

    app = AppTest.from_file(
        str(ROOT / "src/eduecon_radar/ui/app.py"), default_timeout=45
    ).run()
    app.segmented_control(key="edr_primary_view").set_value("文献工作台").run()
    app.segmented_control[1].set_value("检索核验").run()
    assert not app.exception
    rendered = "\n".join(str(item.value) for item in app.markdown)
    assert "完整性核验结果" in rendered
    assert "发现可能遗漏" in rendered
    assert "独立来源中有 1 篇可比记录" in rendered
    assert "A possible missing paper" in rendered
    assert 'href="https://doi.org/10.1000/missing"' in rendered
    assert "主来源论文" in rendered
    assert "独立来源可比" in rendered
    assert "已确认重合" in rendered
    assert "疑似遗漏" in rendered
    assert "重新核验" in {item.label for item in app.button}
    toggle_labels = {item.label for item in app.toggle}
    assert "技术详情" in toggle_labels
    assert "查看本次获取明细" in toggle_labels
    assert not app.dataframe
    next(item for item in app.toggle if item.label == "技术详情").set_value(True).run()
    assert app.dataframe
