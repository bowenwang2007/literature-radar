"""Phase 7D — unified paper lifecycle regression tests."""

from __future__ import annotations

import json
from datetime import date
from pathlib import Path

import httpx

from eduecon_radar.db import Database, utcnow
from eduecon_radar.schemas import PaperMetadata
from eduecon_radar.services.abstract_enrichment import (
    AbstractCandidate,
    AbstractEnrichmentService,
    AbstractRateLimited,
)
from eduecon_radar.services.local_pipeline import (
    LocalPipelineServices,
    ensure_paper_statuses,
    latest_sync_run,
    process_batch,
    reprocess_library_paper,
    retranslate_abstract,
    snapshot_counts,
    sync_run_papers,
)
from eduecon_radar.services.migrations import (
    apply_overnight_migration,
    apply_product_safety_migration,
    apply_profile_migration,
    apply_translation_migration,
    apply_verification_intake_migration,
    apply_workflow_migration,
)
from eduecon_radar.services.paper_lifecycle_state import (
    abstract_state_label,
    library_reprocess_label,
    paper_lifecycle_state,
)
from eduecon_radar.services.personal_mode import PersonalJournalSyncService

ROOT = Path(__file__).resolve().parents[1]


def _canonical_assessment(database, paper_id: int, profile_id: str = "education_economics"):
    """The judgement lives in the profile table; the run row keeps only state."""
    from eduecon_radar.services import profile_store

    assessment = profile_store.assessments_for(database, profile_id, (paper_id,)).get(paper_id)
    assert assessment is not None, "no profile assessment stored"
    return assessment.classification


def _database(tmp_path: Path) -> Database:
    database = Database(tmp_path / "phase7d.db")
    database.init_schema()
    apply_translation_migration(database, ROOT / "migrations/0001_add_paper_translations.sql")
    apply_workflow_migration(database, ROOT / "migrations/0002_add_workflow_batches.sql")
    apply_overnight_migration(database, ROOT / "migrations/0010_add_overnight_pipeline.sql")
    apply_product_safety_migration(
        database, ROOT / "migrations/0011_add_product_safety_and_verification.sql"
    )
    apply_verification_intake_migration(
        database, ROOT / "migrations/0012_add_verification_intake.sql"
    )
    apply_profile_migration(
        database, ROOT / "migrations/0013_add_profile_scoped_assessment.sql"
    )
    return database


class _Journal:
    journal_key = "economics_of_education_review"
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
        self.records = list(records)
        self.last_fetch_stats = {
            "provider_returned_count": len(self.records),
            "pages_requested": 1,
            "raw_candidate_count": len(self.records),
            "stopped_reason": "provider_exhausted",
        }

    def fetch(self, _issn, _since, limit, *, until=None):
        del until
        return list(self.records[:limit])


def _paper(
    doi: str = "10.1000/phase7d-base",
    title: str = "Teacher incentives and student achievement",
    abstract: str | None = "Teacher incentives affect student achievement.",
) -> PaperMetadata:
    return PaperMetadata(
        doi=doi,
        title=title,
        abstract=abstract,
        authors=["Researcher"],
        issue_date=date(2026, 6, 1),
        source_metadata={"DOI": doi, "ISSN": ["0272-7757"]},
    )


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


class _FakeEnrichment:
    def __init__(self, database: Database, *, abstract: str | None = None):
        self.database = database
        self.abstract = abstract
        self.calls: list[tuple[int, bool]] = []

    def enrich_papers(
        self, paper_ids, *, workflow_batch_id=None, force_refresh=False
    ):
        for paper_id in paper_ids:
            self.calls.append((int(paper_id), bool(force_refresh)))
            if self.abstract is None:
                continue
            with self.database.connect() as connection:
                row = connection.execute(
                    "SELECT abstract FROM papers WHERE id=?", (paper_id,)
                ).fetchone()
                if not row or not (row["abstract"] or "").strip():
                    connection.execute(
                        "UPDATE papers SET abstract=?,updated_at=? WHERE id=?",
                        (self.abstract, utcnow(), paper_id),
                    )
        return _EnrichmentStats()


class _EnrichmentStats:
    enriched = 1
    failed = 0
    total = 1


class _FakeTitles:
    def __init__(self, database: Database):
        self.database = database

    def run_batch_titles(self, paper_ids, *, task_id=None):
        for paper_id in paper_ids:
            with self.database.connect() as connection:
                row = connection.execute(
                    "SELECT title FROM papers WHERE id=?", (paper_id,)
                ).fetchone()
                connection.execute(
                    """INSERT INTO paper_translations
                    (paper_id,chinese_title,title_translation_status,
                     title_translation_source,abstract_translation_status,updated_at)
                    VALUES (?,?,?,?,?,?)
                    ON CONFLICT(paper_id) DO UPDATE SET
                      chinese_title=excluded.chinese_title,
                      title_translation_status=excluded.title_translation_status,
                      title_translation_source=excluded.title_translation_source,
                      updated_at=excluded.updated_at""",
                    (
                        paper_id,
                        f"中文：{row['title'][:10]}",
                        "ai_generated",
                        "ai_generated",
                        "pending",
                        utcnow(),
                    ),
                )
        return _TitleResult()


class _TitleResult:
    translated = 1
    pending = 0
    failed = 0
    ai_calls = 1


class _FakeTranslation:
    def __init__(self, database: Database, *, failed: bool = False):
        self.database = database
        self.failed = failed

    def run(self, parameters):
        if self.failed:
            return _TranslationStats(failed=1, failure_reasons=["provider down"])
        for paper_id in parameters.paper_ids:
            with self.database.connect() as connection:
                row = connection.execute(
                    "SELECT abstract FROM papers WHERE id=?", (paper_id,)
                ).fetchone()
                connection.execute(
                    """INSERT INTO paper_translations
                    (paper_id,chinese_abstract,abstract_translation_status,
                     abstract_translation_source,updated_at)
                    VALUES (?,?,?,?,?)
                    ON CONFLICT(paper_id) DO UPDATE SET
                      chinese_abstract=excluded.chinese_abstract,
                      abstract_translation_status=excluded.abstract_translation_status,
                      abstract_translation_source=excluded.abstract_translation_source,
                      updated_at=excluded.updated_at""",
                    (
                        paper_id,
                        f"中文摘要（{str(row['abstract'] or '')[:12]}）",
                        "ai_generated",
                        "ai_generated",
                        utcnow(),
                    ),
                )
        return _TranslationStats(failed=0)


class _TranslationStats:
    def __init__(self, failed: int, failure_reasons: list[str] | None = None):
        self.failed = failed
        self.ai_calls = 1
        self.input_tokens = 10
        self.output_tokens = 10
        self.failure_reasons = failure_reasons or []


class _ScreeningOutcome:
    def __init__(
        self,
        *,
        status: str = "classified",
        decision: str = "relevant",
        basis: str = "title_and_abstract",
        input_tokens: int = 0,
        output_tokens: int = 0,
    ):
        self.status = status
        self.ai_calls = 1
        self.input_tokens = input_tokens
        self.output_tokens = output_tokens
        self.classification = {
            "decision": decision,
            "screening_basis": basis,
            "education_centrality": "central",
            "economic_relevance": "substantive",
            "confidence": "high",
            "evidence": ["Teacher incentives"],
            "reason": "fixture",
        }


class _FakeScreening:
    def __init__(
        self,
        *,
        basis: str = "title_and_abstract",
        decision: str = "relevant",
        fail: bool = False,
        input_tokens: int = 0,
        output_tokens: int = 0,
    ):
        self.basis = basis
        self.decision = decision
        self.fail = fail
        self.input_tokens = input_tokens
        self.output_tokens = output_tokens
        self.calls: list[tuple[int, bool]] = []

    def screen_paper(self, paper_id, *, task_id=None, force=False):
        self.calls.append((int(paper_id), bool(force)))
        if self.fail:
            raise RuntimeError("provider 429")
        return _ScreeningOutcome(
            decision=self.decision,
            basis=self.basis,
            input_tokens=self.input_tokens,
            output_tokens=self.output_tokens,
        )


def _services(
    database: Database,
    *,
    abstract: str | None = None,
    screening_basis: str = "title_and_abstract",
    screening_decision: str = "relevant",
    screening_fail: bool = False,
) -> LocalPipelineServices:
    return LocalPipelineServices(
        enrichment=_FakeEnrichment(database, abstract=abstract),
        translation=_FakeTranslation(database),
        screening=_FakeScreening(
            basis=screening_basis,
            decision=screening_decision,
            fail=screening_fail,
        ),
        batch_titles=_FakeTitles(database),
    )


def test_abstract_state_separates_en_and_cn(tmp_path: Path) -> None:
    database = _database(tmp_path)
    run_id, paper_id = _sync(database, _paper(abstract="English abstract."))
    state = paper_lifecycle_state(database, paper_id)
    assert state is not None
    assert state.abstract.has_en_abstract is True
    assert state.abstract.has_cn_abstract is False
    assert abstract_state_label(state.abstract) == "缺中文摘要"
    assert library_reprocess_label(state) == "生成中文摘要并重新分析"

    with database.connect() as connection:
        connection.execute("UPDATE papers SET abstract=NULL WHERE id=?", (paper_id,))
    state = paper_lifecycle_state(database, paper_id)
    # No enrichment record exists, so retrieval was never run for this paper.
    # The label must say so instead of implying every source was tried.
    assert abstract_state_label(state.abstract) == "缺英文摘要（未尝试获取）"
    assert library_reprocess_label(state) == "补全并重新分析"

    # With a recorded, non-retryable attempt the label becomes plain absence.
    with database.connect() as connection:
        connection.execute(
            "UPDATE papers SET source_metadata_json=? WHERE id=?",
            (
                json.dumps(
                    {
                        "abstract_enrichment": {
                            "status": "failed",
                            "retryable": False,
                            "execution_state": "strategy_exhausted",
                        }
                    }
                ),
                paper_id,
            ),
        )
    state = paper_lifecycle_state(database, paper_id)
    assert abstract_state_label(state.abstract) == "缺英文摘要"

    # A retryable attempt reports that retrying is possible.
    with database.connect() as connection:
        connection.execute(
            "UPDATE papers SET source_metadata_json=? WHERE id=?",
            (
                json.dumps(
                    {
                        "abstract_enrichment": {
                            "status": "failed",
                            "retryable": True,
                            "execution_state": "retryable_error",
                        }
                    }
                ),
                paper_id,
            ),
        )
    state = paper_lifecycle_state(database, paper_id)
    assert abstract_state_label(state.abstract) == "英文摘要可重试"

    with database.connect() as connection:
        connection.execute(
            """INSERT INTO paper_translations
            (paper_id,chinese_abstract,abstract_translation_status,
             abstract_translation_source,updated_at)
            VALUES (?,?,?,?,?)""",
            (paper_id, "中文摘要", "human_edited", "human_edited", utcnow()),
        )
    state = paper_lifecycle_state(database, paper_id)
    assert state.abstract.has_en_abstract is False
    assert state.abstract.has_cn_abstract is True
    assert state.abstract.cn_translation_source == "human_edited"


def test_en_without_cn_completes_and_screens(tmp_path: Path) -> None:
    database = _database(tmp_path)
    run_id, paper_id = _sync(database, _paper(abstract="English abstract."))
    result = process_batch(
        database, run_id, (paper_id,), _services(database)
    )
    assert paper_id in result["completed"]
    with database.connect() as connection:
        translation = connection.execute(
            "SELECT chinese_abstract FROM paper_translations WHERE paper_id=?",
            (paper_id,),
        ).fetchone()
        status = connection.execute(
            """SELECT status,classification_json FROM paper_processing_status
            WHERE sync_run_id=? AND paper_id=?""",
            (run_id, paper_id),
        ).fetchone()
    assert translation["chinese_abstract"].startswith("中文摘要")
    assert status["status"] == "classified"
    # The run snapshot no longer carries a judgement; the profile does.
    assert json.loads(status["classification_json"]) == {}
    assert _canonical_assessment(database, paper_id)["screening_basis"] == (
        "title_and_abstract"
    )


def test_retranslate_triggers_automatic_rescreen(tmp_path: Path) -> None:
    database = _database(tmp_path)
    run_id, paper_id = _sync(database, _paper(abstract="English abstract."))
    services = _services(database)
    process_batch(database, run_id, (paper_id,), services)
    screening = services.screening
    assert screening is not None
    before = len(screening.calls)
    result = retranslate_abstract(database, run_id, paper_id, services)
    assert result["ai_calls"] >= 1
    assert len(screening.calls) == before + 1
    assert screening.calls[-1][1] is True  # forced re-screen
    with database.connect() as connection:
        status = connection.execute(
            "SELECT status FROM paper_processing_status "
            "WHERE sync_run_id=? AND paper_id=?",
            (run_id, paper_id),
        ).fetchone()
    assert status["status"] == "classified"


def test_no_abstract_still_gets_title_only_judgment(tmp_path: Path) -> None:
    database = _database(tmp_path)
    run_id, paper_id = _sync(database, _paper(abstract=None))
    services = _services(
        database,
        abstract=None,
        screening_basis="title_only",
        screening_decision="uncertain",
    )
    result = process_batch(database, run_id, (paper_id,), services)
    assert paper_id in result["needs_abstract"]
    assert paper_id not in result["completed"]
    with database.connect() as connection:
        status = connection.execute(
            """SELECT status,stage,classification_json FROM paper_processing_status
            WHERE sync_run_id=? AND paper_id=?""",
            (run_id, paper_id),
        ).fetchone()
    assert status["status"] == "classified"
    assert status["stage"] == "title_only_screening"
    assert json.loads(status["classification_json"]) == {}
    classification = _canonical_assessment(database, paper_id)
    assert classification["decision"] == "uncertain"
    assert classification["screening_basis"] == "title_only"


def test_title_only_upgrades_when_abstract_arrives(tmp_path: Path) -> None:
    database = _database(tmp_path)
    run_id, paper_id = _sync(database, _paper(abstract=None))
    services = _services(
        database,
        abstract=None,
        screening_basis="title_only",
        screening_decision="irrelevant",
    )
    process_batch(database, run_id, (paper_id,), services)
    assert _canonical_assessment(database, paper_id)["screening_basis"] == "title_only"

    services = _services(database, abstract="New English abstract.")
    result = process_batch(database, run_id, (paper_id,), services)
    assert paper_id in result["completed"]
    classification = _canonical_assessment(database, paper_id)
    with database.connect() as connection:
        translation = connection.execute(
            "SELECT chinese_abstract FROM paper_translations WHERE paper_id=?",
            (paper_id,),
        ).fetchone()
    assert classification["screening_basis"] == "title_and_abstract"
    assert classification["decision"] == "relevant"
    assert translation["chinese_abstract"].startswith("中文摘要")


def test_human_decision_not_overwritten_by_new_ai(tmp_path: Path) -> None:
    database = _database(tmp_path)
    run_id, paper_id = _sync(database, _paper(abstract="English abstract."))
    with database.connect() as connection:
        connection.execute(
            """UPDATE paper_processing_status SET human_decision='related',
            human_note='keep',human_decided_at=? WHERE sync_run_id=? AND paper_id=?""",
            (utcnow(), run_id, paper_id),
        )
    services = _services(database, screening_decision="irrelevant")
    process_batch(database, run_id, (paper_id,), services)
    with database.connect() as connection:
        row = connection.execute(
            "SELECT human_decision,human_note FROM paper_processing_status "
            "WHERE sync_run_id=? AND paper_id=?",
            (run_id, paper_id),
        ).fetchone()
    assert row["human_decision"] == "related"
    assert row["human_note"] == "keep"


def test_snapshot_counts_split_abstract_and_judgment_basis(tmp_path: Path) -> None:
    database = _database(tmp_path)
    run_id, paper_id = _sync(database, _paper(abstract="English abstract."))
    process_batch(database, run_id, (paper_id,), _services(database))
    papers = sync_run_papers(database, run_id)
    counts = snapshot_counts(papers)
    assert counts["english_abstracts"] == 1
    assert counts["chinese_abstracts"] == 1
    assert counts["missing_abstract"] == 0
    assert counts["missing_cn_abstract"] == 0
    assert counts["automatic_judgments"] == 1
    assert counts["full_judgments"] == 1
    assert counts["title_only_judgments"] == 0

    database2 = _database(tmp_path / "second")
    run2, paper2 = _sync(database2, _paper(abstract=None))
    process_batch(
        database2,
        run2,
        (paper2,),
        _services(
            database2,
            abstract=None,
            screening_basis="title_only",
            screening_decision="uncertain",
        ),
    )
    counts2 = snapshot_counts(sync_run_papers(database2, run2))
    assert counts2["missing_abstract"] == 1
    assert counts2["title_only_judgments"] == 1
    assert counts2["automatic_judgments"] == 1


def test_library_reprocess_completes_missing_chain(tmp_path: Path) -> None:
    database = _database(tmp_path)
    run_id, paper_id = _sync(database, _paper(abstract="English abstract."))
    with database.connect() as connection:
        matched_before = connection.execute(
            "SELECT matched_paper_ids FROM sync_runs WHERE id=?", (run_id,)
        ).fetchone()[0]
    services = _services(database)
    result = reprocess_library_paper(database, paper_id, services)
    assert result["status"]["status"] == "classified"
    with database.connect() as connection:
        translation = connection.execute(
            "SELECT chinese_abstract FROM paper_translations WHERE paper_id=?",
            (paper_id,),
        ).fetchone()
        matched_after = connection.execute(
            "SELECT matched_paper_ids FROM sync_runs WHERE id=?", (run_id,)
        ).fetchone()[0]
    assert translation["chinese_abstract"].startswith("中文摘要")
    assert matched_before == matched_after

    database2 = _database(tmp_path / "second")
    run2, paper2 = _sync(database2, _paper(abstract=None))
    result2 = reprocess_library_paper(
        database2,
        paper2,
        _services(
            database2,
            abstract=(
                "This paper studies how performance-based teacher incentives "
                "affect student achievement using a large administrative panel."
            ),
        ),
    )
    assert result2["status"]["status"] == "classified"
    with database2.connect() as connection:
        row = connection.execute(
            "SELECT abstract FROM papers WHERE id=?", (paper2,)
        ).fetchone()
    assert "teacher incentives" in row["abstract"].casefold()


def test_force_refresh_bypasses_negative_cache(tmp_path: Path) -> None:
    database = _database(tmp_path)
    run_id, paper_id = _sync(database, _paper(abstract=None))

    class SequenceAdapter:
        source_name = "seq"

        def __init__(self):
            self.calls = 0

        def fetch(self, paper, context=None):
            self.calls += 1
            if self.calls == 1:
                return None
            return AbstractCandidate(
                text=(
                    "This paper studies how performance-based teacher incentives "
                    "affect student achievement and teacher effort using a large "
                    "administrative panel dataset with a regression discontinuity "
                    "design. The results show significant improvements in test "
                    "scores for students in low-performing schools."
                ),
                source="seq",
                source_url="https://example.com/seq",
                raw_title=paper.title,
                raw_doi=paper.doi,
                raw_authors=tuple(paper.authors),
                raw_year=paper.year,
                raw_journal=paper.journal,
            )

        def fetch_title_candidates(self, paper, context=None):
            return ()

    adapter = SequenceAdapter()
    enrichment = AbstractEnrichmentService(
        database,
        adapters=[adapter],
        policy_version="phase7d-test",
        source_configs={},
    )
    first = enrichment.enrich_papers((paper_id,))
    assert first.enriched == 0
    assert adapter.calls == 1
    # A normal retry within the negative-cache TTL reuses the prior miss.
    second = enrichment.enrich_papers((paper_id,))
    assert second.enriched == 0
    assert adapter.calls == 1
    # An explicit user refresh bypasses the negative cache and succeeds.
    third = enrichment.enrich_papers((paper_id,), force_refresh=True)
    assert third.enriched == 1
    assert adapter.calls == 2
    with database.connect() as connection:
        row = connection.execute(
            "SELECT abstract FROM papers WHERE id=?", (paper_id,)
        ).fetchone()
    assert "teacher incentives" in row["abstract"].casefold()


def test_provider_fuse_is_run_scoped_and_resets_next_service(tmp_path: Path) -> None:
    database = _database(tmp_path)
    run_id, paper_id = _sync(database, _paper(abstract=None))

    class FuseAdapter:
        source_name = "fuse"

        def __init__(self):
            self.calls = 0

        def fetch(self, paper, context=None):
            self.calls += 1
            if self.calls == 1:
                raise AbstractRateLimited("source rate limited")
            return AbstractCandidate(
                text=(
                    "This paper studies how performance-based teacher incentives "
                    "affect student achievement and teacher effort using a large "
                    "administrative panel dataset with a regression discontinuity "
                    "design. The results show significant improvements in test "
                    "scores for students in low-performing schools."
                ),
                source="fuse",
                source_url="https://example.com/fuse",
                raw_title=paper.title,
                raw_doi=paper.doi,
                raw_authors=tuple(paper.authors),
                raw_year=paper.year,
                raw_journal=paper.journal,
            )

        def fetch_title_candidates(self, paper, context=None):
            return ()

    adapter = FuseAdapter()
    first_service = AbstractEnrichmentService(
        database,
        adapters=[adapter],
        policy_version="phase7d-fuse",
        source_configs={},
    )
    first = first_service.enrich_papers((paper_id,))
    assert first.enriched == 0
    assert adapter.calls == 1
    # A brand-new service (a fresh user click) resets the in-memory fuse; the
    # refresh policy also ignores the persisted rate-limit cache entry.
    second_service = AbstractEnrichmentService(
        database,
        adapters=[adapter],
        policy_version="phase7d-fuse",
        source_configs={},
    )
    second = second_service.enrich_papers((paper_id,), force_refresh=True)
    assert second.enriched == 1
    assert adapter.calls == 2


def test_render_never_calls_providers_or_ai(tmp_path: Path, monkeypatch) -> None:
    from streamlit.testing.v1 import AppTest

    from eduecon_radar.services.deepseek import DeepSeekClient
    from eduecon_radar.services.retrieval_verification import OpenAlexVerificationSource

    database = _database(tmp_path)
    _sync(database, _paper(abstract="English abstract."))
    monkeypatch.setenv("DATABASE_URL", f"sqlite:///{database.path.as_posix()}")
    monkeypatch.setenv("DEEPSEEK_API_KEY", "test-only-no-network")
    monkeypatch.setenv("LITERATURE_RADAR_MODE", "local")
    monkeypatch.setattr(
        "eduecon_radar.services.product_access.local_request_allowed", lambda _host: True
    )

    def forbidden(*_args, **_kwargs):
        raise AssertionError("ordinary render must not call providers or AI")

    monkeypatch.setattr(OpenAlexVerificationSource, "fetch", forbidden)
    monkeypatch.setattr(DeepSeekClient, "request_json", forbidden)
    monkeypatch.setattr(httpx.Client, "send", forbidden)
    app = AppTest.from_file(
        str(ROOT / "src/eduecon_radar/ui/app.py"), default_timeout=60
    ).run()
    assert not app.exception
    app.segmented_control[0].set_value("论文库").run()
    assert not app.exception
    next(item for item in app.toggle if item.label == "查看摘要与详情").set_value(
        True
    ).run()
    assert not app.exception
    assert any("重新分析" in item.label for item in app.button)
    with database.connect() as connection:
        assert connection.execute("SELECT COUNT(*) FROM papers").fetchone()[0] == 1


def test_ui_human_suggestion_not_written_until_save(
    tmp_path: Path, monkeypatch
) -> None:
    from streamlit.testing.v1 import AppTest

    from eduecon_radar.services.deepseek import DeepSeekClient
    from eduecon_radar.services.retrieval_verification import OpenAlexVerificationSource

    database = _database(tmp_path)
    run_id, paper_id = _sync(database, _paper(abstract="English abstract."))
    with database.connect() as connection:
        connection.execute(
            """UPDATE paper_processing_status SET status='classified',stage='complete',
            classification_json=?,human_decision='undecided' WHERE sync_run_id=? AND paper_id=?""",
            (
                json.dumps(
                    {
                        "decision": "relevant",
                        "screening_basis": "title_and_abstract",
                        "education_centrality": "central",
                        "economic_relevance": "substantive",
                        "confidence": "high",
                        "evidence": [],
                        "reason": "fixture",
                    }
                ),
                run_id,
                paper_id,
            ),
        )
    monkeypatch.setenv("DATABASE_URL", f"sqlite:///{database.path.as_posix()}")
    monkeypatch.setenv("DEEPSEEK_API_KEY", "")
    monkeypatch.setenv("LITERATURE_RADAR_MODE", "local")
    monkeypatch.setattr(
        "eduecon_radar.services.product_access.local_request_allowed", lambda _host: True
    )

    def forbidden(*_args, **_kwargs):
        raise AssertionError("render must not call providers or AI")

    monkeypatch.setattr(OpenAlexVerificationSource, "fetch", forbidden)
    monkeypatch.setattr(DeepSeekClient, "request_json", forbidden)
    monkeypatch.setattr(httpx.Client, "send", forbidden)
    app = AppTest.from_file(
        str(ROOT / "src/eduecon_radar/ui/app.py"), default_timeout=60
    ).run()
    app.segmented_control(key="edr_primary_view").set_value("文献工作台").run()
    assert not app.exception
    # The review decision is found by its label: the retrieval panel also has a
    # radio group (journal scope), so positional lookup would be fragile.
    decision_radio = next(item for item in app.radio if item.label == "人工决定")
    assert decision_radio.value == "未决定"
    assert any("AI 建议：相关" in str(item.value) for item in app.markdown)
    with database.connect() as connection:
        row = connection.execute(
            "SELECT human_decision FROM paper_processing_status "
            "WHERE sync_run_id=? AND paper_id=?",
            (run_id, paper_id),
        ).fetchone()
    assert row["human_decision"] == "undecided"
    decision_radio.set_value("相关").run()
    # One save control only: "保存并下一篇" also saves the final paper, so there
    # is no second button that does almost the same thing.
    next(item for item in app.button if item.label == "保存并下一篇").click().run()
    assert not app.exception
    with database.connect() as connection:
        row = connection.execute(
            "SELECT human_decision FROM paper_processing_status "
            "WHERE sync_run_id=? AND paper_id=?",
            (run_id, paper_id),
        ).fetchone()
    assert row["human_decision"] == "related"


def test_library_export_scope_preset_and_natural_filename(
    tmp_path: Path,
) -> None:
    from eduecon_radar.ui.views.paper_library import (
        _export_scope_papers,
        _preset_rows,
        _rows_to_csv,
        _rows_to_markdown,
    )

    database = _database(tmp_path)
    run_id, paper_id = _sync(database, _paper(abstract="English abstract."))
    process_batch(database, run_id, (paper_id,), _services(database))
    with database.connect() as connection:
        connection.execute(
            "UPDATE paper_processing_status SET human_decision='related' "
            "WHERE sync_run_id=? AND paper_id=?",
            (run_id, paper_id),
        )
    defaults = {
        "journal": "全部",
        "abstract_state": "全部",
        "decision": "全部",
        "auto_decision": "全部",
        "year": "全部",
        "search": "",
    }
    related = _export_scope_papers(
        database, "人工相关", filters=defaults, has_status=True
    )
    assert len(related) == 1
    pending = _export_scope_papers(
        database, "待审核", filters=defaults, has_status=True
    )
    assert pending == []
    all_papers = _export_scope_papers(
        database, "论文库全部", filters=defaults, has_status=True
    )
    assert len(all_papers) == 1
    rows = _preset_rows(all_papers, "简洁版")
    assert rows[0]["DOI"] == "10.1000/phase7d-base"
    assert "中文摘要" not in rows[0]
    research = _preset_rows(all_papers, "研究版")
    assert research[0]["自动判断"] == "相关"
    assert research[0]["人工判断"] == "相关"
    csv_text = _rows_to_csv(rows)
    assert "英文标题" in csv_text
    assert _rows_to_markdown(rows).startswith("| 中文标题")


def test_library_reprocess_button_completes_paper_on_click(
    tmp_path: Path, monkeypatch
) -> None:
    from streamlit.testing.v1 import AppTest

    from eduecon_radar.services import pipeline_builder
    from eduecon_radar.services.deepseek import DeepSeekClient
    from eduecon_radar.services.retrieval_verification import OpenAlexVerificationSource

    database = _database(tmp_path)
    run_id, paper_id = _sync(database, _paper(abstract="English abstract."))
    monkeypatch.setenv("DATABASE_URL", f"sqlite:///{database.path.as_posix()}")
    monkeypatch.setenv("DEEPSEEK_API_KEY", "test-only-no-network")
    monkeypatch.setenv("LITERATURE_RADAR_MODE", "local")
    monkeypatch.setattr(
        "eduecon_radar.services.product_access.local_request_allowed", lambda _host: True
    )

    def forbidden(*_args, **_kwargs):
        raise AssertionError("unexpected provider call")

    monkeypatch.setattr(OpenAlexVerificationSource, "fetch", forbidden)
    monkeypatch.setattr(DeepSeekClient, "request_json", forbidden)
    monkeypatch.setattr(httpx.Client, "send", forbidden)
    monkeypatch.setattr(
        pipeline_builder,
        "build_local_pipeline_services",
        lambda _database, _settings, *, profile: _services(database),
    )
    app = AppTest.from_file(
        str(ROOT / "src/eduecon_radar/ui/app.py"), default_timeout=90
    ).run()
    assert not app.exception
    app.segmented_control[0].set_value("论文库").run()
    assert not app.exception
    next(item for item in app.toggle if item.label == "查看摘要与详情").set_value(
        True
    ).run()
    button = next(
        item for item in app.button if "生成中文摘要并重新分析" in item.label
    )
    button.click().run()
    assert not app.exception
    with database.connect() as connection:
        translation = connection.execute(
            "SELECT chinese_abstract FROM paper_translations WHERE paper_id=?",
            (paper_id,),
        ).fetchone()
        status = connection.execute(
            "SELECT status FROM paper_processing_status "
            "WHERE sync_run_id=? AND paper_id=?",
            (run_id, paper_id),
        ).fetchone()
    assert translation["chinese_abstract"].startswith("中文摘要")
    assert status["status"] == "classified"


def test_repeated_sync_preserves_paper_review(tmp_path: Path) -> None:
    from eduecon_radar.services.local_pipeline import save_human_decisions

    database = _database(tmp_path)
    old_run, paper_id = _sync(database, _paper())
    save_human_decisions(database, old_run, {paper_id: ("related", "keep this")})
    new_run, same_id = _sync(database, _paper())
    assert same_id == paper_id
    assert new_run != old_run
    new_paper = sync_run_papers(database, new_run)[0]
    assert new_paper["human_decision"] == "related"
    assert new_paper["reviewer_notes"] == "keep this"
    save_human_decisions(database, old_run, {paper_id: ("uncertain", "reconsider")})
    assert sync_run_papers(database, new_run)[0]["human_decision"] == "uncertain"
    assert save_human_decisions(database, old_run, {999999: ("related", None)}) == ()


def test_existing_abstract_requires_upgrade_of_title_judgment(tmp_path: Path) -> None:
    database = _database(tmp_path)
    run_id, paper_id = _sync(database, _paper())
    with database.connect() as connection:
        connection.execute(
            "UPDATE paper_processing_status SET classification_json=? WHERE paper_id=?",
            (json.dumps({"decision": "relevant", "screening_basis": "title_only"}), paper_id),
        )
    state = paper_lifecycle_state(database, paper_id)
    from dataclasses import replace

    state = replace(state, abstract=replace(state.abstract, has_cn_abstract=True))
    assert library_reprocess_label(state) == "重新分析"


def test_unknown_and_metadata_basis_are_not_full_screening() -> None:
    from eduecon_radar.services.paper_lifecycle_state import (
        ScreeningState,
        screening_state_label,
    )

    for basis in (None, "deterministic_metadata_rule"):
        state = ScreeningState(True, "irrelevant", basis, None)
        assert screening_state_label(state) != "完整判断"


def test_removed_paper_does_not_break_historical_run(tmp_path: Path) -> None:
    from eduecon_radar.services.paper_lifecycle import remove_paper

    database = _database(tmp_path)
    run_id, paper_id = _sync(database, _paper())
    with database.connect() as connection:
        before = connection.execute(
            "SELECT matched_paper_ids FROM sync_runs WHERE id=?", (run_id,)
        ).fetchone()[0]
    remove_paper(database, paper_id)
    assert ensure_paper_statuses(database, run_id) == ()
    with database.connect() as connection:
        assert connection.execute(
            "SELECT matched_paper_ids FROM sync_runs WHERE id=?", (run_id,)
        ).fetchone()[0] == before
        assert connection.execute("PRAGMA foreign_key_check").fetchall() == []


def test_empty_independent_index_is_not_confirmation(tmp_path: Path) -> None:
    from eduecon_radar.services.retrieval_verification import verify_sync_run

    class EmptySource:
        source_name = "OpenAlex"
        query_count = 1

        def fetch(self, *_args):
            return ()

    database = _database(tmp_path)
    run_id, _ = _sync(database, _paper())
    result = verify_sync_run(
        database, run_id, journal_issns=(("0272-7757",),), source=EmptySource()
    )
    assert result.status == "unavailable"
    assert result.independent_count == 0
    assert result.details["insufficient_independent_evidence"]


def test_stale_translation_does_not_count_as_complete() -> None:
    from eduecon_radar.services.paper_lifecycle_state import translation_ready

    assert not translation_ready("old Chinese text", "stale", "ai_generated")
    assert translation_ready("human text", "stale", "human_edited")
    counts = snapshot_counts([{
        "abstract": "new English text", "chinese_abstract": "old Chinese text",
        "abstract_translation_status": "stale", "abstract_translation_source": "ai_generated",
        "processing": {"status": "classified", "classification": {"decision": "relevant"}},
    }])
    assert counts["missing_cn_abstract"] == 1
    assert counts["full_judgments"] == 0


def test_title_screening_failure_remains_retryable_and_counts_call(tmp_path: Path) -> None:
    from types import SimpleNamespace

    class FailingScreening:
        def screen_paper(self, *args, **kwargs):
            return SimpleNamespace(
                status="needs_retry", ai_calls=1, input_tokens=7, output_tokens=0
            )

    database = _database(tmp_path)
    run_id, paper_id = _sync(database, _paper(abstract=None))
    result = process_batch(
        database, run_id, (paper_id,), LocalPipelineServices(screening=FailingScreening())
    )
    assert result["needs_retry"] == [paper_id]
    assert result["needs_abstract"] == []
    assert result["ai_calls"] == 1
    assert result["input_tokens"] == 7
    assert sync_run_papers(database, run_id)[0]["processing"]["status"] == "needs_retry"

def test_workbench_fetch_analyze_and_review_under_selected_profile(tmp_path, monkeypatch):
    from dataclasses import replace

    from streamlit.testing.v1 import AppTest

    from eduecon_radar.services import search_queue
    from eduecon_radar.services.crossref import CrossrefClient

    database = _database(tmp_path)
    monkeypatch.setenv("DATABASE_URL", f"sqlite:///{database.path.as_posix()}")
    monkeypatch.setenv("DEEPSEEK_API_KEY", "offline-test")
    monkeypatch.setenv("LITERATURE_RADAR_MODE", "local")
    monkeypatch.setattr(
        "eduecon_radar.services.product_access.local_request_allowed", lambda _host: True
    )
    paper = _paper(abstract=None)
    monkeypatch.setattr(CrossrefClient, "fetch", lambda *_args, **_kwargs: [paper])
    directions = []

    def services(_db, _settings, profile):
        directions.append(profile.profile_id)
        return replace(
            _services(database, abstract="Retrieved abstract."), profile_id=profile.profile_id,
        )

    monkeypatch.setattr(search_queue, "build_local_pipeline_services", services)
    monkeypatch.setattr(search_queue.SearchQueue, "start", lambda *_: None)
    app = AppTest.from_file(str(ROOT / "src/eduecon_radar/ui/app.py"), default_timeout=60).run()
    app.segmented_control(key="edr_primary_view").set_value("文献工作台").run()
    app.selectbox(key="edr_profile_select").set_value("labor_economics").run()
    app.radio(key="local_journal_scope").set_value("全部期刊库").run()
    app.multiselect(key="local_journal_picker").set_value(["economics_of_education_review"]).run()
    app.button(key="local_fetch").click().run()
    assert not app.exception and not app.error
    queue = search_queue.SearchQueue(database)
    from eduecon_radar.config import Settings

    queue.work(Settings.load(ROOT))
    app.run()
    assert directions and set(directions) == {"labor_economics"}
    run = latest_sync_run(database)
    run_id = int(run["id"])
    with database.connect() as connection:
        row = connection.execute(
            "SELECT profile_id FROM sync_runs WHERE id=?", (run_id,),
        ).fetchone()
        assert row["profile_id"] == "labor_economics"
    papers = sync_run_papers(database, run_id)
    assert len(papers) == 1
    assert papers[0]["abstract"] == "Retrieved abstract."
    assert papers[0]["chinese_abstract"]
    assert papers[0]["processing"]["classification"]["decision"]
    assert papers[0]["human_decision"] == "undecided"
    app.radio(key=f"local_review_{run_id}_{papers[0]['id']}_decision").set_value("不确定").run()
    app.button(key=f"local_review_save_next_{run_id}").click().run()
    assert sync_run_papers(database, run_id)[0]["human_decision"] == "uncertain"
    app.segmented_control(key=f"batch_stage_{run_id}").set_value("导出").run()
    assert not app.error and len(app.get("download_button")) == 4
