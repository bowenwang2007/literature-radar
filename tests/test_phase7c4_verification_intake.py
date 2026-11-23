"""Phase 7C4 — verification-candidate intake acceptance tests."""

from __future__ import annotations

import json
from dataclasses import dataclass
from datetime import date
from pathlib import Path

import httpx
import pytest

from eduecon_radar.db import Database, utcnow
from eduecon_radar.schemas import PaperMetadata
from eduecon_radar.services.doi import canonicalize_doi
from eduecon_radar.services.local_pipeline import (
    LocalPipelineServices,
    ensure_paper_statuses,
    latest_sync_run,
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
from eduecon_radar.services.paper_lifecycle import (
    paper_is_tombstoned,
    remove_paper,
)
from eduecon_radar.services.personal_mode import PersonalJournalSyncService
from eduecon_radar.services.profile_registry import default_profile_registry
from eduecon_radar.services.retrieval_verification import (
    IndependentWork,
    verify_sync_run,
)
from eduecon_radar.services.verification_intake import (
    CandidateMissingIdentity,
    CandidatePreviouslyRemoved,
    candidate_intake_state,
    candidate_intake_states,
    intake_verification_candidate,
    intake_verification_candidates,
)
from eduecon_radar.ui.views.paper_library import _library_papers

ROOT = Path(__file__).resolve().parents[1]


def _database(tmp_path: Path) -> Database:
    database = Database(tmp_path / "phase7c4.db")
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


def _registry():
    return default_profile_registry()


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

    def fetch_work(self, doi):
        normalized = canonicalize_doi(doi)
        for record in self.records:
            if canonicalize_doi(record.get("DOI")) == normalized:
                return record
        return None


def _base_paper(
    doi: str = "10.1000/phase7c4-base",
    title: str = "A baseline retrieved paper",
) -> PaperMetadata:
    return PaperMetadata(
        doi=doi,
        title=title,
        abstract="Evidence from the baseline.",
        authors=["Researcher"],
        issue_date=date(2026, 6, 1),
        source_metadata={"DOI": doi, "ISSN": ["0272-7757"]},
    )


def _sync(database: Database, *papers: PaperMetadata) -> tuple[int, int]:
    PersonalJournalSyncService(database, _Crossref(papers)).sync(
        (_Journal(),),
        since=date(2026, 5, 1),
        until=date(2026, 8, 1),
        per_journal_limit=100,
    )
    run = latest_sync_run(database)
    assert run is not None
    run_id = int(run["id"])
    paper_ids = [int(item) for item in json.loads(run["matched_paper_ids"] or "[]")]
    ensure_paper_statuses(database, run_id)
    return run_id, paper_ids[0]


def _candidate(
    doi: str,
    title: str,
    publication_date: str = "2026-07-25",
    *,
    source_id: str = "W0000000001",
) -> dict[str, object]:
    return {
        "source_id": source_id,
        "doi": doi,
        "title": title,
        "publication_date": publication_date,
        "work_type": "article",
        "is_paratext": False,
        "reason": "独立来源存在可比记录，主检索中未找到相同 DOI",
        "gap_classification": "date_basis_difference",
        "crossref_dates": {"created": "2026-07-20", "published_online": "2026-07-25"},
    }


def _crossref_record(
    doi: str,
    title: str,
    *,
    abstract: str | None = None,
    created: str = "2026-07-20",
) -> dict[str, object]:
    return {
        "DOI": doi,
        "title": [title],
        "abstract": abstract,
        "author": [{"given": "Jane", "family": "Researcher"}],
        "publisher": "Elsevier",
        "type": "journal-article",
        "created": {"date-parts": [[2026, 7, 20]]},
        "published-online": {"date-parts": [[2026, 7, 25]]},
        "issued": {"date-parts": [[2026, 7, 25]]},
        "volume": "102",
        "issue": "4",
        "page": "102818",
        "URL": "https://doi.org/" + doi,
        "ISSN": ["0272-7757", "1873-7382"],
        "container-title": ["Economics of Education Review"],
        "alternative-id": ["S0272775726000818"],
    }


def _openalex_work(doi: str, title: str, source_id: str = "W0000000001") -> dict[str, object]:
    return {
        "id": source_id,
        "doi": "https://doi.org/" + doi,
        "title": title,
        "display_name": title,
        "publication_date": "2026-07-25",
        "type": "article",
        "biblio": {"volume": "102", "issue": "4", "first_page": "1", "last_page": "20"},
        "primary_location": {
            "landing_page_url": "https://openalex.org/" + source_id,
            "source": {"issn": ["0272-7757"], "issn_l": "0272-7757"},
        },
        "best_oa_location": None,
        "authorships": [
            {"author": {"display_name": "Jane Researcher"}},
        ],
        "abstract_inverted_index": None,
    }


class _FakeOpenAlex:
    def __init__(self, work: dict[str, object] | None):
        self.work = work

    def fetch_by_doi(self, doi):
        if self.work and canonicalize_doi(self.work.get("doi")) == canonicalize_doi(doi):
            return PaperMetadata(
                doi=doi,
                title=str(self.work.get("title") or ""),
                source_metadata=dict(self.work),
            )
        return None

    def fetch_by_id(self, source_id):
        if self.work and self.work.get("id") == source_id:
            return PaperMetadata(
                doi=self.work.get("doi"),
                title=str(self.work.get("title") or ""),
                source_metadata=dict(self.work),
            )
        return None


class _FakeEnrichment:
    def __init__(self, *, fail: bool = False, abstract: str = "Fetched English abstract."):
        self.fail = fail
        self.abstract = abstract
        self.calls: list[int] = []

    def enrich_papers(
        self, paper_ids, *, workflow_batch_id=None, force_refresh=False
    ):
        if self.fail:
            raise RuntimeError("provider 429 rate limited")
        for paper_id in paper_ids:
            self.calls.append(int(paper_id))
            with self.database.connect() as connection:
                row = connection.execute(
                    "SELECT abstract,source_metadata_json FROM papers WHERE id=?",
                    (paper_id,),
                ).fetchone()
                if not row or not (row["abstract"] or "").strip():
                    metadata = json.loads(row["source_metadata_json"] or "{}")
                    metadata["abstract_source"] = "openalex"
                    metadata["abstract_enrichment"] = {}
                    connection.execute(
                        "UPDATE papers SET abstract=?,source_metadata_json=?,updated_at=? "
                        "WHERE id=?",
                        (
                            self.abstract,
                            json.dumps(metadata, ensure_ascii=False),
                            utcnow(),
                            paper_id,
                        ),
                    )
        return _EnrichmentStats()


class _EnrichmentStats:
    enriched = 1
    failed = 0
    total = 1


class _FakeBatchTitles:
    def __init__(self, database: Database, *, fail: bool = False):
        self.database = database
        self.fail = fail

    def run_batch_titles(self, paper_ids, *, task_id=None):
        if self.fail:
            raise RuntimeError("deepseek unavailable")
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
                        f"中文标题 {row['title'][:12]}",
                        "ai_generated",
                        "ai_generated",
                        "pending",
                        utcnow(),
                    ),
                )
        return _TitleResult(translated=len(paper_ids))


class _TitleResult:
    def __init__(self, translated: int):
        self.translated = translated
        self.pending = 0
        self.failed = 0
        self.ai_calls = 1


class _FakeTranslation:
    def __init__(self, database: Database):
        self.database = database

    def run(self, parameters):
        for paper_id in parameters.paper_ids:
            with self.database.connect() as connection:
                row = connection.execute(
                    "SELECT abstract FROM papers WHERE id=?", (paper_id,)
                ).fetchone()
                abstract = str(row["abstract"] or "") if row else ""
                connection.execute(
                    """INSERT INTO paper_translations
                    (paper_id,chinese_title,title_translation_status,
                     abstract_translation_status,abstract_translation_source,
                     chinese_abstract,updated_at)
                    VALUES (?,?,?,?,?,?,?)
                    ON CONFLICT(paper_id) DO UPDATE SET
                      abstract_translation_status=excluded.abstract_translation_status,
                      abstract_translation_source=excluded.abstract_translation_source,
                      chinese_abstract=excluded.chinese_abstract,
                      updated_at=excluded.updated_at""",
                    (
                        paper_id,
                        None,
                        "pending",
                        "ai_generated",
                        "ai_generated",
                        "中文摘要（" + abstract[:16] + "）",
                        utcnow(),
                    ),
                )
        return _TranslationStats(failed=0)


class _TranslationStats:
    def __init__(self, failed: int):
        self.failed = failed
        self.ai_calls = 1
        self.input_tokens = 10
        self.output_tokens = 10
        self.failure_reasons: list[str] = []


@dataclass
class _ScreeningOutcome:
    status: str = "classified"
    classification: dict[str, object] = None  # type: ignore[assignment]
    ai_calls: int = 1

    def __post_init__(self) -> None:
        if self.classification is None:
            self.classification = {
                "decision": "relevant",
                "education_centrality": "central",
                "economic_relevance": "substantive",
                "confidence": "high",
                "evidence": ["Remote instruction"],
                "reason": "Education policy outcome study.",
            }


class _FakeScreening:
    def __init__(self, *, fail: bool = False):
        self.fail = fail

    def screen_paper(self, paper_id, *, task_id=None, force=False):
        del paper_id, task_id, force
        if self.fail:
            raise RuntimeError("provider 429 rate limited")
        return _ScreeningOutcome()


def _services(
    database: Database,
    *,
    enrichment_fail: bool = False,
    title_fail: bool = False,
    screening_fail: bool = False,
    no_ai: bool = False,
) -> LocalPipelineServices:
    enrichment = _FakeEnrichment(fail=enrichment_fail)
    enrichment.database = database
    if no_ai:
        return LocalPipelineServices(
            enrichment=enrichment,
            translation=None,
            screening=None,
            batch_titles=None,
        )
    return LocalPipelineServices(
        enrichment=enrichment,
        translation=_FakeTranslation(database),
        screening=_FakeScreening(fail=screening_fail),
        batch_titles=_FakeBatchTitles(database, fail=title_fail),
    )


def _verify_with_candidate(
    database: Database, run_id: int, candidate: dict[str, object]
) -> None:
    verify_sync_run(
        database,
        run_id,
        journal_issns=(("0272-7757", "1873-7382"),),
        source=_CandidateSource(candidate),
    )


class _CandidateSource:
    source_name = "Fixture index"
    query_count = 1

    def __init__(self, candidate: dict[str, object]):
        self.candidate = candidate

    def fetch(self, _issns, _since, _until):
        return (
            IndependentWork(
                str(self.candidate.get("source_id") or ""),
                str(self.candidate.get("doi") or "") or None,
                str(self.candidate.get("title") or ""),
                str(self.candidate.get("publication_date") or ""),
                str(self.candidate.get("work_type") or ""),
                False,
            ),
        )


def test_intake_creates_paper_runs_full_chain_and_keeps_history(
    tmp_path: Path,
) -> None:
    database = _database(tmp_path)
    run_id, base_paper_id = _sync(database, _base_paper())
    candidate = _candidate(
        " DOI:10.1016/J.ECONEDUREV.2026.102818 ",
        "Just the right amount of caution?",
    )
    crossref = _Crossref(
        [
            _crossref_record(
                "10.1016/j.econedurev.2026.102818",
                "Just the right amount of caution?",
            ),
        ]
    )
    openalex = _FakeOpenAlex(
        _openalex_work("10.1016/j.econedurev.2026.102818", "Just the right amount of caution?")
    )
    with database.connect() as connection:
        run = connection.execute(
            "SELECT matched_paper_ids,inserted_paper_ids FROM sync_runs WHERE id=?",
            (run_id,),
        ).fetchone()
        matched_before = run["matched_paper_ids"]
        inserted_before = run["inserted_paper_ids"]

    result = intake_verification_candidate(
        database,
        run_id,
        candidate,
        registry=_registry(),
        crossref=crossref,
        openalex=openalex,
        services=_services(database),
    )
    assert result.outcome == "created"
    assert result.paper_id != base_paper_id
    assert result.review_ready is True
    assert not result.errors

    with database.connect() as connection:
        assert connection.execute("SELECT COUNT(*) FROM papers").fetchone()[0] == 2
        paper = connection.execute(
            "SELECT * FROM papers WHERE id=?", (result.paper_id,)
        ).fetchone()
        assert paper["doi"] == "10.1016/j.econedurev.2026.102818"
        assert paper["abstract"] == "Fetched English abstract."
        status = connection.execute(
            """SELECT * FROM paper_processing_status
            WHERE sync_run_id=? AND paper_id=?""",
            (run_id, result.paper_id),
        ).fetchone()
        assert status["status"] == "classified"
        assert status["human_decision"] == "undecided"
        assert status["classification_json"] in ("", "{}")
        translation = connection.execute(
            "SELECT * FROM paper_translations WHERE paper_id=?", (result.paper_id,)
        ).fetchone()
        assert translation["chinese_title"].startswith("中文标题")
        assert translation["chinese_abstract"].startswith("中文摘要")
        metadata = json.loads(paper["source_metadata_json"])
        intake = metadata["eduecon_verification_intake"]
        assert intake["source"] == "retrieval_verification"
        assert intake["discovered_by"] == "OpenAlex"
        assert intake["verification_sync_run_id"] == run_id
        assert intake["candidate"]["doi"].strip().lower().endswith("102818")
        assert run["matched_paper_ids"] == matched_before
        assert run["inserted_paper_ids"] == inserted_before
    from eduecon_radar.services import profile_store

    stored = profile_store.assessments_for(
        database, "education_economics", (result.paper_id,)
    )[result.paper_id]
    assert stored.classification["decision"] == "relevant"

    papers = sync_run_papers(database, run_id)
    added = next(item for item in papers if int(item["id"]) == result.paper_id)
    assert added["human_decision"] == "undecided"
    assert added["processing"]["status"] == "classified"
    assert added["chinese_title"].startswith("中文标题")
    assert added["chinese_abstract"].startswith("中文摘要")

    library = _library_papers(
        database,
        journal="全部",
        abstract_state="全部",
        decision="全部",
        auto_decision="全部",
        year="全部",
        search="",
    )
    assert {int(item["id"]) for item in library} == {base_paper_id, result.paper_id}


def test_intake_dedup_repeated_click_is_idempotent(tmp_path: Path) -> None:
    database = _database(tmp_path)
    run_id, _ = _sync(database, _base_paper())
    candidate = _candidate("10.1000/missing", "A possible missing paper")
    crossref = _Crossref(
        [
            _crossref_record("10.1000/missing", "A possible missing paper"),
        ]
    )
    services = _services(database)
    first = intake_verification_candidate(
        database,
        run_id,
        candidate,
        registry=_registry(),
        crossref=crossref,
        openalex=_FakeOpenAlex(None),
        services=services,
    )
    assert first.outcome == "created"
    second = intake_verification_candidate(
        database,
        run_id,
        candidate,
        registry=_registry(),
        crossref=crossref,
        openalex=_FakeOpenAlex(None),
        services=services,
    )
    assert second.outcome == "already_in_library"
    assert second.paper_id == first.paper_id
    with database.connect() as connection:
        assert connection.execute("SELECT COUNT(*) FROM papers").fetchone()[0] == 2
        assert connection.execute(
            """SELECT COUNT(*) FROM paper_processing_status
            WHERE sync_run_id=? AND paper_id=?""",
            (run_id, first.paper_id),
        ).fetchone()[0] == 1


def test_intake_existing_paper_unattached_attaches_without_duplicate(
    tmp_path: Path,
) -> None:
    database = _database(tmp_path)
    run_id, _ = _sync(database, _base_paper())
    candidate = _candidate("10.1000/missing", "A possible missing paper")
    crossref = _Crossref(
        [
            _crossref_record("10.1000/missing", "A possible missing paper"),
        ]
    )
    services = _services(database)
    first = intake_verification_candidate(
        database,
        run_id,
        candidate,
        registry=_registry(),
        crossref=crossref,
        openalex=_FakeOpenAlex(None),
        services=services,
    )
    # Simulate the "new primary retrieval already added it, verification cache
    # is stale" situation: the paper exists but is not attached to this run.
    with database.connect() as connection:
        connection.execute(
            "DELETE FROM paper_processing_status WHERE sync_run_id=? AND paper_id=?",
            (run_id, first.paper_id),
        )
    state = candidate_intake_state(database, run_id, candidate)
    assert state.in_library is True
    assert state.attached_to_run is False
    second = intake_verification_candidate(
        database,
        run_id,
        candidate,
        registry=_registry(),
        crossref=None,
        openalex=None,
        services=services,
    )
    assert second.outcome == "attached_existing"
    assert second.paper_id == first.paper_id
    with database.connect() as connection:
        assert connection.execute("SELECT COUNT(*) FROM papers").fetchone()[0] == 2
        status = connection.execute(
            "SELECT status FROM paper_processing_status WHERE sync_run_id=? AND paper_id=?",
            (run_id, first.paper_id),
        ).fetchone()
        assert status["status"] == "classified"


def test_intake_previously_removed_requires_explicit_restore(tmp_path: Path) -> None:
    database = _database(tmp_path)
    run_id, _ = _sync(database, _base_paper())
    candidate = _candidate("10.1000/missing", "A possible missing paper")
    crossref = _Crossref(
        [
            _crossref_record("10.1000/missing", "A possible missing paper"),
        ]
    )
    services = _services(database)
    first = intake_verification_candidate(
        database,
        run_id,
        candidate,
        registry=_registry(),
        crossref=crossref,
        openalex=_FakeOpenAlex(None),
        services=services,
    )
    remove_paper(database, first.paper_id)
    state = candidate_intake_state(database, run_id, candidate)
    assert state.tombstoned is True
    assert state.in_library is False

    with pytest.raises(CandidatePreviouslyRemoved):
        intake_verification_candidate(
            database,
            run_id,
            candidate,
            registry=_registry(),
            crossref=crossref,
            openalex=_FakeOpenAlex(None),
            services=services,
        )
    restored = intake_verification_candidate(
        database,
        run_id,
        candidate,
        registry=_registry(),
        crossref=crossref,
        openalex=_FakeOpenAlex(None),
        services=services,
        allow_restore=True,
    )
    assert restored.outcome == "restored"
    with database.connect() as connection:
        assert connection.execute(
            "SELECT COUNT(*) FROM papers WHERE doi='10.1000/missing'"
        ).fetchone()[0] == 1
        tombstone = connection.execute(
            "SELECT restored_at FROM paper_tombstones WHERE doi=?",
            ("10.1000/missing",),
        ).fetchone()
        assert tombstone["restored_at"] is not None
    paper = PaperMetadata(
        doi="10.1000/missing",
        title="A possible missing paper",
        authors=["Jane Researcher"],
        issue_date=date(2026, 7, 25),
        source_metadata={"DOI": "10.1000/missing", "ISSN": ["0272-7757"]},
    )
    with database.connect() as connection:
        journal_id = connection.execute(
            "SELECT id FROM journals WHERE short_name='EER'"
        ).fetchone()[0]
    assert paper_is_tombstoned(database, paper, journal_id) is False
    results = PersonalJournalSyncService(
        database, _Crossref([paper])
    ).sync(
        (_Journal(),),
        since=date(2026, 5, 1),
        until=date(2026, 8, 1),
        per_journal_limit=100,
    )
    assert results[0].ignored == 0


def test_intake_missing_identity_rejected_and_openalex_rescue(tmp_path: Path) -> None:
    database = _database(tmp_path)
    run_id, _ = _sync(database, _base_paper())
    no_doi = {"source_id": "W0000000001", "title": "No DOI candidate"}
    with pytest.raises(CandidateMissingIdentity):
        intake_verification_candidate(
            database,
            run_id,
            no_doi,
            registry=_registry(),
            crossref=None,
            openalex=None,
            services=_services(database),
        )
    rescued = intake_verification_candidate(
        database,
        run_id,
        no_doi,
        registry=_registry(),
        crossref=None,
        openalex=_FakeOpenAlex(
            _openalex_work("10.1000/rescued", "No DOI candidate", "W0000000001")
        ),
        services=_services(database),
    )
    assert rescued.outcome == "created"
    assert rescued.doi == "10.1000/rescued"


def test_partial_failures_preserve_paper_and_record_status(tmp_path: Path) -> None:
    database = _database(tmp_path)
    run_id, _ = _sync(database, _base_paper())
    candidate = _candidate("10.1000/missing", "A possible missing paper")
    crossref = _Crossref(
        [
            _crossref_record("10.1000/missing", "A possible missing paper"),
        ]
    )
    # Enrichment and screening both fail; the paper must stay in the library
    # with a clear needs_retry state and the title translation preserved.
    services = _services(
        database,
        enrichment_fail=True,
        screening_fail=True,
    )
    result = intake_verification_candidate(
        database,
        run_id,
        candidate,
        registry=_registry(),
        crossref=crossref,
        openalex=_FakeOpenAlex(None),
        services=services,
    )
    assert result.outcome == "created"
    assert result.review_ready is False
    assert any("自动处理失败" in message for message in result.errors)
    with database.connect() as connection:
        assert connection.execute("SELECT COUNT(*) FROM papers").fetchone()[0] == 2
        status = connection.execute(
            """SELECT status,error_type,error_message FROM paper_processing_status
            WHERE sync_run_id=? AND paper_id=?""",
            (run_id, result.paper_id),
        ).fetchone()
        assert status["status"] == "needs_retry"
        assert status["error_type"] in {"intake_chain", "intake_title_translation"}
        translation = connection.execute(
            "SELECT chinese_title FROM paper_translations WHERE paper_id=?",
            (result.paper_id,),
        ).fetchone()
        assert translation["chinese_title"].startswith("中文标题")


def test_title_translation_failure_keeps_paper_and_chain_continues(
    tmp_path: Path,
) -> None:
    database = _database(tmp_path)
    run_id, _ = _sync(database, _base_paper())
    candidate = _candidate("10.1000/missing", "A possible missing paper")
    crossref = _Crossref(
        [
            _crossref_record("10.1000/missing", "A possible missing paper"),
        ]
    )
    result = intake_verification_candidate(
        database,
        run_id,
        candidate,
        registry=_registry(),
        crossref=crossref,
        openalex=_FakeOpenAlex(None),
        services=_services(database, title_fail=True),
    )
    assert result.outcome == "created"
    assert any("标题翻译失败" in message for message in result.errors)
    with database.connect() as connection:
        status = connection.execute(
            "SELECT status,error_type FROM paper_processing_status "
            "WHERE sync_run_id=? AND paper_id=?",
            (run_id, result.paper_id),
        ).fetchone()
        assert status["status"] == "classified"
        assert status["error_type"] == "intake_title_translation"
        assert connection.execute("SELECT COUNT(*) FROM papers").fetchone()[0] == 2


def test_intake_without_ai_is_pending_not_rolled_back(tmp_path: Path) -> None:
    database = _database(tmp_path)
    run_id, _ = _sync(database, _base_paper())
    candidate = _candidate("10.1000/missing", "A possible missing paper")
    crossref = _Crossref(
        [
            _crossref_record("10.1000/missing", "A possible missing paper"),
        ]
    )
    result = intake_verification_candidate(
        database,
        run_id,
        candidate,
        registry=_registry(),
        crossref=crossref,
        openalex=_FakeOpenAlex(None),
        services=_services(database, no_ai=True),
    )
    assert result.outcome == "created"
    with database.connect() as connection:
        assert connection.execute("SELECT COUNT(*) FROM papers").fetchone()[0] == 2
        status = connection.execute(
            "SELECT status,stage,error_type FROM paper_processing_status "
            "WHERE sync_run_id=? AND paper_id=?",
            (run_id, result.paper_id),
        ).fetchone()
        assert status["status"] == "needs_retry"
        title_json = json.loads(
            connection.execute(
                "SELECT title_translation_json FROM paper_processing_status "
                "WHERE sync_run_id=? AND paper_id=?",
                (run_id, result.paper_id),
            ).fetchone()[0]
        )
        assert title_json["status"] == "title_translation_pending"


def test_batch_intake_isolates_failures(tmp_path: Path) -> None:
    database = _database(tmp_path)
    run_id, _ = _sync(database, _base_paper())
    good = _candidate("10.1000/good", "A good candidate")
    bad = {
        "source_id": "W9999999999",
        "title": "Unresolvable candidate",
        "publication_date": "2026-07-01",
    }
    crossref = _Crossref(
        [
            _crossref_record("10.1000/good", "A good candidate"),
        ]
    )
    results = intake_verification_candidates(
        database,
        run_id,
        (bad, good),
        registry=_registry(),
        crossref=crossref,
        openalex=_FakeOpenAlex(None),
        services=_services(database),
    )
    outcomes = {result.outcome for result in results}
    assert outcomes == {"skipped", "created"}
    skipped = next(result for result in results if result.outcome == "skipped")
    assert skipped.paper_id == -1
    assert any("DOI" in message for message in skipped.errors)
    with database.connect() as connection:
        assert connection.execute("SELECT COUNT(*) FROM papers").fetchone()[0] == 2


def test_batch_intake_survives_unexpected_journal_error(tmp_path: Path) -> None:
    database = _database(tmp_path)
    run_id, _ = _sync(database, _base_paper())
    with database.connect() as connection:
        connection.execute(
            "UPDATE sync_runs SET selected_journal_keys=? WHERE id=?",
            (json.dumps(["missing_journal_key"]), run_id),
        )
    good = _candidate("10.1000/good", "A good candidate")
    bad = _candidate("10.9999/none", "No journal candidate")
    crossref = _Crossref(
        [
            _crossref_record("10.1000/good", "A good candidate"),
        ]
    )
    results = intake_verification_candidates(
        database,
        run_id,
        (bad, good),
        registry=_registry(),
        crossref=crossref,
        openalex=_FakeOpenAlex(None),
        services=_services(database),
    )
    assert {result.outcome for result in results} == {"skipped", "created"}
    skipped = next(result for result in results if result.outcome == "skipped")
    assert any("期刊" in message for message in skipped.errors)
    with database.connect() as connection:
        assert connection.execute("SELECT COUNT(*) FROM papers").fetchone()[0] == 2


def test_journal_resolution_fallback_without_run_selection(tmp_path: Path) -> None:
    from eduecon_radar.services.verification_intake import (
        JournalResolutionError,
        _resolve_journal,
    )

    database = _database(tmp_path)
    # A legacy-style run dict with no selected keys and no provider ISSNs must
    # fail with a friendly journal error instead of an uncaught KeyError.
    with pytest.raises(JournalResolutionError):
        _resolve_journal(
            database,
            _registry(),
            {},
            crossref_record=None,
            openalex_work=None,
            doi="10.1000/no-journal",
        )
    journal_id, journal_name = _resolve_journal(
        database,
        _registry(),
        {},
        crossref_record=_crossref_record("10.1000/missing", "A possible missing paper"),
        openalex_work=None,
        doi="10.1000/missing",
    )
    assert journal_name == "Economics of Education Review"
    assert journal_id > 0


def test_ordinary_render_never_intakes_or_calls_providers(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    from streamlit.testing.v1 import AppTest

    from eduecon_radar.services.deepseek import DeepSeekClient
    from eduecon_radar.services.retrieval_verification import OpenAlexVerificationSource
    from eduecon_radar.ui.views import personal_mode as ui_personal_mode

    database = _database(tmp_path)
    run_id, _ = _sync(database, _base_paper())
    candidate = _candidate("10.1000/missing", "A possible missing paper")
    _verify_with_candidate(database, run_id, candidate)
    monkeypatch.setenv("DATABASE_URL", f"sqlite:///{database.path.as_posix()}")
    monkeypatch.setenv("DEEPSEEK_API_KEY", "")
    monkeypatch.setenv("LITERATURE_RADAR_MODE", "local")
    monkeypatch.setattr(
        "eduecon_radar.services.product_access.local_request_allowed", lambda _host: True
    )

    def forbidden(*_args, **_kwargs):
        raise AssertionError("ordinary render must not query providers or AI")

    monkeypatch.setattr(OpenAlexVerificationSource, "fetch", forbidden)
    monkeypatch.setattr(DeepSeekClient, "request_json", forbidden)
    monkeypatch.setattr(httpx.Client, "send", forbidden)
    monkeypatch.setattr(ui_personal_mode, "_build_services", forbidden)
    monkeypatch.setattr(ui_personal_mode, "_intake_metadata_clients", forbidden)
    app = AppTest.from_file(
        str(ROOT / "src/eduecon_radar/ui/app.py"), default_timeout=60
    ).run()
    app.segmented_control(key="edr_primary_view").set_value("文献工作台").run()
    app.segmented_control[1].set_value("检索核验").run()
    assert not app.exception
    with database.connect() as connection:
        assert connection.execute("SELECT COUNT(*) FROM papers").fetchone()[0] == 1
    assert "加入并处理" in {item.label for item in app.button}


def test_ui_click_intakes_candidate_and_shows_review_entry(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    from streamlit.testing.v1 import AppTest

    from eduecon_radar.services.deepseek import DeepSeekClient
    from eduecon_radar.services.retrieval_verification import OpenAlexVerificationSource
    from eduecon_radar.ui.views import personal_mode as ui_personal_mode

    database = _database(tmp_path)
    run_id, _ = _sync(database, _base_paper())
    candidate = _candidate("10.1000/missing", "A possible missing paper")
    _verify_with_candidate(database, run_id, candidate)
    monkeypatch.setenv("DATABASE_URL", f"sqlite:///{database.path.as_posix()}")
    monkeypatch.setenv("DEEPSEEK_API_KEY", "")
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
        ui_personal_mode,
        "_build_services",
        lambda _database, _settings, _profile=None: _services(database),
    )
    monkeypatch.setattr(
        ui_personal_mode,
        "_intake_metadata_clients",
        lambda _settings: (
            _Crossref(
                [
                    _crossref_record("10.1000/missing", "A possible missing paper"),
                ]
            ),
            _FakeOpenAlex(None),
        ),
    )
    app = AppTest.from_file(
        str(ROOT / "src/eduecon_radar/ui/app.py"), default_timeout=90
    ).run()
    app.segmented_control(key="edr_primary_view").set_value("文献工作台").run()
    app.segmented_control[1].set_value("检索核验").run()
    assert not app.exception
    intake_button = next(item for item in app.button if item.label == "加入并处理")
    intake_button.click().run()
    assert not app.exception
    with database.connect() as connection:
        assert connection.execute("SELECT COUNT(*) FROM papers").fetchone()[0] == 2
        status = connection.execute(
            """SELECT status FROM paper_processing_status
            WHERE sync_run_id=? AND paper_id IN
            (SELECT id FROM papers WHERE doi='10.1000/missing')""",
            (run_id,),
        ).fetchone()
        assert status["status"] == "classified"
    labels = {item.label for item in app.button}
    assert "去审核" in labels


def test_ui_intake_notice_survives_rerun_for_skipped_candidate(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    from streamlit.testing.v1 import AppTest

    from eduecon_radar.services.deepseek import DeepSeekClient
    from eduecon_radar.ui.views import personal_mode as ui_personal_mode

    database = _database(tmp_path)
    run_id, _ = _sync(database, _base_paper())
    good = _candidate("10.1000/good", "A good candidate")
    bad = _candidate("10.9999/none", "No journal candidate")
    verify_sync_run(
        database,
        run_id,
        journal_issns=(("0272-7757",),),
        source=_TwoCandidateSource(good, bad),
    )
    with database.connect() as connection:
        connection.execute(
            "UPDATE sync_runs SET selected_journal_keys=? WHERE id=?",
            (json.dumps(["missing_journal_key"]), run_id),
        )
    monkeypatch.setenv("DATABASE_URL", f"sqlite:///{database.path.as_posix()}")
    monkeypatch.setenv("DEEPSEEK_API_KEY", "")
    monkeypatch.setenv("LITERATURE_RADAR_MODE", "local")
    monkeypatch.setattr(
        "eduecon_radar.services.product_access.local_request_allowed", lambda _host: True
    )

    def forbidden(*_args, **_kwargs):
        raise AssertionError("unexpected provider call")

    monkeypatch.setattr(DeepSeekClient, "request_json", forbidden)
    monkeypatch.setattr(httpx.Client, "send", forbidden)
    monkeypatch.setattr(
        ui_personal_mode,
        "_build_services",
        lambda _database, _settings, _profile=None: _services(database),
    )
    monkeypatch.setattr(
        ui_personal_mode,
        "_intake_metadata_clients",
        lambda _settings: (
            _Crossref(
                [
                    _crossref_record("10.1000/good", "A good candidate"),
                ]
            ),
            _FakeOpenAlex(None),
        ),
    )
    app = AppTest.from_file(
        str(ROOT / "src/eduecon_radar/ui/app.py"), default_timeout=90
    ).run()
    app.segmented_control(key="edr_primary_view").set_value("文献工作台").run()
    app.segmented_control[1].set_value("检索核验").run()
    assert not app.exception
    batch = next(item for item in app.button if "全部加入并处理" in item.label)
    batch.click().run()
    assert not app.exception
    with database.connect() as connection:
        assert connection.execute("SELECT COUNT(*) FROM papers").fetchone()[0] == 2
        assert connection.execute(
            "SELECT COUNT(*) FROM papers WHERE doi='10.1000/good'"
        ).fetchone()[0] == 1
        assert connection.execute(
            "SELECT COUNT(*) FROM papers WHERE doi='10.9999/none'"
        ).fetchone()[0] == 0
    assert app.error
    assert any("期刊" in str(item.value) for item in app.error)


class _TwoCandidateSource:
    source_name = "Fixture index"
    query_count = 1

    def __init__(self, first: dict[str, object], second: dict[str, object]):
        self.first = first
        self.second = second

    def fetch(self, _issns, _since, _until):
        return (
            IndependentWork(
                str(self.first.get("source_id") or ""),
                str(self.first.get("doi") or "") or None,
                str(self.first.get("title") or ""),
                str(self.first.get("publication_date") or ""),
                str(self.first.get("work_type") or ""),
                False,
            ),
            IndependentWork(
                str(self.second.get("source_id") or ""),
                str(self.second.get("doi") or "") or None,
                str(self.second.get("title") or ""),
                str(self.second.get("publication_date") or ""),
                str(self.second.get("work_type") or ""),
                False,
            ),
        )


def test_state_helper_is_database_only_and_correct(tmp_path: Path) -> None:
    database = _database(tmp_path)
    run_id, _ = _sync(database, _base_paper())
    candidate = _candidate("10.1000/missing", "A possible missing paper")
    state = candidate_intake_state(database, run_id, candidate)
    assert state.in_library is False
    assert state.tombstoned is False
    crossref = _Crossref(
        [
            _crossref_record("10.1000/missing", "A possible missing paper"),
        ]
    )
    intake_verification_candidate(
        database,
        run_id,
        candidate,
        registry=_registry(),
        crossref=crossref,
        openalex=_FakeOpenAlex(None),
        services=_services(database),
    )
    state = candidate_intake_state(database, run_id, candidate)
    assert state.in_library is True
    assert state.attached_to_run is True
    assert state.review_ready is True
    assert state.doi == "10.1000/missing"
    assert state.human_decision == "undecided"


def test_batch_state_helper_matches_single_and_tolerates_malformed(
    tmp_path: Path,
) -> None:
    database = _database(tmp_path)
    run_id, _ = _sync(database, _base_paper())
    candidate = _candidate("10.1000/missing", "A possible missing paper")
    crossref = _Crossref(
        [
            _crossref_record("10.1000/missing", "A possible missing paper"),
        ]
    )
    intake_verification_candidate(
        database,
        run_id,
        candidate,
        registry=_registry(),
        crossref=crossref,
        openalex=_FakeOpenAlex(None),
        services=_services(database),
    )
    malformed = {"title": "broken candidate"}  # no DOI, unexpected shape
    states = candidate_intake_states(
        database, run_id, (candidate, malformed)
    )
    assert states[0] == candidate_intake_state(database, run_id, candidate)
    assert states[0].review_ready is True
    assert states[1].in_library is False
    assert states[1].tombstoned is False
    assert states[1].doi is None


def test_title_matching_does_not_create_duplicate(tmp_path: Path) -> None:
    database = _database(tmp_path)
    run_id, base_paper_id = _sync(database, _base_paper())
    # Same title and author but a different DOI string representation must
    # still resolve to the existing row instead of a second paper.
    duplicate = _candidate("10.1000/phase7c4-base", "A baseline retrieved paper")
    result = intake_verification_candidate(
        database,
        run_id,
        duplicate,
        registry=_registry(),
        crossref=_Crossref(
            [
                _crossref_record("10.1000/phase7c4-base", "A baseline retrieved paper"),
            ]
        ),
        openalex=_FakeOpenAlex(None),
        services=_services(database),
    )
    assert result.outcome == "attached_existing"
    assert result.paper_id == base_paper_id
    with database.connect() as connection:
        assert connection.execute("SELECT COUNT(*) FROM papers").fetchone()[0] == 1
