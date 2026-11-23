import json
from contextlib import suppress
from datetime import UTC, datetime, timedelta
from pathlib import Path

import httpx

from eduecon_radar.db import Database, utcnow
from eduecon_radar.schemas import AbstractTranslationResult
from eduecon_radar.services.abstract_enrichment import (
    AbstractCandidate,
    AbstractEnrichmentService,
    AbstractRateLimited,
    AbstractSourceAdapter,
    AbstractTemporaryFailure,
    AbstractWorkerInterrupted,
    CandidateMatchDecision,
    EnrichmentPaperInput,
    evaluate_candidate_match,
)
from eduecon_radar.services.deepseek import DeepSeekResponse
from eduecon_radar.services.migrations import (
    apply_translation_migration,
    apply_workflow_migration,
)
from eduecon_radar.services.prototype_workflow import (
    save_manual_abstract,
    task_processing_counts,
)
from eduecon_radar.services.resilient_http import (
    ResilientHttpClient,
    parse_datetime,
    parse_retry_after_seconds,
)
from eduecon_radar.services.translation import TranslationParameters, TranslationService

ROOT = Path(__file__).resolve().parents[1]


def _paper() -> EnrichmentPaperInput:
    return EnrichmentPaperInput(
        1,
        "https://doi.org/10.1000/ABC",
        "Returns to Education: Evidence from Schools",
        ["Jane Smith", "Alex Doe"],
        "Journal of Education Economics",
        2024,
        "1234-5678",
    )


def _candidate(**changes: object) -> AbstractCandidate:
    values: dict[str, object] = {
        "text": "This study estimates returns to education using longitudinal student data. " * 5,
        "source": "fixture",
        "raw_doi": "10.1000/abc",
        "raw_title": "Returns to Education: Evidence from Schools",
        "raw_year": 2024,
        "raw_authors": ("Jane Smith", "Alex Doe"),
        "raw_journal": "Journal of Education Economics",
    }
    values.update(changes)
    return AbstractCandidate(**values)


def _database(tmp_path, *, abstract: str | None = None, metadata: dict | None = None):
    database = Database(tmp_path / "h2.db")
    database.init_schema()
    apply_translation_migration(
        database, ROOT / "migrations" / "0001_add_paper_translations.sql"
    )
    apply_workflow_migration(
        database, ROOT / "migrations" / "0002_add_workflow_batches.sql"
    )
    now = utcnow()
    with database.connect() as connection:
        connection.execute(
            "INSERT INTO journals(id,name,short_name,issn,priority,created_at,updated_at) "
            "VALUES(1,'Journal of Education Economics','JEE','1234-5678','high',?,?)",
            (now, now),
        )
        connection.execute(
            "INSERT INTO papers(id,doi,fingerprint,title,abstract,authors_json,journal_id,"
            "issue_date,source_metadata_json,first_seen_at,last_seen_at,created_at,updated_at) "
            "VALUES(1,'10.1000/abc','fp','Returns to Education: Evidence from Schools',"
            "?,? ,1,'2024-05-01',?,?,?,?,?)",
            (
                abstract,
                '["Jane Smith","Alex Doe"]',
                json.dumps(metadata or {}),
                now,
                now,
                now,
                now,
            ),
        )
        connection.execute(
            "INSERT INTO workflow_batches"
            "(id,name,source_type,status,source_config_json,screening_policy,"
            "translation_policy,created_at,updated_at,result_summary_json) "
            "VALUES(1,'h2','sync','completed','{}','{}','{}',?,?,'{}')",
            (now, now),
        )
        connection.execute(
            "INSERT INTO workflow_batch_items(batch_id,paper_id,position,added_at) "
            "VALUES(1,1,1,?)",
            (now,),
        )
        connection.execute(
            "INSERT INTO reviews(paper_id,status) VALUES(1,'pending')"
        )
    return database


class _FixtureAdapter(AbstractSourceAdapter):
    def __init__(
        self,
        name: str,
        *,
        doi: AbstractCandidate | Exception | None = None,
        title: tuple[AbstractCandidate, ...] | Exception = (),
        calls: list[tuple[str, str]] | None = None,
    ) -> None:
        self._name = name
        self.doi = doi
        self.title = title
        self.calls = calls if calls is not None else []

    @property
    def source_name(self) -> str:
        return self._name

    def fetch(self, paper, context=None):
        self.calls.append((self.source_name, "doi"))
        if isinstance(self.doi, Exception):
            raise self.doi
        return self.doi

    def fetch_title_candidates(self, paper, context=None):
        self.calls.append((self.source_name, "title"))
        if isinstance(self.title, Exception):
            raise self.title
        return self.title


def _audit(database: Database) -> dict:
    with database.connect() as connection:
        value = connection.execute(
            "SELECT source_metadata_json FROM papers WHERE id=1"
        ).fetchone()[0]
    return json.loads(value)["abstract_enrichment"]


def test_title_match_decisions_are_deterministic_and_evidence_based():
    assert evaluate_candidate_match(_paper(), _candidate(), title_fallback=True).decision is (
        CandidateMatchDecision.ACCEPTED
    )
    assert evaluate_candidate_match(
        _paper(), _candidate(raw_doi="10.9999/conflict"), title_fallback=True
    ).decision is CandidateMatchDecision.REJECTED_CONFLICT
    assert evaluate_candidate_match(
        _paper(), _candidate(raw_year=2018), title_fallback=True
    ).decision is CandidateMatchDecision.REJECTED_CONFLICT
    assert evaluate_candidate_match(
        _paper(), _candidate(raw_authors=("Different Person",)), title_fallback=True
    ).decision is CandidateMatchDecision.REJECTED_CONFLICT
    assert evaluate_candidate_match(
        _paper(), _candidate(raw_doi=None), title_fallback=True
    ).decision is CandidateMatchDecision.AMBIGUOUS_REQUIRES_MANUAL_REVIEW
    assert evaluate_candidate_match(
        _paper(), _candidate(raw_title="Returns to Educ"), title_fallback=True
    ).decision is CandidateMatchDecision.REJECTED_LOW_CONFIDENCE


def test_matched_without_abstract_continues_doi_chain_and_short_circuits_success(tmp_path):
    calls: list[tuple[str, str]] = []
    empty = _candidate(text="", source="first")
    found = _candidate(source="second")
    database = _database(tmp_path)
    service = AbstractEnrichmentService(
        database,
        adapters=[
            _FixtureAdapter("first", doi=empty, calls=calls),
            _FixtureAdapter("second", doi=found, calls=calls),
        ],
        policy_version="h2",
    )

    stats = service.enrich_papers((1,))

    assert stats.enriched == 1
    assert calls == [("first", "doi"), ("second", "doi")]
    attempts = _audit(database)["attempts"]
    assert [item["result_state"] for item in attempts] == [
        "matched_without_abstract",
        "success_with_abstract",
    ]


def test_all_doi_queries_finish_before_title_fallback(tmp_path):
    calls: list[tuple[str, str]] = []
    database = _database(tmp_path)
    service = AbstractEnrichmentService(
        database,
        adapters=[
            _FixtureAdapter("first", title=(_candidate(source="first"),), calls=calls),
            _FixtureAdapter("second", calls=calls),
        ],
        policy_version="h2",
    )

    service.enrich_papers((1,))

    assert calls == [("first", "doi"), ("second", "doi"), ("first", "title")]
    assert [item["query_mode"] for item in _audit(database)["attempts"]] == [
        "doi",
        "doi",
        "title",
    ]


def test_ambiguous_title_candidate_is_audited_but_never_saved(tmp_path):
    database = _database(tmp_path)
    ambiguous = _candidate(raw_doi=None)
    service = AbstractEnrichmentService(
        database,
        adapters=[_FixtureAdapter("fixture", title=(ambiguous,))],
        policy_version="h2",
    )

    service.enrich_papers((1,))

    with database.connect() as connection:
        assert connection.execute("SELECT abstract FROM papers WHERE id=1").fetchone()[0] is None
    attempt = _audit(database)["attempts"][-1]
    assert attempt["result_state"] == "low_confidence_match"
    assert (
        attempt["rejected_candidates"][0]["match_decision"]
        == "ambiguous_requires_manual_review"
    )


def test_provider_query_cache_deduplicates_and_policy_upgrade_retries(tmp_path):
    calls: list[tuple[str, str]] = []
    database = _database(tmp_path)
    adapter = _FixtureAdapter("fixture", calls=calls)
    service = AbstractEnrichmentService(
        database, adapters=[adapter], policy_version="h2-v1"
    )
    service.enrich_papers((1,))
    first_count = len(calls)

    service.enrich_papers((1,))
    assert len(calls) == first_count
    assert all(item["cache_hit"] for item in _audit(database)["attempts"])

    AbstractEnrichmentService(
        database, adapters=[adapter], policy_version="h2-v2"
    ).enrich_papers((1,))
    assert len(calls) > first_count


def test_successful_abstract_is_reused_without_another_provider_call(tmp_path):
    calls: list[tuple[str, str]] = []
    database = _database(tmp_path)
    service = AbstractEnrichmentService(
        database,
        adapters=[_FixtureAdapter("fixture", doi=_candidate(), calls=calls)],
        policy_version="h2",
    )
    first = service.enrich_papers((1,))
    call_count = len(calls)
    second = service.enrich_papers((1,))

    assert first.enriched == 1
    assert second.skipped_have_abstract == 1
    assert len(calls) == call_count


def test_retryable_cache_is_short_and_backoff_is_capped(tmp_path):
    now = datetime(2026, 7, 29, tzinfo=UTC)
    current = [now]
    calls: list[tuple[str, str]] = []
    adapter = _FixtureAdapter(
        "fixture",
        doi=AbstractTemporaryFailure("temporary"),
        calls=calls,
    )
    database = _database(tmp_path)
    service = AbstractEnrichmentService(
        database,
        adapters=[adapter],
        policy_version="h2",
        clock=lambda: current[0],
        jitter=lambda: 0.0,
        max_backoff_seconds=8.0,
    )

    service.enrich_papers((1,))
    retry_at = datetime.fromisoformat(_audit(database)["attempts"][0]["next_retry_at"])
    assert timedelta(0) < retry_at - now <= timedelta(seconds=8)
    first_count = len(calls)
    service.enrich_papers((1,))
    assert len(calls) == first_count
    current[0] = retry_at + timedelta(milliseconds=1)
    service.enrich_papers((1,))
    assert len(calls) > first_count


def test_rate_limit_timeout_and_interruption_have_distinct_retryable_states(tmp_path):
    errors = (
        (
            AbstractRateLimited("limited", http_status=429, retry_after=30),
            "rate_limited",
        ),
        (AbstractTemporaryFailure("timeout", failure_kind="timeout"), "timeout"),
        (AbstractWorkerInterrupted("closed"), "interrupted"),
    )
    for index, (error, expected) in enumerate(errors):
        database = _database(tmp_path / str(index))
        AbstractEnrichmentService(
            database,
            adapters=[_FixtureAdapter("fixture", doi=error)],
            policy_version="h2",
        ).enrich_papers((1,))
        attempt = _audit(database)["attempts"][0]
        assert attempt["result_state"] == expected
        assert attempt["retryable"] is True


def test_retry_after_larger_than_inline_limit_is_not_slept():
    sleeps: list[float] = []
    request = httpx.Request("GET", "https://example.test/work")
    transport = httpx.MockTransport(
        lambda _: httpx.Response(429, headers={"Retry-After": "120"}, request=request)
    )
    client = ResilientHttpClient(
        httpx.Client(transport=transport),
        max_retries=2,
        sleeper=sleeps.append,
        rate_limit_per_second=0,
        max_inline_sleep_seconds=5,
    )

    with suppress(httpx.HTTPStatusError):
        client.get("https://example.test/work")

    assert sleeps == []


def test_provider_year_string_does_not_break_candidate_matching():
    evaluation = evaluate_candidate_match(
        _paper(),
        _candidate(raw_year="2024"),
        title_fallback=True,
    )

    assert evaluation.decision is CandidateMatchDecision.ACCEPTED
    assert evaluation.evidence["year_delta"] == 0


def test_parse_datetime_accepts_aware_naive_iso_and_none():
    aware = datetime(2026, 7, 29, 12, 0, tzinfo=UTC)

    assert parse_datetime(aware) == aware
    assert parse_datetime("2026-07-29T12:00:00Z") == aware
    assert parse_datetime("2026-07-29T12:00:00") == aware
    assert parse_datetime(None) is None


def test_retry_after_accepts_int_string_iso_and_none():
    now = datetime(2026, 7, 29, 12, 0, tzinfo=UTC)

    assert parse_retry_after_seconds(7, now=now) == 7.0
    assert parse_retry_after_seconds("7", now=now) == 7.0
    assert parse_retry_after_seconds("2026-07-29T12:00:09Z", now=now) == 9.0
    assert parse_retry_after_seconds(None, now=now) is None


def test_string_retry_after_and_iso_clock_produce_aware_retry_time(tmp_path):
    database = _database(tmp_path)
    service = AbstractEnrichmentService(
        database,
        adapters=[
            _FixtureAdapter(
                "fixture",
                doi=AbstractRateLimited("limited", retry_after="12"),
            )
        ],
        clock=lambda: "2026-07-29T12:00:00Z",
        jitter=lambda: 0.0,
    )

    service.enrich_papers((1,))
    attempt = _audit(database)["attempts"][0]

    assert attempt["retry_after"] == 12.0
    assert attempt["last_attempt_at"] == "2026-07-29T12:00:00+00:00"
    assert attempt["next_retry_at"] == "2026-07-29T12:00:12+00:00"


def test_manual_abstract_change_marks_ai_translation_stale(tmp_path):
    database = _database(
        tmp_path,
        abstract="Old English source abstract. " * 20,
        metadata={"abstract_source": "manual"},
    )
    now = utcnow()
    with database.connect() as connection:
        connection.execute(
            """INSERT INTO paper_translations
            (paper_id,chinese_abstract,title_translation_status,abstract_translation_status,
             abstract_translation_source,translated_at,updated_at)
            VALUES(1,'旧中文摘要','pending','ai_generated','ai_generated',?,?)""",
            (now, now),
        )

    save_manual_abstract(
        database,
        1,
        "New human supplied English source abstract with changed content. " * 10,
    )

    with database.connect() as connection:
        row = connection.execute(
            """SELECT pt.chinese_abstract,pt.abstract_translation_status,
            p.source_metadata_json FROM paper_translations pt
            JOIN papers p ON p.id=pt.paper_id WHERE pt.paper_id=1"""
        ).fetchone()
    assert row["chinese_abstract"] == "旧中文摘要"
    assert row["abstract_translation_status"] == "pending"
    assert (
        json.loads(row["source_metadata_json"])["abstract_translation_provenance"][
            "status"
        ]
        == "stale"
    )

    class FakeTranslationClient:
        def request_json(self, **kwargs):
            assert kwargs["schema"] is AbstractTranslationResult
            return DeepSeekResponse(
                data=AbstractTranslationResult(chinese_abstract="更新后的中文摘要"),
                model_name="fixture",
                input_tokens=1,
                output_tokens=1,
            )

    service = TranslationService(
        database,
        ROOT / "prompts",
        FakeTranslationClient(),
        "fixture-translation",
    )
    stats = service.run(
        TranslationParameters(
            review_status=None,
            limit=1,
            missing_only=True,
            overwrite_human=False,
            paper_ids=(1,),
            translate_titles=False,
            translate_abstracts=True,
            task_id=1,
        )
    )
    assert stats.abstracts_translated == 1
    with database.connect() as connection:
        translated = connection.execute(
            """SELECT pt.chinese_abstract,pt.abstract_translation_status,
            p.source_metadata_json FROM paper_translations pt
            JOIN papers p ON p.id=pt.paper_id WHERE pt.paper_id=1"""
        ).fetchone()
    assert translated["chinese_abstract"] == "更新后的中文摘要"
    assert translated["abstract_translation_status"] == "ai_generated"
    translation_provenance = json.loads(translated["source_metadata_json"])[
        "abstract_translation_provenance"
    ]
    assert translation_provenance["status"] == "current"
    assert translation_provenance["source_abstract_sha256"]


def test_task_processing_counts_distinguish_manual_pending_and_exhausted(tmp_path):
    database = _database(
        tmp_path,
        abstract="Human supplied English source abstract. " * 10,
        metadata={"abstract_source": "manual"},
    )
    counts = task_processing_counts(database, 1)
    assert counts["manual_abstracts"] == 1
    assert counts["automatic_abstracts"] == 0
    assert counts["chinese_translation_pending"] == 1
    assert counts["needs_manual_abstract"] == 0


def test_task_processing_counts_match_exhausted_provider_audit(tmp_path):
    database = _database(tmp_path)
    AbstractEnrichmentService(
        database,
        adapters=[_FixtureAdapter("fixture")],
        policy_version="h2",
    ).enrich_papers((1,))

    counts = task_processing_counts(database, 1)
    assert counts["english_abstracts"] == 0
    assert counts["strategy_exhausted"] == 1
    assert counts["needs_manual_abstract"] == 1
    assert counts["retryable_failures"] == 0
