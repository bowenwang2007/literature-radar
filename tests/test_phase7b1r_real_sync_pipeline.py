from __future__ import annotations

import hashlib
import json
from datetime import date
from pathlib import Path

import httpx

from eduecon_radar.db import Database, utcnow
from eduecon_radar.schemas import JournalConfig, PaperMetadata, TitleTranslationResult
from eduecon_radar.services.abstract_enrichment import (
    AbstractAuthenticationError,
    AbstractCandidate,
    AbstractEnrichmentService,
    AbstractSourceAdapter,
    CrossrefEnrichmentAdapter,
    EnrichmentPaperInput,
    OpenAlexEnrichmentAdapter,
)
from eduecon_radar.services.abstract_normalization import normalize_abstract
from eduecon_radar.services.deepseek import DeepSeekError, DeepSeekResponse
from eduecon_radar.services.migrations import (
    apply_p0_hardening_migration,
    apply_translation_migration,
    apply_workflow_migration,
)
from eduecon_radar.services.personal_mode import (
    PersonalJournalSyncService,
    create_personal_task,
    enrich_personal_task_abstracts,
    load_personal_state,
    personal_task_snapshot,
    translate_personal_task_titles,
)
from eduecon_radar.services.profile_registry import default_profile_registry
from eduecon_radar.services.translation import TranslationService

ROOT = Path(__file__).resolve().parents[1]


def _database(tmp_path: Path) -> Database:
    database = Database(tmp_path / "phase7b1r.db")
    database.init_schema()
    apply_translation_migration(
        database, ROOT / "migrations/0001_add_paper_translations.sql"
    )
    apply_workflow_migration(
        database, ROOT / "migrations/0002_add_workflow_batches.sql"
    )
    apply_p0_hardening_migration(
        database, ROOT / "migrations/0007_add_runtime_leases_and_task_reviews.sql"
    )
    return database


def _registry():
    return default_profile_registry()


class FakeCrossref:
    def __init__(self, records: list[PaperMetadata]) -> None:
        self.records = records

    def fetch(self, _issn, _since, limit, *, until=None):
        del until
        return list(self.records[:limit])


def _records(count: int = 5) -> list[PaperMetadata]:
    return [
        PaperMetadata(
            doi=f"10.1000/exact-{index}",
            title=f"Education paper {index}",
            abstract=None,
            authors=["Researcher"],
            publisher="Fixture Publisher",
            issue_date=date(2026, 7, index),
            source_metadata={
                "DOI": f"10.1000/exact-{index}",
                "ISSN": ["0272-7757"],
                "container-title": ["Economics of Education Review"],
            },
        )
        for index in range(1, count + 1)
    ]


def _sync(database: Database, records: list[PaperMetadata]):
    journal = _registry().by_key("economics_of_education_review")
    result = PersonalJournalSyncService(database, FakeCrossref(records)).sync(
        (journal,),
        since=date(2026, 5, 1),
        until=date(2026, 7, 31),
        per_journal_limit=5,
    )[0]
    return journal, result


def test_sync_persists_exact_matched_ids_for_new_and_unchanged_records(tmp_path) -> None:
    database = _database(tmp_path)
    journal, first = _sync(database, _records())
    assert first.found_count == 5
    assert len(first.matched_paper_ids) == 5
    assert len(set(first.matched_paper_ids)) == 5
    assert first.inserted == 5
    assert first.found == (
        first.inserted
        + first.updated
        + first.unchanged
        + first.rejected
        + first.duplicates
    )

    _, second = _sync(database, _records())
    assert second.matched_paper_ids == first.matched_paper_ids
    assert second.unchanged == 5
    assert second.inserted == second.updated == 0
    created = create_personal_task(
        database,
        journal_results=(second,),
        selected_journals=(journal,),
        since=date(2026, 5, 1),
        until=date(2026, 7, 31),
        per_journal_limit=5,
    )
    with database.connect() as connection:
        row = connection.execute(
            "SELECT source_config_json FROM workflow_batches WHERE id=?",
            (created.task_id,),
        ).fetchone()
    source = json.loads(row[0])
    assert source["runtime_mode"] == "personal_local"
    assert source["matched_paper_ids"] == first.matched_paper_ids
    assert set(load_personal_state(database, created.task_id)["items"]) == {
        str(value) for value in first.matched_paper_ids
    }


class FixtureTranslationClient:
    def __init__(self, *, fail_title: str | None = None, copy_title: bool = False):
        self.fail_title = fail_title
        self.copy_title = copy_title
        self.calls = 0

    def request_json(self, **kwargs):
        self.calls += 1
        prompt = kwargs["user_prompt"]
        original = prompt.split("Original title:", 1)[-1].strip()
        if self.fail_title and self.fail_title in prompt:
            raise DeepSeekError("fixture title failure")
        translated = original if self.copy_title else f"教育研究标题{self.calls}"
        return DeepSeekResponse(
            data=TitleTranslationResult(chinese_title=translated),
            model_name="fixture-title",
            input_tokens=2,
            output_tokens=2,
        )


def _task_with_records(database: Database, records: list[PaperMetadata]):
    journal, result = _sync(database, records)
    return create_personal_task(
        database,
        journal_results=(result,),
        selected_journals=(journal,),
        since=date(2026, 5, 1),
        until=date(2026, 7, 31),
        per_journal_limit=5,
    )


def test_immediate_title_translation_isolated_validated_and_persistent(tmp_path) -> None:
    database = _database(tmp_path)
    created = _task_with_records(database, _records(3))
    client = FixtureTranslationClient(fail_title="Education paper 2")
    service = TranslationService(database, ROOT / "prompts", client, "fixture-title")
    result = translate_personal_task_titles(database, created.task_id, service)
    assert result.translated == 2
    assert result.failed == 1
    snapshot = personal_task_snapshot(database, created.task_id)
    translated = [paper for paper in snapshot.papers if paper["chinese_title"]]
    assert len(translated) == 2
    with database.connect() as connection:
        metadata = json.loads(
            connection.execute(
                "SELECT source_metadata_json FROM papers WHERE id=?",
                (translated[0]["id"],),
            ).fetchone()[0]
        )
    provenance = metadata["title_translation_provenance"]
    assert provenance["model_name"] == "fixture-title"
    assert provenance["prompt_version"]
    assert provenance["source_title_sha256"] == hashlib.sha256(
        translated[0]["title"].encode()
    ).hexdigest()


def test_api_missing_and_copied_title_never_report_success(tmp_path) -> None:
    database = _database(tmp_path)
    created = _task_with_records(database, _records(1))
    pending = translate_personal_task_titles(database, created.task_id, None)
    assert pending.pending == 1
    item = load_personal_state(database, created.task_id)["items"][
        str(created.paper_ids[0])
    ]
    assert item["title_translation"]["status"] == "title_translation_pending"
    copied = translate_personal_task_titles(
        database,
        created.task_id,
        TranslationService(
            database,
            ROOT / "prompts",
            FixtureTranslationClient(copy_title=True),
            "fixture-title",
        ),
    )
    assert copied.failed == 1
    assert personal_task_snapshot(database, created.task_id).papers[0][
        "chinese_title"
    ] is None


class ErrorAdapter(AbstractSourceAdapter):
    source_name = "first_provider"

    def fetch(self, paper, context=None):
        raise AbstractAuthenticationError("not authorized", http_status=401)


class SuccessAdapter(AbstractSourceAdapter):
    source_name = "second_provider"

    def fetch(self, paper, context=None):
        return AbstractCandidate(
            "This study estimates the economic effects of education policy on student outcomes "
            "using detailed administrative data and a transparent empirical research design.",
            self.source_name,
            raw_doi=paper.doi,
            raw_title=paper.title,
            raw_authors=tuple(paper.authors),
            raw_year=paper.year,
            raw_journal=paper.journal,
        )


def test_global_abstract_chain_continues_and_ui_snapshot_reads_saved_result(tmp_path) -> None:
    database = _database(tmp_path)
    created = _task_with_records(database, _records(1))
    stats = enrich_personal_task_abstracts(
        database,
        created.task_id,
        AbstractEnrichmentService(
            database,
            adapters=[ErrorAdapter(), SuccessAdapter()],
            jitter=lambda: 0.0,
        ),
    )
    assert stats.enriched == 1
    snapshot = personal_task_snapshot(database, created.task_id)
    paper = snapshot.papers[0]
    assert paper["abstract"].startswith("This study estimates")
    assert paper["abstract_source"] == "second_provider"
    attempts = paper["abstract_enrichment"]["attempts"]
    assert [item["provider"] for item in attempts] == [
        "first_provider",
        "second_provider",
    ]
    assert attempts[0]["status"] == "unauthorized"
    assert attempts[1]["status"] == "success"
    assert paper["processing"]["abstract_retrieval"]["status"] == "abstract_ready"


class TitleFallbackAdapter(AbstractSourceAdapter):
    source_name = "title_fallback"

    def __init__(self) -> None:
        self.doi_calls = 0
        self.title_calls = 0

    def fetch(self, paper, context=None):
        self.doi_calls += 1
        return None

    def fetch_title_candidates(self, paper, context=None):
        self.title_calls += 1
        return (
            AbstractCandidate(
                "This paper studies education investments, learning, teacher behavior, and "
                "economic outcomes with a carefully validated empirical identification strategy.",
                self.source_name,
                raw_title=paper.title,
                raw_authors=tuple(paper.authors),
                raw_year=paper.year,
                raw_journal=paper.journal,
            ),
        )


def test_invalid_doi_skips_doi_request_and_uses_strict_title_fallback(tmp_path) -> None:
    database = _database(tmp_path)
    records = _records(1)
    records[0].doi = None
    records[0].source_metadata["DOI"] = "not-a-doi"
    created = _task_with_records(database, records)
    adapter = TitleFallbackAdapter()
    enrich_personal_task_abstracts(
        database,
        created.task_id,
        AbstractEnrichmentService(database, adapters=[adapter]),
    )
    assert adapter.doi_calls == 0
    assert adapter.title_calls == 1
    audit = personal_task_snapshot(database, created.task_id).papers[0][
        "abstract_enrichment"
    ]
    assert audit["doi_status"] == "invalid_doi"
    assert audit["attempts"][0]["query_mode"] == "title"


def test_provider_requests_explicitly_select_abstract_fields() -> None:
    requests: list[httpx.Request] = []

    def handler(request: httpx.Request) -> httpx.Response:
        requests.append(request)
        payload = {"message": {}} if "crossref" in request.url.host else {"results": []}
        return httpx.Response(200, request=request, json=payload)

    client = httpx.Client(transport=httpx.MockTransport(handler))
    paper = EnrichmentPaperInput(
        1,
        "https://doi.org/10.1000/ABC",
        "Education title",
        ["Researcher"],
        "Economics of Education Review",
        2026,
        "0272-7757",
    )
    crossref = CrossrefEnrichmentAdapter(client=client)
    crossref.fetch(paper)
    crossref.fetch_title_candidates(paper)
    OpenAlexEnrichmentAdapter(client=client).fetch(paper)
    assert "select" not in requests[0].url.params
    assert "abstract" in requests[1].url.params["select"].split(",")
    assert "abstract_inverted_index" in requests[2].url.params["select"].split(",")


def test_nested_and_paragraph_abstract_fields_normalize_to_plain_text() -> None:
    value = {
        "indexed_abstract": [
            "<jats:p>First &amp; second paragraph with education evidence.</jats:p>",
            "<p>Another paragraph with economic analysis and results.</p>",
        ]
    }
    normalized = normalize_abstract(value)
    assert "First & second paragraph" in normalized
    assert "Another paragraph" in normalized
    assert "<" not in normalized


def test_manual_title_is_preserved_when_task_titles_are_reprocessed(tmp_path) -> None:
    database = _database(tmp_path)
    created = _task_with_records(database, _records(1))
    now = utcnow()
    with database.connect() as connection:
        connection.execute(
            """INSERT INTO paper_translations
            (paper_id,chinese_title,title_translation_status,
             abstract_translation_status,title_translation_source,updated_at)
            VALUES (?,'人工标题','human_edited','unavailable','human_edited',?)""",
            (created.paper_ids[0], now),
        )
    client = FixtureTranslationClient()
    result = translate_personal_task_titles(
        database,
        created.task_id,
        TranslationService(database, ROOT / "prompts", client, "fixture-title"),
    )
    assert result.translated == 1
    assert client.calls == 0
    assert personal_task_snapshot(database, created.task_id).papers[0][
        "chinese_title"
    ] == "人工标题"


def test_source_title_change_invalidates_only_machine_translation(tmp_path) -> None:
    database = _database(tmp_path)
    created = _task_with_records(database, _records(1))
    translate_personal_task_titles(
        database,
        created.task_id,
        TranslationService(
            database, ROOT / "prompts", FixtureTranslationClient(), "fixture-title"
        ),
    )
    journal_id = database.upsert_journal(
        JournalConfig(
            name="Economics of Education Review",
            short_name="EER",
            issn="0272-7757",
            eissn="1873-7382",
        )
    )
    database.upsert_paper(
        PaperMetadata(
            doi="10.1000/exact-1",
            title="Education paper 1 revised",
            authors=["Researcher"],
            issue_date=date(2026, 7, 1),
            source_metadata={"DOI": "10.1000/exact-1"},
        ),
        journal_id,
    )
    with database.connect() as connection:
        row = connection.execute(
            "SELECT chinese_title,title_translation_status "
            "FROM paper_translations WHERE paper_id=?",
            (created.paper_ids[0],),
        ).fetchone()
    assert row["chinese_title"] is None
    assert row["title_translation_status"] == "pending"

    with database.connect() as connection:
        connection.execute(
            """UPDATE paper_translations SET chinese_title='人工修订标题',
            title_translation_status='human_edited',
            title_translation_source='human_edited' WHERE paper_id=?""",
            (created.paper_ids[0],),
        )
    database.upsert_paper(
        PaperMetadata(
            doi="10.1000/exact-1",
            title="Education paper 1 final punctuation!",
            authors=["Researcher"],
            issue_date=date(2026, 7, 1),
            source_metadata={"DOI": "10.1000/exact-1"},
        ),
        journal_id,
    )
    with database.connect() as connection:
        preserved = connection.execute(
            "SELECT chinese_title,title_translation_status "
            "FROM paper_translations WHERE paper_id=?",
            (created.paper_ids[0],),
        ).fetchone()
    assert preserved["chinese_title"] == "人工修订标题"
    assert preserved["title_translation_status"] == "human_edited"
