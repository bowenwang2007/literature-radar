from __future__ import annotations

import json
from datetime import date, datetime
from pathlib import Path

from eduecon_radar.db import Database
from eduecon_radar.schemas import PaperMetadata
from eduecon_radar.services.abstract_enrichment import (
    AbstractCandidate,
    AbstractEnrichmentService,
    AbstractRateLimited,
    AbstractSourceAdapter,
)
from eduecon_radar.services.abstract_source_config import AbstractSourceConfig
from eduecon_radar.services.migrations import (
    apply_overnight_migration,
    apply_p0_hardening_migration,
    apply_profile_migration,
    apply_translation_migration,
    apply_workflow_migration,
    overnight_schema_ready,
)
from eduecon_radar.services.profile_registry import default_profile_registry

ROOT = Path(__file__).resolve().parents[1]

ABSTRACT = (
    "This study estimates the economic effects of education policy on student outcomes "
    "using detailed administrative data and a transparent empirical research design."
)


def _database(tmp_path: Path) -> Database:
    database = Database(tmp_path / "fuse.db")
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
    for index in range(1, 4):
        paper_id, _ = database.upsert_paper(
            PaperMetadata(
                doi=f"10.1000/fuse-{index}",
                title=f"Education policy paper {index}",
                abstract=None,
                authors=["Researcher"],
                issue_date=date(2026, 7, index),
                source_metadata={"DOI": f"10.1000/fuse-{index}"},
            ),
            journal_id,
        )
        ids.append(paper_id)
    apply_profile_migration(
        database, ROOT / "migrations/0013_add_profile_scoped_assessment.sql"
    )
    return database


class RateLimitedAdapter(AbstractSourceAdapter):
    source_name = "semantic_scholar"

    def __init__(self) -> None:
        self.calls = 0

    def fetch(self, paper, context=None):
        self.calls += 1
        raise AbstractRateLimited("slow down", http_status=429, retry_after=120)


class SuccessAdapter(AbstractSourceAdapter):
    source_name = "openalex"

    def __init__(self) -> None:
        self.calls = 0

    def fetch(self, paper, context=None):
        self.calls += 1
        return AbstractCandidate(
            ABSTRACT,
            self.source_name,
            raw_doi=paper.doi,
            raw_title=paper.title,
            raw_authors=tuple(paper.authors),
            raw_year=paper.year,
            raw_journal=paper.journal,
        )


def _attempts(database: Database, paper_id: int) -> list[dict]:
    with database.connect() as connection:
        row = connection.execute(
            "SELECT source_metadata_json FROM papers WHERE id=?", (paper_id,)
        ).fetchone()
    return json.loads(row[0] or "{}").get("abstract_enrichment", {}).get("attempts", [])


def test_429_sets_provider_fuse_and_skips_remaining_papers(tmp_path) -> None:
    database = _database(tmp_path)
    limited = RateLimitedAdapter()
    success = SuccessAdapter()
    service = AbstractEnrichmentService(
        database,
        adapters=[limited, success],
        jitter=lambda: 0.0,
    )
    with database.connect() as connection:
        paper_ids = tuple(
            int(row[0])
            for row in connection.execute("SELECT id FROM papers ORDER BY id").fetchall()
        )
    stats = service.enrich_papers(paper_ids)
    assert limited.calls == 1
    assert success.calls == 3
    assert stats.enriched == 3
    first = _attempts(database, paper_ids[0])
    assert first[0]["result_state"] == "rate_limited"
    assert first[0]["status"] == "rate_limited"
    cooldown_until = first[0]["provider_cooldown_until"]
    assert cooldown_until
    attempt_at = datetime.fromisoformat(first[0]["last_attempt_at"])
    cooldown = datetime.fromisoformat(cooldown_until)
    assert (cooldown - attempt_at).total_seconds() >= 86400
    for paper_id in paper_ids[1:]:
        attempts = _attempts(database, paper_id)
        fused = next(
            item
            for item in attempts
            if item.get("provider") == "semantic_scholar"
        )
        assert fused["result_state"] == "provider_fused_skipped"
        assert fused["fused_skip"] is True
        assert fused["status"] == "rate_limited"
    with database.connect() as connection:
        logs = connection.execute(
            "SELECT provider,status,fused_skip FROM provider_call_log ORDER BY id"
        ).fetchall()
    assert len(logs) >= 5
    fused_logs = [row for row in logs if row["fused_skip"] == 1]
    assert len(fused_logs) == 2
    assert all(row["provider"] == "semantic_scholar" for row in fused_logs)


def test_same_doi_uses_in_memory_cache_across_papers(tmp_path) -> None:
    database = _database(tmp_path)
    with database.connect() as connection:
        connection.execute("UPDATE papers SET doi='10.1000/shared' WHERE id=1")
        connection.execute(
            """UPDATE papers SET doi=NULL,
            source_metadata_json='{"DOI":"10.1000/shared"}' WHERE id=2"""
        )
    success = SuccessAdapter()
    service = AbstractEnrichmentService(database, adapters=[success])
    stats = service.enrich_papers((1, 2))
    assert success.calls == 1
    assert stats.enriched == 2
    assert stats.cache_hits == 1


def test_negative_cache_prevents_repeat_not_found(tmp_path) -> None:
    database = _database(tmp_path)

    class NotFoundAdapter(AbstractSourceAdapter):
        source_name = "crossref"

        def __init__(self) -> None:
            self.calls = 0

        def fetch(self, paper, context=None):
            self.calls += 1
            return None

    adapter = NotFoundAdapter()
    service = AbstractEnrichmentService(database, adapters=[adapter])
    with database.connect() as connection:
        connection.execute("UPDATE papers SET doi='10.1000/shared' WHERE id=1")
        connection.execute(
            """UPDATE papers SET doi=NULL,
            source_metadata_json='{"DOI":"10.1000/shared"}' WHERE id=2"""
        )
    stats = service.enrich_papers((1, 2))
    assert adapter.calls == 1
    assert stats.cache_hits == 1
    assert stats.failed == 2


def test_disabled_provider_is_audit_metadata_not_a_fake_provider_call(tmp_path) -> None:
    database = _database(tmp_path)
    config = {
        "publisher_metadata": AbstractSourceConfig(
            name="publisher_metadata",
            enabled=False,
            priority=3,
            timeout_seconds=15,
            max_retries=1,
            rate_limit_per_second=1,
            requires_api_key=False,
            quality_threshold=0.9,
            allow_title_fallback=False,
            feature_status="guarded",
        )
    }
    service = AbstractEnrichmentService(
        database,
        adapters=[],
        source_configs=config,
    )
    stats = service.enrich_papers((1,))
    attempts = _attempts(database, 1)
    assert attempts == []
    assert "provider_not_configured" not in stats.by_result_state
    with database.connect() as connection:
        metadata = json.loads(
            connection.execute(
                "SELECT source_metadata_json FROM papers WHERE id=1"
            ).fetchone()[0]
        )
        count = connection.execute(
            "SELECT COUNT(*) FROM provider_call_log WHERE provider='publisher_metadata'"
        ).fetchone()[0]
    assert metadata["abstract_enrichment"]["disabled_sources"] == [
        "publisher_metadata"
    ]
    assert count == 0


def test_overnight_schema_ready_reflects_migration(tmp_path) -> None:
    from eduecon_radar.db import Database as Db

    plain = Db(tmp_path / "plain.db")
    plain.init_schema()
    assert not overnight_schema_ready(plain)
