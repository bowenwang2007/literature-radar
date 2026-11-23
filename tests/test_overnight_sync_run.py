from __future__ import annotations

import json
from datetime import date
from pathlib import Path

from eduecon_radar.db import Database
from eduecon_radar.schemas import PaperMetadata
from eduecon_radar.services.migrations import (
    apply_overnight_migration,
    apply_p0_hardening_migration,
    apply_profile_migration,
    apply_translation_migration,
    apply_workflow_migration,
    overnight_schema_ready,
)
from eduecon_radar.services.personal_mode import PersonalJournalSyncService
from eduecon_radar.services.profile_registry import default_profile_registry

ROOT = Path(__file__).resolve().parents[1]


def _database(tmp_path: Path, *, overnight: bool = True) -> Database:
    database = Database(tmp_path / "sync_run.db")
    database.init_schema()
    apply_translation_migration(database, ROOT / "migrations/0001_add_paper_translations.sql")
    apply_workflow_migration(database, ROOT / "migrations/0002_add_workflow_batches.sql")
    apply_p0_hardening_migration(
        database, ROOT / "migrations/0007_add_runtime_leases_and_task_reviews.sql"
    )
    if overnight:
        apply_overnight_migration(
            database, ROOT / "migrations/0010_add_overnight_pipeline.sql"
        )
    apply_profile_migration(
        database, ROOT / "migrations/0013_add_profile_scoped_assessment.sql"
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


def _records(count: int = 3, *, revised: bool = False) -> list[PaperMetadata]:
    return [
        PaperMetadata(
            doi=f"10.1000/overnight-{index}",
            title=(
                f"Education paper {index} revised"
                if revised and index == 2
                else f"Education paper {index}"
            ),
            abstract=None,
            authors=["Researcher"],
            publisher="Fixture Publisher",
            issue_date=date(2026, 7, index),
            source_metadata={
                "DOI": f"10.1000/overnight-{index}",
                "ISSN": ["0272-7757"],
                "container-title": ["Economics of Education Review"],
            },
        )
        for index in range(1, count + 1)
    ]


def _sync(database: Database, records: list[PaperMetadata]):
    journal = _registry().by_key("economics_of_education_review")
    results = PersonalJournalSyncService(database, FakeCrossref(records)).sync(
        (journal,),
        since=date(2026, 5, 1),
        until=date(2026, 7, 31),
        per_journal_limit=5,
    )
    return journal, results


def _latest_sync_run(database: Database) -> dict:
    with database.connect() as connection:
        row = connection.execute(
            "SELECT * FROM sync_runs ORDER BY id DESC LIMIT 1"
        ).fetchone()
    return dict(row)


def test_overnight_sync_persists_precise_set_and_status(tmp_path) -> None:
    database = _database(tmp_path)
    journal, first_results = _sync(database, _records())
    first = _latest_sync_run(database)
    assert first["overall_status"] == "success"
    assert json.loads(first["selected_journal_keys"]) == [journal.journal_key]
    assert first["query_start_date"] == "2026-05-01"
    assert first["query_end_date"] == "2026-07-31"
    assert first["requested_limit"] == 5
    matched = json.loads(first["matched_paper_ids"])
    assert len(matched) == 3
    assert json.loads(first["inserted_paper_ids"]) == matched
    assert json.loads(first["updated_paper_ids"]) == []
    assert json.loads(first["unchanged_paper_ids"]) == []

    _, second_results = _sync(database, _records())
    second = _latest_sync_run(database)
    assert json.loads(second["matched_paper_ids"]) == matched
    assert json.loads(second["inserted_paper_ids"]) == []
    assert json.loads(second["unchanged_paper_ids"]) == matched

    _, third_results = _sync(database, _records(revised=True))
    third = _latest_sync_run(database)
    assert json.loads(third["matched_paper_ids"]) == matched
    assert json.loads(third["updated_paper_ids"]) == [matched[1]]
    assert json.loads(third["unchanged_paper_ids"]) == [matched[0], matched[2]]
    assert first_results[0].matched_paper_ids == matched
    assert second_results[0].matched_paper_ids == matched
    assert third_results[0].matched_paper_ids == matched


def test_overnight_sync_records_partial_errors_and_rejected_candidates(tmp_path) -> None:
    database = _database(tmp_path)

    class FailingCrossref:
        def fetch(self, _issn, _since, limit, *, until=None):
            del limit, until
            raise RuntimeError("provider exploded")

    journal = _registry().by_key("economics_of_education_review")
    PersonalJournalSyncService(database, FailingCrossref()).sync(
        (journal,),
        since=date(2026, 5, 1),
        until=date(2026, 7, 31),
        per_journal_limit=5,
    )
    row = _latest_sync_run(database)
    assert row["overall_status"] == "partial"
    assert json.loads(row["query_errors"])
    assert json.loads(row["rejected_candidates"])


def test_overnight_sync_falls_back_to_legacy_row_without_migration(tmp_path) -> None:
    database = _database(tmp_path, overnight=False)
    assert not overnight_schema_ready(database)
    _sync(database, _records())
    row = _latest_sync_run(database)
    assert row["status"] == "success"
    assert row["records_inserted"] == 3
    assert "selected_journal_keys" not in row
