from datetime import date

import pytest

from eduecon_radar.db import Database
from eduecon_radar.journals import JournalFile
from eduecon_radar.schemas import PaperMetadata
from eduecon_radar.services.ingestion import SyncService
from eduecon_radar.services.review import dashboard_stats
from eduecon_radar.services.web_sync import (
    SyncParameters,
    can_run_formal_sync,
    claim_formal_sync,
    dry_run_is_valid,
    latest_sync_run,
    run_sync,
    summarize_formal_sync,
)


def journal_file() -> JournalFile:
    return JournalFile.model_validate(
        {
            "ranking": {"name": "test", "frozen_at": "2026-01-01"},
            "journals": [
                {"name": "American Economic Review", "short_name": "AER", "issn": "1"},
                {"name": "Quarterly Journal", "short_name": "QJE", "issn": "2"},
            ],
        }
    )


class FakeSyncService:
    def __init__(self, result=None):
        self.result = result or {
            "journals_checked": 1,
            "records_fetched": 3,
            "records_inserted": 0,
            "records_updated": 0,
            "records_failed": 0,
        }
        self.calls = []

    def sync(self, selected, **kwargs):
        self.calls.append((selected, kwargs))
        return self.result


class FakeCrossref:
    def __init__(self, papers):
        self.papers = papers

    def fetch(self, issn, since, limit=None):
        return self.papers[:limit]


def test_sync_parameters_require_journals_and_positive_limit():
    with pytest.raises(ValueError, match="at least one journal"):
        SyncParameters((), date(2026, 7, 1), 5)
    with pytest.raises(ValueError, match="between 1 and 1000"):
        SyncParameters(("AER",), date(2026, 7, 1), 0)
    with pytest.raises(ValueError, match="between 1 and 1000"):
        SyncParameters(("AER",), date(2026, 7, 1), 1001)


def test_since_is_stored_as_iso_date_in_confirmation_signature():
    parameters = SyncParameters(("AER",), date(2026, 7, 1), 5)
    assert parameters.signature == (("AER",), "2026-07-01", 5)


def test_dry_run_reuses_sync_service_with_selected_journals():
    service = FakeSyncService()
    parameters = SyncParameters(("QJE", "AER"), date(2026, 7, 1), 5)

    result = run_sync(service, journal_file(), parameters, dry_run=True)

    selected, kwargs = service.calls[0]
    assert [journal.short_name for journal in selected.journals] == ["AER", "QJE"]
    assert kwargs == {"since": date(2026, 7, 1), "limit": 5, "dry_run": True}
    assert result["records_fetched"] == 3


def test_formal_sync_reuses_same_service_with_dry_run_false():
    service = FakeSyncService()
    parameters = SyncParameters(("AER",), date(2026, 7, 1), 5)

    run_sync(service, journal_file(), parameters, dry_run=False)

    assert service.calls[0][1]["dry_run"] is False


def test_formal_confirmation_requires_matching_successful_dry_run_and_checkbox():
    original = SyncParameters(("AER",), date(2026, 7, 1), 5)
    changed = SyncParameters(("AER",), date(2026, 7, 1), 6)
    successful = {"records_failed": 0}

    assert not dry_run_is_valid(original, None, None)
    assert dry_run_is_valid(original, original.signature, successful)
    assert not dry_run_is_valid(changed, original.signature, successful)
    assert not can_run_formal_sync(
        original,
        original.signature,
        successful,
        confirmed=False,
        running=False,
    )
    assert can_run_formal_sync(
        original,
        original.signature,
        successful,
        confirmed=True,
        running=False,
    )
    assert not dry_run_is_valid(original, original.signature, {"records_failed": 1})


def test_formal_sync_signature_can_only_be_claimed_once():
    state = {}
    signature = (("AER",), "2026-07-01", 5)
    assert claim_formal_sync(state, signature) is True
    state["data_sync_running"] = False
    assert claim_formal_sync(state, signature) is False


def test_formal_summary_calculates_four_insert_delta_and_ai_usage():
    summary = summarize_formal_sync(
        {
            "journals_checked": 1,
            "records_fetched": 4,
            "records_inserted": 4,
            "records_updated": 0,
            "records_failed": 0,
        },
        {"total": 10, "awaiting_screening": 2},
        {"total": 14, "awaiting_screening": 6, "ai_calls": 0},
        {"status": "success", "error_summary": None, "ai_calls": 0},
    )
    assert summary["total_delta"] == 4
    assert summary["awaiting_screening_delta"] == 4
    assert summary["ai_called"] is False


def test_partial_sync_summary_preserves_failure_status_and_error():
    summary = summarize_formal_sync(
        {"records_failed": 1},
        {"total": 10, "awaiting_screening": 2},
        {"total": 10, "awaiting_screening": 2},
        {"status": "partial", "error_summary": "AER: HTTP 500", "ai_calls": 0},
    )
    assert summary["records_failed"] == 1
    assert summary["run_status"] == "partial"
    assert summary["error_summary"] == "AER: HTTP 500"


def test_real_service_insert_delta_uses_temporary_database_only(tmp_path):
    db = Database(tmp_path / "web-sync.db")
    db.init_schema()
    papers = [PaperMetadata(doi=f"10.1/{index}", title=f"Paper {index}") for index in range(4)]
    service = SyncService(db, FakeCrossref(papers))
    parameters = SyncParameters(("AER",), date(2026, 7, 1), 4)
    before = dashboard_stats(db)

    stats = run_sync(service, journal_file(), parameters, dry_run=False)
    after = dashboard_stats(db)
    summary = summarize_formal_sync(stats, before, after, latest_sync_run(db))

    assert summary["records_inserted"] == 4
    assert summary["total_delta"] == 4
    assert summary["awaiting_screening_delta"] == 4
    assert summary["ai_called"] is False
