from __future__ import annotations

from collections.abc import MutableMapping
from dataclasses import dataclass
from datetime import date

from eduecon_radar.db import Database
from eduecon_radar.journals import JournalFile
from eduecon_radar.services.ingestion import SyncService


@dataclass(frozen=True)
class SyncParameters:
    journal_short_names: tuple[str, ...]
    since: date
    limit: int

    def __post_init__(self) -> None:
        names = tuple(sorted({name.strip() for name in self.journal_short_names if name.strip()}))
        object.__setattr__(self, "journal_short_names", names)
        if not names:
            raise ValueError("Select at least one journal")
        if not isinstance(self.since, date):
            raise ValueError("since must be a date")
        if not 1 <= self.limit <= 1000:
            raise ValueError("limit must be between 1 and 1000")

    @property
    def signature(self) -> tuple[tuple[str, ...], str, int]:
        return self.journal_short_names, self.since.isoformat(), self.limit


def selected_journal_file(journal_file: JournalFile, parameters: SyncParameters) -> JournalFile:
    requested = set(parameters.journal_short_names)
    available = {journal.short_name for journal in journal_file.journals if journal.enabled}
    unknown = requested - available
    if unknown:
        raise ValueError(f"Unknown or disabled journals: {sorted(unknown)}")
    selected = [
        journal
        for journal in journal_file.journals
        if journal.enabled and journal.short_name in requested
    ]
    return JournalFile(ranking=journal_file.ranking, journals=selected)


def run_sync(
    service: SyncService,
    journal_file: JournalFile,
    parameters: SyncParameters,
    *,
    dry_run: bool,
) -> dict[str, object]:
    return service.sync(
        selected_journal_file(journal_file, parameters),
        since=parameters.since,
        limit=parameters.limit,
        dry_run=dry_run,
    )


def dry_run_is_valid(
    parameters: SyncParameters,
    dry_run_signature: object,
    dry_run_result: dict[str, object] | None,
) -> bool:
    return bool(
        dry_run_result is not None
        and dry_run_signature == parameters.signature
        and dry_run_result.get("records_failed", 0) == 0
    )


def can_run_formal_sync(
    parameters: SyncParameters,
    dry_run_signature: object,
    dry_run_result: dict[str, object] | None,
    *,
    confirmed: bool,
    running: bool,
) -> bool:
    return (
        confirmed
        and not running
        and dry_run_is_valid(
            parameters,
            dry_run_signature,
            dry_run_result,
        )
    )


def claim_formal_sync(
    state: MutableMapping[str, object],
    signature: tuple[tuple[str, ...], str, int],
) -> bool:
    if state.get("data_sync_running") or state.get("data_sync_consumed_signature") == signature:
        return False
    state["data_sync_running"] = True
    state["data_sync_consumed_signature"] = signature
    return True


def finish_formal_sync(state: MutableMapping[str, object]) -> None:
    state["data_sync_running"] = False


def summarize_formal_sync(
    stats: dict[str, object],
    before: dict[str, object],
    after: dict[str, object],
    run_details: dict[str, object] | None,
) -> dict[str, object]:
    total_before = int(before.get("total", 0))
    total_after = int(after.get("total", 0))
    awaiting_before = int(before.get("awaiting_screening", 0))
    awaiting_after = int(after.get("awaiting_screening", 0))
    ai_calls = int((run_details or {}).get("ai_calls", after.get("ai_calls", 0)) or 0)
    return {
        **stats,
        "total_before": total_before,
        "total_after": total_after,
        "total_delta": total_after - total_before,
        "awaiting_screening_before": awaiting_before,
        "awaiting_screening_after": awaiting_after,
        "awaiting_screening_delta": awaiting_after - awaiting_before,
        "ai_calls": ai_calls,
        "ai_called": ai_calls > 0,
        "run_status": (run_details or {}).get("status"),
        "error_summary": (run_details or {}).get("error_summary"),
    }


def latest_sync_run(database: Database) -> dict[str, object] | None:
    with database.connect() as connection:
        row = connection.execute(
            """SELECT status,error_summary,ai_calls,finished_at FROM sync_runs
            ORDER BY id DESC LIMIT 1"""
        ).fetchone()
    return dict(row) if row else None
