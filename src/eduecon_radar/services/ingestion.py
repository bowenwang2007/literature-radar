# --- not part of the product core: see docs/PRODUCT_CORE.md ---
# LEGACY (v0.4 SyncService, writes the legacy ``screenings`` chain).
# Product retrieval uses ``services/journal_sync.py``.

from __future__ import annotations

import logging
from datetime import date, timedelta

from eduecon_radar.db import Database, utcnow
from eduecon_radar.journals import JournalFile
from eduecon_radar.schemas import PaperMetadata
from eduecon_radar.services.crossref import CrossrefClient
from eduecon_radar.services.migrations import product_safety_schema_ready
from eduecon_radar.services.paper_lifecycle import paper_is_tombstoned
from eduecon_radar.services.unpaywall import UnpaywallClient

LOGGER = logging.getLogger(__name__)


class SyncService:
    def __init__(
        self,
        database: Database,
        crossref: CrossrefClient,
        unpaywall: UnpaywallClient | None = None,
    ) -> None:
        self.database = database
        self.crossref = crossref
        self.unpaywall = unpaywall

    def _effective_since(self, journal_id: int | None, requested: date | None) -> date:
        if requested:
            return requested
        if journal_id is None:
            return date.today() - timedelta(days=30)
        with self.database.connect() as connection:
            row = connection.execute(
                "SELECT last_synced_at FROM journals WHERE id=?", (journal_id,)
            ).fetchone()
        if row and row["last_synced_at"]:
            return date.fromisoformat(row["last_synced_at"][:10])
        return date.today() - timedelta(days=30)

    def sync(
        self,
        journal_file: JournalFile,
        *,
        since: date | None = None,
        until: date | None = None,
        journal_short_name: str | None = None,
        limit: int | None = None,
        dry_run: bool = False,
    ) -> dict[str, object]:
        stats: dict[str, object] = {
            "journals_checked": 0,
            "records_fetched": 0,
            "records_without_abstract": 0,
            "records_inserted": 0,
            "records_updated": 0,
            "records_failed": 0,
            "records_ignored": 0,
            "inserted_paper_ids": [],
            "updated_paper_ids": [],
            "affected_paper_ids": [],
        }
        run_id = None if dry_run else self.database.start_sync_run()
        errors: list[str] = []
        try:
            for journal in journal_file.journals:
                if not journal.enabled:
                    continue
                if journal_short_name and journal.short_name.lower() != journal_short_name.lower():
                    continue
                if dry_run:
                    with self.database.connect() as connection:
                        existing = connection.execute(
                            "SELECT id FROM journals WHERE short_name=?", (journal.short_name,)
                        ).fetchone()
                    journal_id = int(existing["id"]) if existing else None
                else:
                    journal_id = self.database.upsert_journal(journal)
                effective_since = self._effective_since(journal_id, since)
                stats["journals_checked"] = int(stats["journals_checked"]) + 1
                try:
                    if until is None:
                        papers = self.crossref.fetch(journal.issn, effective_since, limit)
                    else:
                        papers = self.crossref.fetch(
                            journal.issn, effective_since, limit, until=until
                        )
                except Exception as exc:
                    stats["records_failed"] = int(stats["records_failed"]) + 1
                    errors.append(f"{journal.short_name}: {exc}")
                    LOGGER.exception("Crossref sync failed for %s", journal.short_name)
                    continue
                stats["records_fetched"] = int(stats["records_fetched"]) + len(papers)
                stats["records_without_abstract"] = int(stats["records_without_abstract"]) + sum(
                    not paper.abstract for paper in papers
                )
                if dry_run:
                    continue
                assert journal_id is not None
                for paper in papers:
                    try:
                        if product_safety_schema_ready(
                            self.database
                        ) and paper_is_tombstoned(self.database, paper, journal_id):
                            stats["records_ignored"] = int(stats["records_ignored"]) + 1
                            continue
                        self._enrich_open_access(paper)
                        paper_id, inserted = self.database.upsert_paper(paper, journal_id)
                        key = "records_inserted" if inserted else "records_updated"
                        stats[key] = int(stats[key]) + 1
                        id_key = "inserted_paper_ids" if inserted else "updated_paper_ids"
                        ids = stats[id_key]
                        assert isinstance(ids, list)
                        ids.append(paper_id)
                        affected = stats["affected_paper_ids"]
                        assert isinstance(affected, list)
                        affected.append(paper_id)
                    except Exception as exc:
                        stats["records_failed"] = int(stats["records_failed"]) + 1
                        errors.append(f"{journal.short_name}/{paper.doi or paper.title}: {exc}")
                with self.database.connect() as connection:
                    connection.execute(
                        "UPDATE journals SET last_synced_at=?,updated_at=? WHERE id=?",
                        (utcnow(), utcnow(), journal_id),
                    )
            if run_id is not None:
                status = "partial" if errors else "success"
                self.database.finish_sync_run(
                    run_id, status=status, stats=stats, error="\n".join(errors) or None
                )
            return stats
        except Exception as exc:
            if run_id is not None:
                self.database.finish_sync_run(run_id, status="failed", stats=stats, error=str(exc))
            raise

    def _enrich_open_access(self, paper: PaperMetadata) -> None:
        if not self.unpaywall or not paper.doi or paper.open_access_url:
            return
        try:
            paper.open_access_url = self.unpaywall.find_open_access_url(paper.doi)
        except Exception:
            LOGGER.warning("Unpaywall lookup failed for DOI %s", paper.doi)
