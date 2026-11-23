"""Product retrieval: ISSN-first, multi-journal Crossref sync.

This is the retrieval half of the canonical product path.  It used to live in
``services/personal_mode.py`` together with the retired task pipeline, which
meant that opening the workbench imported the whole historical stack.  Only the
retrieval code moved here; nothing in this module imports a task, pilot,
runtime-bundle or domain module.

The sync writes an auditable snapshot row into ``sync_runs`` (the exact
``matched_paper_ids`` set plus provider diagnostics) and upserts the global
paper facts.  It never writes an assessment: judging a paper is the profile's
job (see ``services/profile_assessment.py``).
"""

from __future__ import annotations

import hashlib
import json
import logging
from dataclasses import dataclass, field
from datetime import date

from eduecon_radar.db import Database, utcnow
from eduecon_radar.schemas import PaperMetadata
from eduecon_radar.services.crossref import CrossrefClient
from eduecon_radar.services.deduplication import normalize_doi, normalize_title
from eduecon_radar.services.journal_registry import RegistryJournal, normalize_journal_name
from eduecon_radar.services.migrations import overnight_schema_ready, product_safety_schema_ready
from eduecon_radar.services.paper_lifecycle import paper_is_tombstoned

LOGGER = logging.getLogger(__name__)


@dataclass
class JournalSyncResult:
    journal_key: str
    canonical_name: str
    query_start_date: str = ""
    query_end_date: str = ""
    requested_limit: int = 0
    provider: str = "crossref"
    found: int = 0
    found_count: int = 0
    provider_returned_count: int = 0
    pages_requested: int = 0
    raw_candidate_count: int = 0
    normalized_count: int = 0
    stopped_reason: str | None = None
    inserted: int = 0
    updated: int = 0
    unchanged: int = 0
    rejected: int = 0
    ignored: int = 0
    duplicates: int = 0
    with_abstract: int = 0
    missing_abstract: int = 0
    query_errors: list[str] = field(default_factory=list)
    paper_ids: list[int] = field(default_factory=list)
    matched_paper_ids: list[int] = field(default_factory=list)
    inserted_paper_ids: list[int] = field(default_factory=list)
    updated_paper_ids: list[int] = field(default_factory=list)
    unchanged_paper_ids: list[int] = field(default_factory=list)
    duplicate_paper_ids: list[int] = field(default_factory=list)


def _safe_error(exc: Exception) -> str:
    return str(exc).replace("Authorization", "[redacted]")[:500]


class JournalSyncService:
    """ISSN-first multi-journal retrieval with per-journal isolation."""

    def __init__(
        self,
        database: Database,
        crossref: CrossrefClient,
        *,
        profile_id: str = "education_economics",
    ) -> None:
        self.database = database
        self.crossref = crossref
        self.profile_id = profile_id
        self.last_run_id: int | None = None

    def sync(
        self,
        journals: tuple[RegistryJournal, ...],
        *,
        since: date,
        until: date,
        per_journal_limit: int | None,
    ) -> tuple[JournalSyncResult, ...]:
        results = self.collect(
            journals, since=since, until=until, per_journal_limit=per_journal_limit,
        )
        self.last_run_id = self.record(journals, since, until, per_journal_limit, list(results))
        return results

    def collect(
        self,
        journals: tuple[RegistryJournal, ...],
        *,
        since: date,
        until: date,
        per_journal_limit: int | None,
    ) -> tuple[JournalSyncResult, ...]:
        if until < since:
            raise ValueError("End date cannot be earlier than start date")
        if per_journal_limit is not None and per_journal_limit < 1:
            raise ValueError("Per-journal limit must be positive")
        globally_seen: dict[str, int | None] = {}
        results: list[JournalSyncResult] = []
        for journal in journals:
            result = JournalSyncResult(
                journal.journal_key,
                journal.canonical_name,
                query_start_date=since.isoformat(),
                query_end_date=until.isoformat(),
                requested_limit=per_journal_limit or 0,
            )
            results.append(result)
            journal_id = self.database.upsert_journal(journal.as_journal_config())
            candidates: list[PaperMetadata] = []
            journal_seen: set[str] = set()
            duplicate_keys: list[str] = []
            for issn in journal.issns:
                remaining = per_journal_limit - len(candidates) if per_journal_limit else None
                if remaining is not None and remaining <= 0:
                    break
                try:
                    fetched = self.crossref.fetch(issn, since, remaining, until=until)
                except Exception as exc:
                    result.query_errors.append(f"{issn}: {_safe_error(exc)}")
                    LOGGER.exception(
                        "Journal retrieval failed for %s/%s",
                        journal.journal_key,
                        issn,
                    )
                    continue
                result.found += len(fetched)
                result.found_count += len(fetched)
                fetch_stats = getattr(self.crossref, "last_fetch_stats", None)
                if isinstance(fetch_stats, dict):
                    if fetch_stats.get("error_summary"):
                        detail = str(fetch_stats["error_summary"])[:500]
                        result.query_errors.append(f"{issn}: {detail}")
                    result.provider_returned_count += int(
                        fetch_stats.get("provider_returned_count") or 0
                    )
                    result.pages_requested += int(fetch_stats.get("pages_requested") or 0)
                    result.raw_candidate_count += int(
                        fetch_stats.get("raw_candidate_count") or 0
                    )
                    result.normalized_count += len(fetched)
                    result.stopped_reason = str(
                        fetch_stats.get("stopped_reason") or result.stopped_reason
                    )
                for paper in fetched:
                    if not _record_matches_journal(paper, journal, queried_issn=issn):
                        result.rejected += 1
                        result.query_errors.append(
                            f"{issn}: provider record identity could not be confirmed"
                        )
                        continue
                    key = _paper_key(paper)
                    if key in journal_seen or key in globally_seen:
                        result.duplicates += 1
                        duplicate_keys.append(key)
                        continue
                    journal_seen.add(key)
                    globally_seen[key] = None
                    _record_journal_match(paper, journal, issn)
                    if product_safety_schema_ready(
                        self.database
                    ) and paper_is_tombstoned(self.database, paper, journal_id):
                        result.ignored += 1
                        continue
                    candidates.append(paper)
                    if per_journal_limit is not None and len(candidates) >= per_journal_limit:
                        break
            for paper in candidates:
                existing_id, existing_updated_at = _existing_paper_state(
                    self.database, paper, journal_id
                )
                try:
                    paper_id, inserted = self.database.upsert_paper(paper, journal_id)
                except Exception as exc:
                    result.rejected += 1
                    result.query_errors.append(
                        f"{paper.doi or paper.title}: {_safe_error(exc)}"
                    )
                    continue
                result.paper_ids.append(paper_id)
                result.matched_paper_ids.append(paper_id)
                globally_seen[_paper_key(paper)] = paper_id
                if inserted:
                    result.inserted += 1
                    result.inserted_paper_ids.append(paper_id)
                else:
                    with self.database.connect() as connection:
                        current = connection.execute(
                            "SELECT updated_at FROM papers WHERE id=?", (paper_id,)
                        ).fetchone()
                    current_updated_at = current["updated_at"] if current else None
                    if (
                        existing_id == paper_id
                        and existing_updated_at == current_updated_at
                    ):
                        result.unchanged += 1
                        result.unchanged_paper_ids.append(paper_id)
                    else:
                        result.updated += 1
                        result.updated_paper_ids.append(paper_id)
                if paper.abstract:
                    result.with_abstract += 1
                    _record_initial_abstract_provenance(self.database, paper_id, paper)
                else:
                    result.missing_abstract += 1
            result.duplicate_paper_ids = [
                paper_id
                for key in duplicate_keys
                if (paper_id := globally_seen.get(key)) is not None
            ]
            result.paper_ids = list(dict.fromkeys(result.paper_ids))
            result.matched_paper_ids = list(dict.fromkeys(result.matched_paper_ids))
        return tuple(results)

    def record(
        self,
        journals: tuple[RegistryJournal, ...],
        since: date,
        until: date,
        per_journal_limit: int | None,
        results: list[JournalSyncResult],
        *,
        started_at: str | None = None,
    ) -> int | None:
        """Persist the precise sync set; falls back to the legacy sync_runs row."""
        matched: list[int] = []
        inserted: list[int] = []
        updated: list[int] = []
        unchanged: list[int] = []
        duplicates: list[int] = []
        rejected: list[dict[str, object]] = []
        query_errors: list[str] = []
        provider_returned = pages = raw_candidates = normalized = 0
        stopped_reason: str | None = None
        for result in results:
            matched.extend(result.matched_paper_ids or result.paper_ids or [])
            inserted.extend(result.inserted_paper_ids)
            updated.extend(result.updated_paper_ids)
            unchanged.extend(result.unchanged_paper_ids)
            duplicates.extend(result.duplicate_paper_ids)
            rejected.extend(
                {
                    "journal_key": result.journal_key,
                    "reason": error,
                }
                for error in result.query_errors
            )
            query_errors.extend(result.query_errors)
            provider_returned += result.provider_returned_count
            pages += result.pages_requested
            raw_candidates += result.raw_candidate_count
            normalized += result.normalized_count
            stopped_reason = result.stopped_reason or stopped_reason
        overall = (
            "failed"
            if not results
            else "partial"
            if any(result.query_errors for result in results)
            else "success"
        )
        now = utcnow()
        if overnight_schema_ready(self.database):
            values: dict[str, object] = {
                "started_at": started_at or now,
                "finished_at": now,
                "status": "running",
                "journals_checked": len(journals),
                "records_fetched": sum(result.found_count for result in results),
                "records_inserted": len(inserted),
                "records_updated": len(updated),
                "records_failed": len(rejected),
                "ai_calls": 0,
                "input_tokens": 0,
                "output_tokens": 0,
                "estimated_cost": 0.0,
                "error_summary": None,
                "selected_journal_keys": json.dumps(
                    [journal.journal_key for journal in journals], ensure_ascii=False
                ),
                "query_start_date": since.isoformat(),
                "query_end_date": until.isoformat(),
                "requested_limit": per_journal_limit or 0,
                "matched_paper_ids": json.dumps(list(dict.fromkeys(matched))),
                "inserted_paper_ids": json.dumps(list(dict.fromkeys(inserted))),
                "updated_paper_ids": json.dumps(list(dict.fromkeys(updated))),
                "unchanged_paper_ids": json.dumps(list(dict.fromkeys(unchanged))),
                "duplicate_paper_ids": json.dumps(list(dict.fromkeys(duplicates))),
                "rejected_candidates": json.dumps(rejected, ensure_ascii=False),
                "query_errors": json.dumps(
                    list(dict.fromkeys(query_errors)), ensure_ascii=False
                ),
                "overall_status": overall,
                "provider_returned_count": provider_returned,
                "pages_requested": pages,
                "raw_candidate_count": raw_candidates,
                "normalized_count": normalized,
                "stopped_reason": stopped_reason,
                # Written only where the profile migration is present, so an
                # un-migrated database keeps working unchanged.
                "profile_id": self.profile_id,
            }
            with self.database.connect() as connection:
                if started_at is not None:
                    connection.execute("BEGIN IMMEDIATE")
                    prior = connection.execute(
                        "SELECT id FROM sync_runs WHERE started_at=?", (started_at,)
                    ).fetchone()
                    if prior is not None:
                        return int(prior["id"])
                columns = {
                    str(row["name"])
                    for row in connection.execute("PRAGMA table_info(sync_runs)")
                }
                usable = {key: value for key, value in values.items() if key in columns}
                placeholders = ",".join("?" * len(usable))
                cursor = connection.execute(
                    f"INSERT INTO sync_runs ({','.join(usable)}) VALUES ({placeholders})",
                    tuple(usable.values()),
                )
                run_id = int(cursor.lastrowid)
                connection.execute(
                    "UPDATE sync_runs SET status=?,finished_at=?,error_summary=? WHERE id=?",
                    (overall, now, None, run_id),
                )
            return run_id
        run_id = self.database.start_sync_run()
        self.database.finish_sync_run(
            run_id,
            status=overall,
            stats={
                "journals_checked": len(journals),
                "records_fetched": sum(result.found_count for result in results),
                "records_inserted": len(inserted),
                "records_updated": len(updated),
                "records_failed": len(rejected),
                "ai_calls": 0,
                "input_tokens": 0,
                "output_tokens": 0,
                "estimated_cost": 0.0,
            },
            error="; ".join(query_errors[:10]) or None,
        )
        return run_id


#: Historical name; the class is the same object.
PersonalJournalSyncService = JournalSyncService


def _record_matches_journal(
    paper: PaperMetadata, journal: RegistryJournal, *, queried_issn: str
) -> bool:
    metadata = paper.source_metadata
    returned_issns = set()
    for value in metadata.get("ISSN") or []:
        try:
            normalized = _safe_normalize_issn(str(value))
        except ValueError:
            normalized = None
        if normalized:
            returned_issns.add(normalized)
    if returned_issns:
        return bool(returned_issns.intersection(journal.issns))
    titles = metadata.get("container-title") or []
    if titles:
        expected = {
            normalize_journal_name(value)
            for value in (journal.canonical_name, journal.short_name, *journal.aliases)
        }
        return any(normalize_journal_name(str(title)) in expected for title in titles)
    return queried_issn in journal.issns


def _record_journal_match(
    paper: PaperMetadata, journal: RegistryJournal, queried_issn: str
) -> None:
    returned_names = [str(item) for item in paper.source_metadata.get("container-title") or []]
    returned_issns = [str(item) for item in paper.source_metadata.get("ISSN") or []]
    paper.source_metadata["eduecon_journal_match"] = {
        "journal_key": journal.journal_key,
        "canonical_name": journal.canonical_name,
        "queried_issn": queried_issn,
        "returned_issns": returned_issns,
        "returned_names": returned_names,
    }


def _record_initial_abstract_provenance(
    database: Database,
    paper_id: int,
    paper: PaperMetadata,
) -> None:
    if not paper.abstract:
        return
    with database.connect() as connection:
        row = connection.execute(
            "SELECT abstract,source_metadata_json FROM papers WHERE id=?", (paper_id,)
        ).fetchone()
        if not row or row["abstract"] != paper.abstract:
            return
        metadata = json.loads(row["source_metadata_json"] or "{}")
        if metadata.get("abstract_source") == "manual":
            return
        metadata["abstract_source"] = "crossref_initial_metadata"
        metadata["abstract_provenance"] = {
            "source": "crossref_initial_metadata",
            "query_mode": "issn_journal_sync",
            "query_identifier": paper.doi,
            "matched_doi": paper.doi,
            "matched_title": paper.title,
            "confidence": 1.0,
            "content_sha256": hashlib.sha256(paper.abstract.encode("utf-8")).hexdigest(),
            "recorded_at": utcnow(),
        }
        connection.execute(
            "UPDATE papers SET source_metadata_json=? WHERE id=?",
            (json.dumps(metadata, ensure_ascii=False), paper_id),
        )


def _paper_key(paper: PaperMetadata) -> str:
    doi = normalize_doi(paper.doi)
    return f"doi:{doi}" if doi else f"title:{normalize_title(paper.title)}"


def _existing_paper_state(
    database: Database,
    paper: PaperMetadata,
    journal_id: int,
) -> tuple[int | None, str | None]:
    doi = normalize_doi(paper.doi)
    with database.connect() as connection:
        if doi:
            row = connection.execute(
                "SELECT id,updated_at FROM papers WHERE doi=?", (doi,)
            ).fetchone()
            if row:
                return int(row["id"]), row["updated_at"]
        rows = connection.execute(
            "SELECT id,title,authors_json,updated_at FROM papers WHERE journal_id=?",
            (journal_id,),
        ).fetchall()
    first_author = normalize_title(paper.authors[0]) if paper.authors else ""
    for row in rows:
        authors = json.loads(row["authors_json"] or "[]")
        existing_author = normalize_title(authors[0]) if authors else ""
        if normalize_title(row["title"]) == normalize_title(paper.title) and (
            not first_author or first_author == existing_author
        ):
            return int(row["id"]), row["updated_at"]
    return None, None


def _safe_normalize_issn(value: str) -> str | None:
    from eduecon_radar.services.journal_registry import normalize_issn

    return normalize_issn(value)
