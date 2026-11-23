from __future__ import annotations

import json
import sqlite3
from dataclasses import dataclass

from eduecon_radar.db import Database, utcnow
from eduecon_radar.schemas import PaperMetadata
from eduecon_radar.services.deduplication import fingerprint_paper, normalize_doi
from eduecon_radar.services.migrations import require_product_safety_schema


@dataclass(frozen=True)
class PaperRemovalPreview:
    paper_id: int
    title: str
    doi: str | None
    has_human_data: bool
    related_rows: int


def _paper_reference_tables(connection: sqlite3.Connection) -> tuple[str, ...]:
    tables: list[str] = []
    rows = connection.execute(
        "SELECT name FROM sqlite_master WHERE type='table' AND name NOT LIKE 'sqlite_%'"
    ).fetchall()
    for row in rows:
        table = str(row[0])
        if table == "paper_tombstones":
            continue
        foreign_keys = connection.execute(f"PRAGMA foreign_key_list({table})").fetchall()
        if any(str(item[2]) == "papers" and str(item[3]) == "paper_id" for item in foreign_keys):
            tables.append(table)
    return tuple(sorted(tables))


def _has_column(
    connection: sqlite3.Connection, table: str, column: str
) -> bool:
    return any(
        str(row[1]) == column
        for row in connection.execute(f"PRAGMA table_info({table})").fetchall()
    )


def restore_paper_tombstone(
    database: Database,
    *,
    doi: str | None = None,
    fingerprint: str | None = None,
) -> bool:
    """Mark a tombstone restored so it stops suppressing future syncs.

    The archive stays in place as history; only the resync suppression rule is
    lifted after the user explicitly confirms the re-admission.
    """
    require_product_safety_schema(database)
    with database.connect() as connection:
        if not _has_column(connection, "paper_tombstones", "restored_at"):
            raise RuntimeError(
                "Phase 7C4 tombstone restore schema is not installed; "
                "apply migrations/0012_add_verification_intake.sql first."
            )
        cursor = connection.execute(
            """UPDATE paper_tombstones SET restored_at=?
            WHERE restored_at IS NULL
              AND ((? IS NOT NULL AND doi=?) OR (? IS NOT NULL AND fingerprint=?))""",
            (utcnow(), doi, doi, fingerprint, fingerprint),
        )
        return cursor.rowcount > 0


def _related_rows(
    connection: sqlite3.Connection, paper_id: int
) -> dict[str, list[dict[str, object]]]:
    return {
        table: [dict(row) for row in connection.execute(
            f"SELECT * FROM {table} WHERE paper_id=?", (paper_id,)
        ).fetchall()]
        for table in _paper_reference_tables(connection)
    }


def _has_human_data(related: dict[str, list[dict[str, object]]]) -> bool:
    for row in related.get("reviews", []):
        if (
            row.get("status") not in {None, "pending"}
            or row.get("reviewer_note")
            or row.get("edited_chinese_title")
            or row.get("edited_chinese_summary")
            or row.get("verified_numbers")
            or row.get("verified_causal_language")
        ):
            return True
    for row in related.get("paper_translations", []):
        if row.get("title_translation_source") == "human_edited" or row.get(
            "abstract_translation_source"
        ) == "human_edited":
            return True
    for row in related.get("paper_processing_status", []):
        if row.get("human_decision") not in {None, "undecided"} or row.get("human_note"):
            return True
    return bool(
        related.get("workflow_task_reviews")
        or related.get("workflow_task_review_migration_conflicts")
    )


def preview_paper_removal(database: Database, paper_id: int) -> PaperRemovalPreview:
    require_product_safety_schema(database)
    with database.connect() as connection:
        paper = connection.execute("SELECT * FROM papers WHERE id=?", (paper_id,)).fetchone()
        if paper is None:
            raise ValueError("论文不存在或已被移除")
        related = _related_rows(connection, paper_id)
    return PaperRemovalPreview(
        paper_id=paper_id,
        title=str(paper["title"]),
        doi=str(paper["doi"]) if paper["doi"] else None,
        has_human_data=_has_human_data(related),
        related_rows=sum(len(rows) for rows in related.values()),
    )


def remove_paper(database: Database, paper_id: int) -> PaperRemovalPreview:
    """Archive and remove one paper atomically; cascade deletes all dependent rows."""
    require_product_safety_schema(database)
    with database.connect() as connection:
        paper = connection.execute("SELECT * FROM papers WHERE id=?", (paper_id,)).fetchone()
        if paper is None:
            raise ValueError("论文不存在或已被移除")
        related = _related_rows(connection, paper_id)
        preview = PaperRemovalPreview(
            paper_id=paper_id,
            title=str(paper["title"]),
            doi=str(paper["doi"]) if paper["doi"] else None,
            has_human_data=_has_human_data(related),
            related_rows=sum(len(rows) for rows in related.values()),
        )
        archive = {"paper": dict(paper), "related": related, "schema_version": 1}
        connection.execute(
            """INSERT INTO paper_tombstones
            (doi,fingerprint,journal_id,title,has_human_data,archived_json,removed_at)
            VALUES (?,?,?,?,?,?,?)""",
            (
                paper["doi"],
                paper["fingerprint"],
                paper["journal_id"],
                paper["title"],
                int(preview.has_human_data),
                json.dumps(archive, ensure_ascii=False, default=str),
                utcnow(),
            ),
        )
        sync_run_ids = {
            int(row["sync_run_id"])
            for row in related.get("paper_processing_status", [])
            if row.get("sync_run_id") is not None
        }
        for sync_run_id in sync_run_ids:
            connection.execute(
                "DELETE FROM retrieval_verifications WHERE sync_run_id=?",
                (sync_run_id,),
            )
        deleted = connection.execute("DELETE FROM papers WHERE id=?", (paper_id,))
        if deleted.rowcount != 1:
            raise RuntimeError("论文移除未完成")
    return preview


def paper_is_tombstoned(
    database: Database, paper: PaperMetadata, journal_id: int
) -> bool:
    """Return whether a retrieved record was previously removed by the user."""
    require_product_safety_schema(database)
    doi = normalize_doi(paper.doi)
    year = (
        (paper.issue_date or paper.online_date).year
        if (paper.issue_date or paper.online_date)
        else None
    )
    fingerprint = fingerprint_paper(
        paper.title, paper.authors[0] if paper.authors else "", year, str(journal_id)
    )
    with database.connect() as connection:
        restored_supported = _has_column(connection, "paper_tombstones", "restored_at")
        suppressed = (
            "AND restored_at IS NULL"
            if restored_supported
            else ""
        )
        row = connection.execute(
            f"""SELECT 1 FROM paper_tombstones
            WHERE ((? IS NOT NULL AND doi=?) OR fingerprint=?) {suppressed} LIMIT 1""",
            (doi, doi, fingerprint),
        ).fetchone()
    return row is not None
