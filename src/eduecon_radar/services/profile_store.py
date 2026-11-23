"""Profile-scoped storage: assessments, human reviews and membership.

Semantics this module enforces:

* **Paper facts are global.**  Nothing here writes ``papers`` or
  ``paper_translations``; those are shared by every profile.
* **An assessment belongs to a profile.**  ``paper_profile_assessments`` has one
  row per (profile, paper); re-assessing under a different profile never touches
  the other profile's judgement.
* **A human review belongs to a profile.**  ``paper_profile_reviews`` mirrors
  that, so the same paper can be "related" for one profile and "irrelevant" for
  another.
* **A retrieval run is a discovery snapshot.**  ``sync_runs.profile_id`` records
  which profile asked for it, and membership of a profile is derived from those
  runs plus anything explicitly assessed or reviewed under it.  There is no
  separate membership table to keep in sync.

The legacy run-scoped columns in ``paper_processing_status`` remain readable for
historical rows (``legacy_assessment``), but the product path reads and writes
the profile tables.
"""

from __future__ import annotations

import hashlib
import json
import sqlite3
from contextlib import closing
from dataclasses import dataclass
from pathlib import Path

from eduecon_radar.db import Database, utcnow
from eduecon_radar.services.migrations import (
    DEFAULT_PROFILE_ID as DEFAULT_PROFILE_ID,
)
from eduecon_radar.services.migrations import (
    ProductSafetyMigrationRequired,
)

HUMAN_DECISIONS = ("undecided", "related", "irrelevant", "uncertain")


class ProfileStorageError(RuntimeError):
    pass


@dataclass(frozen=True)
class Assessment:
    profile_id: str
    paper_id: int
    decision: str | None
    screening_basis: str | None
    classification: dict[str, object]
    model_name: str | None
    prompt_version: str | None
    source_run_id: int | None
    assessed_at: str | None
    input_fingerprint: str | None = None

    @property
    def is_judgment(self) -> bool:
        return str(self.decision or "") in {"relevant", "irrelevant", "uncertain"}


@dataclass(frozen=True)
class Review:
    profile_id: str
    paper_id: int
    human_decision: str
    human_note: str | None
    decided_at: str | None


def profile_schema_ready(database: Database) -> bool:
    """True when migration 0013 is present."""
    if not database.path.is_file():
        return False
    uri = database.path.resolve().as_uri() + "?mode=ro"
    with closing(sqlite3.connect(uri, uri=True)) as connection:
        tables = {
            row[0]
            for row in connection.execute(
                "SELECT name FROM sqlite_master WHERE type='table' AND name IN "
                "('paper_profile_assessments','paper_profile_reviews')"
            ).fetchall()
        }
        if tables != {"paper_profile_assessments", "paper_profile_reviews"}:
            return False
        columns = {
            str(row[1]) for row in connection.execute("PRAGMA table_info(sync_runs)")
        }
    return "profile_id" in columns


def require_profile_schema(database: Database) -> None:
    if not profile_schema_ready(database):
        raise ProductSafetyMigrationRequired(
            "Profile-aware schema (migration 0013) is not installed. Back up the "
            "database, then run: eduecon-radar migrate-database --profile production "
            "--database <path> --apply"
        )


def assessment_fingerprint(
    profile_id: str, prompt_version: str, title: str, abstract: str | None
) -> str:
    """Identity of the *input* an assessment was produced from.

    A changed abstract (for example a newly enriched one) changes the
    fingerprint, which is how a title-only judgement is recognised as stale once
    the abstract arrives.
    """
    payload = json.dumps(
        {
            "profile": profile_id,
            "prompt": prompt_version,
            "title": title or "",
            "abstract": abstract or "",
        },
        ensure_ascii=False,
        sort_keys=True,
        separators=(",", ":"),
    )
    return hashlib.sha256(payload.encode("utf-8")).hexdigest()


def reflect_saved_assessment(paper: dict, fingerprint: str | None) -> None:
    """A rediscovery can already have complete, evidence-matched research in this profile."""
    processing = paper["processing"]
    if processing["status"] != "pending" or not fingerprint:
        return
    if not all(paper.get(key) for key in ("abstract", "chinese_abstract", "chinese_title")):
        return
    if processing["classification"].get("decision") not in {"relevant", "irrelevant", "uncertain"}:
        return
    current = assessment_fingerprint(
        paper["profile_id"], processing.get("prompt_version") or "", paper["title"],
        paper["abstract"],
    )
    if current == fingerprint:
        processing.update(status="classified", stage="complete")


def record_assessment(
    database: Database,
    *,
    profile_id: str,
    paper_id: int,
    classification: dict[str, object],
    model_name: str | None,
    prompt_version: str | None,
    source_run_id: int | None = None,
    input_fingerprint: str | None = None,
) -> None:
    """Upsert the profile's assessment for one paper."""
    decision = classification.get("decision")
    basis = classification.get("screening_basis")
    now = utcnow()
    with database.connect() as connection:
        connection.execute(
            """INSERT INTO paper_profile_assessments
            (profile_id,paper_id,decision,screening_basis,classification_json,
             model_name,prompt_version,input_fingerprint,source_run_id,assessed_at,updated_at)
            VALUES (?,?,?,?,?,?,?,?,?,?,?)
            ON CONFLICT(profile_id,paper_id) DO UPDATE SET
              decision=excluded.decision,
              screening_basis=excluded.screening_basis,
              classification_json=excluded.classification_json,
              model_name=excluded.model_name,
              prompt_version=excluded.prompt_version,
              input_fingerprint=excluded.input_fingerprint,
              source_run_id=excluded.source_run_id,
              assessed_at=excluded.assessed_at,
              updated_at=excluded.updated_at""",
            (
                profile_id,
                paper_id,
                str(decision) if decision else None,
                str(basis) if basis else None,
                json.dumps(classification, ensure_ascii=False),
                model_name,
                prompt_version,
                input_fingerprint,
                source_run_id,
                now,
                now,
            ),
        )


def clear_assessment(database: Database, *, profile_id: str, paper_id: int) -> None:
    with database.connect() as connection:
        connection.execute(
            "DELETE FROM paper_profile_assessments WHERE profile_id=? AND paper_id=?",
            (profile_id, paper_id),
        )


def record_human_reviews(
    database: Database,
    *,
    profile_id: str,
    drafts: dict[int, tuple[str, str | None]],
    paper_ids: tuple[int, ...] | None = None,
) -> tuple[int, ...]:
    """Save human decisions for one profile; returns the papers actually changed.

    ``paper_ids`` limits the scope to the papers the operator was looking at;
    when omitted, any paper in ``drafts`` is accepted (used by the backfill and
    by targeted library actions).
    """
    normalized: dict[int, tuple[str, str | None]] = {}
    for paper_id, (decision, note) in drafts.items():
        value = str(decision or "").strip().casefold() or "undecided"
        if value not in HUMAN_DECISIONS:
            raise ProfileStorageError(f"Unsupported human decision: {decision!r}")
        normalized[int(paper_id)] = (value, (note or "").strip() or None)
    if paper_ids is not None:
        allowed = {int(item) for item in paper_ids}
        normalized = {
            paper_id: draft
            for paper_id, draft in normalized.items()
            if paper_id in allowed
        }
    saved: list[int] = []
    now = utcnow()
    with database.connect() as connection:
        known = {
            int(row["id"])
            for row in connection.execute("SELECT id FROM papers").fetchall()
        }
        normalized = {
            paper_id: draft
            for paper_id, draft in normalized.items()
            if paper_id in known
        }
        existing = {
            int(row["paper_id"]): (str(row["human_decision"]), row["human_note"])
            for row in connection.execute(
                "SELECT paper_id,human_decision,human_note FROM paper_profile_reviews "
                "WHERE profile_id=?",
                (profile_id,),
            )
        }
        for paper_id, (decision, note) in normalized.items():
            if decision == "undecided":
                continue
            if existing.get(paper_id) == (decision, note):
                continue
            connection.execute(
                """INSERT INTO paper_profile_reviews
                (profile_id,paper_id,human_decision,human_note,decided_at,updated_at)
                VALUES (?,?,?,?,?,?)
                ON CONFLICT(profile_id,paper_id) DO UPDATE SET
                  human_decision=excluded.human_decision,
                  human_note=excluded.human_note,
                  decided_at=excluded.decided_at,
                  updated_at=excluded.updated_at""",
                (profile_id, paper_id, decision, note, now, now),
            )
            saved.append(paper_id)
    return tuple(saved)


def assessments_for(
    database: Database, profile_id: str, paper_ids: tuple[int, ...]
) -> dict[int, Assessment]:
    if not paper_ids:
        return {}
    placeholders = ",".join("?" * len(paper_ids))
    with database.connect() as connection:
        rows = connection.execute(
            f"""SELECT * FROM paper_profile_assessments
            WHERE profile_id=? AND paper_id IN ({placeholders})""",
            (profile_id, *paper_ids),
        ).fetchall()
    return {int(row["paper_id"]): _assessment(profile_id, row) for row in rows}


def reviews_for(
    database: Database, profile_id: str, paper_ids: tuple[int, ...]
) -> dict[int, Review]:
    if not paper_ids:
        return {}
    placeholders = ",".join("?" * len(paper_ids))
    with database.connect() as connection:
        rows = connection.execute(
            f"""SELECT * FROM paper_profile_reviews
            WHERE profile_id=? AND paper_id IN ({placeholders})""",
            (profile_id, *paper_ids),
        ).fetchall()
    return {
        int(row["paper_id"]): Review(
            profile_id=profile_id,
            paper_id=int(row["paper_id"]),
            human_decision=str(row["human_decision"] or "undecided"),
            human_note=row["human_note"],
            decided_at=row["decided_at"],
        )
        for row in rows
    }


def _assessment(profile_id: str, row: sqlite3.Row) -> Assessment:
    return Assessment(
        profile_id=profile_id,
        paper_id=int(row["paper_id"]),
        decision=row["decision"],
        screening_basis=row["screening_basis"],
        classification=json.loads(row["classification_json"] or "{}"),
        model_name=row["model_name"],
        prompt_version=row["prompt_version"],
        source_run_id=row["source_run_id"],
        assessed_at=row["assessed_at"],
        input_fingerprint=row["input_fingerprint"],
    )


def sync_run_profile(database: Database, run_id: int) -> str | None:
    with database.connect() as connection:
        row = connection.execute(
            "SELECT profile_id FROM sync_runs WHERE id=?", (run_id,)
        ).fetchone()
    return str(row["profile_id"]) if row and row["profile_id"] else None


#: Derived profile membership.  Takes the profile id three times.  Kept here so
#: the definition of "belongs to this profile" exists once, even though both the
#: library query and this module need it.
#:
#: A retrieval run is the discovery record, so its ``matched_paper_ids`` counts
#: even before a processing row exists for the paper; processing rows are
#: included as well so a paper attached to a run without being in the matched
#: set (verification intake) still belongs to the profile.
MEMBERSHIP_SQL = """SELECT CAST(j.value AS INTEGER) AS paper_id
          FROM sync_runs sr, json_each(sr.matched_paper_ids) j
         WHERE sr.profile_id = ? AND json_valid(sr.matched_paper_ids)
    UNION SELECT ps.paper_id FROM paper_processing_status ps
          JOIN sync_runs sr2 ON sr2.id = ps.sync_run_id
         WHERE sr2.profile_id = ?
    UNION SELECT paper_id FROM paper_profile_assessments WHERE profile_id = ?
    UNION SELECT paper_id FROM paper_profile_reviews WHERE profile_id = ?"""

#: Number of bind parameters :data:`MEMBERSHIP_SQL` expects.
MEMBERSHIP_PARAMS = 4


def profile_paper_ids(database: Database, profile_id: str) -> tuple[int, ...]:
    """Papers that belong to a profile.

    Membership is derived, never stored twice: a paper belongs to a profile when
    a retrieval run made for that profile discovered it, or when it has been
    assessed or reviewed under that profile.
    """
    with database.connect() as connection:
        rows = connection.execute(
            f"SELECT paper_id FROM ({MEMBERSHIP_SQL})",
            (profile_id,) * MEMBERSHIP_PARAMS,
        ).fetchall()
    return tuple(sorted({int(row["paper_id"]) for row in rows}))


def _legacy_runs(database: Database) -> list[dict[str, object]]:
    """Retrieval runs of a database that has no ``sync_runs.profile_id`` yet.

    An un-migrated database keeps working exactly as before: every run belongs
    to the default profile until the profile migration is applied.  A database
    that predates the overnight pipeline has no retrieval columns at all, which
    is a normal state, not an error.
    """
    try:
        with database.connect() as connection:
            rows = connection.execute(
                """SELECT id,started_at,finished_at,status,selected_journal_keys,
                query_start_date,query_end_date,requested_limit,matched_paper_ids,
                provider_returned_count,pages_requested,raw_candidate_count,
                normalized_count,stopped_reason
                FROM sync_runs WHERE requested_limit IS NOT NULL
                ORDER BY id DESC"""
            ).fetchall()
    except sqlite3.OperationalError:
        return []
    return [dict(row) for row in rows]


def latest_run_for_profile(database: Database, profile_id: str) -> dict[str, object] | None:
    runs = list_runs_for_profile(database, profile_id)
    return runs[0] if runs else None


def list_runs_for_profile(database: Database, profile_id: str) -> list[dict[str, object]]:
    try:
        with database.connect() as connection:
            rows = connection.execute(
                """SELECT id,started_at,finished_at,status,selected_journal_keys,
                query_start_date,query_end_date,requested_limit,matched_paper_ids,
                provider_returned_count,pages_requested,raw_candidate_count,
                normalized_count,stopped_reason,profile_id
                FROM sync_runs
                WHERE requested_limit IS NOT NULL AND profile_id=?
                ORDER BY id DESC""",
                (profile_id,),
            ).fetchall()
    except sqlite3.OperationalError:
        return _legacy_runs(database)
    return [dict(row) for row in rows]


def migrate_legacy_assessments(database: Database, *, profile_id: str) -> dict[str, int]:
    """Move existing run-scoped judgements into the default profile.

    Rules, in order of importance:

    * the stored ``classification_json`` is copied verbatim, so a judgement whose
      basis was never recorded stays ``historical unknown`` instead of being
      dressed up as a complete judgement;
    * human decisions and notes are copied as they are, never reinterpreted;
    * the latest run that actually produced a judgement wins;
    * running twice changes nothing.
    """
    if not profile_schema_ready(database):
        raise ProfileStorageError("Profile schema (0013) is not installed")
    assessments = 0
    reviews = 0
    with database.connect() as connection:
        rows = connection.execute(
            """SELECT paper_id,classification_json,model_name,prompt_version,
                      classified_at,sync_run_id,human_decision,human_note,human_decided_at
            FROM paper_processing_status
            ORDER BY paper_id, sync_run_id"""
        ).fetchall()
        latest: dict[int, sqlite3.Row] = {}
        for row in rows:
            classification = json.loads(row["classification_json"] or "{}")
            if classification.get("decision"):
                latest[int(row["paper_id"])] = row
        for paper_id, row in latest.items():
            classification = json.loads(row["classification_json"] or "{}")
            connection.execute(
                """INSERT INTO paper_profile_assessments
                (profile_id,paper_id,decision,screening_basis,classification_json,
                 model_name,prompt_version,input_fingerprint,source_run_id,assessed_at,updated_at)
                VALUES (?,?,?,?,?,?,?,NULL,?,?,?)
                ON CONFLICT(profile_id,paper_id) DO NOTHING""",
                (
                    profile_id,
                    paper_id,
                    classification.get("decision"),
                    classification.get("screening_basis"),
                    json.dumps(classification, ensure_ascii=False),
                    row["model_name"],
                    row["prompt_version"],
                    row["sync_run_id"],
                    row["classified_at"],
                    utcnow(),
                ),
            )
            assessments += 1
        human_rows = connection.execute(
            """SELECT paper_id,human_decision,human_note,human_decided_at
            FROM paper_processing_status
            WHERE human_decision IS NOT NULL AND human_decision <> 'undecided'
            ORDER BY paper_id, COALESCE(human_decided_at,''), sync_run_id"""
        ).fetchall()
        for row in human_rows:
            connection.execute(
                """INSERT INTO paper_profile_reviews
                (profile_id,paper_id,human_decision,human_note,decided_at,updated_at)
                VALUES (?,?,?,?,?,?)
                ON CONFLICT(profile_id,paper_id) DO NOTHING""",
                (
                    profile_id,
                    int(row["paper_id"]),
                    str(row["human_decision"]),
                    row["human_note"],
                    row["human_decided_at"],
                    utcnow(),
                ),
            )
            reviews += 1
        connection.execute(
            "UPDATE sync_runs SET profile_id=? WHERE profile_id IS NULL OR profile_id=''",
            (profile_id,),
        )
    return {
        "profile_id_assessments": assessments,
        "profile_id_reviews": reviews,
        "assessments_stored": _count(database, "paper_profile_assessments", profile_id),
        "reviews_stored": _count(database, "paper_profile_reviews", profile_id),
    }


def _count(database: Database, table: str, profile_id: str) -> int:
    with database.connect() as connection:
        row = connection.execute(
            f"SELECT COUNT(*) AS n FROM {table} WHERE profile_id=?",
            (profile_id,),
        ).fetchone()
    return int(row["n"]) if row else 0


def assessment_drift(database: Database) -> tuple[dict[str, object], ...]:
    """Rows where the run snapshot contradicts the canonical judgement.

    After the single-authority cutover the product never writes a judgement into
    ``paper_processing_status``, so a differing snapshot next to a profile
    assessment means either a historical row or a writer that bypassed the
    canonical path.  Read-only, and cheap enough to run on every check.
    """
    if not profile_schema_ready(database):
        return ()
    with database.connect() as connection:
        rows = connection.execute(
            """SELECT ps.sync_run_id, ps.paper_id, ps.classification_json,
                      pa.classification_json AS canonical_json, pa.profile_id
            FROM paper_processing_status ps
            JOIN paper_profile_assessments pa ON pa.paper_id = ps.paper_id
            JOIN sync_runs sr ON sr.id = ps.sync_run_id
            WHERE sr.profile_id = pa.profile_id
              AND ps.classification_json IS NOT NULL
              AND ps.classification_json NOT IN ('', '{}')"""
        ).fetchall()
    drift: list[dict[str, object]] = []
    for row in rows:
        try:
            snapshot = json.loads(row["classification_json"] or "{}")
            canonical = json.loads(row["canonical_json"] or "{}")
        except ValueError:
            continue
        if snapshot.get("decision") == canonical.get("decision"):
            continue
        drift.append(
            {
                "kind": "snapshot_contradicts_assessment",
                "profile_id": row["profile_id"],
                "paper_id": int(row["paper_id"]),
                "sync_run_id": int(row["sync_run_id"]),
                "snapshot_decision": snapshot.get("decision"),
                "assessment_decision": canonical.get("decision"),
            }
        )
    return tuple(drift)


def profile_coverage(database: Database, profile_id: str) -> dict[str, int]:
    """Counts that make a profile's real coverage visible."""
    with database.connect() as connection:
        assessments = connection.execute(
            """SELECT COUNT(*) AS n,
                      SUM(CASE WHEN decision IS NOT NULL THEN 1 ELSE 0 END) AS judged
                 FROM paper_profile_assessments WHERE profile_id=?""",
            (profile_id,),
        ).fetchone()
        reviews = connection.execute(
            """SELECT COUNT(*) AS n,
                      SUM(CASE WHEN human_decision <> 'undecided' THEN 1 ELSE 0 END) AS decided
                 FROM paper_profile_reviews WHERE profile_id=?""",
            (profile_id,),
        ).fetchone()
    return {
        "papers": len(profile_paper_ids(database, profile_id)),
        "assessments": int(assessments["n"] or 0),
        "judged": int(assessments["judged"] or 0),
        "reviews": int(reviews["n"] or 0),
        "decided": int(reviews["decided"] or 0),
    }


def path_has_profile_schema(database_path: Path) -> bool:
    return profile_schema_ready(Database(database_path))
