from __future__ import annotations

import json
import sqlite3
from collections.abc import Iterator
from contextlib import contextmanager
from datetime import UTC, datetime
from pathlib import Path

from eduecon_radar.schemas import JournalConfig, PaperMetadata
from eduecon_radar.services.deduplication import fingerprint_paper, normalize_doi, normalize_title
from eduecon_radar.services.metadata_provenance import (
    PROVENANCE_FIELDS,
    current_source_priority,
    deep_merge_metadata,
    load_source_priorities,
    merge_field,
    provenance_schema_ready,
    record_provenance,
    source_identity,
    value_is_empty,
)

SCHEMA = """
PRAGMA foreign_keys = ON;
CREATE TABLE IF NOT EXISTS journals (
 id INTEGER PRIMARY KEY, name TEXT NOT NULL, short_name TEXT NOT NULL UNIQUE,
 issn TEXT NOT NULL, eissn TEXT, rank INTEGER, rank_source TEXT, priority TEXT NOT NULL,
 enabled INTEGER NOT NULL DEFAULT 1, last_synced_at TEXT, created_at TEXT NOT NULL,
 updated_at TEXT NOT NULL
);
CREATE TABLE IF NOT EXISTS papers (
 id INTEGER PRIMARY KEY, doi TEXT UNIQUE, fingerprint TEXT NOT NULL UNIQUE,
 title TEXT NOT NULL, abstract TEXT, authors_json TEXT NOT NULL DEFAULT '[]',
 journal_id INTEGER NOT NULL REFERENCES journals(id), publisher TEXT, article_type TEXT,
 online_date TEXT, issue_date TEXT, volume TEXT, issue TEXT, pages TEXT, source_url TEXT,
 open_access_url TEXT, keywords_json TEXT NOT NULL DEFAULT '[]',
 jel_codes_json TEXT NOT NULL DEFAULT '[]', source_metadata_json TEXT NOT NULL DEFAULT '{}',
 first_seen_at TEXT NOT NULL, last_seen_at TEXT NOT NULL, created_at TEXT NOT NULL,
 updated_at TEXT NOT NULL
);
CREATE TABLE IF NOT EXISTS screenings (
 id INTEGER PRIMARY KEY, paper_id INTEGER NOT NULL REFERENCES papers(id) ON DELETE CASCADE,
 rule_score REAL NOT NULL DEFAULT 0, rule_hits_json TEXT NOT NULL DEFAULT '{}',
 ai_decision TEXT NOT NULL DEFAULT 'unprocessed', ai_confidence REAL, ai_reason TEXT,
 topics_json TEXT NOT NULL DEFAULT '[]', education_level TEXT, country_or_region TEXT,
 method TEXT, research_question TEXT, data_and_sample TEXT, identification_strategy TEXT,
 main_result TEXT, heterogeneity TEXT, policy_implication TEXT, chinese_title TEXT,
 chinese_summary TEXT, missing_information_json TEXT NOT NULL DEFAULT '[]',
 model_name TEXT NOT NULL, prompt_version TEXT NOT NULL, screened_at TEXT NOT NULL,
 UNIQUE(paper_id, model_name, prompt_version)
);
CREATE TABLE IF NOT EXISTS reviews (
 id INTEGER PRIMARY KEY, paper_id INTEGER NOT NULL UNIQUE REFERENCES papers(id) ON DELETE CASCADE,
 status TEXT NOT NULL DEFAULT 'pending', reviewer_note TEXT, edited_chinese_title TEXT,
 edited_chinese_summary TEXT, verified_numbers INTEGER NOT NULL DEFAULT 0,
 verified_causal_language INTEGER NOT NULL DEFAULT 0, reviewed_at TEXT
);
CREATE TABLE IF NOT EXISTS sync_runs (
 id INTEGER PRIMARY KEY, started_at TEXT NOT NULL, finished_at TEXT, status TEXT NOT NULL,
 journals_checked INTEGER NOT NULL DEFAULT 0, records_fetched INTEGER NOT NULL DEFAULT 0,
 records_inserted INTEGER NOT NULL DEFAULT 0, records_updated INTEGER NOT NULL DEFAULT 0,
 records_failed INTEGER NOT NULL DEFAULT 0, ai_calls INTEGER NOT NULL DEFAULT 0,
 input_tokens INTEGER NOT NULL DEFAULT 0, output_tokens INTEGER NOT NULL DEFAULT 0,
 estimated_cost REAL NOT NULL DEFAULT 0, error_summary TEXT
);
CREATE INDEX IF NOT EXISTS idx_papers_journal ON papers(journal_id);
CREATE INDEX IF NOT EXISTS idx_screenings_decision ON screenings(ai_decision);
CREATE INDEX IF NOT EXISTS idx_reviews_status ON reviews(status);
"""


def utcnow() -> str:
    return datetime.now(UTC).isoformat()


class Database:
    def __init__(self, path: Path | str) -> None:
        self.path = Path(path)

    @contextmanager
    def connect(self) -> Iterator[sqlite3.Connection]:
        self.path.parent.mkdir(parents=True, exist_ok=True)
        connection = sqlite3.connect(self.path, timeout=30)
        connection.row_factory = sqlite3.Row
        connection.execute("PRAGMA foreign_keys = ON")
        try:
            yield connection
            connection.commit()
        except Exception:
            connection.rollback()
            raise
        finally:
            connection.close()

    def init_schema(self) -> None:
        with self.connect() as connection:
            connection.executescript(SCHEMA)

    def upsert_journal(self, journal: JournalConfig) -> int:
        now = utcnow()
        with self.connect() as connection:
            connection.execute(
                """INSERT INTO journals
                (name,short_name,issn,eissn,rank,rank_source,priority,enabled,created_at,updated_at)
                VALUES (?,?,?,?,?,?,?,?,?,?)
                ON CONFLICT(short_name) DO UPDATE SET name=excluded.name,issn=excluded.issn,
                eissn=excluded.eissn,rank=excluded.rank,rank_source=excluded.rank_source,
                priority=excluded.priority,enabled=excluded.enabled,updated_at=excluded.updated_at""",
                (
                    journal.name,
                    journal.short_name,
                    journal.issn,
                    journal.eissn,
                    journal.rank,
                    journal.rank_source,
                    journal.priority,
                    int(journal.enabled),
                    now,
                    now,
                ),
            )
            row = connection.execute(
                "SELECT id FROM journals WHERE short_name=?", (journal.short_name,)
            ).fetchone()
            assert row is not None
            return int(row["id"])

    def upsert_paper(self, paper: PaperMetadata, journal_id: int) -> tuple[int, bool]:
        now = utcnow()
        doi = normalize_doi(paper.doi)
        year = (
            (paper.issue_date or paper.online_date).year
            if (paper.issue_date or paper.online_date)
            else None
        )
        fingerprint = fingerprint_paper(
            paper.title, paper.authors[0] if paper.authors else "", year, str(journal_id)
        )
        with self.connect() as connection:
            existing = (
                connection.execute("SELECT * FROM papers WHERE doi=?", (doi,)).fetchone()
                if doi
                else None
            )
            if existing is None:
                existing = connection.execute(
                    "SELECT * FROM papers WHERE fingerprint=?", (fingerprint,)
                ).fetchone()
            if existing is None:
                # A year can change between online-first and issue assignment. This conservative
                # exact-title check avoids a duplicate without fuzzy-merging unrelated records.
                candidates = connection.execute(
                    "SELECT id,title,authors_json FROM papers WHERE journal_id=?",
                    (journal_id,),
                ).fetchall()
                first_author = normalize_title(paper.authors[0]) if paper.authors else ""
                for candidate in candidates:
                    authors = json.loads(candidate["authors_json"])
                    old_author = normalize_title(authors[0]) if authors else ""
                    if normalize_title(candidate["title"]) == normalize_title(paper.title) and (
                        not first_author or first_author == old_author
                    ):
                        existing = connection.execute(
                            "SELECT * FROM papers WHERE id=?", (int(candidate["id"]),)
                        ).fetchone()
                        break
            values = (
                doi,
                fingerprint,
                paper.title,
                paper.abstract,
                json.dumps(paper.authors, ensure_ascii=False),
                journal_id,
                paper.publisher,
                paper.article_type,
                paper.online_date.isoformat() if paper.online_date else None,
                paper.issue_date.isoformat() if paper.issue_date else None,
                paper.volume,
                paper.issue,
                paper.pages,
                paper.source_url,
                paper.open_access_url,
                json.dumps(paper.keywords, ensure_ascii=False),
                json.dumps(paper.jel_codes, ensure_ascii=False),
                json.dumps(paper.source_metadata, ensure_ascii=False),
            )
            if existing:
                paper_id = int(existing["id"])
                _, priorities = load_source_priorities()
                source_name, source_record_id = source_identity(paper.source_metadata)
                incoming_priority = priorities.get(source_name, 0)
                incoming = {
                    "doi": values[0],
                    "title": values[2],
                    "abstract": values[3],
                    "authors_json": values[4],
                    "publisher": values[6],
                    "article_type": values[7],
                    "online_date": values[8],
                    "issue_date": values[9],
                    "volume": values[10],
                    "issue": values[11],
                    "pages": values[12],
                    "source_url": values[13],
                    "open_access_url": values[14],
                    "keywords_json": values[15],
                    "jel_codes_json": values[16],
                }
                has_provenance = provenance_schema_ready(connection)
                decisions = {}
                merged = {}
                for field_name in PROVENANCE_FIELDS:
                    priority = (
                        current_source_priority(connection, paper_id, field_name)
                        if has_provenance
                        else 0
                    )
                    decision = merge_field(
                        field_name,
                        existing[field_name],
                        incoming[field_name],
                        current_priority=priority,
                        incoming_priority=incoming_priority,
                    )
                    decisions[field_name] = decision
                    merged[field_name] = decision.value
                existing_metadata = json.loads(existing["source_metadata_json"] or "{}")
                merged_metadata = deep_merge_metadata(
                    existing_metadata, paper.source_metadata
                )
                title_changed = merged["title"] != existing["title"]
                abstract_changed = merged["abstract"] != existing["abstract"]
                translation_table = connection.execute(
                    "SELECT 1 FROM sqlite_master "
                    "WHERE type='table' AND name='paper_translations'"
                ).fetchone()
                if translation_table and (title_changed or abstract_changed):
                    translation = connection.execute(
                        "SELECT title_translation_source,abstract_translation_source "
                        "FROM paper_translations WHERE paper_id=?",
                        (paper_id,),
                    ).fetchone()
                    if translation and title_changed and (
                        translation["title_translation_source"] != "human_edited"
                    ):
                        connection.execute(
                            "UPDATE paper_translations SET chinese_title=NULL,"
                            "title_translation_status='pending',"
                            "title_translation_source=NULL,updated_at=? WHERE paper_id=?",
                            (now, paper_id),
                        )
                        provenance = merged_metadata.get(
                            "title_translation_provenance"
                        )
                        provenance = (
                            dict(provenance) if isinstance(provenance, dict) else {}
                        )
                        provenance.update(status="stale", stale_at=now)
                        merged_metadata["title_translation_provenance"] = provenance
                    if translation and abstract_changed and (
                        translation["abstract_translation_source"] != "human_edited"
                    ):
                        connection.execute(
                            "UPDATE paper_translations SET chinese_abstract=NULL,"
                            "abstract_translation_status='pending',"
                            "abstract_translation_source=NULL,updated_at=? WHERE paper_id=?",
                            (now, paper_id),
                        )
                        provenance = merged_metadata.get(
                            "abstract_translation_provenance"
                        )
                        provenance = (
                            dict(provenance) if isinstance(provenance, dict) else {}
                        )
                        provenance.update(status="stale", stale_at=now)
                        merged_metadata["abstract_translation_provenance"] = provenance
                merged["source_metadata_json"] = json.dumps(
                    merged_metadata, ensure_ascii=False
                )
                merged_year = merged["issue_date"] or merged["online_date"]
                merged_year_value = int(str(merged_year)[:4]) if merged_year else None
                merged_authors = json.loads(merged["authors_json"] or "[]")
                merged_fingerprint = fingerprint_paper(
                    merged["title"],
                    merged_authors[0] if merged_authors else "",
                    merged_year_value,
                    str(journal_id),
                )
                content_changed = (
                    merged_fingerprint != existing["fingerprint"]
                    or journal_id != existing["journal_id"]
                    or any(
                        merged[field_name] != existing[field_name]
                        for field_name in PROVENANCE_FIELDS
                    )
                    or merged["source_metadata_json"] != existing["source_metadata_json"]
                )
                if content_changed:
                    connection.execute(
                        """UPDATE papers SET doi=?,fingerprint=?,title=?,abstract=?,
                        authors_json=?,journal_id=?,publisher=?,article_type=?,online_date=?,
                        issue_date=?,volume=?,issue=?,pages=?,source_url=?,open_access_url=?,
                        keywords_json=?,jel_codes_json=?,source_metadata_json=?,last_seen_at=?,
                        updated_at=? WHERE id=?""",
                        (
                            merged["doi"],
                            merged_fingerprint,
                            merged["title"],
                            merged["abstract"],
                            merged["authors_json"],
                            journal_id,
                            merged["publisher"],
                            merged["article_type"],
                            merged["online_date"],
                            merged["issue_date"],
                            merged["volume"],
                            merged["issue"],
                            merged["pages"],
                            merged["source_url"],
                            merged["open_access_url"],
                            merged["keywords_json"],
                            merged["jel_codes_json"],
                            merged["source_metadata_json"],
                            now,
                            now,
                            paper_id,
                        ),
                    )
                else:
                    connection.execute(
                        "UPDATE papers SET last_seen_at=? WHERE id=?", (now, paper_id)
                    )
                if has_provenance:
                    for field_name, decision in decisions.items():
                        if decision.decision is None:
                            continue
                        record_provenance(
                            connection,
                            paper_id=paper_id,
                            field_name=field_name,
                            value=decision.value,
                            source_name=source_name,
                            source_record_id=source_record_id,
                            source_priority=incoming_priority,
                            decision=decision.decision,
                            reason=decision.reason,
                            quality_flags=decision.quality_flags,
                            is_current=decision.changed,
                        )
                return paper_id, False
            cursor = connection.execute(
                """INSERT INTO papers
                (doi,fingerprint,title,abstract,authors_json,journal_id,publisher,article_type,
                online_date,issue_date,volume,issue,pages,source_url,open_access_url,keywords_json,
                jel_codes_json,source_metadata_json,first_seen_at,last_seen_at,created_at,updated_at)
                VALUES (?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?)""",
                (*values, now, now, now, now),
            )
            paper_id = int(cursor.lastrowid)
            connection.execute(
                "INSERT INTO reviews (paper_id,status) VALUES (?,'pending')", (paper_id,)
            )
            if provenance_schema_ready(connection):
                _, priorities = load_source_priorities()
                source_name, source_record_id = source_identity(paper.source_metadata)
                priority = priorities.get(source_name, 0)
                for field_name, value in zip(
                    PROVENANCE_FIELDS,
                    (
                        doi,
                        paper.title,
                        paper.abstract,
                        json.dumps(paper.authors, ensure_ascii=False),
                        paper.publisher,
                        paper.article_type,
                        paper.online_date.isoformat() if paper.online_date else None,
                        paper.issue_date.isoformat() if paper.issue_date else None,
                        paper.volume,
                        paper.issue,
                        paper.pages,
                        paper.source_url,
                        paper.open_access_url,
                        json.dumps(paper.keywords, ensure_ascii=False),
                        json.dumps(paper.jel_codes, ensure_ascii=False),
                    ),
                    strict=True,
                ):
                    # ``_empty`` (not a bare ``is not None``) because an empty
                    # JSON container — "[]" keywords, "[]" JEL codes — is the
                    # absence of a fact.  Recording it as an accepted value made
                    # the provenance table claim a source had supplied "no JEL
                    # codes", which is not something any source asserted.
                    if not value_is_empty(value):
                        record_provenance(
                            connection,
                            paper_id=paper_id,
                            field_name=field_name,
                            value=value,
                            source_name=source_name,
                            source_record_id=source_record_id,
                            source_priority=priority,
                            decision="accepted",
                            reason="initial_insert",
                            is_current=True,
                        )
            return paper_id, True

    def start_sync_run(self) -> int:
        with self.connect() as connection:
            cursor = connection.execute(
                "INSERT INTO sync_runs (started_at,status) VALUES (?,'running')", (utcnow(),)
            )
            return int(cursor.lastrowid)

    def finish_sync_run(
        self, run_id: int, *, status: str, stats: dict, error: str | None = None
    ) -> None:
        allowed = {
            "journals_checked",
            "records_fetched",
            "records_inserted",
            "records_updated",
            "records_failed",
            "ai_calls",
            "input_tokens",
            "output_tokens",
            "estimated_cost",
        }
        fields = {key: value for key, value in stats.items() if key in allowed}
        assignments = [f"{key}=?" for key in fields]
        values = list(fields.values())
        assignments.extend(["finished_at=?", "status=?", "error_summary=?"])
        values.extend([utcnow(), status, error, run_id])
        with self.connect() as connection:
            connection.execute(
                f"UPDATE sync_runs SET {','.join(assignments)} WHERE id=?",
                values,  # noqa: S608
            )
