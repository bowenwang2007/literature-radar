"""Build a shared seed from cited papers and isolate each reader's writable database."""
import json
import sqlite3
from contextlib import closing, contextmanager
from pathlib import Path
from shutil import copyfile
from tempfile import TemporaryDirectory
from threading import BoundedSemaphore

from eduecon_radar.db import Database
from eduecon_radar.schemas import JournalConfig, PaperMetadata
from eduecon_radar.services.local_pipeline import ensure_paper_statuses_for_paper_ids
from eduecon_radar.services.migration_integrity import migrate_database

PROFILES = {
    "education_economics": ("schools", "classroom", "teachers"),
    "labor_economics": ("schools", "teachers", "trade"),
}


class ReadOnlyCatalog(Database):
    @contextmanager
    def connect(self):
        uri = self.path.resolve().as_uri() + "?mode=ro&immutable=1"
        with closing(sqlite3.connect(uri, uri=True)) as conn:
            conn.row_factory = sqlite3.Row
            conn.execute("PRAGMA query_only=ON")
            yield conn


class HostedSettings:
    """Network-backed operations are unavailable in an anonymous workspace."""
    hosted = True
    deepseek_api_key = None


class WorkspaceLimit(Exception):
    pass


class WorkspacePool:
    def __init__(self, source: Path, limit: int = 32):
        self._slots = BoundedSemaphore(limit)
        self._directory = TemporaryDirectory(prefix="radar-catalog-")
        self.seed = build_collection(Path(self._directory.name), source)

    def open(self):
        if not self._slots.acquire(blocking=False):
            raise WorkspaceLimit
        try:
            directory = TemporaryDirectory(prefix="radar-reader-")
            path = Path(directory.name) / "papers.sqlite"
            copyfile(self.seed.path, path)
            return directory, Database(path)
        except BaseException:
            self._slots.release()
            raise

    def close(self, directory):
        try:
            directory.cleanup()
        finally:
            self._slots.release()


def build_collection(directory: Path, source: Path) -> ReadOnlyCatalog:
    path = directory / "collection.sqlite"
    if path.exists():
        raise FileExistsError(path)
    records = json.loads(source.read_text(encoding="utf-8-sig"))
    expected = {key for group in PROFILES.values() for key in group}
    if {r["id"] for r in records} != expected or len(records) != len(expected):
        raise ValueError("Unexpected collection membership")
    migrate_database(path, "production", dry_run=False)
    db = Database(path)
    ids = {}
    journals = {}
    for r in records:
        journal = r["journal"]
        if journal not in journals:
            acronym, issn = ("QJE", "0033-5533") if "Quarterly" in journal else ("AER", "0002-8282")
            journals[journal] = db.upsert_journal(JournalConfig(
                name=journal, short_name=acronym, issn=issn,
            ))
        url = "https://doi.org/" + r["doi"]
        paper_id, _ = db.upsert_paper(PaperMetadata(
            doi=r["doi"], title=r["title"], authors=r["authors"], source_url=url,
            abstract=r["excerpt"] or None,
            source_metadata={"abstract_source": "publisher_excerpt" if r["excerpt"] else None,
                             "reading_notes": r["notes"]},
        ), journals[journal])
        ids[r["id"]] = paper_id
        with db.connect() as conn:
            # Keep the precision supplied by the journal (year-month or a complete date).
            conn.execute("UPDATE papers SET issue_date=? WHERE id=?", (r["published"], paper_id))
            conn.execute(
                """INSERT INTO paper_translations(paper_id,chinese_title,title_translation_status,
                title_translation_source,chinese_abstract,abstract_translation_status,
                abstract_translation_source,updated_at) VALUES(?,?,'human_edited','human_edited',
                ?,?,?,CURRENT_TIMESTAMP)""",
                (paper_id, r["zh"], r["translation"] or None,
                 "human_edited" if r["translation"] else "unavailable",
                 "human_edited" if r["translation"] else None),
            )
    for profile, members in PROFILES.items():
        paper_ids = tuple(ids[k] for k in members)
        with db.connect() as conn:
            run_id = conn.execute(
                "INSERT INTO sync_runs(started_at,status,requested_limit,"
                "profile_id,matched_paper_ids) "
                "VALUES('2026-11-22T00:00:00Z','success',0,?,?)",
                (profile, json.dumps(paper_ids)),
            ).lastrowid
        ensure_paper_statuses_for_paper_ids(db, run_id, paper_ids)
    return ReadOnlyCatalog(path)
