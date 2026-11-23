from datetime import date

from eduecon_radar.db import Database
from eduecon_radar.journals import JournalFile
from eduecon_radar.schemas import PaperMetadata
from eduecon_radar.services.crossref import parse_crossref_item
from eduecon_radar.services.ingestion import SyncService


class FakeCrossref:
    def __init__(self, papers):
        self.papers = papers
        self.calls = 0

    def fetch(self, issn, since, limit=None):
        self.calls += 1
        return self.papers[:limit] if limit else self.papers


class ExplodingUnpaywall:
    def find_open_access_url(self, doi):
        raise AssertionError("dry-run must not call Unpaywall")


def journal_file():
    return JournalFile.model_validate(
        {
            "ranking": {"name": "test", "frozen_at": "2026-01-01"},
            "journals": [{"name": "Journal", "short_name": "J", "issn": "0000-0000"}],
        }
    )


def test_parse_crossref_missing_abstract():
    paper = parse_crossref_item(
        {
            "DOI": "10.1/x",
            "title": ["Schooling"],
            "author": [{"given": "A", "family": "Scholar"}],
            "published-online": {"date-parts": [[2026, 1, 2]]},
        }
    )
    assert paper.abstract is None
    assert paper.authors == ["A Scholar"]
    assert paper.online_date == date(2026, 1, 2)


def test_sync_is_idempotent(tmp_path):
    db = Database(tmp_path / "radar.db")
    db.init_schema()
    papers = [PaperMetadata(doi="10.1/X", title="Schooling", authors=["A Scholar"])]
    service = SyncService(db, FakeCrossref(papers))
    first = service.sync(journal_file(), since=date(2026, 1, 1))
    second = service.sync(journal_file(), since=date(2026, 1, 1))
    assert first["records_inserted"] == 1
    assert second["records_updated"] == 1
    assert first["inserted_paper_ids"] == first["affected_paper_ids"]
    assert second["updated_paper_ids"] == second["affected_paper_ids"]
    assert first["inserted_paper_ids"] == second["updated_paper_ids"]
    with db.connect() as connection:
        assert connection.execute("SELECT COUNT(*) FROM papers").fetchone()[0] == 1


def test_sync_preserves_existing_review(tmp_path):
    db = Database(tmp_path / "radar.db")
    db.init_schema()
    paper = PaperMetadata(doi="10.1/X", title="Schooling", authors=["A Scholar"])
    service = SyncService(db, FakeCrossref([paper]))
    service.sync(journal_file(), since=date(2026, 1, 1))
    with db.connect() as connection:
        connection.execute("UPDATE reviews SET edited_chinese_title='人工标题',status='approved'")
    service.sync(journal_file(), since=date(2026, 1, 1))
    with db.connect() as connection:
        row = connection.execute("SELECT * FROM reviews").fetchone()
        assert row["edited_chinese_title"] == "人工标题"
        assert row["status"] == "approved"


def test_dry_run_does_not_write_database(tmp_path):
    db = Database(tmp_path / "radar.db")
    db.init_schema()
    service = SyncService(
        db, FakeCrossref([PaperMetadata(title="School study", abstract="Education")])
    )
    stats = service.sync(journal_file(), since=date(2026, 1, 1), dry_run=True)
    assert stats["records_fetched"] == 1
    with db.connect() as connection:
        assert connection.execute("SELECT COUNT(*) FROM journals").fetchone()[0] == 0
        assert connection.execute("SELECT COUNT(*) FROM papers").fetchone()[0] == 0
        assert connection.execute("SELECT COUNT(*) FROM sync_runs").fetchone()[0] == 0


def test_dry_run_does_not_call_unpaywall(tmp_path):
    db = Database(tmp_path / "radar.db")
    db.init_schema()
    service = SyncService(
        db,
        FakeCrossref([PaperMetadata(doi="10.1/x", title="School study")]),
        ExplodingUnpaywall(),
    )
    service.sync(journal_file(), since=date(2026, 1, 1), dry_run=True)


def test_sync_does_not_change_existing_screening_or_review(tmp_path):
    db = Database(tmp_path / "radar.db")
    db.init_schema()
    paper = PaperMetadata(doi="10.1/X", title="Schooling", authors=["A Scholar"])
    service = SyncService(db, FakeCrossref([paper]))
    service.sync(journal_file(), since=date(2026, 1, 1))
    with db.connect() as connection:
        paper_id = connection.execute("SELECT id FROM papers").fetchone()["id"]
        connection.execute(
            """INSERT INTO screenings
            (paper_id,ai_decision,model_name,prompt_version,screened_at)
            VALUES (?,'include','test-model','v1','2026-07-01T00:00:00+00:00')""",
            (paper_id,),
        )
        connection.execute(
            "UPDATE reviews SET status='approved',reviewer_note='keep' WHERE paper_id=?",
            (paper_id,),
        )
        screening_before = dict(connection.execute("SELECT * FROM screenings").fetchone())
        review_before = dict(connection.execute("SELECT * FROM reviews").fetchone())

    service.sync(journal_file(), since=date(2026, 1, 1))

    with db.connect() as connection:
        screening_after = dict(connection.execute("SELECT * FROM screenings").fetchone())
        review_after = dict(connection.execute("SELECT * FROM reviews").fetchone())
    assert screening_after == screening_before
    assert review_after == review_before
