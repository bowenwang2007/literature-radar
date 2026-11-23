from datetime import date

from eduecon_radar.db import Database
from eduecon_radar.schemas import JournalConfig, PaperMetadata


def test_init_and_online_first_update_is_idempotent(tmp_path):
    db = Database(tmp_path / "radar.db")
    db.init_schema()
    journal_id = db.upsert_journal(
        JournalConfig(name="Test Journal", short_name="TJ", issn="0000-0000")
    )
    online = PaperMetadata(
        doi="https://doi.org/10.1234/ABC.",
        title="A paper",
        authors=["Ada Lovelace"],
        online_date=date(2025, 1, 1),
    )
    first_id, inserted = db.upsert_paper(online, journal_id)
    assert inserted is True
    issue = online.model_copy(
        update={"issue_date": date(2026, 2, 1), "volume": "10", "issue": "2", "pages": "1-20"}
    )
    second_id, inserted = db.upsert_paper(issue, journal_id)
    assert inserted is False
    assert second_id == first_id
    with db.connect() as connection:
        row = connection.execute("SELECT * FROM papers").fetchone()
        assert row["doi"] == "10.1234/abc."
        assert row["volume"] == "10"
        assert connection.execute("SELECT COUNT(*) FROM papers").fetchone()[0] == 1


def test_no_doi_online_first_year_change_does_not_duplicate(tmp_path):
    db = Database(tmp_path / "radar.db")
    db.init_schema()
    journal_id = db.upsert_journal(
        JournalConfig(name="Test Journal", short_name="TJ", issn="0000-0000")
    )
    first = PaperMetadata(title="Same Title", authors=["A Author"], online_date=date(2025, 1, 1))
    final = first.model_copy(update={"issue_date": date(2026, 1, 1), "volume": "2"})
    first_id, _ = db.upsert_paper(first, journal_id)
    second_id, inserted = db.upsert_paper(final, journal_id)
    assert inserted is False
    assert first_id == second_id


def test_ascii_case_variants_of_doi_do_not_duplicate(tmp_path):
    db = Database(tmp_path / "doi-case.db")
    db.init_schema()
    journal_id = db.upsert_journal(
        JournalConfig(name="Test Journal", short_name="TJ", issn="0000-0000")
    )
    first_id, inserted = db.upsert_paper(
        PaperMetadata(doi="10.1000/ABC", title="First title"), journal_id
    )
    assert inserted is True

    second_id, inserted = db.upsert_paper(
        PaperMetadata(doi="10.1000/abc", title="Updated title"), journal_id
    )

    assert inserted is False
    assert second_id == first_id
    with db.connect() as connection:
        assert connection.execute("SELECT COUNT(*) FROM papers").fetchone()[0] == 1
