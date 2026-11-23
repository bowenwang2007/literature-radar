from datetime import date

import httpx

from eduecon_radar.config import Settings
from eduecon_radar.db import Database
from eduecon_radar.services import search_queue
from eduecon_radar.services.crossref import CrossrefClient
from eduecon_radar.services.local_pipeline import save_human_decisions, sync_run_papers
from eduecon_radar.services.migration_integrity import migrate_database
from eduecon_radar.services.profile_registry import load_profile_registry
from eduecon_radar.services.search_queue import SearchQueue


def test_same_cursor_can_advance_and_real_repeated_page_is_partial():
    calls = []

    def handler(request):
        calls.append(request.url.params["cursor"])
        start = 0 if len(calls) == 1 else 100
        items = [
            {"DOI": f"10.1000/{i}", "title": [f"Title {i}"]} for i in range(start, start + 100)
        ]
        return httpx.Response(200, json={"message": {"items": items, "next-cursor": "cursor"}})

    client = CrossrefClient(httpx.Client(transport=httpx.MockTransport(handler)))
    papers = client.fetch("0002-8282", date(2025, 1, 1))
    assert len(papers) == 200
    assert calls == ["*", "cursor", "cursor"]
    assert client.last_fetch_stats["error_summary"]


def test_resume_after_batch_commit_does_not_refetch_or_replace_review(tmp_path, monkeypatch):
    from contextlib import closing

    from eduecon_radar.schemas import PaperMetadata

    db = Database(tmp_path / "library.db")
    migrate_database(db.path, "production", dry_run=False)
    q = SearchQueue(db)
    profile = load_profile_registry().profile("labor_economics")
    journal = profile.registry().by_key("labour_economics")
    issued = date(2025, 1, 1)
    job_id = q.submit(profile.profile_id, (journal,), issued, issued, False)
    assert q.submit(profile.profile_id, (journal,), issued, issued, False) == job_id
    calls = []

    def fetch(self, issn, since, limit, *, until):
        assert limit is None
        calls.append(issn)
        return [
            PaperMetadata(
                title="Test paper",
                doi="10.1000/test",
                authors=[],
                issue_date=issued,
                source_metadata={"ISSN": [issn]},
            )
        ]

    monkeypatch.setattr(CrossrefClient, "fetch", fetch)
    attach = search_queue.ensure_paper_statuses

    def interrupt(db, run):
        (paper,) = attach(db, run)
        save_human_decisions(db, run, {paper: ("uncertain", "my note")})
        raise InterruptedError()

    q.pause(job_id)
    q.work(Settings.load())
    assert not calls
    q.restart(job_id)
    with closing(q.connect()) as c:
        job = dict(c.execute("SELECT * FROM jobs WHERE id=?", (job_id,)).fetchone())
    monkeypatch.setattr(search_queue, "ensure_paper_statuses", interrupt)
    import pytest

    with pytest.raises(InterruptedError):
        q.step(job, Settings.load())
    before = list(calls)
    monkeypatch.setattr(search_queue, "ensure_paper_statuses", attach)
    SearchQueue(db).work(Settings.load())
    result = q.latest("labor_economics")
    assert result["phase"] == "done" and before == calls
    papers = sync_run_papers(db, result["run_id"])
    assert [(p["human_decision"], p["reviewer_notes"]) for p in papers] == [
        ("uncertain", "my note")
    ]
    with db.connect() as c:
        assert c.execute("SELECT count(*) FROM sync_runs").fetchone()[0] == 1
