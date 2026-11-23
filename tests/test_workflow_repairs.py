import json
from contextlib import closing
from datetime import date
from types import SimpleNamespace

from eduecon_radar.db import Database
from eduecon_radar.schemas import PaperMetadata
from eduecon_radar.services import pipeline_builder
from eduecon_radar.services.local_pipeline import (
    ensure_paper_statuses,
    resumable_paper_ids,
    save_human_decisions,
    sync_run_papers,
)
from eduecon_radar.services.migration_integrity import migrate_database
from eduecon_radar.services.profile_registry import load_profile_registry
from eduecon_radar.services.search_queue import BATCH_REQUEST_LIMIT, SearchQueue
from eduecon_radar.ui.views.daily_radar import FILTERS
from eduecon_radar.ui.views.paper_library import (
    _library_papers,
    _rows_to_markdown,
    resolve_library_scope,
)


def test_free_followup_survives_restart_with_exhausted_ai_budget(tmp_path, monkeypatch):
    db = Database(tmp_path / "library.db")
    migrate_database(db.path, "production", dry_run=False)
    profile = "education_economics"
    journal = (
        load_profile_registry().profile(profile).registry().by_key("economics_of_education_review")
    )
    jid = db.upsert_journal(journal.as_journal_config())
    pid, _ = db.upsert_paper(PaperMetadata(title="Education", doi="10.1000/test", authors=[]), jid)
    q = SearchQueue(db)
    job_id = q.submit(profile, (journal,), date.today(), date.today(), True)

    def job():
        with closing(q.connect()) as c:
            return dict(c.execute("SELECT * FROM jobs WHERE id=?", (job_id,)).fetchone())

    with db.connect() as c:
        rid = c.execute(
            "INSERT INTO sync_runs(started_at,profile_id,matched_paper_ids,status)"
            " VALUES(?,?,?,'success')",
            (job()["started"], profile, json.dumps([pid])),
        ).lastrowid
    ensure_paper_statuses(db, rid)
    save_human_decisions(db, rid, {pid: ("uncertain", "keep this note")})
    q.update(job_id, run_id=rid, phase="analysis")
    for _ in range(BATCH_REQUEST_LIMIT):
        assert q._budget(job_id, reserve=True)
    cfg = SimpleNamespace(deepseek_api_key=None)
    q.step(job(), cfg)
    assert job()["phase"] == "abstracts"

    def enrich(ids, **_):
        assert ids == (pid,)
        with db.connect() as c:
            c.execute("UPDATE papers SET abstract='New public evidence' WHERE id=?", (pid,))

    monkeypatch.setattr(
        pipeline_builder,
        "build_enrichment_service",
        lambda *_: SimpleNamespace(enrich_papers=enrich),
    )
    q = SearchQueue(db)
    q.step(job(), cfg)
    q.step(job(), cfg)
    assert job()["phase"] == "done" and "已补摘要，待分析 1 篇" in job()["progress"]
    assert pid in resumable_paper_ids(db, rid)
    p = sync_run_papers(db, rid)[0]
    assert p["human_decision"] == "uncertain" and p["reviewer_notes"] == "keep this note"
    with db.connect() as c:
        c.execute(
            "INSERT INTO paper_translations(paper_id,chinese_abstract,abstract_translation_status,"
            "abstract_translation_source,updated_at) VALUES(?,?,?,'ai_generated','t')",
            (pid, "obsolete", "failed"),
        )
    assert not sync_run_papers(db, rid)[0]["chinese_abstract"]
    assert not _library_papers(db, **FILTERS, scope=resolve_library_scope(db, profile))[0][
        "chinese_abstract"
    ]
    with db.connect() as c:
        c.execute(
            "UPDATE paper_translations SET abstract_translation_source='human_edited' "
            "WHERE paper_id=?",
            (pid,),
        )
    assert sync_run_papers(db, rid)[0]["chinese_abstract"] == "obsolete"


def test_markdown_preserves_long_notes_and_abstracts():
    note = "中文研究笔记" * 100 + "END"
    result = _rows_to_markdown([{"备注": note, "摘要": "part | one\r\npart two"}])
    assert note in result and "part \\| one part two" in result
