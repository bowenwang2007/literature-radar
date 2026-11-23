from contextlib import closing
from datetime import date, timedelta
from types import SimpleNamespace

import pytest
from pydantic import BaseModel

from eduecon_radar.db import Database
from eduecon_radar.services.deepseek import DeepSeekClient, limit_batch_requests
from eduecon_radar.services.migration_integrity import migrate_database
from eduecon_radar.services.profile_registry import load_profile_registry
from eduecon_radar.services.reading_desk import (
    ReadingDesk,
    matching_topics,
    parse_topics,
    to_ris,
    tracking_window,
    update_watch,
)
from eduecon_radar.services.search_queue import SearchQueue


def test_watch_retry_resume_overlap_and_independent_scopes(tmp_path):
    db = Database(tmp_path / "library.db")
    migrate_database(db.path, "production", dry_run=False)
    queue = SearchQueue(db)
    desk = ReadingDesk(db)
    pid = "labor_economics"
    today = date(2026, 11, 22)
    journal = load_profile_registry().profile(pid).registry().by_key("labour_economics")
    desk.save_watch(pid, [journal.journal_key], ["wages"], True, today)
    watch = ReadingDesk(db).watch(pid)
    assert not desk.watch("education_economics")
    desk.save_watch(pid, [journal.journal_key], ["RCT"], True, today)
    assert desk.watch(pid)["series"] == watch["series"]

    def check(day, retry=False):
        return update_watch(db, queue, pid, watch, [journal], day, retry_partial=retry)

    first = check(today)
    assert first == check(today)
    queue.update(first, phase="done")
    assert check(today) == first
    retry = check(today, True)
    assert retry != first
    with closing(queue.connect()) as conn:
        started = conn.execute("SELECT started FROM jobs WHERE id=?", (retry,)).fetchone()[0]
    with db.connect() as conn:
        rid = conn.execute(
            "INSERT INTO sync_runs(started_at,status,profile_id,query_end_date) VALUES(?,?,?,?)",
            (started, "success", pid, today.isoformat()),
        ).lastrowid
    queue.update(retry, phase="done", run_id=rid)
    assert check(today, True) == retry
    tomorrow = today + timedelta(days=1)
    assert tracking_window(db, queue, pid, watch, tomorrow)[:2] == (
        today - timedelta(days=6),
        today,
    )
    assert check(tomorrow) != retry


def test_shelf_and_ris_keep_source_and_direction(tmp_path):
    desk = ReadingDesk(Database(tmp_path / "library.db"))
    paper = {
        "id": 7,
        "title": "The age of wages",
        "doi": "10.1000/x",
        "abstract": "中文摘要",
        "authors": [{"given": "Mary", "family": "Li"}, "Working Group"],
        "journal": "Journal",
        "publication_date": "2025-01-01",
    }
    topics = parse_topics("wages，WAGES,中文, wage")
    assert matching_topics(paper, topics) == ["wages", "中文"]
    desk.bookmark("labor", paper, True)
    desk.bookmark("labor", paper, True)
    assert desk.shelf("labor") == {7: "10.1000/x"} and desk.shelf("education") == {}
    assert "AB  - " not in to_ris([paper], include_abstract=False)
    ris = to_ris([paper])
    for literal in (
        "TI  - The age of wages",
        "AU  - Li, Mary",
        "AU  - Working Group",
        "DO  - 10.1000/x",
        "AB  - 中文摘要",
        "ER  - ",
    ):
        assert literal in ris.split("\r\n")
    desk.bookmark("labor", paper, False)
    assert not desk.shelf("labor")


def test_physical_attempt_cap_survives_worker_restart(tmp_path, monkeypatch):
    from eduecon_radar.services import search_queue

    monkeypatch.setattr(search_queue, "BATCH_REQUEST_LIMIT", 2)
    queue = SearchQueue(Database(tmp_path / "library.db"))
    calls = []

    class Schema(BaseModel):
        result: str

    def failing(**_):
        calls.append(1)
        raise TimeoutError("test transport timed out")

    client = DeepSeekClient(
        api_key="test",
        base_url="",
        max_attempts=4,
        sleeper=lambda _: None,
        client=SimpleNamespace(chat=SimpleNamespace(completions=SimpleNamespace(create=failing))),
    )
    for _ in range(2):
        with (
            limit_batch_requests(lambda queue=queue: queue._budget(1, reserve=True)),
            pytest.raises(RuntimeError),
        ):
            client.request_json(model="test", system_prompt="", user_prompt="", schema=Schema)
        queue = SearchQueue(queue.db)
    assert len(calls) == 2


def test_empty_daily_start_has_no_implicit_search_or_paid_request(tmp_path, monkeypatch):
    from pathlib import Path

    from streamlit.testing.v1 import AppTest

    db = Database(tmp_path / "clean.db")
    migrate_database(db.path, "production", dry_run=False)
    monkeypatch.setenv("DATABASE_URL", "sqlite:///" + db.path.as_posix())
    monkeypatch.setenv("DEEPSEEK_API_KEY", " ")
    monkeypatch.setenv("LITERATURE_RADAR_MODE", "local")
    monkeypatch.setattr(
        "eduecon_radar.services.product_access.local_request_allowed", lambda _host: True
    )
    app_path = Path(__file__).resolve().parents[1] / "src/eduecon_radar/ui/app.py"
    app = AppTest.from_file(str(app_path), default_timeout=60).run()
    assert not app.exception and not app.error
    assert app.segmented_control(key="edr_primary_view").value == "今日雷达"
    for section in ("近期发表", "本周新收", "待读书架"):
        app.segmented_control(key="radar_section_education_economics").set_value(section).run()
        assert not app.exception and not app.error
    assert not SearchQueue(db).path.exists()
    assert not ReadingDesk(db).path.exists()


def test_recent_online_paper_and_repeat_discovery_have_separate_dates(tmp_path):
    from eduecon_radar.schemas import PaperMetadata
    from eduecon_radar.ui.views.paper_library import radar_page, resolve_library_scope

    db = Database(tmp_path / "early-access.db")
    migrate_database(db.path, "production", dry_run=False)
    pid = "labor_economics"
    journal = load_profile_registry().profile(pid).registry().by_key("labour_economics")
    paper = PaperMetadata(
        title="Wage adjustment",
        doi="10.1000/early-online",
        authors=[],
        online_date=date(2026, 11, 15),
        issue_date=date(2027, 1, 1),
    )
    paper_id, _ = db.upsert_paper(paper, db.upsert_journal(journal.as_journal_config()))
    with db.connect() as conn:
        for stamp in ("2026-01-01", "2026-11-22"):
            conn.execute(
                "INSERT INTO sync_runs(started_at,status,profile_id,requested_limit,"
                "matched_paper_ids) VALUES(?,?,?,?,?)",
                (stamp, "success", pid, 0, f"[{paper_id}]"),
            )
    scope = resolve_library_scope(db, pid)
    recent, total = radar_page(db, scope, "近期发表", ["wage"], 5, date(2026, 11, 22))
    assert total == 1 and recent[0][0]["publication_date"] == "2026-11-15"
    assert recent[0][1] == ["wage"]
    assert radar_page(db, scope, "本周新收", [], 5, date(2026, 11, 22)) == ([], 0)
