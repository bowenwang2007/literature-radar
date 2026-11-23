import hashlib
import sqlite3
from pathlib import Path

import pytest

from eduecon_radar.services.local_pipeline import save_human_decisions, sync_run_papers
from eduecon_radar.services.reading_desk import ReadingDesk, to_ris
from eduecon_radar.ui.collection import WorkspaceLimit, WorkspacePool
from eduecon_radar.ui.views.daily_radar import FILTERS
from eduecon_radar.ui.views.paper_library import _library_papers, resolve_library_scope

ROOT = Path(__file__).resolve().parents[1]
ENTRY = ROOT / "src/eduecon_radar/ui/app.py"


def test_public_workspaces_are_independent_and_bounded(monkeypatch, tmp_path):
    private = tmp_path / "personal.db"
    monkeypatch.setenv("DATABASE_URL", "sqlite:///" + private.as_posix())
    pool = WorkspacePool(ROOT / "config/public_papers.json", limit=2)
    digest = hashlib.sha256(pool.seed.path.read_bytes()).digest()
    with pool.seed.connect() as conn, pytest.raises(sqlite3.OperationalError, match="readonly"):
        conn.execute("UPDATE papers SET title='new'")
    first, db = pool.open()
    second, other = pool.open()
    with pytest.raises(WorkspaceLimit):
        pool.open()
    edu = "education_economics"
    try:
        paper = sync_run_papers(db, 1)[0]
        pid = paper["id"]
        save_human_decisions(db, 1, {pid: ("uncertain", "check methods")})
        ReadingDesk(db).bookmark(edu, paper, True)
        assert not ReadingDesk(other).shelf(edu)
        assert sync_run_papers(other, 1)[0]["human_decision"] == "undecided"
        assert sync_run_papers(db, 2)[0]["human_decision"] == "undecided"
        filters = dict(FILTERS, year="2001", search=paper["doi"])
        found = _library_papers(db, **filters, scope=resolve_library_scope(db, edu))
        assert found[0]["reviewer_notes"] == "check methods"
        assert "[Abstract excerpt]" in to_ris(found)
    finally:
        pool.close(first)
        pool.close(second)
    assert not db.path.exists() and not other.path.exists() and not private.exists()
    third, fresh = pool.open()
    assert sync_run_papers(fresh, 1)[0]["reviewer_notes"] is None
    pool.close(third)
    assert hashlib.sha256(pool.seed.path.read_bytes()).digest() == digest


def test_default_entry_shares_real_review_without_providers(monkeypatch):
    import httpx
    from streamlit.testing.v1 import AppTest

    from eduecon_radar.config import Settings
    from eduecon_radar.services.search_queue import SearchQueue

    def forbidden(*args, **kwargs):
        pytest.fail("The hosted workspace entered a private or network-backed operation")

    monkeypatch.delenv("LITERATURE_RADAR_MODE", raising=False)
    monkeypatch.setenv("DEEPSEEK_API_KEY", "unused-test-value")
    monkeypatch.setattr(Settings, "load", forbidden)
    monkeypatch.setattr(SearchQueue, "start", forbidden)
    monkeypatch.setattr(httpx.Client, "send", forbidden)
    app = AppTest.from_file(str(ENTRY), default_timeout=30).run()
    assert not app.exception
    app.segmented_control(key="edr_primary_view").set_value("文献工作台").run()
    assert not app.exception
    assert any(button.label == "保存并下一篇" for button in app.button)
    assert not any(button.label in {"获取论文", "重新获取"} for button in app.button)
    app.segmented_control(key="edr_primary_view").set_value("论文库").run()
    assert not app.exception


def test_first_run_publishes_once_and_keeps_existing_research(tmp_path):
    from concurrent.futures import ThreadPoolExecutor

    from eduecon_radar.services.migration_integrity import create_personal_database

    db = tmp_path / "my library" / "papers.db"

    def open_new():
        try:
            create_personal_database(db)
        except FileExistsError:
            return False
        return True

    with ThreadPoolExecutor(2) as executor:
        assert sorted(executor.map(lambda _: open_new(), range(2))) == [False, True]
    with sqlite3.connect(db) as conn:
        assert conn.execute("SELECT COUNT(*) FROM papers").fetchone()[0] == 0
    before = hashlib.sha256(db.read_bytes()).digest()
    assert not open_new()
    assert hashlib.sha256(db.read_bytes()).digest() == before


@pytest.mark.parametrize("failure", [WorkspaceLimit(), OSError("private scratch path")])
def test_visitor_can_retry_workspace_open(monkeypatch, failure):
    from streamlit.testing.v1 import AppTest

    from eduecon_radar.ui import hosted_workspace

    monkeypatch.delenv("LITERATURE_RADAR_MODE", raising=False)
    acquire = hosted_workspace.session_workspace
    attempts = 0

    def unavailable_once():
        nonlocal attempts
        attempts += 1
        if attempts == 1:
            raise failure
        return acquire()

    monkeypatch.setattr(hosted_workspace, "session_workspace", unavailable_once)
    app = AppTest.from_file(str(ENTRY), default_timeout=30).run()
    assert not app.exception
    assert all("private scratch path" not in x.value for x in app.error)
    next(b for b in app.button if b.label == "重试").click().run()
    assert not app.error and not app.exception
    assert app.segmented_control(key="edr_primary_view").value == "今日雷达"


def test_browser_static_image_is_served_with_compression(tmp_path):
    """AppTest does not fetch browser assets through the Streamlit HTTP middleware."""
    import socket
    import subprocess
    import sys
    import time

    import httpx

    with socket.socket() as probe:
        probe.bind(("127.0.0.1", 0))
        port = probe.getsockname()[1]
    process = subprocess.Popen(
        [sys.executable, "-m", "streamlit", "run", str(ENTRY), "--server.headless=true",
         "--server.address=127.0.0.1", f"--server.port={port}"],
        cwd=tmp_path, stdout=subprocess.DEVNULL, stderr=subprocess.STDOUT,
    )
    try:
        with httpx.Client(base_url=f"http://127.0.0.1:{port}", trust_env=False) as client:
            deadline = time.monotonic() + 45
            while True:
                assert process.poll() is None, "Streamlit exited before it served the page"
                try:
                    response = client.get("/favicon.png", headers={"Accept-Encoding": "gzip"})
                    break
                except httpx.ConnectError:
                    if time.monotonic() >= deadline:
                        raise
                    time.sleep(0.1)
            assert response.status_code == 200
            assert response.content.startswith(b"\x89PNG\r\n\x1a\n")
    finally:
        process.terminate()
        process.wait(timeout=15)
