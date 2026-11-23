from __future__ import annotations

import os
import sqlite3
import tempfile
from pathlib import Path
from urllib.parse import unquote, urlparse

import pytest

ROOT = Path(__file__).resolve().parents[1]
PRODUCTION_DATABASE = (ROOT / "data" / "eduecon_radar.db").resolve()


def _writable(directory: Path) -> bool:
    """True when a fresh file can actually be created inside ``directory``."""
    try:
        directory.mkdir(parents=True, exist_ok=True)
        probe = directory / ".write_probe"
        probe.write_text("", encoding="utf-8")
        probe.unlink()
    except OSError:
        return False
    return True


def pytest_configure(config: pytest.Config) -> None:
    """Keep pytest's temp files somewhere writable.

    Some sandboxes hand out a temp directory that can be created but not
    written to (and then fail during teardown cleanup, masking the real test
    summary).  Redirect to an explicitly configured or in-project directory
    only when the default one is genuinely unusable, so normal runs keep
    using the platform temp location.
    """
    configured = os.environ.get("EDUECON_TEST_TMPDIR")
    candidates = (
        [Path(configured)] if configured else []
    ) + [ROOT / ".test-temp" / "pytest-run", Path(tempfile.gettempdir())]

    current = Path(tempfile.gettempdir()).resolve()
    if _writable(current) and not configured:
        return
    for candidate in candidates:
        if _writable(candidate):
            tempfile.tempdir = str(candidate)
            return


def _sqlite_path(value: object) -> Path | None:
    if isinstance(value, Path):
        return value.resolve()
    if not isinstance(value, str) or value == ":memory:":
        return None
    if value.startswith("file:"):
        parsed = urlparse(value)
        raw = unquote(parsed.path)
        if len(raw) >= 3 and raw[0] == "/" and raw[2] == ":":
            raw = raw[1:]
        return Path(raw).resolve()
    return Path(value).resolve()


@pytest.fixture(autouse=True)
def forbid_production_database_connections(monkeypatch):
    """Tests may hash production files, but may never open the production SQLite DB."""
    original = sqlite3.connect

    def guarded_connect(database, *args, **kwargs):
        path = _sqlite_path(database)
        if path == PRODUCTION_DATABASE:
            pytest.fail(
                "Tests must not open the production database; use tmp_path and DATABASE_URL."
            )
        return original(database, *args, **kwargs)

    monkeypatch.setattr(sqlite3, "connect", guarded_connect)
